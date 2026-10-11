// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/runtime.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/publication.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
namespace mmq = strata::prefill::mmq;
namespace pub = strata::prefill::publication;
namespace {
constexpr size_t guard = 64, slots = 16;
constexpr unsigned char sentinel = 0xa5;
struct Layout { int layer, gt, dt; size_t blob; };
constexpr Layout layouts[] = {{0,18,20,2176000},{1,22,42,1510400},{2,22,20,1971200},
    {12,18,42,1715200},{17,21,20,2329600},{21,21,42,1868800},{47,23,20,2662400}};
void progress(const char* text) { std::puts(text); std::fflush(stdout); }
template<class F> void checked(const char* op, F action) noexcept {
    progress(op);
    try { action(); }
    catch (const std::exception& e) { pub::cleanup_require(false, op, [&] { return e.what(); }); }
    catch (...) { pub::cleanup_require(false, op, [] { return "nonstandard exception"; }); }
}
void require(bool ok, const char* op) { pub::require(ok, op, [] { return "probe contract mismatch"; }); }
// The byte pattern includes expert and position; the oracle never reads a GPU result or source buffer.
unsigned char pattern(size_t expert, size_t pos) {
    uint64_t x = uint64_t(pos) + UINT64_C(0x9e3779b97f4a7c15) * (expert + 1);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return static_cast<unsigned char>(x ^ (x >> 31));
}
struct Counts { uint64_t cases=0, refusals=0, individuals=0, groups=0, copy_bytes=0,
    compared=0, uploads=0, reads=0, drains=0, frees=0; } counts;
void arm(strata::core::Runtime& rt, int layer, size_t h, size_t d, size_t blob, bool tiny) {
    const size_t up = tiny ? h + 32 : h, down = tiny ? up + h + 48 : 2*h;
    require(blob >= down + d, "blob extent");
    const size_t ss = blob + 32, gs = 2*h + 48, ds = d + 80;
    const size_t sb = guard + slots*ss + guard, gb = guard + slots*gs + guard,
        db = guard + slots*ds + guard, rb = std::max({sb,gb,db});
    const size_t device_bytes = sb+gb+db;
    // Three exact-size arrays: initialized source, shared readback, retained individual output.
    const size_t host_bytes = sb+rb+gb+db;
    std::printf("ARM_PLAN layer=%d half=%zu down=%zu blob=%zu up_off=%zu down_off=%zu "
        "device_live=%zu device_limit=134217728 host_live=%zu host_limit=536870912 "
        "group_vectors=%zu group_grid=%zu individual_grid=%zu\n", layer,h,d,blob,up,down,
        device_bytes,host_bytes,(2*h+d)/16,std::min(size_t{65536},((2*h+d)/16+255)/256*256),
        std::min(size_t{262144},((2*h+d)/16+255)/256*256));
    std::fflush(stdout);
    require(device_bytes <= size_t{128}*1024*1024 && host_bytes <= size_t{512}*1024*1024,"budgets");
    std::unique_ptr<unsigned char[]> host, read, individual;
    std::array<void*,3> ptr{};
    auto drain = [&] { checked("TERMINAL_OR_OPERATION_DRAIN", [&] { rt.finish(); }); ++counts.drains; };
    auto release = [&] { for (void*& p : ptr) if (p) {
        checked("RELEASE_AFTER_TERMINAL_DRAIN", [&] { sycl::free(p,rt.context()); }); p=nullptr; ++counts.frees;
    } };
    try {
        progress("HOST_ALLOCATION_BEGIN");
        host.reset(new unsigned char[sb]); read.reset(new unsigned char[rb]);
        individual.reset(new unsigned char[gb+db]);
        std::fill_n(host.get(),sb,sentinel);
        for (size_t q=0;q<slots;++q) for (size_t i=0;i<blob;++i)
            host[guard+q*ss+i]=pattern(q,i);
        auto& queue=rt.compute();
        const size_t sizes[]={sb,gb,db};
        for (size_t i=0;i<3;++i) {
            checked("DEVICE_ALLOCATION_BEGIN", [&] { ptr[i]=sycl::aligned_alloc_device(16,sizes[i],queue); });
            require(ptr[i]!=nullptr,"allocation");
            require(sycl::get_pointer_type(ptr[i],rt.context())==sycl::usm::alloc::device &&
                strata::core::Runtime::owner(ptr[i])==&rt,"allocation ownership");
        }
        auto copy = [&](void* dst,const void* src,size_t bytes,const char* phase) {
            checked(phase,[&] { auto event=queue.memcpy(dst,src,bytes);
                progress("COPY_WAIT_BEGIN"); rt.wait(event,"MMQ byte probe copy"); });
        };
        copy(ptr[0],host.get(),sb,"INPUT_UPLOAD_BEGIN"); counts.uploads+=sb;
        auto* gu=static_cast<unsigned char*>(ptr[1])+guard;
        auto* dd=static_cast<unsigned char*>(ptr[2])+guard;
        mmq::GatherGroup g{};
        for (size_t q=0;q<slots;++q) g.blob[q]=static_cast<unsigned char*>(ptr[0])+guard+q*ss;
        auto verify = [&](bool is_down,int first,int n) {
            const size_t bytes=is_down?db:gb, stride=is_down?ds:gs, live=is_down?d:2*h;
            copy(read.get(),ptr[is_down?2:1],bytes,"OUTPUT_READBACK_BEGIN"); counts.reads+=bytes;
            size_t failures=0;
            for(size_t i=0;i<bytes;++i) {
                unsigned char expected=sentinel;
                if(i>=guard && i<bytes-guard) {
                    size_t q=(i-guard)/stride, pos=(i-guard)%stride;
                    if(q>=static_cast<size_t>(first) && q<static_cast<size_t>(n) && pos<live) {
                        size_t source=is_down?down+pos:(pos<h?pos:up+pos-h);
                        expected=pattern(q,source);
                    }
                }
                if(read[i]!=expected) { if(failures<8) std::fprintf(stderr,
                    "BYTE_FAIL layer=%d down=%d offset=%zu actual=%u expected=%u\n",layer,
                    int(is_down),i,unsigned(read[i]),unsigned(expected)); ++failures; }
            }
            counts.compared+=bytes;
            require(failures==0,"full output byte oracle");
        };
        const int bounds[][2]={{0,1},{0,7},{3,7},{0,16},{5,16}};
        for(const auto& b:bounds) {
            g.first=b[0]; g.n=b[1];
            std::printf("CASE_BEGIN layer=%d first=%d n=%d copied_experts=%d copied_bytes=%zu "
                "copied_words32=%zu copied_vectors16=%zu\n",layer,g.first,g.n,g.n-g.first,
                size_t(g.n-g.first)*(2*h+d),size_t(g.n-g.first)*(2*h+d)/4,
                size_t(g.n-g.first)*(2*h+d)/16); std::fflush(stdout);
            for(int mode=0;mode<2;++mode) {
                checked("OUTPUT_SENTINEL_SUBMISSION",[&] { queue.memset(ptr[1],sentinel,gb);
                    queue.memset(ptr[2],sentinel,db); });
                checked(mode?"GROUP_SUBMISSION_BEGIN":"INDIVIDUAL_SUBMISSION_BEGIN",[&] {
                    if(mode) require(mmq::gather_native_group(g,up,h,down,d,gu,gs,dd,ds,&queue),"valid group acceptance");
                    else for(int q=g.first;q<g.n;++q) mmq::gather_native(g.blob[q],g.blob[q]+up,h,
                        g.blob[q]+down,d,gu+size_t(q)*gs,dd+size_t(q)*ds,&queue);
                }); drain();
                verify(false,g.first,g.n);
                if(!mode) std::copy_n(read.get(),gb,individual.get());
                else require(std::equal(read.get(),read.get()+gb,individual.get()),"individual/group GU parity");
                verify(true,g.first,g.n);
                if(!mode) std::copy_n(read.get(),db,individual.get()+gb);
                else require(std::equal(read.get(),read.get()+db,individual.get()+gb),"individual/group down parity");
                counts.copy_bytes+=size_t(g.n-g.first)*(2*h+d);
                if(mode) ++counts.groups; else counts.individuals+=uint64_t(g.n-g.first);
            }
            ++counts.cases; progress("CASE_PASS");
        }
        // Live, in-bounds pointers throughout. Production rejects these before touching a queue.
        checked("REFUSAL_SENTINEL_SUBMISSION",[&] {queue.memset(ptr[1],sentinel,gb);queue.memset(ptr[2],sentinel,db);});
        for(int bad=0;bad<14;++bad) {
            g.first=0;g.n=1;
            size_t u=up,hh=h,doff=down,ddbytes=d,gstride=gs,dstride=ds;
            void* gout=gu;void* dout=dd; const auto* saved=g.blob[0];
            if(bad==0) g.first=-1; if(bad==1) g.n=0; if(bad==2) g.first=g.n;
            if(bad==3) g.n=17; if(bad==4) g.first=2;
            if(bad==5) ++u; if(bad==6) ++hh; if(bad==7) ++doff; if(bad==8) ++ddbytes;
            if(bad==9) ++gstride; if(bad==10) ++dstride;
            if(bad==11) gout=gu+1; if(bad==12) dout=dd+1; if(bad==13) g.blob[0]=saved+1;
            checked("REFUSAL_CALL_NO_SUBMISSION_EXPECTED",[&] { require(!mmq::gather_native_group(
                g,u,hh,doff,ddbytes,gout,gstride,dout,dstride,&queue),"invalid group refused"); });
            g.blob[0]=saved; ++counts.refusals;
        }
        drain();verify(false,0,0);verify(true,0,0);
        copy(read.get(),ptr[0],sb,"FULL_INPUT_READBACK_BEGIN");counts.reads+=sb;
        require(std::equal(read.get(),read.get()+sb,host.get()),"full input unchanged");counts.compared+=sb;
        drain();release();progress("ARM_PASS");
    } catch (...) { drain();release();throw; }
}
}
int main(int argc,char**) {
    if(argc!=1) { std::fputs("No arguments accepted\n",stderr);return 2; }
    strata::core::Runtime* rt=nullptr;
    checked("RUNTIME_INITIALIZATION",[&] {rt=&strata::core::Runtime::get();});
    try {
        auto& q=rt->compute();
        require(rt->device().is_gpu() && rt->device().has(sycl::aspect::usm_device_allocations) &&
            q.has_property<sycl::property::queue::in_order>() && q.get_context()==rt->context() &&
            q.get_device()==rt->device() && strata::core::Runtime::owner_of_stream(&q)==rt,"checked compute queue");
        arm(*rt,-1,48,80,256,true);
        for(const auto& l:layouts) {
            const size_t h=mmq::matrix_bytes(l.gt,640,2560),d=mmq::matrix_bytes(l.dt,2560,640);
            require(2*h+d==l.blob,"pinned native blob extent");
            require((2*h+d)/16>65536,"real group grid-stride coverage");
            arm(*rt,l.layer,h,d,l.blob,false);
        }
        checked("FINAL_TERMINAL_DRAIN",[&] {rt->finish();});++counts.drains;
        require(counts.cases==40 && counts.refusals==112 && counts.individuals==312 &&
            counts.groups==40 && counts.drains==97 && counts.frees==24,"finite receipt counts");
        std::printf("PASS arms=8 cases=%llu refusals=%llu individual_calls=%llu grouped_calls=%llu "
            "copied_bytes=%llu compared_bytes=%llu uploaded_bytes=%llu readback_bytes=%llu drains=%llu frees=%llu\n",
            (unsigned long long)counts.cases,(unsigned long long)counts.refusals,
            (unsigned long long)counts.individuals,(unsigned long long)counts.groups,
            (unsigned long long)counts.copy_bytes,(unsigned long long)counts.compared,
            (unsigned long long)counts.uploads,(unsigned long long)counts.reads,
            (unsigned long long)counts.drains,(unsigned long long)counts.frees);
        std::fflush(stdout);std::_Exit(0);
    } catch(const std::exception& e) {
        checked("FAILURE_TERMINAL_DRAIN",[&] {rt->finish();});
        std::fprintf(stderr,"FAIL %s\n",e.what());std::fflush(stderr);std::_Exit(1);
    } catch(...) {
        checked("FAILURE_TERMINAL_DRAIN",[&] {rt->finish();});
        std::fputs("FAIL nonstandard exception\n",stderr);std::fflush(stderr);std::_Exit(1);
    }
}
