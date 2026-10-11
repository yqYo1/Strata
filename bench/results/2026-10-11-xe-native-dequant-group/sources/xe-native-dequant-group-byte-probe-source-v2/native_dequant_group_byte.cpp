// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// SOURCE ONLY. No build, runtime, correctness or performance qualification.
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_kernels.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <vector>
namespace k = strata::kernels;
namespace {
constexpr size_t guard=64, slots=16, MiB=1024*1024;
constexpr unsigned char fill=0xa5;
constexpr int types[]={16,17,18,20,21,22,29,23,11,42,12,13,7,8,6,2,3,14};
uint64_t cases=0, refusals=0, reads=0, readbytes=0, individuals=0, groups=0;
void progress(const char* s) {std::puts(s);std::fflush(stdout);}
[[noreturn]] void fatal(const char* phase,const char* why) noexcept {
    std::fprintf(stderr,"FAIL phase=%s reason=%s completion_unconfirmed=1 release_skipped=1\n",phase,why);
    std::fflush(stderr);std::_Exit(74);
}
void require(bool b,const char* why) {if(!b) throw std::runtime_error(why);}
template<class F> void checked(const char* phase,F f) noexcept {
    progress(phase);
    try {f();} catch(const std::exception& e){fatal(phase,e.what());}catch(...){fatal(phase,"nonstandard exception");}
}
uint64_t mix(uint64_t x) {x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
// FNV-1a-64 is an identity aid, not a replacement for the full byte comparison.
uint64_t hash(const unsigned char* p,size_t n) {uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;++i)h=(h^p[i])*UINT64_C(1099511628211);return h;}
void put16(unsigned char* p,uint16_t x) {p[0]=static_cast<unsigned char>(x);p[1]=static_cast<unsigned char>(x>>8);}
struct Format {size_t block, scale; int halves; bool packed;};
#define F(T) Format{sizeof(T),offsetof(T,d),1,false}
Format format(int t) {
    switch(t) {
        case 16:return F(block_iq2_xxs);case 17:return F(block_iq2_xs);case 18:return F(block_iq3_xxs);
        case 20:return F(block_iq4_nl);case 21:return F(block_iq3_s);case 22:return F(block_iq2_s);
        case 23:return F(block_iq4_xs);case 11:return F(block_q3_K);case 42:return F(block_q2_0);
        case 12:return {sizeof(block_q4_K),0,2,false};case 13:return {sizeof(block_q5_K),0,2,false};
        case 7:return {sizeof(block_q5_1),0,2,false};case 3:return {sizeof(block_q4_1),0,2,false};
        case 8:return F(block_q8_0);case 6:return F(block_q5_0);case 2:return F(block_q4_0);case 14:return F(block_q6_K);
        case 29:return {sizeof(block_iq1_m),offsetof(block_iq1_m,scales),0,true};
        default:throw std::runtime_error("format");
    }
}
#undef F
// All code/index bits fit the native decoder's fixed-width integer fields. Scales
// are independently rewritten; random exponent bits must never enter an fp16 scale.
// Main finite scales are small enough for finite decoded fp16 with all integer codes.
// First blocks explicitly exercise signed zero, subnormal and normal boundaries,
// largest finite (may overflow output), infinity and NaN. Output finiteness is NOT
// assumed for that last edge set; full bytes still must match the old API.
void native(unsigned char* p,size_t bytes,int t,int expert,int projection) {
    Format f=format(t);require(bytes%f.block==0,"native block extent");
    constexpr uint16_t edges[]={0x0000,0x8000,0x0001,0x03ff,0x0400,0x8400,0x7bff,0xfbff,0x7c00,0xfc00,0x7e01};
    for(size_t b=0;b<bytes/f.block;++b) {
        auto* q=p+b*f.block;
        uint64_t seed=mix((uint64_t(expert+1)<<48)^(uint64_t(projection+1)<<40)^b^uint64_t(t));
        for(size_t j=0;j<f.block;++j)q[j]=static_cast<unsigned char>(mix(seed+j));
        uint16_t scale=b<std::size(edges)?edges[b]:static_cast<uint16_t>(0x1800|(seed&0x3ff)|((seed>>12&1)<<15));
        if(f.packed) {
            for(int j=0;j<4;++j) {
                uint16_t v=uint16_t(q[f.scale+2*j])|(uint16_t(q[f.scale+2*j+1])<<8);
                v=uint16_t((v&0x0fff)|(((scale>>(4*j))&15)<<12));put16(q+f.scale+2*j,v);
            }
        } else for(int j=0;j<f.halves;++j) {
            uint16_t s=j?static_cast<uint16_t>(scale^0x8000):scale;
            put16(q+f.scale+2*j,s);
        }
    }
}
void compare(const std::vector<unsigned char>& got,const std::vector<unsigned char>& want,const char* label) {
    require(got.size()==want.size(),"compare extent");size_t mismatches=0;
    for(size_t i=0;i<got.size();++i) if(got[i]!=want[i]) {
        std::fprintf(stderr,"DIFF label=%s exact_byte_offset=%zu got=%u reference=%u\n",label,i,unsigned(got[i]),unsigned(want[i]));
        if(++mismatches==1024) {std::fflush(stderr);throw std::runtime_error("1024 exact offsets recorded; difference budget reached, incomplete comparison");}
    }
    std::printf("BYTE_COMPARE label=%s bytes=%zu input_independent=1 reference_fnv1a64=%016llx output_fnv1a64=%016llx differences=%zu\n",label,got.size(),(unsigned long long)hash(want.data(),want.size()),(unsigned long long)hash(got.data(),got.size()),mismatches);std::fflush(stdout);
    require(!mismatches,"byte mismatch");
}
void empty_without_runtime() {
    progress("EMPTY_BEFORE_RUNTIME_BEGIN");
    for(int first=0;first<=16;++first) {
        k::IqDequantGroup g{};g.first=first;
        // Intentionally invalid stream: accepted empty calls must not inspect it.
        for(bool gu:{false,true}) require(gu?k::iq_dequant_gu_group_f16(-1,g,-1,-1,nullptr,0,0,reinterpret_cast<void*>(uintptr_t{1})):k::iq_dequant_down_group_f16(-1,g,-1,-1,nullptr,0,0,reinterpret_cast<void*>(uintptr_t{1})),"empty before Runtime");
    }
    progress("EMPTY_BEFORE_RUNTIME_PASS");
}
void arm(strata::core::Runtime& rt,int type,int64_t ff,int64_t emb,bool gu,bool explicit_stream) {
    auto& q=rt.compute();void* stream=explicit_stream?&q:nullptr;
    size_t matrix=size_t(ff)*size_t(emb),values=matrix*(gu?2:1),live=2*values;
    size_t native_bytes=k::iq_row_bytes(type,int64_t(matrix));
    size_t ss=(native_bytes+15)/16*16+32, projections=gu?2:1;
    size_t sb=2*guard+slots*projections*ss, maxstride=values+24;
    size_t ob=2*guard+slots*maxstride*2, rb=slots*live;
    const size_t device_budget=sb+ob,host_budget=2*sb+3*ob+rb;
    require(device_budget<=512*MiB && host_budget<=768*MiB,"payload budgets");
    std::printf("ARM_BEGIN type=%d ff=%lld emb=%lld gu=%d explicit_stream=%d device_payload=%zu host_upper_bound=%zu\n",type,(long long)ff,(long long)emb,int(gu),int(explicit_stream),device_budget,host_budget);std::fflush(stdout);
    std::vector<unsigned char> input(sb,fill),source_read(sb),reference(rb),read(ob),expected(ob,fill),initial(ob,fill);
    for(size_t e=0;e<slots;++e)for(size_t p=0;p<projections;++p)native(input.data()+guard+(e*projections+p)*ss,native_bytes,type,int(e),int(p)+(gu?0:2));
    std::printf("INPUT bytes=%zu fnv1a64=%016llx\n",sb,(unsigned long long)hash(input.data(),sb));std::fflush(stdout);
    void* src=nullptr;void* out=nullptr;
    checked("ALLOCATE_DEVICE",[&]{src=sycl::aligned_alloc_device(16,sb,q);out=sycl::aligned_alloc_device(16,ob,q);});
    require(src&&out,"allocation");
    for(void* p:{src,out})require(sycl::get_pointer_type(p,rt.context())==sycl::usm::alloc::device && strata::core::Runtime::owner(p)==&rt,"USM provenance");
    auto drain=[&]{checked("CHECKED_RUNTIME_DRAIN",[&]{rt.finish();});};
    auto copy=[&](void* dst,const void* from,size_t bytes){checked("CHECKED_COPY",[&]{auto e=q.memcpy(dst,from,bytes);rt.wait(e,"native group byte copy");});};
    auto readback=[&](std::vector<unsigned char>& dest,void* from){require(readbytes+dest.size()<=UINT64_C(1099511627776),"1TiB finite readback budget");drain();copy(dest.data(),from,dest.size());++reads;readbytes+=dest.size();};
    copy(src,input.data(),sb);drain();
    auto* y=reinterpret_cast<uint16_t*>(static_cast<unsigned char*>(out)+guard);
    k::IqDequantGroup base{};
    for(size_t e=0;e<slots;++e) {
        const void* p=static_cast<unsigned char*>(src)+guard+e*projections*ss;
        base.gate[e]=p;base.down[e]=p;base.up[e]=static_cast<unsigned char*>(src)+guard+(e*projections+1)*ss;
        base.gate_bytes[e]=base.down_bytes[e]=native_bytes;base.up_bytes[e]=native_bytes;
    }
    // Retain a dense independently computed reference for every distinct source.
    // Individual and group use the same actual device sources, never a host decoder.
    copy(out,initial.data(),ob);drain();
    checked("UNCHANGED_INDIVIDUAL_REFERENCE_SUBMIT",[&]{for(int e=0;e<16;++e){
        if(gu)k::iq_dequant_gu_f16(type,base.gate[e],base.up[e],ff,emb,y+size_t(e)*values,stream);
        else k::iq_dequant_f16(type,base.down[e],int64_t(matrix),y+size_t(e)*values,stream);++individuals;
    }});drain();readback(read,out);
    std::copy_n(read.data()+guard,rb,reference.data());
    // Individual guard check; all rb live bytes are saved without finiteness assumptions.
    require(std::all_of(read.begin(),read.begin()+guard,[](auto b){return b==fill;})&&std::all_of(read.begin()+guard+rb,read.end(),[](auto b){return b==fill;}),"individual guards");
    auto call=[&](int t,const k::IqDequantGroup& g,int64_t f,int64_t em,uint16_t* dst,size_t stride,size_t capacity) {
        return gu?k::iq_dequant_gu_group_f16(t,g,f,em,dst,stride,capacity,stream):k::iq_dequant_down_group_f16(t,g,f,em,dst,stride,capacity,stream);
    };
    for(int n=1;n<=16;++n)for(int padded=0;padded<2;++padded)for(int reused=0;reused<2;++reused) {
        // Every n gets both zero first and maximal legal nonzero first (when n<16).
        for(int first:{0,16-n}) {
            if(first==0 && n==16) { /* duplicate is deliberate and counted */ }
            auto g=base;g.first=first;g.n=n;size_t stride=values+(padded?24:0);
            std::fill(expected.begin(),expected.end(),fill);
            for(int e=first;e<first+n;++e) {
                int from=reused?(e%3):e;
                g.gate[e]=base.gate[from];g.up[e]=base.up[from];g.down[e]=base.down[from];
                std::copy_n(reference.data()+size_t(from)*live,live,expected.data()+guard+size_t(e)*stride*2);
            }
            // Capacity exactly ends at last live byte or includes all padded slots/guards.
            size_t capacity=padded?ob-2*guard:(size_t(first+n-1)*stride+values)*2;
            std::printf("CASE_BEGIN type=%d gu=%d n=%d first=%d padded=%d reused=%d stride_elements=%zu declared_capacity=%zu\n",type,int(gu),n,first,padded,reused,stride,capacity);std::fflush(stdout);
            copy(out,initial.data(),ob);drain();
            checked("GROUP_SUBMIT",[&]{require(call(type,g,ff,emb,y,stride,capacity),"legal group refused");});++groups;
            // A true return is not completion; both null/explicit paths drain here.
            drain();readback(read,out);compare(read,expected,"group_full_output_padding_inactive_guards");++cases;progress("CASE_PASS");
        }
    }
    // All metadata below uses real allocations except the separately identified
    // overflow/refusal-only values. Every mutation fails a check BEFORE queue_for.
    copy(out,initial.data(),ob);drain();
    for(int bad=0;bad<26;++bad) {
        auto g=base;g.first=0;g.n=1;int t=type;int64_t f=ff,em=emb;
        auto* dst=y;size_t stride=values,cap=ob-2*guard;
        switch(bad) {
            case 0:g.first=-1;break;case 1:g.n=-1;break;case 2:g.first=16;break;case 3:g.n=17;break;
            case 4:t=30;break;case 5:f=0;break;case 6:f=-1;break;case 7:em=0;break;case 8:em=257;break;
            case 9:dst=nullptr;break;case 10:dst=reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(y)+2);break;
            case 11:stride=values-8;break;case 12:stride=values+1;break;case 13:cap=live-1;break;
            case 14:g.gate[0]=g.down[0]=nullptr;break;
            case 15:g.gate[0]=g.down[0]=static_cast<unsigned char*>(src)+guard+1;break;
            case 16:g.gate_bytes[0]=g.down_bytes[0]=native_bytes-1;break;
            case 17:g.gate[0]=g.down[0]=y;break;
            case 18:if(gu)g.up[0]=nullptr;else g.down_bytes[0]=0;break;
            case 19:if(gu)g.up_bytes[0]=native_bytes-1;else stride=0;break;
            // Refusal-only overflow metadata; no fake pointer is ever admitted.
            case 20:f=std::numeric_limits<int64_t>::max();break;
            case 21:g.first=15;stride=std::numeric_limits<size_t>::max()-7;break;
            case 22:if(gu)g.up[0]=static_cast<unsigned char*>(src)+guard+1;else em=-256;break;
            case 23:if(gu)g.up[0]=y;else cap=0;break;
            case 24:g.first=17;g.n=0;break;
            case 25:g.first=1;g.n=16;break;
        }
        std::printf("REFUSAL_BEGIN case=%d statically_prequeue=1 fake_dimension_only=%d\n",bad,bad==20||bad==21);std::fflush(stdout);
        checked("REFUSAL_CALL",[&]{require(!call(t,g,f,em,dst,stride,cap),"refusal unexpectedly admitted");});++refusals;
    }
    drain();readback(read,out);compare(read,initial,"refusal_guard_preservation");
    readback(source_read,src);compare(source_read,input,"full_source_and_source_guards_unchanged");
    drain();checked("FREE_AFTER_CHECKED_TERMINAL_DRAIN",[&]{sycl::free(out,rt.context());sycl::free(src,rt.context());});progress("ARM_PASS");
}
}
int main(int argc,char** argv) {
    const bool smoke=argc==2 && std::strcmp(argv[1],"--smoke")==0;
    const bool small=argc==2 && std::strcmp(argv[1],"--small")==0;
    const bool full=argc==2 && std::strcmp(argv[1],"--full")==0;
    if(!smoke&&!small&&!full){std::fputs("Use --smoke, --small or --full\n",stderr);return 2;}
    // No Runtime initialization before the n==0 API qualification.
    checked("EMPTY_NO_RUNTIME",[]{empty_without_runtime();});
    strata::core::Runtime* rt=nullptr;
    checked("RUNTIME_INITIALIZATION",[&]{rt=&strata::core::Runtime::get();});
    checked("ALL_ARMS",[&]{
        auto& q=rt->compute();require(rt->device().is_gpu()&&rt->device().has(sycl::aspect::usm_device_allocations)&&q.has_property<sycl::property::queue::in_order>()&&q.get_context()==rt->context()&&q.get_device()==rt->device()&&strata::core::Runtime::owner_of_stream(&q)==rt,"owned in-order Runtime GPU stream");
        if(smoke) arm(*rt,18,3,768,true,true);
        else for(int t:types)for(bool explicit_stream:{false,true})for(bool gu:{false,true}) {
            if(full) arm(*rt,t,640,2560,gu,explicit_stream);
            arm(*rt,t,3,768,gu,explicit_stream); // 9 superblocks: partial eight-block launch tail
        }
        checked("FINAL_TERMINAL_DRAIN",[&]{rt->finish();});
        const unsigned expected_arms=smoke?1:(full?144:72);
        require(cases==expected_arms*128 && groups==cases && individuals==expected_arms*16 && refusals==expected_arms*26,"finite case counts");
        std::printf("PASS arms=%u cases=%llu individual_calls=%llu group_calls=%llu refusals=%llu readbacks=%llu read_bytes=%llu tracing_qualified=0 timing_qualified=0\n",expected_arms,(unsigned long long)cases,(unsigned long long)individuals,(unsigned long long)groups,(unsigned long long)refusals,(unsigned long long)reads,(unsigned long long)readbytes);std::fflush(stdout);
    });std::_Exit(0);
}
