// Proposal fixture, deliberately not built/run by the implementation agent.
#include "layer26-phase-audit-v1/layer26_phase_audit.hpp.in"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sys/stat.h>
using A = strata::prefill::detail::Layer26PhaseAudit;
namespace {
std::string root;
unsigned serial = 0;
std::string output(const char* rows = "1,6") {
    std::string p = root + "/case-" + std::to_string(serial++);
    assert(::setenv("STRATA_PREFILL_PHASE_PATH", p.c_str(), 1)==0);
    assert(::setenv("STRATA_PREFILL_PHASE_ROWS", rows, 1)==0);
    return p;
}
template<class F> void rejects(F f) {
    bool threw=false;
    try { f(); } catch (const std::exception&) { threw=true; }
    assert(threw);
}
std::vector<unsigned char> bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f);
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
void full(bool low) {
    const std::string path=output();
    A audit(8, 100, 4, low);
    size_t copies=0;
    A::Copy copy=[&](void* dst,const void* src,size_t n,bool device) {
        assert(device); ++copies; std::memcpy(dst,src,n); return true;
    };
    std::vector<unsigned char> expected;
    std::vector<unsigned char> input(4*40960);
    // Preserve all FP32 bit patterns, including signed0/subnormal/Inf/NaN payloads.
    const uint32_t bits[]={0,0x80000000u,1,0x7f800000u,0xff800000u,0x7fc12345u};
    for (size_t i=0;i<input.size();i+=4) std::memcpy(input.data()+i,bits+(i/4)%6,4);
    for (int start : {0,4}) {
        for (unsigned t=0;t<A::tensor_count;++t) {
            const auto tensor=static_cast<A::Tensor>(t);
            if (!low && tensor==A::ffn_input_low) continue;
            if (tensor==A::grouping) {
                std::vector<int32_t> ids(40),slots(40),src(40),cnt(512),off(513),order;
                for (int e=0;e<512;++e) { cnt[size_t(e)]=e<10 ? 4 : 0; off[size_t(e)]=std::min(e,10)*4; }
                off[512]=40;
                for (int e=0;e<10;++e) order.push_back(e);
                for (int row=0;row<4;++row) for(int k=0;k<10;++k) {
                    const int i=row*10+k;
                    ids[size_t(i)]=k; slots[size_t(i)]=k*4+row; src[size_t(k*4+row)]=row;
                }
                assert(audit.capture_grouping(26,start,4,ids.data(),slots.data(),src.data(),cnt,off,order));
                const int row=(start ? 6 : 1)-start;
                std::array<int32_t,70> ledger{};
                for(int k=0;k<10;++k) {
                    const int32_t fields[]={k,k*4+row,row,4,k*4,k,row};
                    std::copy(fields,fields+7,ledger.begin()+k*7);
                }
                const auto* p=reinterpret_cast<const unsigned char*>(ledger.data());
                expected.insert(expected.end(),p,p+280);
            } else {
                assert(audit.capture(tensor,26,start,4,input.data(),copy));
                const size_t independent_bytes[]={5120,10240,40960,5120,5120,5120,2048,40,40,280,10240,40960};
                const size_t len=independent_bytes[t];
                assert(len==A::elements(tensor)*A::item_bytes(tensor));
                const size_t offset=size_t((start ? 6 : 1)-start)*len;
                expected.insert(expected.end(),input.begin()+offset,input.begin()+offset+len);
            }
        }
    }
    audit.complete();
    assert(expected.size()==size_t(low ? 250576 : 240336));
    assert(bytes(path+".raw")==expected);
    assert(copies==size_t(low ? 22 : 20));
    const auto text=bytes(path);
    const std::string log(text.begin(),text.end());
    assert(log.find("status=complete coverage=complete")!=std::string::npos);
    assert(log.find("memcmp=not_compared adoption=unqualified")!=std::string::npos);
    struct stat st{};
    assert(::stat(path.c_str(),&st)==0 && (st.st_mode&0777)==0600);
    assert(::stat((path+".raw").c_str(),&st)==0 && (st.st_mode&0777)==0600);
    rejects([&]{ audit.complete(); });
}
}
int main(int argc,char** argv) {
    assert(argc==2);
    const std::string name=std::string(argv[1])+"/phase-XXXXXX";
    std::vector<char> dir(name.begin(),name.end());dir.push_back(0);
    const char* created=::mkdtemp(dir.data()); assert(created); root=created;
    full(false); full(true);
    for (const char* csv : {"", "1,", "-1", "8", "1,1", "0,1,2,3,4,5,6", "1x", "9223372036854775808"}) {
        output(csv); rejects([]{ A a(8,0,4,false); });
    }
    output("0"); rejects([]{ A a(262145,0,4,false); });
    output("0"); rejects([]{ A a(8,262140,4,false); });
    output("0"); rejects([]{ A a(8,0,8193,false); });
    output("0");
    { A a(8,0,4,false); rejects([]{ A b(8,0,4,false); }); }
    // A pre-existing raw file must never be overwritten, even if log open succeeds.
    const std::string raw_collision=output("0");
    { std::ofstream f(raw_collision+".raw"); f << "sentinel"; }
    rejects([]{ A a(8,0,4,false); });
    assert(bytes(raw_collision+".raw")==std::vector<unsigned char>({'s','e','n','t','i','n','e','l'}));
    output("0");
    {
        A a(8,0,4,false);
        std::vector<unsigned char> input(4*5120);
        unsigned calls=0;
        A::Copy copy=[&](void* d,const void* s,size_t n,bool){++calls; std::memcpy(d,s,n);return true;};
        assert(a.capture(A::gdn_input,25,0,4,nullptr,{})); assert(calls==0);
        assert(a.capture(A::gdn_input,26,4,4,nullptr,{})); assert(calls==0);
        rejects([&]{ a.capture(A::gdn_out,26,0,4,input.data(),copy); });
        assert(a.capture(A::gdn_input,26,0,4,input.data(),copy));
        rejects([&]{ a.capture(A::gdn_input,26,0,4,input.data(),copy); });
        rejects([&]{ a.complete(); });
        rejects([&]{ a.capture(A::gdn_out,26,1,4,input.data(),copy); });
    }
    output("0");
    {
        auto a=std::make_unique<A>(8,0,4,false);
        std::vector<unsigned char> input(4*5120);
        void* pending=nullptr; size_t pending_bytes=0;
        A::Copy fail=[&](void* d,const void*,size_t n,bool){pending=d;pending_bytes=n;return false;};
        assert(!a->capture(A::gdn_input,26,0,4,input.data(),fail));
        // Fake retirement while the owner remains alive. Actual SYCL lease is a Main gate.
        assert(pending && pending_bytes==5120); std::memset(pending,0xa5,pending_bytes);
        rejects([&]{ a->complete(); });
    }
    // Independent malformed-group cases: reject before any invalid pointer read.
    output("0");
    {
        A a(4,0,4,false);std::vector<unsigned char> input(4*40960);
        A::Copy copy=[](void* d,const void* q,size_t n,bool){std::memcpy(d,q,n);return true;};
        for(unsigned t=0;t<unsigned(A::grouping);++t) {
            if(t==A::ffn_input_low)continue;
            assert(a.capture(static_cast<A::Tensor>(t),26,0,4,input.data(),copy));
        }
        std::vector<int32_t> ids(40),slots(40),src(40),cnt(512),off(513),order;
        for(int e=0;e<512;++e){cnt[e]=e<10?4:0;off[e]=std::min(e,10)*4;}off[512]=40;
        for(int e=0;e<10;++e)order.push_back(e);
        for(int row=0;row<4;++row)for(int k=0;k<10;++k){ids[row*10+k]=k;slots[row*10+k]=k*4+row;src[k*4+row]=row;}
        auto invalid=[&]{rejects([&]{a.capture_grouping(26,0,4,ids.data(),slots.data(),src.data(),cnt,off,order);});};
        ids[0]=-1;invalid();ids[0]=512;invalid();ids[0]=0;
        slots[0]=-1;invalid();slots[0]=40;invalid();slots[0]=0;
        cnt[0]=0;invalid();cnt[0]=5;invalid();cnt[0]=4;
        off[0]=-1;invalid();off[0]=INT32_MAX;invalid();off[0]=0;
        src[0]=-1;invalid();src[0]=0;
        ids[1]=0;slots[1]=0;invalid();ids[1]=1;slots[1]=4;
        order.erase(order.begin());invalid();order.insert(order.begin(),0);
        auto shortcnt=cnt;shortcnt.pop_back();
        rejects([&]{a.capture_grouping(26,0,4,ids.data(),slots.data(),src.data(),shortcnt,off,order);});
        rejects([&]{a.complete();});
    }
    std::cout << "CPU fixture complete; no GPU/lease/correctness qualification; artifacts=" << root << '\n';
}
