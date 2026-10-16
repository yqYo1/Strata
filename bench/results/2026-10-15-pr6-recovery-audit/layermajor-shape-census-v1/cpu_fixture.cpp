// SOURCE ONLY; Main owns compilation, ASan/UBSan and execution.
#include "layermajor_shape_census.hpp.in"
#include <cassert>
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <climits>
using C = strata::prefill::detail::LayerMajorShapeCensus;
static C::Format fmt(int l) { return {l%3,l%5,1024,256,640*1024,1280*1024,1280*1024+2560*256}; }
static std::array<int32_t,512> residency() {
    std::array<int32_t,512> r{}; for(int e=0;e<512;++e)r[e]=e; return r;
}
struct Ref { std::map<int,uint64_t> hist; uint64_t calls=0, rows=0, chunks=0; };
// Independent oracle derives routes from token IDs rather than observer bins.
static std::array<int32_t,512> routes(int64_t t,int layer,int chunk_index) {
    std::array<int32_t,512> cnt{};
    for(int64_t row=0;row<t;++row) for(int k=0;k<10;++k)
        ++cnt[size_t((row*17+k*31+layer*7+chunk_index*13)%512)];
    return cnt;
}
static void feed(C& c,int layer,int64_t n,int64_t cap,Ref* reference=nullptr) {
    auto res=residency(); c.begin(layer,fmt(layer),res.data(),true);
    int chunk_index=0;
    for(int64_t off=0;off<n;off+=cap,++chunk_index) {
        const int64_t t=std::min(cap,n-off); auto cnt=routes(t,layer,chunk_index);
        if(reference) {
            ++reference->chunks;
            for(int e=0;e<512;++e) {
                ++reference->hist[cnt[e]];
                if(cnt[e]) {++reference->calls;reference->rows+=cnt[e];}
            }
        }
        c.counts(layer,t,cnt.data(),cnt.size());
        // Reverse order deliberately differs from oracle, histogram is a multiset.
        for(int e=511;e>=0;--e) if(cnt[e]) {
            c.issue(true,layer,cnt[e],e,e,-1);
            c.issue(false,layer,cnt[e],e,e,-1);
        }
    }
    c.end(layer);
}
static void silence(C& c) { c.finalized=true; }
static std::string output(C& c) {
    FILE* f=std::tmpfile(); assert(f); c.complete(f);
    assert(std::fseek(f,0,SEEK_END)==0); const long n=std::ftell(f); assert(n>=0);
    assert(size_t(n)<=C::output_cap); std::rewind(f); std::string s(size_t(n),'\0');
    assert(std::fread(s.data(),1,s.size(),f)==s.size()); std::fclose(f);return s;
}
int main() {
    // Actual C++ header, ordinary/tail chunks, per-layer format pairs and exact-ne oracle.
    {
        auto c=std::make_unique<C>(8193,13,8192);
        for(int l=0;l<48;++l) {
            Ref r;feed(*c,l,8193,8192,&r);assert(!c->failure);
            const auto& a=c->record[size_t(l)];
            for(int ne=0;ne<C::bins;++ne) assert(a.hist[size_t(ne)]==r.hist[ne]);
            assert(a.gu_calls==r.calls&&a.down_calls==r.calls);
            assert(a.gu_rows==r.rows&&a.down_rows==r.rows&&a.chunks==r.chunks);
            assert(a.format.gu_type==l%3&&a.format.d_type==l%5);
        }
        auto s=output(*c);assert(!c->failure);
        assert(s.find("\"complete\":true")!=std::string::npos);
        assert(s.find("\"layer\":47")!=std::string::npos);
        assert(s.find("\"gu_logical_nk\":[1280,2560]")!=std::string::npos);
    }
    // Full physical context boundary and nonzero position validated with int64.
    {
        auto c=std::make_unique<C>(262144,0,8192);
        feed(*c,0,262144,8192); assert(!c->failure);
        assert(c->record[0].gu_rows==262144ull*10);
        silence(*c); // Deliberately only one layer; no request completion assertion.
        auto bad=std::make_unique<C>(262144,1,8192);assert(bad->failure);silence(*bad);
        auto big=std::make_unique<C>(1,INT64_MAX,1);assert(big->failure);silence(*big);
    }
    auto res=residency();
    // Conservation corruption: table row mass; exact-ne mismatch; missing Down.
    {
        auto c=std::make_unique<C>(2,0,2);c->begin(0,fmt(0),res.data(),true);
        auto cnt=routes(2,0,0);++cnt[0];c->counts(0,2,cnt.data(),cnt.size());
        assert(c->failure);assert(output(*c).find("route_conservation")!=std::string::npos);
    }
    {
        auto c=std::make_unique<C>(2,0,2);c->begin(0,fmt(0),res.data(),true);
        auto cnt=routes(2,0,0);c->counts(0,2,cnt.data(),cnt.size());
        c->issue(true,0,2,0,0,-1);assert(c->failure);silence(*c);
    }
    {
        auto c=std::make_unique<C>(1,0,1);c->begin(0,fmt(0),res.data(),true);
        auto cnt=routes(1,0,0);c->counts(0,1,cnt.data(),cnt.size());
        for(int e=0;e<512;++e)if(cnt[e])c->issue(true,0,cnt[e],e,e,-1);
        c->end(0);assert(c->failure);assert(output(*c).find("missing_issuer_call")!=std::string::npos);
    }
    // Accumulator overflow must qualify false, never wrap or throw.
    {
        auto c=std::make_unique<C>(1,0,1);c->begin(0,fmt(0),res.data(),true);
        c->record[0].hist[0]=UINT64_MAX;auto cnt=routes(1,0,0);
        c->counts(0,1,cnt.data(),cnt.size());assert(c->failure);
        assert(output(*c).find("counter_overflow")!=std::string::npos);
    }
    // Format extent underflow, residency and unsupported threaded ownership.
    for(int which=0;which<3;++which) {
        auto c=std::make_unique<C>(1,0,1);auto f=fmt(0);auto r=res;
        if(which==0)f.down_off=f.up_off-1;
        if(which==1)r[7]=-1;
        c->begin(0,f,r.data(),which!=2);assert(c->failure);silence(*c);
    }
    // Dense pathological output: size preflight fails before any layer bytes.
    {
        auto c=std::make_unique<C>(1,0,1);
        for(auto& r:c->record)r.hist.fill(UINT64_MAX);
        C::Writer w{nullptr};c->serialize(w);assert(!w.ok&&w.bytes<=C::output_cap);
        // Deliberately synthetic formatter fault injection, not a qualified request.
        c->retired=48;auto s=output(*c);
        assert(s.find("output_overflow")!=std::string::npos);
        assert(s.find("\"layers\"")==std::string::npos);
    }
    // Borrowed pointer restored before owner destruction, on normal and exceptional exit.
    {
        auto old=std::make_unique<C>(1,0,1);silence(*old);C* ptr=old.get();
        {
            auto owner=std::make_unique<C>(1,0,1);silence(*owner);
            try {
                strata::prefill::detail::LayerMajorShapeBorrow borrow(ptr,owner.get());
                assert(ptr==owner.get());throw 9;
            }catch(int){}
            assert(ptr==old.get());
        }
        assert(ptr==old.get());
    }
    // Partial request produces an explicit failure and no success token.
    {
        auto c=std::make_unique<C>(1,0,1);feed(*c,0,1,1);
        auto s=output(*c);assert(s.find("request_incomplete")!=std::string::npos);
        assert(s.find("\"complete\":true")==std::string::npos);
    }
}
