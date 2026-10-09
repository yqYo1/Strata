// Independent CPU-only diagnostic fixture. Root owns compilation and execution.
#include "strata/core/cache_route_pairs.hpp"
#include <cassert>
#include <memory>
#include <string>
#include <thread>
using strata::core::CacheRoutePairs;
static std::string output(CacheRoutePairs& c, const std::array<int32_t,CacheRoutePairs::cells>& res,
                          int64_t h,int64_t look,int64_t off,bool expected,bool failed=false,bool complete=true) {
    FILE* f=std::tmpfile();assert(f);
    assert(c.report(f,res.data(),res.size(),h,look,off,failed,complete)==expected);
    std::rewind(f);std::string out;char b[4096];size_t n;
    while((n=std::fread(b,1,sizeof b,f))!=0){out.append(b,n);assert(out.size()<4*1024*1024);}
    std::fclose(f);return out;
}
int main() {
    unsetenv("STRATA_CACHE_ROUTE_PAIRS");assert(!CacheRoutePairs::enabled());
    setenv("STRATA_CACHE_ROUTE_PAIRS","0",1);assert(!CacheRoutePairs::enabled());
    setenv("STRATA_CACHE_ROUTE_PAIRS","1",1);assert(CacheRoutePairs::enabled());
    for(const char* s:{"", "01", "true", "2"}) {
        setenv("STRATA_CACHE_ROUTE_PAIRS",s,1);bool rejected=false;
        try{(void)CacheRoutePairs::enabled();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    }
    auto c=std::make_unique<CacheRoutePairs>();std::array<int32_t,CacheRoutePairs::cells> res;res.fill(-1);res[3]=0;
    c->begin(48,512,res.data(),res.size(),true);c->callback();
    c->entry(0,3,0);c->entry(0,3,0);c->entry(0,4,-1);c->entry(0,4,-1);c->entry(0,5,1);
    c->callback();c->entry(0,4,-1);
    auto s=output(*c,res,2,5,1,true);
    assert(s.find("PAIR 0 3 2 2 0 0 1\n")!=std::string::npos);
    assert(s.find("PAIR 0 4 3 0 3 0 2\n")!=std::string::npos);
    assert(s.find("callback_pairs=4")!=std::string::npos);
    c->begin(48,512,res.data(),res.size(),true);c->callback();c->entry(47,511,-1);
    s=output(*c,res,0,1,0,true);assert(s.find("request=2")!=std::string::npos);assert(s.find("PAIR 0 3")==std::string::npos);
    c->begin(48,512,res.data(),res.size(),true);c->callback();c->entry(0,3,0);res[3]=1;
    s=output(*c,res,1,1,0,false);assert(s.find("unchanged=0")!=std::string::npos);
    c->begin(48,512,res.data(),res.size(),true);c->callback();c->entry(48,0,0);c->entry(0,512,0);
    output(*c,res,0,0,0,false);
    c->begin(48,512,res.data(),res.size(),true);
    std::thread wrong([&]{c->callback();c->entry(0,0,0);});wrong.join();output(*c,res,0,0,0,false);
    c->begin(48,512,res.data(),res.size(),true);c->callback();c->entry(0,3,0);output(*c,res,1,2,0,false);
    c->begin(47,512,res.data(),res.size(),true);c->callback();c->entry(0,0,0);
    s=output(*c,res,0,0,0,false);assert(s.find("supported=0")!=std::string::npos);
    c->begin(48,512,res.data(),res.size(),true);output(*c,res,0,0,0,false);
    c->begin(48,512,res.data(),res.size(),true);
    for(int l=0;l<48;++l){c->callback();for(int e=0;e<512;++e)c->entry(l,e,-1);}
    s=output(*c,res,0,24576,0,true);assert(s.find("cells=24576")!=std::string::npos);
    // A cancelled/failed request can have valid partial totals. It must stay incomplete.
    c->begin(48,512,res.data(),res.size(),true);c->callback();c->entry(0,3,0);
    s=output(*c,res,1,1,0,false,true);assert(s.find("complete=0")!=std::string::npos);
    c->begin(48,512,res.data(),res.size(),true);c->callback();c->entry(0,3,0);
    s=output(*c,res,1,1,0,false,false,false);
    assert(s.find("error=0 complete=0")!=std::string::npos);
    // Two tokens with unique top-k IDs: repeated expert7 is one callback pair.
    c->begin(48,512,res.data(),res.size(),true);c->callback();
    c->entry(0,4,0);c->entry(0,7,-1);c->entry(0,8,-1);
    c->entry(0,4,0);c->entry(0,7,-1);c->entry(0,9,-1);
    s=output(*c,res,2,6,0,true);
    assert(s.find("PAIR 0 4 2 2 0 0 1\n")!=std::string::npos);
    assert(s.find("PAIR 0 7 2 0 2 0 1\n")!=std::string::npos);
    assert(s.find("callback_pairs=4")!=std::string::npos);
    // The same expert across two split callbacks creates two callback pairs.
    c->begin(48,512,res.data(),res.size(),true);
    c->callback();c->entry(0,7,-1);c->entry(0,7,-1);
    c->callback();c->entry(0,7,-1);c->entry(0,7,-1);
    s=output(*c,res,0,4,0,true);assert(s.find("PAIR 0 7 4 0 4 0 2\n")!=std::string::npos);
}
