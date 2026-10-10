#include "prefill_route_census.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <thread>
using C = strata::prefill::detail::RouteCensus;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const char* mode = argv[1];
    const bool off = !std::strcmp(mode,"off");
    const bool zero = !std::strcmp(mode,"zero");
    const bool bad_env = !std::strcmp(mode,"bad-env");
    if(off) unsetenv("STRATA_PREFILL_ROUTE_CENSUS");
    else setenv("STRATA_PREFILL_ROUTE_CENSUS",zero?"0":bad_env?"true":"1",1);
    const bool outside = !std::strcmp(mode,"outside-layer-copy");
    const bool mixed = !std::strcmp(mode,"mixed-stream-all");
    const bool mmq = !std::strcmp(mode,"mmq");
    C census(32768,0,0,1,512,false);
    for(int64_t p=0;p<32768;p+=8192) {
        census.begin_chunk(p,8192);
        std::array<int32_t,512> cnt{};
        for(unsigned e=0;e<10;++e)cnt[e]=8192;
        C::Meta m;
        m.chunk=p;m.T=8192;m.K=10;m.layer=0;m.experts=512;
        m.branch=mmq?C::mmq:C::generic_iq;m.mmq_built=mmq;
        m.native=true;m.gu=22;m.down=20;m.H=2560;m.FF=640;
        m.gu_row=820;m.down_row=360;m.up_off=524800;m.down_off=1049600;m.bytes=1971200;
        m.stream_all=mixed;
        census.layer(m,cnt.data());
        if(mixed) {
            for(unsigned e=2;e<=10;++e)census.plan_source(0,e,C::stager_ram);
            std::thread issuer([&]{for(unsigned e=2;e<=10;++e)census.copied(0,e,m.bytes,C::stager_ram);});
            issuer.join();
        }
        for(int e=0;e<10;++e){
            census.executed(0,e,!mixed||e<2);
            if(!mmq){census.product(0,e,0);census.product(0,e,1);}
        }
        if(!std::strcmp(mode,"duplicate-call"))census.executed(0,0,true);
        if(!std::strcmp(mode,"resident-copy")){census.plan_source(0,0,C::pinned);census.copied(0,0,m.bytes,C::pinned);}
        if(outside){census.plan_source(1,0,C::pinned);census.copied(1,0,m.bytes,C::pinned);}
        census.end_chunk();
    }
    census.finish(true);
    return 0;
}
