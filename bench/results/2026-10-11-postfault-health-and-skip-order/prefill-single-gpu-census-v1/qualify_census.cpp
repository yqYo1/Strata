// CPU-only contract uses the EXACT unchanged production census header.
#include "strata/prefill_route_census.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>

using Census=strata::prefill::detail::RouteCensus;
int main(int argc,char**argv) {
    if(argc!=2) return 2;
    const char* name=argv[1];
    const bool off=!std::strcmp(name,"off");
    const bool multi=!std::strcmp(name,"unsupported");
    const bool missing=!std::strcmp(name,"missing-product");
    const bool unrouted=!std::strcmp(name,"unrouted-copy");
    const bool branch=!std::strcmp(name,"unknown-branch");
    int64_t n=65535;
    if(!std::strcmp(name,"valid32")) n=32767;
    else if(!std::strcmp(name,"valid262")) n=262143;
    else if(!std::strcmp(name,"collector-bound")) n=262144;
    else if(std::strcmp(name,"valid64") && !off && !multi && !missing && !unrouted && !branch) return 2;
    if(off) { if(unsetenv("STRATA_PREFILL_ROUTE_CENSUS")) return 2; }
    else if(setenv("STRATA_PREFILL_ROUTE_CENSUS","1",1)) return 2;
    // Constructor's last bool REJECTS unsupported modes. False admits singleGPU.
    Census census(n,0,0,48,512,multi);
    if(census.on != !off) return 1;
    for(int64_t p0=0;p0<n;p0+=8192) {
        const int64_t T=std::min<int64_t>(8192,n-p0);
        census.begin_chunk(p0,T);
        for(unsigned l=0;l<48;++l) {
            Census::Meta m;
            m.chunk=p0;m.T=T;m.K=10;m.layer=l;m.experts=512;
            m.branch=branch?99u:static_cast<unsigned>(Census::generic_iq);
            m.native=true;m.stream_all=false;m.H=2560;m.FF=640;
            m.gu=22;m.down=20;m.gu_row=820;m.down_row=360;
            m.up_off=m.gu_row*640;m.down_off=2*m.up_off;
            m.bytes=m.down_off+2560*m.down_row;
            std::array<int32_t,512> counts{};
            for(unsigned e=0;e<512;++e) counts[e]=int32_t(T*10/512+(e<unsigned(T*10%512)));
            if(unrouted) { counts[1]+=counts[0];counts[0]=0; }
            census.layer(m,counts.data());
            for(unsigned e=0;e<512;++e) {
                if(!counts[e]) {
                    if(unrouted && e==0) {
                        census.plan_source(l,e,Census::pinned);
                        census.copied(l,e,m.bytes,Census::pinned);
                    }
                    continue;
                }
                const bool resident=e%4==0;
                if(!resident) {
                    const unsigned source=e%4==1?Census::pinned:Census::stager_ram;
                    census.plan_source(l,e,source);
                    census.copied(l,e,m.bytes,source);
                }
                census.executed(l,e,resident);
                census.product(l,e,0);
                if(!(missing && e==0)) census.product(l,e,1);
            }
        }
        census.end_chunk();
    }
    census.finish(true);
    std::cout<<"{\"case\":\""<<name<<"\",\"gpu_executed\":false}\n";
}
