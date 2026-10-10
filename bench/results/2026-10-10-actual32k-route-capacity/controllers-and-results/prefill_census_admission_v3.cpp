#include "prefill_route_census.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <sys/syscall.h>
using C = strata::prefill::detail::RouteCensus;
static unsigned allocations=0, fail_alloc=0;
static std::atomic<unsigned long long> clocks{0};
static std::array<size_t,8> sizes{};
static void* alloc(size_t n) noexcept {
    ++allocations;if(allocations<=sizes.size())sizes[allocations-1]=n;
    return allocations==fail_alloc?nullptr:std::malloc(n);
}
void* operator new(size_t n,const std::nothrow_t&) noexcept{return alloc(n);}
void* operator new[](size_t n,const std::nothrow_t&) noexcept{return alloc(n);}
void operator delete(void* p) noexcept{std::free(p);}
void operator delete[](void* p) noexcept{std::free(p);}
void operator delete(void* p,size_t) noexcept{std::free(p);}
void operator delete[](void* p,size_t) noexcept{std::free(p);}
extern "C" int clock_gettime(clockid_t id,timespec* ts) noexcept{
    ++clocks;return static_cast<int>(syscall(SYS_clock_gettime,id,ts));
}
int main(int argc,char** argv){
    if(argc!=2)return 2;const std::string_view mode=argv[1];
    if(mode=="off")unsetenv("STRATA_PREFILL_ROUTE_CENSUS");
    else setenv("STRATA_PREFILL_ROUTE_CENSUS",mode=="zero"?"0":mode=="bad-env"?"true":"1",1);
    fail_alloc=mode=="alloc1"?1:mode=="alloc2"?2:0;
    if(mode=="fatal"){C::fatal_receipt("owned_contract_fatal");std::fflush(stderr);std::_Exit(0);}
    const bool cap=mode=="output-cap";
    const int64_t n=mode=="full-context-host"?262144:32768;
    const int64_t tile=cap?1:8192;
    const int lb=2,le=cap?4:3,ne=mode=="ne10"?10:16;
    allocations=0;clocks=0;
    {
      C census(n,0,lb,le,ne,false);
      if(mode=="copy-before")census.copied(lb,0,1971200,C::pinned);
      if(mode=="early-return")return 0;
      for(int64_t p=0;p<n;p+=tile){
        census.begin_chunk(p,tile);
        for(int l=lb;l<le;++l){
          std::array<int32_t,512> cnt{};for(int e=0;e<10;++e)cnt[e]=static_cast<int32_t>(tile);
          C::Meta m;m.chunk=p;m.T=tile;m.K=10;m.layer=l;m.experts=ne;m.native=true;
          m.branch=mode=="mmq"?C::mmq:C::generic_iq;m.mmq_built=mode=="mmq";
          m.gu=22;m.down=20;m.H=2560;m.FF=640;m.gu_row=820;m.down_row=360;
          m.up_off=524800;m.down_off=1049600;m.bytes=1971200;m.stream_all=mode=="threaded";
          m.fused_only=mode=="fused-fallback";
          if(mode=="meta-small")--m.experts;if(mode=="meta-large")++m.experts;
          if(mode=="unknown-branch")m.branch=19;if(mode=="native-branch")m.branch=C::generic_gguf;
          census.layer(m,mode=="null-cnt"?nullptr:cnt.data());
          if(mode=="duplicate-layer")census.layer(m,cnt.data());
          if(mode=="product-before")census.product(l,0,0);
          if(mode=="threaded"){
            for(int e=2;e<=10;++e)census.plan_source(l,e,C::stager_ram);
            std::thread issuer([&]{for(int e=2;e<=10;++e)census.copied(l,e,m.bytes,C::stager_ram);});issuer.join();
          }
          for(int e=0;e<10;++e){
            if(mode=="missing-call"&&e==0)continue;
            census.executed(l,e,mode!="threaded"||e<2);
            if(mode!="mmq"){
              census.product(l,e,0);if(mode!="missing-down")census.product(l,e,1);
            }
          }
          if(mode=="duplicate-product")census.product(l,0,0);
          if(mode=="unselected-product")census.product(l,10,0);
          if(mode=="resident-copy"){census.plan_source(l,0,C::pinned);census.copied(l,0,m.bytes,C::pinned);}
          if(mode.starts_with("id-")){
            char actor[16]{},axis=0;int value=0;
            if(std::sscanf(argv[1],"id-%15[^-]-%c%d",actor,&axis,&value)!=3)return 3;
            const int il=axis=='l'?value:l,ie=axis=='e'?value:0;
            if(!std::strcmp(actor,"plan"))census.plan_source(il,ie,C::pinned);
            else if(!std::strcmp(actor,"copy"))census.copied(il,ie,m.bytes,C::pinned);
            else if(!std::strcmp(actor,"call"))census.executed(il,ie,true);
            else if(!std::strcmp(actor,"product"))census.product(il,ie,0);
            else if(!std::strcmp(actor,"lookup"))census.copied(l,0,m.bytes,census.stager_source(il,ie));
            else return 4;
          }
        }
        census.end_chunk();
        if(mode=="copy-between"&&p==0)census.copied(lb,0,1971200,C::pinned);
      }
      if(mode=="copy-after")census.copied(lb,0,1971200,C::pinned);
      census.finish(true);
    }
    std::printf("{\"allocations\":%u,\"clock_calls\":%llu,\"allocation_sizes\":[%zu,%zu]}\n",allocations,clocks.load(),sizes[0],sizes[1]);
}
