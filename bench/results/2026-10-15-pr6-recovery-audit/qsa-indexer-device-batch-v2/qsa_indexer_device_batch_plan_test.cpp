#include "strata/kernels/qsa_indexer_device_batch_plan.hpp"
#include <cstdlib>
#include <algorithm>
#include <iostream>
using namespace strata::kernels::detail;
static void need(bool b,const char* message) {
    if(!b) { std::cerr<<"FAIL: "<<message<<'\n'; std::exit(1); }
}
template<class F> static void rejects(F action) {
    bool caught=false; try { action(); } catch(const std::invalid_argument&) { caught=true; }
    need(caught,"invalid device batch geometry/flag must reject");
}
int main() {
    need(!qsa_indexer_device_batch_requested(nullptr) && !qsa_indexer_device_batch_requested("0") &&
         qsa_indexer_device_batch_requested("1"),"flag strict default off");
    for(const char* value:{"","01"," 1","1 ","true","-1","2"})
        rejects([&]{qsa_indexer_device_batch_requested(value);});
    for(int64_t capacity:{int64_t(1),int64_t(4),int64_t(5),int64_t(31),int64_t(262144)})
        for(int64_t p0=0;p0<=std::min<int64_t>(capacity,9);++p0)
            for(int64_t n=0;n<=std::min<int64_t>(capacity-p0,19);++n) {
                const auto p=QsaIndexerDeviceBatchPlan::make(n,p0,100,capacity);
                int64_t complete=0;
                for(int64_t pos=p0;pos<p0+n;++pos) complete+=pos%4==3;
                need(p.completed_blocks==complete && p.block_global==size_t(complete)*256,"one workgroup per completed block");
                need(p.raw_bytes==size_t(n)*512 && p.seed_zero==(n>0 && p0==0),"raw extent and seed decision");
                if(complete) need(p.first_block==p0/4 && (p.first_block+complete)*4<=p.end,"completed row bounds");
            }
    const auto full=QsaIndexerDeviceBatchPlan::make(8192,0,0,262144);
    need(full.completed_blocks==2048 && full.block_global==524288 && full.seed_zero,"8192 rows are one block grid");
    const auto last=QsaIndexerDeviceBatchPlan::make(4,262140,0,262144);
    need(last.first_block==65535 && last.completed_blocks==1 && last.pooled_bytes==65537ull*512,
         "last physical completed row and spare65536 are addressable layout only");
    rejects([]{QsaIndexerDeviceBatchPlan::make(-1,0,0,10);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(1,-1,0,10);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(4,8,0,10);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(1,11,0,10);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(1,0,2,10);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(1,0,-4,10);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(1,0,2147483644,10);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(1,0,0,0);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(1,0,0,int64_t(INT32_MAX)+1);});
    rejects([]{QsaIndexerDeviceBatchPlan::make(INT64_MAX,0,0,262144);});
    rejects([]{qsa_indexer_batch_bytes(UINT64_MAX,512);});
    std::cout<<"PASS native QSA device batch host plan/flag contracts; no GPU qualification\n";
}
