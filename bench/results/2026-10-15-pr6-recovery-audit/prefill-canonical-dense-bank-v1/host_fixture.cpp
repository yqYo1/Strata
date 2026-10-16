// Unexecuted actual-policy fixture. No GPU/runtime/provider/environment changes.
#include "strata/core/canonical_bank_policy.hpp"
#include "strata/core/native_dense_bank_policy.hpp"
#include <cstdlib>
#include <limits>
#include <cstdio>
using namespace strata::core::detail;
void need(bool x){if(!x) std::abort();}
int main(){
    int owner;constexpr int ok=0,oom=2,unknown=999;
    need(classify_bank_allocation(ok,ok,oom,&owner)==BankAllocation::success);
    need(classify_bank_allocation(ok,ok,oom,nullptr)==BankAllocation::uncertain);
    need(classify_bank_allocation(oom,ok,oom,nullptr)==BankAllocation::retryable_oom);
    need(classify_bank_allocation(oom,ok,oom,&owner)==BankAllocation::uncertain);
    need(classify_bank_allocation(unknown,ok,oom,nullptr)==BankAllocation::uncertain);
    need(classify_bank_allocation(unknown,ok,oom,&owner)==BankAllocation::uncertain);
    for(int l=0;l<48;++l){
        const auto prefix="blk."+std::to_string(l)+".";
        need(canonical_layer(prefix+"hc_attn_norm.weight")==l);
        need(canonical_persistent(prefix+"hc_attn_norm.weight"));
        need(canonical_persistent(prefix+"hc_ffn_norm.weight"));
        need(!canonical_persistent(prefix+"hc_attn_up.weight"));
    }
    for(const char* n:{"output_hc_down.weight","output_hc_norm.weight","output_hc_up.weight","blk.1.ple_key.weight","blk.1.ple_value.weight","blk.1.ple_norm_key.weight","blk.1.ple_norm_query.weight","blk.1.ple_norm_conv.weight","blk.1.ple_conv1d.weight"})need(canonical_persistent(n));
    need(!canonical_persistent("blk.10.ple_value.weight"));
    need(canonical_stride(0)==0 && canonical_stride(1)==256 && canonical_stride(256)==256);
    bool threw=false;try{canonical_stride(UINT64_MAX);}catch(...){threw=true;}need(threw);
    for(const char* n:{"blk..x","blk.48.x","blk.-1.x","blk.999999999999.x"}) {threw=false;try{canonical_layer(n);}catch(...){threw=true;}need(threw);}
    std::puts("HOST_ONLY canonical policy/HC/PLE classification; no allocation or model pass");
}
