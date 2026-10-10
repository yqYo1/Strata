#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::prefill {
inline bool route_group_requested(){
 const char* v=std::getenv("STRATA_SYCL_ROUTE_GROUP");
 if(!v||std::strcmp(v,"0")==0)return false;
 if(std::strcmp(v,"1")!=0)throw std::invalid_argument("STRATA_SYCL_ROUTE_GROUP must be 0 or 1");
#if defined(STRATA_SYCL_ROUTE_GROUP)
 return true;
#else
 throw std::invalid_argument("SYCL route grouping unavailable in this build");
#endif
}
// Integer-only per-chunk storage: counts[E], offsets[E+1], fill[E].
struct RouteGroupScratch {int32_t* data=nullptr;size_t words=0;};
struct RouteGroupView {int32_t* counts;int32_t* offsets;int32_t* fill;};
inline size_t route_group_scratch_words(int64_t E){return E>0&&E<=1024?size_t(3*E+1):0;}
inline size_t route_group_scratch_bytes(int64_t E){return route_group_scratch_words(E)*sizeof(int32_t);}
RouteGroupView route_group_view(RouteGroupScratch scratch,int64_t E);
// No submissions if false; unsupported shape/capacity/non-in-order queue falls back
// only at caller's pre-submit choice. Caller owns valid nonoverlapping USM in stream context.
bool route_group_supported(int64_t T,int64_t K,int64_t E,size_t ids_capacity,
 size_t slot_capacity,size_t src_capacity,RouteGroupScratch scratch,void* stream);
void route_group_submit(const int32_t* ids,int32_t* slot,int32_t* src,int64_t T,int64_t K,
 int64_t E,size_t ids_capacity,size_t slot_capacity,size_t src_capacity,
 RouteGroupScratch scratch,void* stream);
} // namespace strata::prefill
