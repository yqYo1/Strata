// Private integer mapping prototype; no fused/MMQ/weight/arithmetic dependency.
#include "strata/prefill/route_group.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <limits>
namespace strata::prefill {
namespace {class RouteCount;class RouteScan;class RoutePlace;
bool shape(int64_t T,int64_t K,int64_t E,size_t ic,size_t sc,size_t rc,RouteGroupScratch scratch){
 if(T<0||K<=0||E<=0||E>1024||T>INT32_MAX/K)return false;
 size_t n=size_t(T*K),words=route_group_scratch_words(E);
 return ic>=n&&sc>=n&&rc>=n&&scratch.data&&scratch.words>=words;
}
bool overlap(const void* a,size_t an,const void* b,size_t bn){
 auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
 if(an>UINTPTR_MAX-x||bn>UINTPTR_MAX-y)throw std::invalid_argument("route group address overflow");
 return x<y+bn&&y<x+an;
}
}
RouteGroupView route_group_view(RouteGroupScratch s,int64_t E){
 auto words=route_group_scratch_words(E);if(!words||!s.data||s.words<words)throw std::invalid_argument("route scratch capacity");
 return {s.data,s.data+E,s.data+2*E+1};
}
bool route_group_supported(int64_t T,int64_t K,int64_t E,size_t ic,size_t sc,size_t rc,RouteGroupScratch s,void* stream){
 if(!shape(T,K,E,ic,sc,rc,s)||!stream)return false;
 return strata::q_of(stream)->is_in_order();
}
void route_group_submit(const int32_t* ids,int32_t* slot,int32_t* src,int64_t T,int64_t K,
 int64_t E,size_t ic,size_t sc,size_t rc,RouteGroupScratch scratch,void* stream){
 if(!route_group_supported(T,K,E,ic,sc,rc,scratch,stream))throw std::invalid_argument("route group shape/capacity/queue");
 if(T==0)return;
 if(!ids||!slot||!src)throw std::invalid_argument("route group null map");
 const void* ptrs[]{ids,slot,src,scratch.data};size_t bytes[]{size_t(T*K)*4,size_t(T*K)*4,size_t(T*K)*4,route_group_scratch_words(E)*4};
 for(int i=0;i<4;++i){if(reinterpret_cast<uintptr_t>(ptrs[i])%alignof(int32_t))throw std::invalid_argument("route group alignment");for(int j=0;j<i;++j)if(overlap(ptrs[i],bytes[i],ptrs[j],bytes[j]))throw std::invalid_argument("route group alias");}
 auto& q=*strata::q_of(stream);const auto v=route_group_view(scratch,E);const size_t n=size_t(T*K);
 // No catch/retry encompasses mutable submissions. Queue orders all phases.
 q.fill(v.counts,int32_t(0),size_t(E));
 q.parallel_for<RouteCount>(sycl::range<1>(n),[=](sycl::id<1> at){int32_t e=ids[at[0]];if(e>=0&&e<E){sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> c(v.counts[e]);c.fetch_add(1);}});
 q.single_task<RouteScan>([=]{int32_t off=0;for(int64_t e=0;e<E;++e){v.offsets[e]=off;v.fill[e]=off;off+=v.counts[e];}v.offsets[E]=off;});
 q.parallel_for<RoutePlace>(sycl::range<1>(n),[=](sycl::id<1> at){size_t i=at[0];int32_t e=ids[i];if(e>=0&&e<E){sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> f(v.fill[e]);int32_t p=f.fetch_add(1);slot[i]=p;src[p]=int32_t(i/size_t(K));}});
 // Invalid IDs skipped; owner must D2H counts, drain and require sum==T*K
 // BEFORE any map consumer. Within-expert atomic order is not host stable.
}
} // namespace strata::prefill
