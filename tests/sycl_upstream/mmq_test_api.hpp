#pragma once
#include "strata/sycl_upstream/mmq_stages.hpp"
#include "strata/sycl_upstream/mmq_product.hpp"
#ifdef STRATA_TEST_PUBLIC_MMQ
#include "strata/sycl_upstream/mmq_adapter.hpp"
// Reuse the same byte/numerical fixtures through the original engine API.
// Only the test adds completion barriers for its event-based assertions.
namespace tested {
namespace api = strata::prefill::mmq;
inline sycl::event mmq_gather_native(sycl::queue& q, const void* g, const void* u, size_t half,
    const void* d, size_t bytes, void* gu, void* dn) {
    api::gather_native(g,u,half,d,bytes,gu,dn,&q); return q.ext_oneapi_submit_barrier();
}
inline bool mmq_gather_native_group(sycl::queue& q, const api::GatherGroup& g, size_t up, size_t half,
    size_t down, size_t bytes, void* gu, size_t gs, void* dn, size_t ds) {
    return api::gather_native_group(g,up,half,down,bytes,gu,gs,dn,ds,&q);
}
inline sycl::event mmq_gather_strata_q2(sycl::queue& q, const uint8_t* b, void* gu, void* dn) {
    api::gather_strata_q2(b,gu,dn,&q); return q.ext_oneapi_submit_barrier();
}
inline sycl::event mmq_swiglu(sycl::queue& q, const float* gu, float* h, int64_t r, int64_t ff, bool i) {
    api::swiglu(gu,h,r,ff,i,&q); return q.ext_oneapi_submit_barrier();
}
inline void mmq_iota(sycl::queue& q, int32_t* dst, int64_t n) { api::iota(dst,n,&q); }
inline void mmq_quantize(sycl::queue& q, const float* x, const int32_t* ids, strata::sycl_upstream::MmqBlock* out,
    ggml_type t, int64_t cols, int64_t ld, int64_t rows) { api::quantize(x,ids,out,t,cols,ld,rows,&q); }
inline sycl::event mmq_product(sycl::queue& q, const strata::sycl_upstream::MmqProduct& p,
    const strata::sycl_upstream::MmqPlan&, float*) {
    static api::Context context;
    context.run({p.weights,p.type,p.weight_rows,p.weight_cols,p.expert_bytes,p.experts,p.x,p.bounds,p.ids,
                 p.total_rows,p.max_rows,p.dst,p.ld_dst}, &q);
    return q.ext_oneapi_submit_barrier();
}
}
#else
namespace tested = strata::sycl_upstream;
#endif
