#pragma once
// Launch syntax and CUDA scalar operations for mechanically lowered original
// kernels. Streams still resolve through the owning native Runtime.
#include "strata/sycl_upstream/cuda_backend.hpp"
#include "strata/sycl_upstream/mmq_loader_shim.hpp"
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/intel/math.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::sycl_upstream::cuda_kernel {
using half = sycl::half;
using half2 = sycl::half2;
using __half = sycl::half;
using original_loaders::int2;
using original_loaders::uint2;
using original_loaders::min;
using original_loaders::make_int2;
using original_loaders::__byte_perm;
using original_loaders::__popc;
using original_loaders::__vcmpne4;
using original_loaders::__vsub4;
struct alignas(8) float2 { float x, y; };
inline float __half2float(half x) { return float(x); }
inline float __low2float(half2 x) { return float(x[0]); }
inline float __high2float(half2 x) { return float(x[1]); }
inline half __low2half(half2 x) { return x[0]; }
inline half __high2half(half2 x) { return x[1]; }
inline float2 __half22float2(half2 x) { return {float(x[0]),float(x[1])}; }
inline half2 make_half2(float a,float b) { return half2(half(a),half(b)); }
inline half __float2half(float x) { return half(x); }
inline float __uint_as_float(uint32_t x) { return sycl::bit_cast<float>(x); }
inline float __fmaf_rn(float a,float b,float c) { return sycl::fma(a,b,c); }
inline float __fmul_rn(float a,float b) { return sycl::ext::intel::math::fmul_rn(a,b); }
inline float __fdiv_rn(float a,float b) { return sycl::ext::intel::math::fdiv_rn(a,b); }
inline float __expf(float a) { return sycl::native::exp(a); }
inline int __dp4a(int a,int b,int c) {
    // The original Pascal fallback has the same signed four-byte operation.
    const auto x=sycl::bit_cast<sycl::vec<int8_t,4>>(a);
    const auto y=sycl::bit_cast<sycl::vec<int8_t,4>>(b);
    for(int i=0;i<4;++i)c+=int(x[i])*int(y[i]);
    return c;
}
inline uint32_t __vsubss4(uint32_t a,uint32_t b) {
    uint32_t out=0;
    for(int i=0;i<4;++i) {
        const int x=int(int8_t(a>>(8*i)))-int(int8_t(b>>(8*i)));
        out|=uint32_t(uint8_t(sycl::clamp(x,-128,127)))<<(8*i);
    }
    return out;
}
inline float __shfl_xor_sync(uint32_t,float x,int mask,int width=32) {
    (void)width; // All lowered original call sites use the complete 32-lane warp.
    return sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(),x,mask);
}
inline float __shfl_down_sync(uint32_t,float x,int delta,int width=32) {
    const auto sg=sycl::ext::oneapi::this_work_item::get_sub_group();
    const float other=sycl::shift_group_left(sg,x,delta);
    return int(sg.get_local_linear_id())+delta<width?other:x;
}
inline void __syncthreads() {
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>());
}
inline dim3 thread_index() {
    const auto it=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return dim3(unsigned(it.get_local_id(2)),unsigned(it.get_local_id(1)),unsigned(it.get_local_id(0)));
}
inline dim3 block_index() {
    const auto it=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return dim3(unsigned(it.get_group(2)),unsigned(it.get_group(1)),unsigned(it.get_group(0)));
}
inline dim3 block_dimensions() {
    const auto it=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return dim3(unsigned(it.get_local_range(2)),unsigned(it.get_local_range(1)),unsigned(it.get_local_range(0)));
}
inline dim3 grid_dimensions() {
    const auto it=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    return dim3(unsigned(it.get_group_range(2)),unsigned(it.get_group_range(1)),unsigned(it.get_group_range(0)));
}
template<bool Warp32,bool Shared,class F> void launch(dim3 grid,dim3 block,size_t shared_bytes,cudaStream_t stream,F fn) {
    // The original wrappers retain their argument/shape/error checks. This
    // callback enqueues one native kernel without waiting or allocating USM.
    (void)cuda::submit(stream,[=](sycl::queue& q) {
        if(!grid.x || !grid.y || !grid.z || !block.x || !block.y || !block.z)
            throw std::invalid_argument("original kernel launch has zero dimensions");
        if constexpr(Warp32) {
            const auto sizes=q.get_device().get_info<sycl::info::device::sub_group_sizes>();
            if(std::find(sizes.begin(),sizes.end(),32)==sizes.end())
                throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),"original kernel needs subgroup size 32");
        }
        const sycl::range<3> local(block.z,block.y,block.x);
        const sycl::range<3> global(size_t(grid.z)*block.z,size_t(grid.y)*block.y,size_t(grid.x)*block.x);
        return q.submit([&](sycl::handler& h) {
            if constexpr(Shared) {
                sycl::local_accessor<float,1> scratch((shared_bytes+3)/4,h);
                if constexpr(Warp32) h.parallel_for(sycl::nd_range<3>(global,local),[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                    fn(scratch.template get_multi_ptr<sycl::access::decorated::no>().get());
                });
                else h.parallel_for(sycl::nd_range<3>(global,local),[=](sycl::nd_item<3>) {
                    fn(scratch.template get_multi_ptr<sycl::access::decorated::no>().get());
                });
            } else {
                if constexpr(Warp32) h.parallel_for(sycl::nd_range<3>(global,local),[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] { fn(nullptr); });
                else h.parallel_for(sycl::nd_range<3>(global,local),[=](sycl::nd_item<3>) { fn(nullptr); });
            }
        });
    });
}
} // namespace strata::sycl_upstream::cuda_kernel
