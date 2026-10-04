// Original dequant group arithmetic, original IQ dispatch, and pinned tables.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl_upstream/cuda_backend.hpp"
#include "upstream_dequant.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

namespace strata::kernels {
namespace {
namespace original = sycl_upstream::original_dequant;
namespace backend = sycl_upstream::cuda;
[[noreturn]] void bad(const char* why) { std::fprintf(stderr,"dequant: %s\n",why);std::exit(1); }
size_t product(int64_t a,int64_t b) {
    if(a<0 || b<0 || (b && uint64_t(a)>std::numeric_limits<size_t>::max()/uint64_t(b)))bad("shape overflow");
    return size_t(a)*size_t(b);
}
template<class F> void dispatch(void* stream,F&& body) {
    const auto error=backend::submit(static_cast<cudaStream_t>(stream),std::forward<F>(body));
    const auto last=cudaGetLastError();
    if(error!=cudaSuccess || last!=cudaSuccess) {
        std::fprintf(stderr,"dequant launch: %s (%s)\n",cudaGetErrorString(error!=cudaSuccess?error:last),backend::backend_error_detail());
        std::exit(1);
    }
}
template<class T> void launch(int type,const void* blocks,int64_t row0,int64_t rows,int64_t cols,T* out,void* stream) {
    int be=0,bb=0;
    if(!original::geometry(type,be,bb) || cols<=0 || cols%be || rows<=0 || row0<0)bad("unsupported type or shape");
    const size_t gpr=size_t(cols/32),total=product(rows,cols/32);
    const size_t row_bytes=product(cols/be,bb);
    if(total>std::numeric_limits<size_t>::max()/32 || uint64_t(row0)>std::numeric_limits<size_t>::max()/row_bytes)bad("shape overflow");
    const auto* input=static_cast<const uint8_t*>(blocks);
    dispatch(stream,[=](sycl::queue& q) {
        return q.parallel_for(sycl::range<1>(total),[=](sycl::id<1> id) {
            const size_t g=id[0],r=g/gpr,gi=g%gpr;
            const auto* b=input+(size_t(row0)+r)*row_bytes;auto* o=out+g*32;
#define STRATA_DEQUANT_CASE(TYPE) case TYPE: original::group32<TYPE>(b,int(gi),o);break
            switch(type) {
                STRATA_DEQUANT_CASE(2);STRATA_DEQUANT_CASE(6);STRATA_DEQUANT_CASE(7);STRATA_DEQUANT_CASE(8);
                STRATA_DEQUANT_CASE(11);STRATA_DEQUANT_CASE(12);STRATA_DEQUANT_CASE(13);STRATA_DEQUANT_CASE(14);
                STRATA_DEQUANT_CASE(20);STRATA_DEQUANT_CASE(23);STRATA_DEQUANT_CASE(42);
            }
#undef STRATA_DEQUANT_CASE
        });
    });
}
bool iq_only(int t) {return t==16 || t==17 || t==18 || t==21 || t==22 || t==29;}
}
bool dequant_bf16_supported(int type) noexcept {int a,b;return original::geometry(type,a,b);}
void dequant_bf16(int t,const void* b,int64_t row0,int64_t rows,int64_t cols,uint16_t* out,void* stream) {launch(t,b,row0,rows,cols,out,stream);}
void dequant_f16(int t,const void* b,int64_t row0,int64_t rows,int64_t cols,uint16_t* out,void* stream) {
    if(iq_only(t)) {
        if(row0<0 || rows<=0 || cols<=0 || cols%256)bad("IQ slice arguments");
        const size_t offset=product(row0,int64_t(iq_row_bytes(t,cols)));
        const size_t n=product(rows,cols);if(n>size_t(INT64_MAX))bad("IQ shape overflow");
        iq_dequant_f16(t,static_cast<const uint8_t*>(b)+offset,int64_t(n),out,stream);
    } else launch(t,b,row0,rows,cols,reinterpret_cast<original::H16*>(out),stream);
}
void dequant_f32(int t,const void* b,int64_t row0,int64_t rows,int64_t cols,float* out,void* stream) {
    if(iq_only(t)) {
        if(row0<0 || rows<=0 || cols<=0 || cols%256)bad("IQ slice arguments");
        const size_t offset=product(row0,int64_t(iq_row_bytes(t,cols)));
        const size_t n=product(rows,cols);if(n>size_t(INT64_MAX))bad("IQ shape overflow");
        iq_dequant_f32(t,static_cast<const uint8_t*>(b)+offset,int64_t(n),out,stream);
    } else launch(t,b,row0,rows,cols,out,stream);
}
}
