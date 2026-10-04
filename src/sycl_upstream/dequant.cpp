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
template<class T> void flat(int type,const void* source,int64_t n,T* dst,void* stream) {
    if(n<0 || n%256 || !(iq_supported(type) || (type==30 && std::is_same_v<T,float>)))bad("IQ flat arguments");
    if(!n)return;
    dispatch(stream,[=](sycl::queue& q) {
        return q.parallel_for(sycl::range<1>(product(n/256,32)),[=](sycl::id<1> id) {
            const size_t block=id[0]/32,lane=id[0]%32;
            original::dq_dispatch(type,source,int64_t(block),dst+block*256,int(lane));
        });
    });
}
}
bool dequant_bf16_supported(int type) noexcept {int a,b;return original::geometry(type,a,b);}
bool iq_supported(int t) noexcept {return t==16 || t==17 || t==18 || t==20 || t==21 || t==22 || t==23 || t==29 || t==42 || t==11 || t==12 || t==13 || t==7 || t==6 || t==8;}
bool embed_type_supported(int t) noexcept {return iq_supported(t) || t==30;}
size_t iq_row_bytes(int t,int64_t n) noexcept {return original::iq_row_bytes(t,n);}
void iq_dequant_f16(int t,const void* src,int64_t n,uint16_t* dst,void* stream) {flat(t,src,n,reinterpret_cast<sycl::half*>(dst),stream);}
void iq_dequant_f32(int t,const void* src,int64_t n,float* dst,void* stream) {flat(t,src,n,dst,stream);}
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
void iq_dequant_gu_f16(int type,const void* gate,const void* up,int64_t ff,int64_t cols,uint16_t* out,void* stream) {
    if(!iq_supported(type) || cols<=0 || cols%256 || ff<0)bad("IQ gate/up arguments");
    if(!ff)return;
    const size_t per_row=size_t(cols/256),blocks=product(ff,cols/256);
    const size_t count=product(int64_t(blocks),64);
    dispatch(stream,[=](sycl::queue& q) {
        return q.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id) {
            const size_t block=id[0]/64,role=(id[0]/32)%2,lane=id[0]%32;
            auto* dst=reinterpret_cast<sycl::half*>(out)+((2*(block/per_row)+role)*per_row+block%per_row)*256;
            original::dq_dispatch(type,role?up:gate,int64_t(block),dst,int(lane));
        });
    });
}
void iq_embed_rows(int type,const void* table,size_t stride,const int32_t* tokens,int64_t rows,int64_t cols,float* out,void* stream) {
    if(rows<=0)return;
    if(!embed_type_supported(type) || cols<=0 || cols%256 || stride<iq_row_bytes(type,cols))bad("IQ embedding arguments");
    const size_t per_row=size_t(cols/256),count=product(rows,cols/8);
    dispatch(stream,[=](sycl::queue& q) {
        return q.parallel_for(sycl::range<1>(count),[=](sycl::id<1> id) {
            const size_t block=id[0]/32,lane=id[0]%32,row=block/per_row;
            const auto* source=static_cast<const uint8_t*>(table)+size_t(tokens[row])*stride;
            original::dq_dispatch(type,source,int64_t(block%per_row),out+row*size_t(cols)+(block%per_row)*256,int(lane));
        });
    });
}
}
