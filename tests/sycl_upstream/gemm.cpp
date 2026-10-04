#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl_upstream/cuda/cublas_v2.h"
#include "strata/sycl_upstream/cuda_backend.hpp"
#include <ggml.h>
#include <bit>
#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>

namespace backend=strata::sycl_upstream::cuda;
using strata::prefill::Gemm;
using namespace std::chrono_literals;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void ck(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorName(error))+": "+backend::backend_error_detail());}
struct Buffer {
    void* p{};size_t bytes;
    explicit Buffer(size_t bytes):bytes(bytes){ck(cudaMalloc(&p,bytes));}
    ~Buffer(){cudaFree(p);}
    template<class T>T* as(){return static_cast<T*>(p);}
    template<class T>void put(const std::vector<T>& v,cudaStream_t s){check(v.size()*sizeof(T)<=bytes,"buffer size");ck(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));}
    template<class T>std::vector<T> get(cudaStream_t s){std::vector<T> v(bytes/sizeof(T));ck(cudaMemcpyAsync(v.data(),p,bytes,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
};
struct Kind {int type,qk,bytes;ggml_to_float_t decode;
    explicit Kind(int type):type(type),qk(int(ggml_blck_size(ggml_type(type)))),bytes(int(ggml_type_size(ggml_type(type)))),decode(ggml_get_type_traits(ggml_type(type))->to_float) {}};
const Kind kinds[]={Kind{42},Kind{2},Kind{6},Kind{7},Kind{8},Kind{11},Kind{12},Kind{13},Kind{14},
                    Kind{20},Kind{23},Kind{16},Kind{17},Kind{22},Kind{18},Kind{21},Kind{29}};
uint32_t rng=0x726bc193;
uint32_t next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
std::vector<uint8_t> weights(const Kind& kind,int rows,int cols){
    const size_t stride=size_t(cols/kind.qk)*kind.bytes;std::vector<uint8_t> out(rows*stride);
    for(int b=0;b<rows*cols/kind.qk;++b){auto* block=out.data()+size_t(b)*kind.bytes;
        for(int j=0;j<kind.bytes;++j)block[j]=uint8_t(next());
        uint16_t d=uint16_t(0x2000+next()%0x1800);if(b%7==0)d=0;else if(b%3==0)d|=0x8000;
        const size_t at=kind.type==11?108:kind.type==14?208:0;
        if(kind.type==29){for(int j=0;j<4;++j){uint16_t sc;std::memcpy(&sc,block+kind.bytes-8+2*j,2);
            sc=(sc&0xfff)|uint16_t(((d>>(4*j))&15)<<12);std::memcpy(block+kind.bytes-8+2*j,&sc,2);}}
        else{std::memcpy(block+at,&d,2);if(kind.type==7 || kind.type==12 || kind.type==13){uint16_t m=0x2600;std::memcpy(block+2,&m,2);}}
    }return out;
}
std::vector<float> decode(const Kind& kind,const std::vector<uint8_t>& w,int rows,int cols){
    std::vector<float> out(size_t(rows)*cols);const size_t stride=size_t(cols/kind.qk)*kind.bytes;
    for(int r=0;r<rows;++r)kind.decode(w.data()+r*stride,out.data()+size_t(r)*cols,cols);return out;
}
uint16_t bits(float v,bool bf16){return bf16?ggml_fp32_to_bf16(v).bits:ggml_fp32_to_fp16(v);}
float value(uint16_t v,bool bf16){return bf16?std::bit_cast<float>(uint32_t(v)<<16):ggml_fp16_to_fp32(v);}
size_t numeric=0,guards=0;double worst_l1=0;
void compare(const std::vector<float>& got,const std::vector<float>& x,const std::vector<float>& w,int t,int n,int k,int ld,float beta,float initial){
    for(int r=0;r<t;++r){for(int c=0;c<n;++c){double sum=beta*initial,l1=std::abs(sum);
        for(int z=0;z<k;++z){double term=double(x[size_t(r)*k+z])*w[size_t(c)*k+z];sum+=term;l1+=std::abs(term);}
        const double error=std::abs(double(got[size_t(r)*ld+c])-sum),scaled=error/(l1+1e-30);worst_l1=std::max(worst_l1,scaled);
        if(!std::isfinite(got[size_t(r)*ld+c]) || error>2e-5*l1+2e-6)throw std::runtime_error("GEMM differs from independent FP64 product");++numeric;
    }for(int c=n;c<ld;++c){check(got[size_t(r)*ld+c]==-1234567.f,"GEMM row padding overwritten");++guards;}}
    for(size_t i=size_t(t)*ld;i<got.size();++i){check(got[i]==-1234567.f,"GEMM trailing guard overwritten");++guards;}
}
struct Gate{std::promise<void> release;std::shared_future<void> future=release.get_future().share();bool opened=false;
    void open(){if(!opened){release.set_value();opened=true;}}~Gate(){open();}};
void hold(cudaStream_t stream,Gate& gate){ck(backend::submit(stream,[future=gate.future](sycl::queue& q){return q.submit([&](sycl::handler& h){h.host_task([future]{future.wait();});});}));}
void cold_graph(cudaStream_t s,cudaStream_t other){
    constexpr int t=3,n=11,k=47,ld=n+2;std::vector<uint16_t>x(t*k,bits(.5f,true)),w(n*k,bits(2,true));std::vector<float>xf(t*k,.5f),wf(n*k,2),initial(t*ld+16,-1234567.f);
    Buffer dx(x.size()*2),dw(w.size()*2),dy(initial.size()*4);dx.put(x,s);dw.put(w,s);dy.put(initial,s);ck(cudaStreamSynchronize(s));
    Gemm gemm;std::string error;check(gemm.init(s,0,error),"cold GEMM init");ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));gemm.bf16(dx.as<uint16_t>(),dw.as<uint16_t>(),dy.as<float>(),t,n,k,ld);
    cudaGraph_t graph{};cudaGraphExec_t exec{};ck(cudaStreamEndCapture(s,&graph));ck(cudaGraphInstantiate(&exec,graph,0ull));ck(cudaGraphDestroy(graph));
    check(dy.get<float>(other)==initial,"GEMM executed during instantiation");ck(cudaGraphLaunch(exec,s));compare(dy.get<float>(s),xf,wf,t,n,k,ld,0,0);ck(cudaGraphExecDestroy(exec));
}

int main()try{
    ggml_context* cpu=ggml_init({16384,nullptr,true});check(cpu,"pinned GGML initialization");ggml_free(cpu);
    ck(cudaSetDevice(0));cudaStream_t s{},other{};ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
    sycl::device device;ck(backend::stream_device(s,&device));std::cout<<"device="<<device.get_info<sycl::info::device::name>()<<'\n';
    // The first oneMKL operation in this process occurs inside capture.
    cold_graph(s,other);
    size_t dequant_values=0,dequant_cases=0,iq_cases=0,gemm_cases=0,graph_cases=1;
    // Independent pinned CPU dequantizers; row slicing and three output kinds.
    for(const auto& kind:kinds)for(int cols:{256,512,2560}){
        constexpr int rows=7,row0=2,slice=3;auto w=weights(kind,rows,cols);auto ref=decode(kind,w,rows,cols);Buffer input(w.size()),f(size_t(slice)*cols*4+64),h(size_t(slice)*cols*2+64),bf(h.bytes);
        input.put(w,s);ck(cudaMemsetAsync(f.p,0xbd,f.bytes,s));ck(cudaMemsetAsync(h.p,0xbd,h.bytes,s));ck(cudaMemsetAsync(bf.p,0xbd,bf.bytes,s));
        strata::kernels::dequant_f32(kind.type,input.p,row0,slice,cols,f.as<float>(),s);
        strata::kernels::dequant_f16(kind.type,input.p,row0,slice,cols,h.as<uint16_t>(),s);
        const bool supported=strata::kernels::dequant_bf16_supported(kind.type);
        if(supported)strata::kernels::dequant_bf16(kind.type,input.p,row0,slice,cols,bf.as<uint16_t>(),s);
        auto fv=f.get<float>(s);auto hv=h.get<uint16_t>(s);auto bv=bf.get<uint16_t>(s);
        for(size_t i=0;i<size_t(slice)*cols;++i){float expected=ref[size_t(row0)*cols+i];
            if(std::abs(fv[i]-expected)>1e-6*std::abs(expected)+1e-30)throw std::runtime_error("FP32 dequant differs from pinned CPU oracle");
            check(hv[i]==bits(expected,false),"FP16 dequant bits differ from pinned CPU conversion");
            if(supported)check(bv[i]==bits(expected,true),"BF16 dequant bits differ from pinned CPU conversion");dequant_values+=supported?3:2;
        }
        for(size_t i=size_t(slice)*cols;i<hv.size();++i){check(hv[i]==0xbdbd && bv[i]==0xbdbd,"dequant 16-bit guard");++guards;}
        for(size_t i=size_t(slice)*cols;i<fv.size();++i){check(std::bit_cast<uint32_t>(fv[i])==0xbdbdbdbd,"dequant FP32 guard");++guards;}
        check(input.get<uint8_t>(s)==w,"dequant modified source");++dequant_cases;
    }
    // Original IQ flat, gate/up interleaving and row-selected embedding APIs.
    // BF16 also reaches the original embedding dispatcher, with padded rows.
    std::vector<Kind> iq_kinds(std::begin(kinds),std::end(kinds));iq_kinds.emplace_back(30);
    for(const auto& kind:iq_kinds)for(int cols:{256,1280}){
        if(!strata::kernels::embed_type_supported(kind.type))continue;
        constexpr int rows=9;const size_t stride=size_t(cols/kind.qk)*kind.bytes;
        auto w=weights(kind,rows,cols),up=weights(kind,rows,cols);auto wf=decode(kind,w,rows,cols),uf=decode(kind,up,rows,cols);
        Buffer input(w.size()),second(up.size()),f(wf.size()*4+64),h(wf.size()*2+64),gu(wf.size()*4+64);input.put(w,s);second.put(up,s);
        ck(cudaMemsetAsync(f.p,0xbd,f.bytes,s));strata::kernels::iq_dequant_f32(kind.type,input.p,int64_t(wf.size()),f.as<float>(),s);
        auto fv=f.get<float>(s);for(size_t i=0;i<wf.size();++i){check(fv[i]==wf[i],"IQ flat FP32 oracle");++dequant_values;}
        for(size_t i=wf.size();i<fv.size();++i){check(std::bit_cast<uint32_t>(fv[i])==0xbdbdbdbd,"IQ flat FP32 guard");++guards;}
        if(kind.type!=30){
            ck(cudaMemsetAsync(h.p,0xbd,h.bytes,s));strata::kernels::iq_dequant_f16(kind.type,input.p,int64_t(wf.size()),h.as<uint16_t>(),s);
            auto hv=h.get<uint16_t>(s);for(size_t i=0;i<wf.size();++i){check(hv[i]==bits(wf[i],false),"IQ flat FP16 oracle");++dequant_values;}
            for(size_t i=wf.size();i<hv.size();++i){check(hv[i]==0xbdbd,"IQ flat FP16 guard");++guards;}
            ck(cudaMemsetAsync(gu.p,0xbd,gu.bytes,s));strata::kernels::iq_dequant_gu_f16(kind.type,input.p,second.p,rows,cols,gu.as<uint16_t>(),s);
            auto gv=gu.get<uint16_t>(s);for(int r=0;r<rows;++r)for(int role=0;role<2;++role)for(int c=0;c<cols;++c){
                check(gv[(size_t(2*r)+role)*cols+c]==bits((role?uf:wf)[size_t(r)*cols+c],false),"IQ gate/up row interleaving oracle");++dequant_values;}
            for(size_t i=2*wf.size();i<gv.size();++i){check(gv[i]==0xbdbd,"IQ gate/up guard");++guards;}
        }
        std::vector<uint8_t> padded(size_t(rows)*(stride+64),0xb7);for(int r=0;r<rows;++r)std::copy_n(w.data()+r*stride,stride,padded.data()+r*(stride+64));
        std::vector<int32_t>ids{8,3,7,3,0,8,1,4,5};Buffer table(padded.size()),tokens(ids.size()*4);table.put(padded,s);tokens.put(ids,s);
        ck(cudaMemsetAsync(f.p,0xbd,f.bytes,s));strata::kernels::iq_embed_rows(kind.type,table.p,stride+64,tokens.as<int32_t>(),rows,cols,f.as<float>(),s);
        fv=f.get<float>(s);for(int r=0;r<rows;++r)for(int c=0;c<cols;++c){check(fv[size_t(r)*cols+c]==wf[size_t(ids[r])*cols+c],"IQ embedding selection/stride oracle");++dequant_values;}
        for(size_t i=wf.size();i<fv.size();++i){check(std::bit_cast<uint32_t>(fv[i])==0xbdbdbdbd,"IQ embedding guard");++guards;}
        check(table.get<uint8_t>(s)==padded && input.get<uint8_t>(s)==w && second.get<uint8_t>(s)==up,"IQ modified input");++iq_cases;
    }
    // Original dense GEMM methods: asymmetric dimensions, beta and padded Y.
    for(bool bf16:{false,true})for(auto shape:{std::array<int,3>{1,3,7},{5,19,33},{65,37,256},{129,1280,640}})for(float beta:{0.f,0.5f,1.f}){
        const auto [t,n,k]=shape;const int ld=n+5;std::vector<uint16_t>x(size_t(t)*k),w(size_t(n)*k);
        for(auto& a:x)a=bits(float(int(next()%1025)-512)/512,bf16);for(auto& a:w)a=bits(float(int(next()%1025)-512)/512,bf16);
        std::vector<float>xf(x.size()),wf(w.size());for(size_t i=0;i<x.size();++i)xf[i]=value(x[i],bf16);for(size_t i=0;i<w.size();++i)wf[i]=value(w[i],bf16);
        Buffer dx(x.size()*2),dw(w.size()*2),dy((size_t(t)*ld+16)*4);dx.put(x,s);dw.put(w,s);
        std::vector<float>initial(dy.bytes/4,-1234567.f);for(int r=0;r<t;++r)std::fill_n(initial.begin()+size_t(r)*ld,n,beta==0?std::numeric_limits<float>::quiet_NaN():.375f);dy.put(initial,s);
        {Gemm gemm;std::string error;check(gemm.init(s,0,error),"original GEMM init");if(bf16)gemm.bf16(dx.as<uint16_t>(),dw.as<uint16_t>(),dy.as<float>(),t,n,k,ld,beta);else gemm.f16(dx.as<uint16_t>(),dw.as<uint16_t>(),dy.as<float>(),t,n,k,ld,beta);
            compare(dy.get<float>(s),xf,wf,t,n,k,ld,beta,.375f);}
        check(dx.get<uint16_t>(s)==x && dw.get<uint16_t>(s)==w,"GEMM modified input");++gemm_cases;
    }
    // All seventeen native formats, through the unchanged original slicing path.
    for(const auto& kind:kinds)for(int scratch_rows:{2,19}){
        constexpr int t=9,n=7,k=512,ld=n+3;auto w=weights(kind,n,k);auto wf=decode(kind,w,n,k);for(auto& v:wf)v=value(bits(v,false),false);
        std::vector<uint16_t>x(t*k);std::vector<float>xf(x.size());for(size_t i=0;i<x.size();++i){x[i]=bits(float(int(next()%257)-128)/128,false);xf[i]=value(x[i],false);}
        Buffer dx(x.size()*2),dw(w.size()),dy((t*ld+16)*4),scratch(size_t(scratch_rows)*k*2),workspace(4096);dx.put(x,s);dw.put(w,s);
        std::vector<float>initial(t*ld+16,-1234567.f);for(int r=0;r<t;++r)std::fill_n(initial.begin()+r*ld,n,.375f);dy.put(initial,s);
        Gemm gemm;std::string error;check(gemm.init_external(s,scratch.as<uint16_t>(),int64_t(scratch_rows)*k,workspace.p,workspace.bytes,error),"original external GEMM init");
        gemm.native(dx.as<uint16_t>(),kind.type,dw.p,dy.as<float>(),t,n,k,ld,.5f);compare(dy.get<float>(s),xf,wf,t,n,k,ld,.5f,.375f);++gemm_cases;
        // Rebind to a different borrowed scratch; replay reads changed weights.
        if(scratch_rows==2){Buffer replacement(size_t(3)*k*2);gemm.rebind(replacement.as<uint16_t>(),3*k,workspace.p,workspace.bytes);
            ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));gemm.native(dx.as<uint16_t>(),kind.type,dw.p,dy.as<float>(),t,n,k,ld,0.f);cudaGraph_t graph{};cudaGraphExec_t exec{};
            ck(cudaStreamEndCapture(s,&graph));ck(cudaGraphInstantiate(&exec,graph,0ull));ck(cudaGraphDestroy(graph));
            for(int repeat=0;repeat<3;++repeat){w=weights(kind,n,k);wf=decode(kind,w,n,k);for(auto& v:wf)v=value(bits(v,false),false);dw.put(w,s);dy.put(initial,s);
                ck(cudaGraphLaunch(exec,s));compare(dy.get<float>(s),xf,wf,t,n,k,ld,0,0);}
            ck(cudaGraphExecDestroy(exec));++graph_cases;
        }
    }
    // Ordinary default-stream GEMM retains legacy ordering. CUDA forbids
    // capture on the legacy stream, so cold capture uses a named stream above.
    {
        constexpr int t=3,n=11,k=47,ld=n+2;std::vector<uint16_t>x(t*k,bits(.5f,true)),w(n*k,bits(2,true));std::vector<float>xf(t*k,.5f),wf(n*k,2),initial(t*ld+16,-1234567.f);
        Buffer dx(x.size()*2),dw(w.size()*2),dy(initial.size()*4);dx.put(x,nullptr);dw.put(w,nullptr);dy.put(initial,nullptr);ck(cudaDeviceSynchronize());
        Gemm gemm;std::string error;check(gemm.init(nullptr,0,error),"default-stream GEMM init");gemm.bf16(dx.as<uint16_t>(),dw.as<uint16_t>(),dy.as<float>(),t,n,k,ld);
        compare(dy.get<float>(nullptr),xf,wf,t,n,k,ld,0,0);++gemm_cases;
    }
    // Transposed/ordinary real operands and the zero-K beta-only product.
    size_t blas_cases=0;
    for(auto ta:{CUBLAS_OP_N,CUBLAS_OP_T,CUBLAS_OP_C})for(auto tb:{CUBLAS_OP_N,CUBLAS_OP_T,CUBLAS_OP_C}){
        constexpr int m=5,n=3,k=7,ldc=8;const int ar=ta==CUBLAS_OP_N?m:k,ac=ta==CUBLAS_OP_N?k:m;
        const int br=tb==CUBLAS_OP_N?k:n,bc=tb==CUBLAS_OP_N?n:k,lda=ar+2,ldb=br+3;
        std::vector<uint16_t>a(size_t(lda)*ac,bits(100,false)),b(size_t(ldb)*bc,bits(100,false));
        for(int j=0;j<ac;++j)for(int i=0;i<ar;++i)a[size_t(j)*lda+i]=bits(float(int(next()%129)-64)/64,false);
        for(int j=0;j<bc;++j)for(int i=0;i<br;++i)b[size_t(j)*ldb+i]=bits(float(int(next()%129)-64)/64,false);
        std::vector<float>initial(n*ldc,-1234567.f);for(int j=0;j<n;++j)std::fill_n(initial.begin()+j*ldc,m,.375f);
        Buffer da(a.size()*2),db(b.size()*2),dc(initial.size()*4);da.put(a,s);db.put(b,s);dc.put(initial,s);
        cublasHandle_t handle{};check(cublasCreate(&handle)==CUBLAS_STATUS_SUCCESS && cublasSetStream(handle,s)==CUBLAS_STATUS_SUCCESS,"orientation BLAS init");
        float alpha=1.25f,beta=.5f;check(cublasGemmEx(handle,ta,tb,m,n,k,&alpha,da.p,CUDA_R_16F,lda,db.p,CUDA_R_16F,ldb,&beta,dc.p,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)==CUBLAS_STATUS_SUCCESS,"orientation BLAS product");
        auto out=dc.get<float>(s);for(int j=0;j<n;++j){for(int i=0;i<m;++i){double sum=beta*.375;
            for(int z=0;z<k;++z)sum+=alpha*double(value(a[ta==CUBLAS_OP_N?size_t(z)*lda+i:size_t(i)*lda+z],false))*value(b[tb==CUBLAS_OP_N?size_t(j)*ldb+z:size_t(z)*ldb+j],false);
            check(std::abs(out[j*ldc+i]-sum)<1e-6,"orientation BLAS FP64 reference");++numeric;
        }for(int i=m;i<ldc;++i){check(out[j*ldc+i]==-1234567.f,"orientation BLAS guard");++guards;}}
        check(cublasDestroy(handle)==CUBLAS_STATUS_SUCCESS,"orientation BLAS destroy");++blas_cases;
    }
    {
        cublasHandle_t handle{};check(cublasCreate(&handle)==CUBLAS_STATUS_SUCCESS && cublasSetStream(handle,s)==CUBLAS_STATUS_SUCCESS,"zero-K BLAS init");
        std::vector<float>initial{2,3,4,-1234567,5,6,7,-1234567};Buffer dc(initial.size()*4);dc.put(initial,s);float alpha=1,beta=.5f;
        check(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,3,2,0,&alpha,nullptr,CUDA_R_16F,1,nullptr,CUDA_R_16F,1,&beta,dc.p,CUDA_R_32F,4,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)==CUBLAS_STATUS_SUCCESS,"zero-K BLAS product");
        const auto out=dc.get<float>(s);for(size_t i=0;i<initial.size();++i){check(out[i]==(i%4==3?initial[i]:initial[i]*beta),"zero-K BLAS output/padding");++numeric;}
        check(cublasDestroy(handle)==CUBLAS_STATUS_SUCCESS,"zero-K BLAS destroy");++blas_cases;
    }
    // cuBLAS setup limits and native submission behind a held callback.
    {
        cublasHandle_t h{};check(cublasCreate(&h)==CUBLAS_STATUS_SUCCESS,"BLAS create");check(cublasSetStream(h,s)==CUBLAS_STATUS_SUCCESS,"BLAS stream");
        check(cublasSetWorkspace(h,nullptr,0)==CUBLAS_STATUS_NOT_SUPPORTED,"workspace falsely accepted");
        Buffer a(512),b(512),c(1024);std::vector<uint16_t>av(256,bits(1,false)),bv(256,bits(2,false));a.put(av,s);b.put(bv,s);ck(cudaStreamSynchronize(s));
        Gate gate;hold(s,gate);float alpha=1,beta=0;auto submit=std::async(std::launch::async,[&]{return cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,16,16,16,&alpha,a.p,CUDA_R_16F,16,b.p,CUDA_R_16F,16,&beta,c.p,CUDA_R_32F,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);});
        const bool ready=submit.wait_for(2s)==std::future_status::ready;if(!ready)gate.open();check(submit.get()==CUBLAS_STATUS_SUCCESS && ready,"GEMM host submission blocked on held stream");
        ck(cudaMemsetAsync(c.p,0,c.bytes,other));ck(cudaStreamSynchronize(other));check(cudaStreamQuery(s)==cudaErrorNotReady,"GEMM synchronized unrelated stream");gate.open();
        const auto out=c.get<float>(s);for(float v:out)check(v==32,"GEMM lost stream dependency");
        check(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,16,16,16,&alpha,a.p,CUDA_R_16F,15,b.p,CUDA_R_16F,16,&beta,c.p,CUDA_R_32F,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)==CUBLAS_STATUS_INVALID_VALUE,"invalid leading dimension accepted");
        check(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,16,16,16,&alpha,a.p,CUDA_R_16F,16,b.p,CUDA_R_16BF,16,&beta,c.p,CUDA_R_32F,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)==CUBLAS_STATUS_NOT_SUPPORTED,"unsupported mixed input types accepted");
        check(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,16,16,16,&alpha,a.as<uint16_t>()+1,CUDA_R_16F,16,b.p,CUDA_R_16F,16,&beta,c.p,CUDA_R_32F,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)==CUBLAS_STATUS_INVALID_VALUE,"undersized input accepted");cudaGetLastError();
        check(c.get<float>(s)==out,"invalid BLAS call changed output");
        check(cublasDestroy(h)==CUBLAS_STATUS_SUCCESS && cublasSetStream(h,s)==CUBLAS_STATUS_NOT_INITIALIZED,"stale BLAS handle");
    }
    ck(cudaStreamDestroy(s));ck(cudaStreamDestroy(other));
    std::cout<<"PASS dequant_cases="<<dequant_cases<<" iq_cases="<<iq_cases<<" dequant_values="<<dequant_values<<" gemm_cases="<<gemm_cases<<" blas_cases="<<blas_cases<<" graph_cases="<<graph_cases<<" numeric_checks="<<numeric<<" guard_checks="<<guards<<" max_product_error_over_L1="<<worst_l1<<'\n';
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
