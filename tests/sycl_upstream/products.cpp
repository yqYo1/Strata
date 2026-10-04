#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/sycl_upstream/cuda_backend.hpp"
#include <ggml.h>
#include <sycl/ext/intel/math.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <set>
#include <vector>

namespace k=strata::kernels;
namespace backend=strata::sycl_upstream::cuda;
using namespace std::chrono_literals;
std::string current_case;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(current_case+": "+why);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorName(e))+": "+backend::backend_error_detail());}
struct Buffer {
    void* p{};size_t bytes;
    explicit Buffer(size_t n):bytes(n){ck(cudaMalloc(&p,n));}
    ~Buffer(){cudaFree(p);}
    template<class T>T* as(){return static_cast<T*>(p);}
    template<class T>void put(const std::vector<T>& v,cudaStream_t s){check(v.size()*sizeof(T)<=bytes,"input extent");ck(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));}
    template<class T>std::vector<T> get(cudaStream_t s){std::vector<T> v(bytes/sizeof(T));ck(cudaMemcpyAsync(v.data(),p,bytes,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
};
struct alignas(4) Q81 {uint16_t d,s;int8_t q[32];};
static_assert(sizeof(Q81)==36);
uint32_t random_state=0x713586ac;
uint32_t next(){random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return random_state;}
float hfloat(uint16_t v){return ggml_fp16_to_fp32(v);}
std::vector<float> input(int rows,int cols){
    std::vector<float> x(size_t(rows)*cols);
    for(auto& v:x)v=float(int(next()%1025)-512)/4096;
    if(cols>=96){std::fill_n(x.begin(),32,0.f);for(int i=32;i<64;++i)x[i]=(i%2?-.25f:.25f);for(int i=64;i<96;++i)x[i]=float(i-80)/64;}
    return x;
}
std::vector<Q81> quantize(const std::vector<float>& x){
    check(x.size()%32==0,"CPU quantizer shape");std::vector<Q81> q(x.size()/32);
    for(size_t b=0;b<q.size();++b){std::array<float,32> sum,amax;
        for(int i=0;i<32;++i){sum[i]=x[b*32+i];amax[i]=std::abs(sum[i]);}
        for(int mask=16;mask;mask>>=1){const auto old=sum,maxold=amax;for(int i=0;i<32;++i){sum[i]=old[i]+old[i^mask];amax[i]=std::max(maxold[i],maxold[i^mask]);}}
        const float d=amax[0]/127.f;q[b].d=ggml_fp32_to_fp16(d);q[b].s=ggml_fp32_to_fp16(sum[0]);
        for(int i=0;i<32;++i)q[b].q[i]=amax[0]==0.f?0:int8_t(std::round(x[b*32+i]/d));
    }return q;
}
struct Weights {
    int type,rows,cols,qk,bb;size_t stride;std::vector<uint8_t> raw;std::vector<float> values;
    Weights(int ty,int r,int c):type(ty),rows(r),cols(c),qk(int(ggml_blck_size(ggml_type(ty)))),bb(int(ggml_type_size(ggml_type(ty)))),stride(size_t(c/qk)*bb),raw(size_t(r)*stride),values(size_t(r)*c){
        for(size_t b=0;b<raw.size()/bb;++b){auto* block=raw.data()+b*bb;for(int i=0;i<bb;++i)block[i]=uint8_t(next());
            uint16_t d=uint16_t(0x1000+next()%0x700);if(b%11==0)d=0;else if(b%3==0)d|=0x8000;
            const size_t at=type==11?108:type==14?208:0;
            if(type==29){for(int i=0;i<4;++i){uint16_t scale;std::memcpy(&scale,block+bb-8+2*i,2);scale=(scale&0xfff)|uint16_t(((d>>(4*i))&15)<<12);std::memcpy(block+bb-8+2*i,&scale,2);}}
            else{std::memcpy(block+at,&d,2);if(type==7 || type==12 || type==13){uint16_t m=0x1000;std::memcpy(block+2,&m,2);}}
        }
        const auto decode=ggml_get_type_traits(ggml_type(type))->to_float;
        for(int row=0;row<rows;++row)decode(raw.data()+size_t(row)*stride,values.data()+size_t(row)*cols,cols);
    }
    std::pair<double,double> dot(int row,const Q81* x)const{
        double total=0,l1=0;
        for(int b=0;b<cols/32;++b){const double scale=hfloat(x[b].d);for(int i=0;i<32;++i){const double term=double(values[size_t(row)*cols+b*32+i])*scale*int(x[b].q[i]);total+=term;l1+=std::abs(term);}}
        // Q4_0/Q5_0 consume the original-input sum in Q8_1 for their
        // zero-point term. Reconstructed integer-code sums are different.
        if(type==2 || type==6)for(int b=0;b<cols/32;++b){int codes=0;for(int i=0;i<32;++i)codes+=x[b].q[i];uint16_t d;std::memcpy(&d,raw.data()+size_t(row)*stride+b*bb,2);
            const double correction=-hfloat(d)*(type==2?8:16)*(double(hfloat(x[b].s))-double(hfloat(x[b].d))*codes);total+=correction;l1+=std::abs(correction);}
        return {total,l1};
    }
};
size_t numeric=0,guards=0,bytes_checked=0,bitwise=0;double maximum_error=0;
void value_check(float got,double reference,double l1,const char* why){
    const double error=std::abs(double(got)-reference);maximum_error=std::max(maximum_error,error/(l1+1e-30));
    if(!std::isfinite(got) || error>3e-5*l1+2e-6){std::cerr<<"got="<<got<<" reference="<<reference<<" L1="<<l1<<'\n';throw std::runtime_error(current_case+": "+why);}++numeric;
}
void product_check(const std::vector<float>& out,const Weights& w,const std::vector<Q81>& x,int cols){
    for(int c=0;c<cols;++c)for(int r=0;r<w.rows;++r){auto [v,l1]=w.dot(r,x.data()+size_t(c)*w.cols/32);value_check(out[size_t(c)*w.rows+r],v,l1,"product differs from pinned CPU dequantization and Q8_1 bytes");}
    for(size_t i=size_t(cols)*w.rows;i<out.size();++i){check(out[i]==-1234567.f,"product guard");++guards;}
}
void equal_bits(const std::vector<float>& a,const std::vector<float>& b,const char* why){check(a.size()==b.size(),"comparison extent");for(size_t i=0;i<a.size();++i){check(std::bit_cast<uint32_t>(a[i])==std::bit_cast<uint32_t>(b[i]),why);++bitwise;}}
void quant_check(const std::vector<uint8_t>& got,const std::vector<Q81>& cpu,const std::vector<float>& x){
    const auto* expected=reinterpret_cast<const uint8_t*>(cpu.data());
    for(size_t i=0;i<cpu.size()*36;++i)if(got[i]!=expected[i]){
        const size_t block=i/36,part=i%36;
        std::cerr<<"quant block="<<block<<" byte="<<part<<" gpu="<<int(got[i])<<" cpu="<<int(expected[i])<<" d="<<hfloat(cpu[block].d)<<" sum="<<hfloat(cpu[block].s);
        if(part>=4){
            float amax=0;for(int lane=0;lane<32;++lane)amax=std::max(amax,std::abs(x[block*32+lane]));
            const float xi=x[block*32+part-4],d=amax/127.f;
            std::cerr<<" input="<<xi<<" amax="<<amax<<" unrounded_d="<<std::hexfloat<<d<<" ratio="<<xi/d<<std::defaultfloat;
            Buffer dx(32*4),dr(32*7*4);std::vector<float> block_x(x.begin()+block*32,x.begin()+(block+1)*32);
            cudaStream_t ds{};ck(cudaStreamCreateWithFlags(&ds,cudaStreamNonBlocking));dx.put(block_x,ds);auto* p=dx.as<float>();auto* r=dr.as<float>();
            ck(backend::submit(ds,[=](sycl::queue& q){return q.parallel_for(sycl::nd_range<1>(32,32),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int lane=it.get_local_linear_id();const float xi=p[lane];float a=sycl::fabs(xi);
                for(int mask=16;mask;mask>>=1)a=sycl::fmax(a,sycl::permute_group_by_xor(it.get_sub_group(),a,mask));
                const float d=a/127.f,v=xi/d,dn=sycl::ext::intel::math::fdiv_rn(a,127.f),vn=sycl::ext::intel::math::fdiv_rn(xi,dn);
                r[lane*7]=a;r[lane*7+1]=d;r[lane*7+2]=v;r[lane*7+3]=roundf(v);r[lane*7+4]=dn;r[lane*7+5]=vn;r[lane*7+6]=roundf(vn);
            });}));
            auto values=dr.get<float>(ds);ck(cudaStreamDestroy(ds));std::cerr<<" GPU_tree[a,d,ratio,round,d_rn,ratio_rn,round_rn]="<<std::hexfloat;
            for(int j=0;j<7;++j)std::cerr<<values[(part-4)*7+j]<<',';std::cerr<<std::defaultfloat;
        }
        std::cerr<<'\n';check(false,"Q8_1 bytes differ from independent CPU tree");
    }bytes_checked+=cpu.size()*36;
}

#include "product_groups.hpp"

void native_f32(int type,const void* w,const float* x,void* scratch,float* y,int ni,int no,int nc,cudaStream_t s){
    using F=void(*)(const void*,const float*,void*,float*,int,int,int,void*);
    F fn{};
    switch(type){
        case 42:fn=k::native_q2_0_f32;break;case 2:fn=k::native_q4_0_f32;break;
        case 6:fn=k::native_q5_0_f32;break;case 8:fn=k::native_q8_0_f32;break;
        case 11:fn=k::native_q3_k_f32;break;case 12:fn=k::native_q4_k_f32;break;
        case 13:fn=k::native_q5_k_f32;break;case 14:fn=k::native_q6_k_f32;break;
        case 20:fn=k::native_iq4_nl_f32;break;case 23:fn=k::native_iq4_xs_f32;break;
        default:check(false,"unknown native F32 composition type");
    }
    fn(w,x,scratch,y,ni,no,nc,s);
}
void product_graph(int type,bool native,cudaStream_t s,cudaStream_t other,size_t& cases){
    current_case="product graph type="+std::to_string(type)+" native="+std::to_string(native);
    constexpr int width=512,rows=5,cols=3;Weights w(type,rows,width);auto x=input(cols,width);auto cpuq=quantize(x);
    Buffer dw(w.raw.size()),dx(x.size()*4),dq(cpuq.size()*36+64),dy((cols*rows+16)*4);
    dw.put(w.raw,s);dx.put(x,s);std::vector<float> initial(dy.bytes/4,-1234567.f);dy.put(initial,s);ck(cudaMemsetAsync(dq.p,0xbd,dq.bytes,s));
    ck(cudaStreamSynchronize(s));k::native_mmvq_set_multi_exact(true);k::iq_set_old_kernels(false);
    cudaGraph_t graph{};cudaGraphExec_t exec{};ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
    if(native)native_f32(type,dw.p,dx.as<float>(),dq.p,dy.as<float>(),width,rows,cols,s);
    else{k::quantize_q8_1_rows(dx.as<float>(),cols,width,dq.p,s);k::iq_mmvq(type,dw.p,dq.p,dy.as<float>(),width,rows,cols,s);}
    ck(cudaStreamEndCapture(s,&graph));size_t count=0;ck(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);ck(cudaGraphGetNodes(graph,nodes.data(),&count));int kernels=0;
    for(auto node:nodes){cudaGraphNodeType nt;ck(cudaGraphNodeGetType(node,&nt));kernels+=nt==cudaGraphNodeTypeKernel;}check(kernels==2,"product capture must have quantization and product kernels");
    ck(cudaGraphInstantiate(&exec,graph,0ull));ck(cudaGraphDestroy(graph));check(dy.get<float>(other)==initial,"product capture executed before launch");
    k::native_mmvq_set_multi_exact(false);k::iq_set_old_kernels(true);
    for(int replay=0;replay<3;++replay){x=input(cols,width);cpuq=quantize(x);dx.put(x,s);dy.put(initial,s);ck(cudaGraphLaunch(exec,s));const auto got=dy.get<float>(s);product_check(got,w,cpuq,cols);auto q=dq.get<uint8_t>(s);quant_check(q,cpuq,x);
        for(size_t i=cpuq.size()*36;i<q.size();++i){check(q[i]==0xbd,"captured Q8_1 guard");++guards;}++cases;}
    ck(cudaGraphExecDestroy(exec));k::native_mmvq_set_multi_exact(true);k::iq_set_old_kernels(false);
}

int main()try{
    auto* ctx=ggml_init({16384,nullptr,true});check(ctx,"GGML CPU initialization");ggml_free(ctx);
    ck(cudaSetDevice(0));cudaStream_t s{},other{};ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
    sycl::device device;ck(backend::stream_device(s,&device));std::cout<<"device="<<device.get_info<sycl::info::device::name>()<<'\n';
    size_t native_cases=0,iq_cases=0,bf16_cases=0,group_cases=0,graph_cases=0;
    // Every native format, both original matrix layouts, every column count,
    // both sides of the small-K branch, and an incomplete four-row group.
    for(int type:{42,2,6,8,11,12,13,14,20,23})for(int width:{512,8192}){
        Weights w(type,5,width);Buffer dw(w.raw.size());dw.put(w.raw,s);
        for(int cols=1;cols<=8;++cols){current_case="native type="+std::to_string(type)+" width="+std::to_string(width)+" cols="+std::to_string(cols);auto x=input(cols,width);auto cpuq=quantize(x);Buffer dx(x.size()*4),dq(cpuq.size()*36+64),dy((size_t(cols)*5+16)*4);dx.put(x,s);ck(cudaMemsetAsync(dq.p,0xbd,dq.bytes,s));
            k::native_quantize_q8_1(dx.as<float>(),dq.p,width,cols,s);auto gpuq=dq.get<uint8_t>(s);quant_check(gpuq,cpuq,x);
            for(size_t i=cpuq.size()*36;i<gpuq.size();++i){check(gpuq[i]==0xbd,"native quantizer guard");++guards;}
            std::vector<float> initial(dy.bytes/4,-1234567.f);std::vector<float> exact;
            for(bool mode:{true,false}){k::native_mmvq_set_multi_exact(mode);dy.put(initial,s);k::native_mmvq(type,dw.p,dq.p,dy.as<float>(),width,5,cols,s);auto out=dy.get<float>(s);product_check(out,w,cpuq,cols);if(mode)exact=out;++native_cases;}
            k::native_mmvq_set_multi_exact(true);
            Buffer one((5+16)*4);std::vector<float> oneinit(21,-1234567.f);
            for(int c=0;c<cols;++c){one.put(oneinit,s);k::native_mmvq(type,dw.p,static_cast<const uint8_t*>(dq.p)+size_t(c)*width/32*36,one.as<float>(),width,5,1,s);auto got=one.get<float>(s);
                for(int r=0;r<5;++r){check(std::bit_cast<uint32_t>(got[r])==std::bit_cast<uint32_t>(exact[c*5+r]),"native exact multi-column differs from individual column");++bitwise;}}
            check(dx.get<float>(s)==x && dw.get<uint8_t>(s)==w.raw,"native product modified inputs");
        }
    }
    // The IQ API has fourteen product types; its broader support query also
    // includes Q3_K for dequantization only. Both old/new paths are retained.
    for(int type:{16,17,18,20,21,22,23,29,42,12,13,7,6,8})for(int width:{256,1280,2560}){
        Weights w(type,5,width);Buffer dw(w.raw.size());dw.put(w.raw,s);
        for(int cols:{1,3,4,5,8,9}){current_case="IQ type="+std::to_string(type)+" width="+std::to_string(width)+" cols="+std::to_string(cols);auto x=input(cols,width);auto cpuq=quantize(x);Buffer dx(x.size()*4),dq(cpuq.size()*36+64),dy((size_t(cols)*5+16)*4);dx.put(x,s);ck(cudaMemsetAsync(dq.p,0xbd,dq.bytes,s));
            k::quantize_q8_1_rows(dx.as<float>(),cols,width,dq.p,s);auto gpuq=dq.get<uint8_t>(s);quant_check(gpuq,cpuq,x);
            for(size_t i=cpuq.size()*36;i<gpuq.size();++i){check(gpuq[i]==0xbd,"IQ quantizer guard");++guards;}
            std::vector<float> initial(dy.bytes/4,-1234567.f),old;
            for(bool mode:{true,false}){k::iq_set_old_kernels(mode);dy.put(initial,s);k::iq_mmvq(type,dw.p,dq.p,dy.as<float>(),width,5,cols,s);auto out=dy.get<float>(s);product_check(out,w,cpuq,cols);if(mode)old=out;else equal_bits(out,old,"IQ decode-once differs from original per-column product");++iq_cases;}
            check(dx.get<float>(s)==x && dw.get<uint8_t>(s)==w.raw,"IQ product modified inputs");
        }
    }
    k::iq_set_old_kernels(false);
    std::cout<<"PASS native and IQ product cases="<<native_cases+iq_cases<<'\n';
    for(int type:{42,2,6,8,11,12,13,14,20,23})product_graph(type,true,s,other,graph_cases);
    for(int type:{16,17,18,20,21,22,23,29,42,12,13,7,6,8})product_graph(type,false,s,other,graph_cases);
    for(int gu:{16,17,18,21,22,23,29,42,12,13,6,8})for(int down:{20,23,42,7,6,8})grouped_test(gu,down,256,256,s,other,group_cases,graph_cases,false);
    grouped_test(42,42,2560,640,s,other,group_cases,graph_cases,false);
    grouped_test(21,20,2560,640,s,other,group_cases,graph_cases,true);
    check(!k::native_expert_supported(14,8,256,256) && !k::native_expert_supported(21,20,257,256),"unsupported group format/dimension admitted");
    std::cout<<"PASS grouped cases="<<group_cases<<" graph replays="<<graph_cases<<'\n';
    // All native MMVF block-size choices, padded token strides, all 1..8
    // token variants, and original BF16 warp/split/naive dispatch branches.
    for(int width:{2,62,64,66,130,194,256,320,384,448,512,640,2560}){
        current_case="BF16 width="+std::to_string(width);
        constexpr int rows=7;const int ldx=width+4,ldy=rows+3;auto x=input(8,ldx);std::vector<uint16_t>w(size_t(rows)*width),xb(x.size());
        for(auto& v:w)v=ggml_fp32_to_bf16(float(int(next()%257)-128)/1024).bits;for(size_t i=0;i<x.size();++i)xb[i]=ggml_fp32_to_bf16(x[i]).bits;
        Buffer dx(x.size()*4),dw(w.size()*2),dy((8*ldy+16)*4),single((rows+16)*4),dxb(xb.size()*2);dx.put(x,s);dw.put(w,s);dxb.put(xb,s);
        std::vector<float> initial(dy.bytes/4,-1234567.f),oneinitial(rows+16,-1234567.f);
        for(int cols=1;cols<=8;++cols){dy.put(initial,s);k::bf16_gemv_fp32_mmvf_multi(dx.as<float>(),ldx,dw.as<uint16_t>(),dy.as<float>(),ldy,width,rows,cols,s);const auto out=dy.get<float>(s);
            for(int c=0;c<cols;++c){single.put(oneinitial,s);k::bf16_gemv_fp32_mmvf(dx.as<float>()+c*ldx,dw.as<uint16_t>(),single.as<float>(),width,rows,s);auto one=single.get<float>(s);
                for(int r=0;r<rows;++r){double sum=0,l1=0;for(int i=0;i<width;++i){double term=double(x[c*ldx+i])*std::bit_cast<float>(uint32_t(w[r*width+i])<<16);sum+=term;l1+=std::abs(term);}value_check(out[c*ldy+r],sum,l1,"native BF16 FP64 reference");check(std::bit_cast<uint32_t>(out[c*ldy+r])==std::bit_cast<uint32_t>(one[r]),"native BF16 multi-token order changed");++bitwise;}
            }
            for(size_t i=0;i<out.size();++i)if(i>=size_t(cols)*ldy || int(i%ldy)>=rows){check(out[i]==-1234567.f,"native BF16 stride/guard");++guards;}++bf16_cases;
        }
        for(int tpr:{1,16,32,64,128,256}){single.put(oneinitial,s);k::bf16_gemv_split(dxb.as<uint16_t>(),dw.as<uint16_t>(),single.as<float>(),width,rows,tpr,s);auto out=single.get<float>(s);
            for(int r=0;r<rows;++r){double sum=0,l1=0;for(int i=0;i<width;++i){double term=double(std::bit_cast<float>(uint32_t(xb[i])<<16))*std::bit_cast<float>(uint32_t(w[r*width+i])<<16);sum+=term;l1+=std::abs(term);}value_check(out[r],sum,l1,"original BF16 split FP64 reference");}for(size_t i=rows;i<out.size();++i){check(out[i]==-1234567.f,"BF16 split guard");++guards;}++bf16_cases;}
    }
    for(int rows:{7,65}){
        current_case="BF16 ordinary rows="+std::to_string(rows);constexpr int width=130;
        auto xf=input(1,width);std::vector<uint16_t> x(width),w(size_t(rows)*width);
        for(int i=0;i<width;++i)x[i]=ggml_fp32_to_bf16(xf[i]).bits;
        for(auto& v:w)v=ggml_fp32_to_bf16(float(int(next()%257)-128)/1024).bits;
        Buffer dx(x.size()*2),dw(w.size()*2),dy((rows+16)*4);dx.put(x,s);dw.put(w,s);std::vector<float> initial(rows+16,-1234567.f);dy.put(initial,s);
        k::bf16_gemv(dx.as<uint16_t>(),dw.as<uint16_t>(),dy.as<float>(),width,rows,s);const auto out=dy.get<float>(s);
        for(int r=0;r<rows;++r){double sum=0,l1=0;for(int i=0;i<width;++i){const double term=double(std::bit_cast<float>(uint32_t(x[i])<<16))*std::bit_cast<float>(uint32_t(w[r*width+i])<<16);sum+=term;l1+=std::abs(term);}value_check(out[r],sum,l1,"original BF16 naive/warp CPU reference");}
        for(size_t i=rows;i<out.size();++i){check(out[i]==-1234567.f,"ordinary BF16 guard");++guards;}++bf16_cases;
    }
    ck(cudaStreamDestroy(s));ck(cudaStreamDestroy(other));
    std::cout<<"PASS native_cases="<<native_cases<<" iq_cases="<<iq_cases<<" bf16_cases="<<bf16_cases<<" group_cases="<<group_cases<<" graph_cases="<<graph_cases<<" numerical_checks="<<numeric<<" byte_checks="<<bytes_checked<<" bitwise_checks="<<bitwise<<" guard_checks="<<guards<<" max_error_over_L1="<<maximum_error<<'\n';
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
