// Main-authored independent chunk64 GDN implementation; repository LICENSE.
// Mathematical/numerical design references are pinned in the package manifest.
// No external kernel source copied. Private proposal, not product adoption.
#include "candidate.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
namespace strata::gdn_chunk64 {
namespace {
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace mi = sycl::ext::intel::experimental::matrix;
using namespace sycl_backend;
constexpr int BT=64, D=128, HK=16, HV=48, C=10240, WG=128;
template<int N> struct alignas(4) Tile { BF values[N]; };
static_assert(sizeof(BF)==2 && std::is_trivially_copyable_v<BF>);
static_assert(sizeof(Tile<4096>)==8192 && alignof(Tile<4096>)>=4);
using Left=mx::joint_matrix<sycl::sub_group,BF,mx::use::a,8,16,mx::layout::row_major>;
using TransposedLeft=mx::joint_matrix<sycl::sub_group,BF,mx::use::a,8,16,mx::layout::col_major>;
using Right=mx::joint_matrix<sycl::sub_group,BF,mx::use::b,16,16,mx::layout::row_major>;
using Transposed=mx::joint_matrix<sycl::sub_group,BF,mx::use::b,16,16,mx::layout::col_major>;
using Acc=mx::joint_matrix<sycl::sub_group,float,mx::use::accumulator,8,16>;
template<bool Transpose=false,bool TransposeA=false>
void product(sycl::sub_group sg, Acc &acc, const BF *a, int lda,
             const BF *b, int ldb, int inner) {
  mx::joint_matrix_fill(sg,acc,0.f);
  for(int k=0;k<inner;k+=16) {
    using A=std::conditional_t<TransposeA,TransposedLeft,Left>;
    A left;
    mx::joint_matrix_load(sg,left,a+(TransposeA?k*lda:k),lda);
    if constexpr(Transpose) {
      Transposed right;
      mx::joint_matrix_load(sg,right,b+k,ldb);
      mx::joint_matrix_mad(sg,acc,left,right,acc);
    } else {
      Right right;
      mx::joint_matrix_load(sg,right,b+k*ldb,ldb);
      mx::joint_matrix_mad(sg,acc,left,right,acc);
    }
  }
}
void require(bool ok) { if(!ok) throw std::invalid_argument("GDN chunk64 contract"); }
void matrix_check(const sycl::device &device) {
  require(device.is_gpu() && device.get_info<sycl::info::device::vendor_id>()==0x8086 &&
          device.has(sycl::aspect::ext_intel_matrix));
  require(device.get_info<sycl::info::device::max_work_group_size>()>=WG &&
          device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]>=WG &&
          device.get_info<sycl::info::device::local_mem_size>()>=32768);
  const auto groups=device.get_info<sycl::info::device::sub_group_sizes>();
  require(std::find(groups.begin(),groups.end(),16)!=groups.end());
  bool found=false;
  for(const auto &x:device.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
    found |= x.atype==mx::matrix_type::bf16 && x.btype==mx::matrix_type::bf16 &&
             x.ctype==mx::matrix_type::fp32 && x.dtype==mx::matrix_type::fp32 &&
             (x.msize==8 || (!x.msize && x.max_msize>=8)) &&
             (x.nsize==16 || (!x.nsize && x.max_nsize>=16)) &&
             (x.ksize==16 || (!x.ksize && x.max_ksize>=16));
  require(found);
}
}
Sizes sizes(int tokens) {
  require(tokens>0 && tokens<=8192);
  const size_t chunks=(tokens+BT-1)/BT, padded=chunks*BT;
  return {padded*HK*D,padded*HV*D,padded*HV*BT,
          chunks*HV*D*D,padded*HV,padded*HV*BT};
}
void run(float *state,const uint16_t *h,const float *gate,const float *beta,
         const uint16_t *z,const float *gamma,float eps,uint16_t *output,
         int tokens,Workspace ws,void *stream) {
  const Sizes n=sizes(tokens);
  require(stream && std::isfinite(eps) && eps>=0.f);
  auto &queue=queue_for(stream);
  matrix_check(queue.get_device());
  const std::initializer_list<Span> outputs={
    {state,size_t(HV)*D*D*4},{output,size_t(tokens)*HV*D*2},
    {ws.q,n.qk*2},{ws.k,n.qk*2},{ws.v,n.values*2},
    {ws.kb,n.values*2},{ws.vb,n.values*2},{ws.inverse,n.inverse*2},
    {ws.w,n.values*2},{ws.u,n.values*2},{ws.history,n.history*2},
    {ws.delta,n.values*2},{ws.raw,n.values*2},{ws.g,n.gates*4},{ws.lower,n.lower*4}};
  const std::initializer_list<Span> inputs={
    {h,size_t(tokens)*C*2},{gate,size_t(tokens)*HV*4},
    {beta,size_t(tokens)*HV*4},{z,size_t(tokens)*HV*D*2},{gamma,D*4}};
  validate_spans(outputs,inputs,4);
  for(auto span:outputs) validate_device_span(span.pointer,span.bytes,queue);
  for(auto span:inputs) validate_device_span(span.pointer,span.bytes,queue);
  const int chunks=(tokens+BT-1)/BT;
  auto event=queue.parallel_for(sycl::range<1>(size_t(chunks)*HV),[=](sycl::id<1> id) {
    const int chunk=int(id[0])/HV, head=int(id[0])%HV;
    float sum=0.f;
    for(int t=0;t<BT;++t) {
      const int token=chunk*BT+t;
      if(token<tokens) sum+=gate[size_t(token)*HV+head];
      ws.g[(size_t(chunk)*HV+head)*BT+t]=sum;
    }
  });
  finish(stream,event);
  event=queue.parallel_for(sycl::range<1>(n.values),[=](sycl::id<1> id) {
    size_t x=id[0]; const int d=x%D; x/=D;
    const int t=x%BT; x/=BT;
    const int head=x%HV, chunk=x/HV, token=chunk*BT+t;
    const bool live=token<tokens;
    const BF k=live?sycl::bit_cast<BF>(h[size_t(token)*C+HK*D+(head%HK)*D+d]):BF(0.f);
    const BF v=live?sycl::bit_cast<BF>(h[size_t(token)*C+2*HK*D+head*D+d]):BF(0.f);
    const float b=live?beta[size_t(token)*HV+head]:0.f;
    const float eg=sycl::exp(ws.g[(size_t(chunk)*HV+head)*BT+t]);
    ws.v[id[0]]=v;
    ws.kb[id[0]]=BF((float(k)*b)*eg);
    ws.vb[id[0]]=BF(float(v)*b);
    if(head<HK) {
      const size_t dst=((size_t(chunk)*HK+head)*BT+t)*D+d;
      ws.q[dst]=live?sycl::bit_cast<BF>(h[size_t(token)*C+head*D+d]):BF(0.f);
      ws.k[dst]=k;
    }
  });
  finish(stream,event);
  // KKT and gate/beta scaling. Four workgroups cover one64x64 matrix.
  event=queue.parallel_for(sycl::nd_range<1>(size_t(chunks)*HV*4*WG,WG),
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int group=item.get_group_linear_id(), matrix=group/4;
      const int head=matrix%HV, chunk=matrix/HV;
      const int tile=(group%4)*8+item.get_sub_group().get_group_linear_id();
      const int row=(tile/4)*8, col=(tile%4)*16;
      const BF *key=ws.k+(size_t(chunk)*HK+head%HK)*BT*D;
      auto sg=item.get_sub_group(); Acc acc;
      product<true>(sg,acc,key+row*D,D,key+col*D,D,D);
      mi::joint_matrix_apply(sg,acc,[=](float &value,size_t r,size_t c) {
        const int i=row+r,j=col+c;
        const bool valid=i>j && chunk*BT+i<tokens && chunk*BT+j<tokens;
        const float b=valid?beta[size_t(chunk*BT+i)*HV+head]:0.f;
        // Mask before exponentiation; never evaluate upper-triangle positive differences.
        const float decay=valid?sycl::exp(ws.g[size_t(matrix)*BT+i]-ws.g[size_t(matrix)*BT+j]):0.f;
        ws.lower[(size_t(matrix)*BT+i)*BT+j]=valid?value*b*decay:0.f;
      });
    });
  finish(stream,event);
  // FP32 forward substitution, then one BF16 inverse storage boundary.
  // This deliberately states the full64 solve contract; it is not asserted
  // identical to SGLang's block16/XPU intermediate rounding.
  event=queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<float,1> inverse(BT*BT,handler);
    handler.parallel_for(sycl::nd_range<1>(size_t(chunks)*HV*64,64),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int matrix=item.get_group_linear_id(), col=item.get_local_linear_id();
        for(int row=0;row<BT;++row) {
          float value=row==col?1.f:0.f;
          if(col<row) for(int p=col;p<row;++p)
            value-=ws.lower[(size_t(matrix)*BT+row)*BT+p]*inverse[p*BT+col];
          inverse[row*BT+col]=value;
          item.barrier(sycl::access::fence_space::local_space);
        }
        for(int row=0;row<BT;++row)
          ws.inverse[(size_t(matrix)*BT+row)*BT+col]=BF(inverse[row*BT+col]);
      });
  });
  finish(stream,event);
  // W/U use native BF16 operands and FP32 matrix accumulation.
  event=queue.parallel_for(sycl::nd_range<1>(size_t(chunks)*HV*8*WG,WG),
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int group=item.get_group_linear_id(), matrix=group/8;
      const int tile=(group%8)*8+item.get_sub_group().get_group_linear_id();
      const int row=(tile/8)*8,col=(tile%8)*16;
      auto sg=item.get_sub_group();Acc acc;
      const BF *a=ws.inverse+size_t(matrix)*BT*BT+row*BT;
      product(sg,acc,a,BT,ws.kb+size_t(matrix)*BT*D+col,D,BT);
      mi::joint_matrix_apply(sg,acc,[=](float &value,size_t r,size_t c) {
        ws.w[(size_t(matrix)*BT+row+r)*D+col+c]=BF(value);
      });
      product(sg,acc,a,BT,ws.vb+size_t(matrix)*BT*D+col,D,BT);
      mi::joint_matrix_apply(sg,acc,[=](float &value,size_t r,size_t c) {
        ws.u[(size_t(matrix)*BT+row+r)*D+col+c]=BF(value);
      });
    });
  finish(stream,event);
  // Persistent carry: one32-value-column slice per workgroup. Only chunk order
  // is serial; each64-token update uses DPAS and keeps the FP32 carry in registers.
  event=queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<Tile<D*32>,1> hs(1,handler);
    sycl::local_accessor<Tile<BT*32>,1> weighted(1,handler);
    sycl::local_accessor<float,1> scratch(D*32,handler);
    handler.parallel_for(sycl::nd_range<1>(HV*4*WG,WG),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int head=item.get_group_linear_id()/4,vblock=item.get_group_linear_id()%4;
        const int tid=item.get_local_linear_id(),rg=tid/32,col=tid%32;
        auto sg=item.get_sub_group();const int subgroup=sg.get_group_linear_id();
        BF *local_h=hs[0].values,*local_v=weighted[0].values;
        float *temp=scratch.template get_multi_ptr<sycl::access::decorated::no>().get();
        float saved[32];
        for(int r=0;r<32;++r) saved[r]=state[(size_t(rg*32+r)*HV+head)*D+vblock*32+col];
        for(int chunk=0;chunk<chunks;++chunk) {
          const size_t matrix=size_t(chunk)*HV+head;
          const int live=sycl::min(BT,tokens-chunk*BT);
          for(int r=0;r<32;++r) {
            const int row=rg*32+r;
            local_h[row*32+col]=BF(saved[r]);
            ws.history[(matrix*D+row)*D+vblock*32+col]=BF(saved[r]);
          }
          item.barrier(sycl::access::fence_space::local_space);
          for(int c=0;c<32;c+=16) {
            Acc acc;
            product(sg,acc,ws.w+matrix*BT*D+subgroup*8*D,D,local_h+c,32,D);
            mx::joint_matrix_store(sg,acc,temp+subgroup*8*32+c,32,mx::layout::row_major);
          }
          item.barrier(sycl::access::fence_space::local_space);
          const float last=ws.g[matrix*BT+live-1];
          for(int i=tid;i<BT*32;i+=WG) {
            const int t=i/32,c=i%32;
            const float delta=t<live?float(ws.u[(matrix*BT+t)*D+vblock*32+c])-temp[i]:0.f;
            ws.delta[(matrix*BT+t)*D+vblock*32+c]=BF(delta);
            local_v[i]=BF(delta*sycl::exp(last-ws.g[matrix*BT+t]));
          }
          item.barrier(sycl::access::fence_space::local_space);
          for(int r=0;r<16;r+=8) for(int c=0;c<32;c+=16) {
            Acc acc;const int row=subgroup*16+r;
            product<false,true>(sg,acc,ws.k+(size_t(chunk)*HK+head%HK)*BT*D+row,D,
                          local_v+c,32,BT);
            mx::joint_matrix_store(sg,acc,temp+row*32+c,32,mx::layout::row_major);
          }
          item.barrier(sycl::access::fence_space::local_space);
          const float decay=sycl::exp(last);
          for(int r=0;r<32;++r) saved[r]=saved[r]*decay+temp[(rg*32+r)*32+col];
          item.barrier(sycl::access::fence_space::local_space);
        }
        for(int r=0;r<32;++r) state[(size_t(rg*32+r)*HV+head)*D+vblock*32+col]=saved[r];
      });
  });
  finish(stream,event);
  // Separate parallel output path; BF16 raw readout precedes normalization.
  event=queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<Tile<BT*BT>,1> scores(1,handler);
    sycl::local_accessor<float,1> first(BT*16,handler);
    handler.parallel_for(sycl::nd_range<1>(size_t(chunks)*HV*WG,WG),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int matrix=item.get_group_linear_id(),head=matrix%HV,chunk=matrix/HV;
        auto sg=item.get_sub_group();const int row=sg.get_group_linear_id()*8;
        const BF *query=ws.q+(size_t(chunk)*HK+head%HK)*BT*D;
        const BF *key=ws.k+(size_t(chunk)*HK+head%HK)*BT*D;
        BF *a=scores[0].values;
        float *temp=first.template get_multi_ptr<sycl::access::decorated::no>().get();
        for(int col=0;col<BT;col+=16) {
          Acc acc;product<true>(sg,acc,query+row*D,D,key+col*D,D,D);
          mi::joint_matrix_apply(sg,acc,[=](float &value,size_t r,size_t c) {
            const int i=row+r,j=col+c;
            const bool live=i>=j && chunk*BT+i<tokens && chunk*BT+j<tokens;
            a[i*BT+j]=BF(live?value*sycl::exp(ws.g[size_t(matrix)*BT+i]-ws.g[size_t(matrix)*BT+j]):0.f);
          });
        }
        item.barrier(sycl::access::fence_space::local_space);
        for(int col=0;col<D;col+=16) {
          Acc acc;product(sg,acc,query+row*D,D,ws.history+size_t(matrix)*D*D+col,D,D);
          mi::joint_matrix_apply(sg,acc,[=](float &value,size_t r,size_t c) {
            temp[(row+r)*16+c]=value*sycl::exp(ws.g[size_t(matrix)*BT+row+r]);
          });
          item.barrier(sycl::access::fence_space::local_space);
          product(sg,acc,a+row*BT,BT,ws.delta+size_t(matrix)*BT*D+col,D,BT);
          mi::joint_matrix_apply(sg,acc,[=](float &value,size_t r,size_t c) {
            const float scale=sycl::rsqrt(128.f);
            ws.raw[(size_t(matrix)*BT+row+r)*D+col+c]=BF(temp[(row+r)*16+c]*scale+value*scale);
          });
          item.barrier(sycl::access::fence_space::local_space);
        }
      });
  });
  finish(stream,event);
  event=queue.parallel_for(sycl::nd_range<1>(size_t(tokens)*HV*D,D),
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int cell=item.get_group_linear_id(),d=item.get_local_linear_id();
      const int token=cell/HV,head=cell%HV;
      const float x=float(ws.raw[((size_t(token/BT)*HV+head)*BT+token%BT)*D+d]);
      const float sum=sycl::reduce_over_group(item.get_group(),x*x,sycl::plus<float>());
      const float sigmoid=1.f/(1.f+sycl::exp(-kernels::f32_from_bf16(z[size_t(cell)*D+d])));
      output[size_t(cell)*D+d]=kernels::f16_from_f32(((x*sycl::rsqrt(sum/128.f+eps))*gamma[d])*sigmoid);
    });
  finish(stream,event);
}
}
