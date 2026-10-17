// Main-authored independent chunk64 GDN implementation; repository LICENSE.
// Numerical boundaries and references: gdn-current-contract-next-v1.ja.md.
// No external kernel source copied. Private proposal, not product adoption.
#include "strata/prefill/gdn_chunk64.hpp"
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
template<bool Transpose=false,bool TransposeA=false,bool ALocal=false,bool BLocal=false>
void product(sycl::sub_group sg, Acc &acc, const BF *a, int lda,
             const BF *b, int ldb, int inner) {
  mx::joint_matrix_fill(sg,acc,0.f);
  for(int k=0;k<inner;k+=16) {
    using A=std::conditional_t<TransposeA,TransposedLeft,Left>;
    A left;
    auto left_pointer=sycl::address_space_cast<ALocal?sycl::access::address_space::local_space:
      sycl::access::address_space::global_space,sycl::access::decorated::no>(a+(TransposeA?k*lda:k));
    mx::joint_matrix_load(sg,left,left_pointer,lda);
    if constexpr(Transpose) {
      Transposed right;
      auto right_pointer=sycl::address_space_cast<BLocal?sycl::access::address_space::local_space:
        sycl::access::address_space::global_space,sycl::access::decorated::no>(b+k);
      mx::joint_matrix_load(sg,right,right_pointer,ldb);
      mx::joint_matrix_mad(sg,acc,left,right,acc);
    } else {
      Right right;
      auto right_pointer=sycl::address_space_cast<BLocal?sycl::access::address_space::local_space:
        sycl::access::address_space::global_space,sycl::access::decorated::no>(b+k*ldb);
      mx::joint_matrix_load(sg,right,right_pointer,ldb);
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
void conv_bf16(float *history,const uint16_t *qkv,const float *weights,
               uint16_t *normalized,int tokens,float eps,void *stream,
               ConvEvents *events) {
  require(tokens>0 && tokens<=8192 && stream && std::isfinite(eps) && eps>=0.f);
  auto &queue=queue_for(stream);
  const auto device=queue.get_device();
  require(device.is_gpu() && device.get_info<sycl::info::device::vendor_id>()==0x8086 &&
          device.get_info<sycl::info::device::max_work_group_size>()>=128 &&
          device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]>=128 &&
          device.get_info<sycl::info::device::local_mem_size>()>=4*sizeof(float));
  const auto groups=device.get_info<sycl::info::device::sub_group_sizes>();
  require(std::find(groups.begin(),groups.end(),32)!=groups.end());
  const size_t elements=size_t(tokens)*C;
  validate_spans({{history,size_t(C)*3*4},{normalized,elements*2}},
                 {{qkv,elements*2},{weights,size_t(C)*4*4}},4);
  for(auto span:{Span{history,size_t(C)*3*4},Span{normalized,elements*2},
                 Span{qkv,elements*2},Span{weights,size_t(C)*4*4}})
    validate_device_span(span.pointer,span.bytes,queue);
  auto event=queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<float,1> partial(4,handler);
    handler.parallel_for(sycl::nd_range<1>(elements,D),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int d=item.get_local_linear_id(),cell=item.get_group_linear_id();
        const int token=cell/(C/D),head=cell%(C/D),channel=head*D+d;
        float value=0.f;
        for(int j=0;j<4;++j) {
          const int source=token-3+j;
          const float x=source<0?history[channel*3+source+3]:
            kernels::f32_from_bf16(qkv[size_t(source)*C+channel]);
          value=j==0?x*weights[channel*4]:sycl::fma(x,weights[channel*4+j],value);
        }
        // Keep the framework activation storage rounding before Q/K L2 math,
        // while fusing the producer/normalizer without a full FP32 tensor.
        value=kernels::f32_from_bf16(kernels::bf16_from_f32(value/(1.f+sycl::exp(-value))));
        if(head<2*HK) {
          float sum=value*value;
          for(int off=16;off;off>>=1)sum+=sycl::permute_group_by_xor(item.get_sub_group(),sum,off);
          if(d%32==0)partial[d/32]=sum;
          item.barrier(sycl::access::fence_space::local_space);
          value*=sycl::rsqrt(((partial[0]+partial[1])+partial[2])+partial[3]+eps);
        }
        normalized[size_t(token)*C+channel]=kernels::bf16_from_f32(value);
      });
  });
  finish(stream,event);
  if(events)events->convolution=event;
  event=queue.parallel_for(sycl::range<1>(C),[=](sycl::id<1> id) {
    const size_t channel=id[0];
    float tail[3];
    for(int j=0;j<3;++j) {
      const int token=tokens-3+j;
      tail[j]=token<0?history[channel*3+size_t(token+3)]:
        kernels::f32_from_bf16(qkv[size_t(token)*C+channel]);
    }
    for(int j=0;j<3;++j)history[channel*3+j]=tail[j];
  });
  finish(stream,event);
  if(events)events->history=event;
}
void run(float *state,const uint16_t *h,const float *gate,const float *beta,
         const uint16_t *z,const float *gamma,float eps,uint16_t *output,
         int tokens,Workspace ws,void *stream,Events *events,bool packed_inverse,bool narrow_carry) {
  const Sizes n=sizes(tokens);
  require(stream && std::isfinite(eps) && eps>=0.f);
  auto &queue=queue_for(stream);
  matrix_check(queue.get_device());
  const std::initializer_list<Span> outputs={
    {state,size_t(HV)*D*D*4},{output,size_t(tokens)*HV*D*2},
    {ws.q,n.qk*2},{ws.k,n.qk*2},
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
  if(events)events->gates=event;
  event=queue.parallel_for(sycl::range<1>(n.values),[=](sycl::id<1> id) {
    size_t x=id[0]; const int d=x%D; x/=D;
    const int t=x%BT; x/=BT;
    const int head=x%HV, chunk=x/HV, token=chunk*BT+t;
    const bool live=token<tokens;
    const BF k=live?sycl::bit_cast<BF>(h[size_t(token)*C+HK*D+(head%HK)*D+d]):BF(0.f);
    const BF v=live?sycl::bit_cast<BF>(h[size_t(token)*C+2*HK*D+head*D+d]):BF(0.f);
    const float b=live?beta[size_t(token)*HV+head]:0.f;
    const float eg=sycl::exp(ws.g[(size_t(chunk)*HV+head)*BT+t]);
    ws.kb[id[0]]=BF((float(k)*b)*eg);
    ws.vb[id[0]]=BF(float(v)*b);
    if(head<HK) {
      const size_t dst=((size_t(chunk)*HK+head)*BT+t)*D+d;
      ws.q[dst]=live?sycl::bit_cast<BF>(h[size_t(token)*C+head*D+d]):BF(0.f);
      ws.k[dst]=k;
    }
  });
  finish(stream,event);
  if(events)events->pack=event;
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
  if(events)events->kkt=event;
  // FP32 forward substitution, then one BF16 inverse storage boundary.
  // This deliberately states the full64 solve contract; it is not asserted
  // identical to SGLang's block16/XPU intermediate rounding.
  auto invert=[&]<bool Packed>(){return queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<float,1> inverse(BT*BT,handler);
    handler.parallel_for(sycl::nd_range<1>(size_t(chunks)*HV*64,64),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int matrix=item.get_group_linear_id(), col=item.get_local_linear_id();
        // The solved inverse occupies the lower triangle; retain the original
        // coefficients in its otherwise-unused upper triangle. XOR the column
        // by row so transpose stores do not share a constant low address.
        auto index=[](int r,int c){return r*BT+(Packed?(c^r):c);};
        if constexpr(Packed){
          for(int i=col;i<BT*BT;i+=64){const int r=i/BT,c=i%BT;
            if(r>c)inverse[index(c,r)]=ws.lower[size_t(matrix)*BT*BT+i];}
          item.barrier(sycl::access::fence_space::local_space);
        }
        for(int row=0;row<BT;++row) {
          float value=row==col?1.f:0.f;
          if(col<row) for(int p=col;p<row;++p)
            value-=(Packed?inverse[index(p,row)]:ws.lower[(size_t(matrix)*BT+row)*BT+p])*inverse[index(p,col)];
          if constexpr(Packed){if(col<=row)inverse[index(row,col)]=value;}
          else inverse[index(row,col)]=value;
          item.barrier(sycl::access::fence_space::local_space);
        }
        for(int row=0;row<BT;++row)
          ws.inverse[(size_t(matrix)*BT+row)*BT+col]=BF(Packed && col>row?0.f:inverse[index(row,col)]);
      });
  });};
  event=packed_inverse?invert.template operator()<true>():invert.template operator()<false>();
  finish(stream,event);
  if(events)events->inverse=event;
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
  if(events)events->wu=event;
  // Persistent carry: one16/32-value-column slice per workgroup. Only chunk order
  // is serial; each64-token update uses DPAS and keeps the FP32 carry in registers.
  auto carry=[&]<int V>(){return queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<Tile<D*V>,1> hs(1,handler);
    sycl::local_accessor<Tile<BT*V>,1> weighted(1,handler);
    sycl::local_accessor<float,1> scratch(D*V,handler);
    handler.parallel_for(sycl::nd_range<1>(HV*(D/V)*WG,WG),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int head=item.get_group_linear_id()/(D/V),vblock=item.get_group_linear_id()%(D/V);
        const int tid=item.get_local_linear_id(),rg=tid/V,col=tid%V;
        auto sg=item.get_sub_group();const int subgroup=sg.get_group_linear_id();
        BF *local_h=hs[0].values,*local_v=weighted[0].values;
        float *temp=scratch.template get_multi_ptr<sycl::access::decorated::no>().get();
        float saved[V];
        for(int r=0;r<V;++r) saved[r]=state[(size_t(rg*V+r)*HV+head)*D+vblock*V+col];
        for(int chunk=0;chunk<chunks;++chunk) {
          const size_t matrix=size_t(chunk)*HV+head;
          const int live=sycl::min(BT,tokens-chunk*BT);
          for(int r=0;r<V;++r) {
            const int row=rg*V+r;
            local_h[row*V+col]=BF(saved[r]);
            ws.history[(matrix*D+row)*D+vblock*V+col]=BF(saved[r]);
          }
          item.barrier(sycl::access::fence_space::local_space);
          for(int c=0;c<V;c+=16) {
            Acc acc;
            product<false,false,false,true>(sg,acc,ws.w+matrix*BT*D+subgroup*8*D,D,local_h+c,V,D);
            mx::joint_matrix_store(sg,acc,scratch.template get_multi_ptr<sycl::access::decorated::no>()+subgroup*8*V+c,V,mx::layout::row_major);
          }
          item.barrier(sycl::access::fence_space::local_space);
          const float last=ws.g[matrix*BT+live-1];
          for(int i=tid;i<BT*V;i+=WG) {
            const int t=i/V,c=i%V;
            const float delta=t<live?float(ws.u[(matrix*BT+t)*D+vblock*V+c])-temp[i]:0.f;
            ws.delta[(matrix*BT+t)*D+vblock*V+c]=BF(delta);
            local_v[i]=BF(delta*sycl::exp(last-ws.g[matrix*BT+t]));
          }
          item.barrier(sycl::access::fence_space::local_space);
          for(int r=0;r<16;r+=8) for(int c=0;c<V;c+=16) {
            Acc acc;const int row=subgroup*16+r;
            product<false,true,false,true>(sg,acc,ws.k+(size_t(chunk)*HK+head%HK)*BT*D+row,D,
                          local_v+c,V,BT);
            mx::joint_matrix_store(sg,acc,scratch.template get_multi_ptr<sycl::access::decorated::no>()+row*V+c,V,mx::layout::row_major);
          }
          item.barrier(sycl::access::fence_space::local_space);
          const float decay=sycl::exp(last);
          for(int r=0;r<V;++r) saved[r]=saved[r]*decay+temp[(rg*V+r)*V+col];
          item.barrier(sycl::access::fence_space::local_space);
        }
        for(int r=0;r<V;++r) state[(size_t(rg*V+r)*HV+head)*D+vblock*V+col]=saved[r];
      });
  });};
  event=narrow_carry?carry.template operator()<16>():carry.template operator()<32>();
  finish(stream,event);
  if(events)events->carry=event;
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
          product<false,false,true,false>(sg,acc,a+row*BT,BT,ws.delta+size_t(matrix)*BT*D+col,D,BT);
          mi::joint_matrix_apply(sg,acc,[=](float &value,size_t r,size_t c) {
            const float scale=sycl::rsqrt(128.f);
            // Add FP32 history/update before scaling. Scaling both cancelling
            // terms separately leaves tiny residuals and costs an extra multiply.
            ws.raw[(size_t(matrix)*BT+row+r)*D+col+c]=BF((temp[(row+r)*16+c]+value)*scale);
          });
          item.barrier(sycl::access::fence_space::local_space);
        }
      });
  });
  finish(stream,event);
  if(events)events->readout=event;
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
  if(events)events->norm=event;
}
}
