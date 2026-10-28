// Main-authored Strata experiment; repository LICENSE (MIT) applies.
// Uses only this fork's independently authored BF16 prefix design at
// 6e40afc1beddd059ee518617f6c1efce1da14aca:src/kernels/sycl/qsa_bf16_prefix.cpp
// SHA256 4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192.
// New pair membership/matrix mapping; no other engine code copied/adapted.
#include "strata/kernels/qsa_pair_attention.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace strata::kernels {
namespace {
namespace mx=sycl::ext::oneapi::experimental::matrix;
namespace mi=sycl::ext::intel::experimental::matrix;
using BFloat=sycl::ext::oneapi::bfloat16;
constexpr int Dim=256, Heads=12, PairHeads=24, Cells=32, Pitch=33, WG=192, SG=16;
template<int Count> struct alignas(4) PairTile { BFloat values[Count]; };
template<int Count> constexpr bool storage() {
  using T=PairTile<Count>;
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>);
  static_assert(alignof(T)>=4 && offsetof(T,values)==0 && sizeof(T)==Count*2);
  return true;
}
static_assert(sizeof(BFloat)==2 && storage<Cells*Dim>() && storage<PairHeads*Cells>());
static_assert(PairHeads*Dim<=Cells*Dim);
constexpr size_t LocalBytes=(Cells*Dim+PairHeads*Cells)*2+
    (PairHeads*Pitch+3*PairHeads+Cells+PairHeads)*4+Cells*8;
static_assert(LocalBytes==21856);

bool available(const sycl::device &device) {
  if (!device.is_gpu() || device.get_info<sycl::info::device::vendor_id>()!=0x8086 ||
      !device.has(sycl::aspect::ext_intel_matrix) ||
      device.get_info<sycl::info::device::max_work_group_size>()<WG ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]<WG ||
      device.get_info<sycl::info::device::local_mem_size>()<LocalBytes+4096) return false;
  const auto groups=device.get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(groups.begin(),groups.end(),SG)==groups.end()) return false;
  const auto fits=[](size_t fixed,size_t maximum,size_t n) {return fixed==n || (!fixed && maximum>=n);};
  for (const auto &c:device.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
    if (c.atype==mx::matrix_type::bf16 && c.btype==mx::matrix_type::bf16 &&
        c.ctype==mx::matrix_type::fp32 && c.dtype==mx::matrix_type::fp32 &&
        fits(c.msize,c.max_msize,8) && fits(c.nsize,c.max_nsize,16) && fits(c.ksize,c.max_ksize,16))
      return true;
  return false;
}
}

bool qsa_pair_attention_prefill(const uint16_t *query,const QsaAttnPools &pools,
    const int32_t *original_ids,const int32_t *steps,int64_t cap,const QsaShapes &shape,
    uint16_t *output,int64_t queries,const QsaPromptBf16Bounds &bounds,int64_t live_cells,
    const QsaBf16PackedScratch &packed,const int32_t *unions,const uint8_t *membership,
    const int32_t *counts,void *stream) {
  if (!qsa_prompt_attn_bf16_validate(query,pools,original_ids,steps,cap,shape,
                                    output,queries,bounds,stream)) return false;
  if (!queries) return true;
  if (queries>262144 || cap>2051 || live_cells<=0 || uint64_t(live_cells)>
      uint64_t(bounds.logical_pages)*uint64_t(shape.page_size))
    throw std::invalid_argument("pair attention exceeds its geometry contract");
  using namespace sycl_backend;
  auto &queue=queue_for(stream);
  if (!available(queue.get_device())) return false;
  const int64_t pairs=(queries+1)/2, stride=2*cap;
  const size_t active_pages=(uint64_t(live_cells)+shape.page_size-1)/shape.page_size;
  const size_t active_rows=checked_count(checked_count(active_pages,2),shape.page_size);
  const size_t packed_bytes=checked_count(checked_count(active_rows,Dim),2);
  const size_t query_bytes=checked_count(checked_count(checked_count(queries,24),Dim),2);
  validate_spans({{output,query_bytes}},{{packed.k,packed_bytes},{packed.v,packed_bytes},
      {packed.bad_rows,checked_count(active_rows,4)},
      {unions,checked_count(checked_count(pairs,stride),4)},
      {membership,checked_count(pairs,stride)},{counts,checked_count(pairs,4)}});
  validate_spans({},{{packed.k,packed_bytes},{packed.v,packed_bytes}},2);
  validate_spans({},{{packed.bad_rows,checked_count(active_rows,4)},
                    {unions,checked_count(checked_count(pairs,stride),4)},
                    {counts,checked_count(pairs,4)}},4);
  for (const auto span:{Span{packed.k,packed_bytes},Span{packed.v,packed_bytes},
                        Span{packed.bad_rows,checked_count(active_rows,4)},
                        Span{unions,checked_count(checked_count(pairs,stride),4)},
                        Span{membership,checked_count(pairs,stride)},Span{counts,checked_count(pairs,4)}})
    validate_device_span(span.pointer,span.bytes,queue);
  auto event=queue.submit([&](sycl::handler &handler) {
    // The initial24Q rows share the larger32x256 K/V arena. Cached fragments
    // own every Q value before the first K write. Arrays are real BFloat objects.
    sycl::local_accessor<PairTile<Cells*Dim>,1> kvstorage(1,handler);
    sycl::local_accessor<PairTile<PairHeads*Cells>,1> pstorage(1,handler);
    sycl::local_accessor<float,1> scores(PairHeads*Pitch,handler),maximum(PairHeads,handler),
        normalizer(PairHeads,handler),alpha(PairHeads,handler);
    sycl::local_accessor<int64_t,1> rows(Cells,handler);
    sycl::local_accessor<int,1> masks(Cells,handler),bad_head(PairHeads,handler);
    handler.parallel_for(sycl::nd_range<1>(size_t(pairs)*2*WG,WG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
      auto kv=sycl::address_space_cast<sycl::access::address_space::local_space,
          sycl::access::decorated::no>(kvstorage[0].values);
      auto probability=sycl::address_space_cast<sycl::access::address_space::local_space,
          sycl::access::decorated::no>(pstorage[0].values);
      const int tid=item.get_local_linear_id(); const auto sg=item.get_sub_group();
      const int subgroup=sg.get_group_linear_id(),lane=sg.get_local_linear_id();
      const int64_t pair=int64_t(item.get_group_linear_id())/2, qa=pair*2;
      const int kv_head=item.get_group_linear_id()%2;
      const int actual=qa+1<queries?2:1;
      const int reported=counts[pair];
      const bool invalid_count=reported<0 || int64_t(reported)>stride;
      const int count=invalid_count?0:reported;
      bool bad_metadata=invalid_count;
      int local_bad=0,local_live=0;
      for (int h=subgroup;h<PairHeads;h+=WG/SG) {
        int bad=0;
        for (int d=lane;d<Dim;d+=SG) {
          float value=0.f;
          if (h/Heads<actual)
            value=sycl::bit_cast<float>(uint32_t(query[((qa+h/Heads)*24+kv_head*Heads+h%Heads)*Dim+d])<<16);
          bad |= !sycl::isfinite(value);
          value=value*.0625f;value=value*1.4426950408f;
          kv[h*Dim+d]=BFloat(value);
        }
        bad=sycl::reduce_over_group(sg,bad,sycl::maximum<int>());
        if (!lane) bad_head[h]=bad;
      }
      if (tid<PairHeads) {maximum[tid]=-INFINITY;normalizer[tid]=0.f;}
      using Acc=mx::joint_matrix<sycl::sub_group,float,mx::use::accumulator,8,16>;
      using Left=mx::joint_matrix<sycl::sub_group,BFloat,mx::use::a,8,16,mx::layout::row_major>;
      using Right=mx::joint_matrix<sycl::sub_group,BFloat,mx::use::b,16,16,mx::layout::row_major>;
      Acc accumulated[4];Left cached_query[Dim/16];
      for (int j=0;j<4;++j) mx::joint_matrix_fill(sg,accumulated[j],0.f);
      item.barrier(sycl::access::fence_space::local_space);
      if (subgroup<6)
        for (int d=0;d<Dim;d+=16)
          mx::joint_matrix_load(sg,cached_query[d/16],kv+(subgroup/2)*8*Dim+d,Dim);
      item.barrier(sycl::access::fence_space::local_space);
      for (int64_t first=0;first<count;first+=Cells) {
        if (tid<Cells) {
          int64_t row=-1;int mask=0;
          if (first+tid<count) {
            const int64_t cell=unions[pair*stride+first+tid];
            mask=membership[pair*stride+first+tid];
            bad_metadata |= mask<=0 || (mask&~((1<<actual)-1));
            if (first+tid>0) bad_metadata |= cell<=unions[pair*stride+first+tid-1];
            for (int q=0;q<actual;++q)
              if ((mask&(1<<q)) && (cell<0 || cell>=steps[(qa+q)*kStepCount+kStepNKv]))
                bad_metadata=true;
            const int64_t logical=cell>=0?cell/shape.page_size:-1;
            const int page=logical>=0 && logical<bounds.logical_pages?pools.page_table[logical]:-1;
            if (cell>=0 && cell<live_cells && page>=0 && page<bounds.physical_pages)
              row=(logical*2+kv_head)*shape.page_size+cell%shape.page_size;
          }
          if (row>=0) {
            local_live|=mask;
            // A bad V row belonging only to B must never enter A's PV as
            // 0*NaN. Diagnose its member queries, and stage zeros for BOTH
            // matrices. The initialized logical-prefix producer marks every
            // nonfinite scale in bad_rows before this consumer starts.
            if (packed.bad_rows[row]) {local_bad|=mask;row=-1;}
          }
          rows[tid]=row;masks[tid]=mask;
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int i=tid;i<Cells*(Dim/4);i+=WG) {
          const int c=i%Cells,d=(i/Cells)*4;const int64_t row=rows[c];
          sycl::vec<uint16_t,4> words{uint16_t(0)};
          if (row>=0) words.load(0,sycl::address_space_cast<sycl::access::address_space::global_space,
              sycl::access::decorated::no>(packed.k+row*Dim+d));
          for (int j=0;j<4;++j) kv[(d+j)*Cells+c]=sycl::bit_cast<BFloat>(uint16_t(words[j]));
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (subgroup<6) {
          const int h=(subgroup/2)*8,c=(subgroup%2)*16;Acc product;Right right;
          mx::joint_matrix_fill(sg,product,0.f);
          for (int d=0;d<Dim;d+=16) {
            mx::joint_matrix_load(sg,right,kv+d*Cells+c,Cells);
            mx::joint_matrix_mad(sg,product,cached_query[d/16],right,product);
          }
          mi::joint_matrix_apply(sg,product,[=](float &value,size_t r,size_t col) {
            scores[(h+r)*Pitch+c+col]=value;
          });
        }
        item.barrier(sycl::access::fence_space::local_space);
        // Each subgroup owns two independent query/head softmax rows.
        // Membership is applied before max, exp, sums and diagnostics.
        // Preserve the selected ordered-exp path's ascending FP32 tile sum;
        // a union changes tile boundaries, not the probability/storage type.
        for (int h=subgroup;h<PairHeads;h+=WG/SG) {
          const int bit=1<<(h/Heads),c0=lane,c1=lane+SG;
          const float s0=scores[h*Pitch+c0],s1=scores[h*Pitch+c1];
          const bool row0=rows[c0]>=0 && (masks[c0]&bit);
          const bool row1=rows[c1]>=0 && (masks[c1]&bit);
          const bool valid0=row0 && sycl::isfinite(s0),valid1=row1 && sycl::isfinite(s1);
          const int bad=(row0 && !valid0) || (row1 && !valid1);
          const int score_bad=sycl::reduce_over_group(sg,bad,sycl::maximum<int>());
          const float tile_max=sycl::reduce_over_group(sg,
              sycl::fmax(valid0?s0:-INFINITY,valid1?s1:-INFINITY),sycl::maximum<float>());
          const float prior=!lane?maximum[h]:-INFINITY;
          const float old_m=sycl::group_broadcast(sg,prior,0),m=sycl::fmax(old_m,tile_max);
          const float p0=valid0?sycl::exp2(s0-m):0.f,p1=valid1?sycl::exp2(s1-m):0.f;
          float sum=0.f;
          for (int c=0;c<Cells;++c) {
            const float p=sycl::select_from_group(sg,c<SG?p0:p1,c%SG);
            if (!lane) sum+=p;
          }
          probability[h*Cells+c0]=BFloat(p0);
          probability[h*Cells+c1]=BFloat(p1);
          if (!lane) {
            const float old_norm=normalizer[h];
            const float a=old_norm>0.f?sycl::exp2(old_m-m):0.f;
            bad_head[h]|=score_bad;maximum[h]=m;alpha[h]=a;
            normalizer[h]=sycl::fma(old_norm,a,sum);
          }
        }
        // Same typed four-value V staging as the selected ordered-exp path.
        // QK readers completed before this K/V-arena overwrite. Probability
        // and softmax writes use disjoint objects until the following barrier.
        for (int i=tid;i<Cells*(Dim/4);i+=WG) {
          const int c=i/(Dim/4),d=(i%(Dim/4))*4;const int64_t row=rows[c];
          sycl::vec<uint16_t,4> words{uint16_t(0)};
          if (row>=0) words.load(0,sycl::address_space_cast<sycl::access::address_space::global_space,
              sycl::access::decorated::no>(packed.v+row*Dim+d));
          for (int j=0;j<4;++j) kv[c*Dim+d+j]=sycl::bit_cast<BFloat>(uint16_t(words[j]));
        }
        item.barrier(sycl::access::fence_space::local_space);
        // Three eight-head blocks use every matrix row: 24 heads across the
        // pair, with four subgroups per block covering disjoint64-dim columns.
        const int h=(subgroup/4)*8,dim_base=(subgroup%4)*64;
        for (int j=0;j<4;++j) {
          Left left;Right right;
          mi::joint_matrix_apply(sg,accumulated[j],[=](float &value,size_t r,size_t) {
            value*=alpha[h+r];
          });
          for (int c=0;c<Cells;c+=16) {
            mx::joint_matrix_load(sg,left,probability+h*Cells+c,Cells);
            mx::joint_matrix_load(sg,right,kv+c*Dim+dim_base+j*16,Dim);
            mx::joint_matrix_mad(sg,accumulated[j],left,right,accumulated[j]);
          }
        }
        // All PV readers finish before next tile's rows, masks or K overwrite.
        item.barrier(sycl::access::fence_space::local_space);
      }
      const int invalid_scales=sycl::reduce_over_group(item.get_group(),local_bad,sycl::bit_or<int>());
      const int any_live=sycl::reduce_over_group(item.get_group(),local_live,sycl::bit_or<int>());
      const bool invalid_metadata=sycl::any_of_group(item.get_group(),bad_metadata);
      const int h=(subgroup/4)*8,dim_base=(subgroup%4)*64;
      for (int j=0;j<4;++j)
        mi::joint_matrix_apply(sg,accumulated[j],[=](float &value,size_t r,size_t c) {
          const int head=h+int(r),q=head/Heads,bit=1<<q;
          if (q>=actual) return;
          uint16_t result=0;
          if (invalid_metadata || ((any_live&bit) &&
              (bad_head[head] || (invalid_scales&bit)))) result=0x7fc0;
          else if (normalizer[head]>0.f)
            result=sycl::bit_cast<uint16_t>(BFloat(value/normalizer[head]));
          output[((qa+q)*24+kv_head*Heads+head%Heads)*Dim+dim_base+j*16+c]=result;
        });
    });
  });
  finish(stream,event);
  return true;
}
} // namespace strata::kernels
