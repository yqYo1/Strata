// Main-authored Strata prefill path; repository LICENSE (MIT) applies.
// Logical-prefix packing is this fork's own selected6e40 implementation,
// qsa_bf16_prefix.cpp SHA4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192.
// Independent typed admission/three-kernel composition;no external engine code.
#include "strata/kernels/qsa_pair_prefix.hpp"
#include "strata/kernels/qsa_pair_union.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace strata::kernels {
bool qsa_bf16_pair_prefix_batch(const uint16_t *query,const QsaAttnPools &pools,
    const int32_t *ids,const int32_t *steps,int64_t cap,const QsaShapes &shape,
    uint16_t *output,int64_t queries,const QsaPromptBf16Bounds &bounds,
    int64_t live_cells,const QsaBf16PackedScratch &packed,
    const QsaPairScratch &pair,void *stream) {
  if (!qsa_prompt_attn_bf16_validate(query,pools,ids,steps,cap,shape,
                                    output,queries,bounds,stream)) return false;
  if (!queries) return true;
  using namespace sycl_backend;
  if (cap>2051 || live_cells<=0 || live_cells>262144 || uint64_t(live_cells)>
      uint64_t(bounds.logical_pages)*uint64_t(shape.page_size))
    throw std::invalid_argument("pair prefix exceeds its geometry contract");
  auto &queue=queue_for(stream);const auto device=queue.get_device();
  const auto groups=device.get_info<sycl::info::device::sub_group_sizes>();
  // The shared validator already verifies the same BF16xBF16/FP32 matrix
  // combination and SG16;add the pair consumer/producer's stronger limits.
  if (!device.is_gpu() || device.get_info<sycl::info::device::vendor_id>()!=0x8086 ||
      !device.has(sycl::aspect::ext_intel_matrix) ||
      device.get_info<sycl::info::device::max_work_group_size>()<192 ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]<192 ||
      device.get_info<sycl::info::device::local_mem_size>()<21856+4096+4096 ||
      std::find(groups.begin(),groups.end(),32)==groups.end()) return false;
  constexpr int Dim=256;
  const size_t rows_count=checked_count(checked_count(bounds.physical_pages,2),shape.page_size);
  const size_t active_pages=(uint64_t(live_cells)+shape.page_size-1)/shape.page_size;
  const size_t active_rows=checked_count(checked_count(active_pages,2),shape.page_size);
  const size_t packed_bytes=checked_count(checked_count(active_rows,Dim),2);
  const size_t query_bytes=checked_count(checked_count(checked_count(queries,24),Dim),2);
  const size_t pairs=(uint64_t(queries)+1)/2,entries=checked_count(pairs,checked_count(cap,2));
  const Span pk{packed.k,packed_bytes},pv{packed.v,packed_bytes},
      bad{packed.bad_rows,checked_count(active_rows,4)},
      unions{pair.unions,checked_count(entries,4)},masks{pair.membership,entries},
      counts{pair.counts,checked_count(pairs,4)},out{output,query_bytes};
  validate_spans({pk,pv,bad,unions,masks,counts,out},
      {{query,query_bytes},{ids,checked_count(checked_count(queries,cap),4)},
       {steps,checked_count(checked_count(queries,kStepCount),4)},
       {pools.page_table,checked_count(bounds.logical_pages,4)},
       {pools.k_q,checked_count(rows_count,Dim)},{pools.v_q,checked_count(rows_count,Dim)},
       {pools.k_scale,checked_count(checked_count(rows_count,4),2)},
       {pools.v_scale,checked_count(checked_count(rows_count,4),2)}},1);
  validate_spans({}, {pk,pv,out},2);validate_spans({}, {bad,unions,counts},4);
  for (const auto span:{pk,pv,bad,unions,masks,counts,out})
    validate_device_span(span.pointer,span.bytes,queue);
  using BFloat=sycl::ext::oneapi::bfloat16;
  // Same logical/physical mapping,rounding,diagnostics and padding as the
  // selected prefix producer. The old attention route itself is unchanged.
  auto prepared=queue.parallel_for(sycl::nd_range<1>(active_rows*64,64),
      [=](sycl::nd_item<1> item) {
    const size_t row=item.get_group_linear_id(),lane=item.get_local_linear_id();
    const size_t logical=row/(2*shape.page_size),
                 head=row/shape.page_size%2,offset=row%shape.page_size;
    const size_t cell=logical*shape.page_size+offset;
    const int page=pools.page_table[logical];
    const bool valid=cell<uint64_t(live_cells) && page>=0 && page<bounds.physical_pages;
    const size_t source_row=valid?(size_t(page)*2+head)*shape.page_size+offset:0;
    if (!lane) {
      int invalid=0;
      if (valid) for (int g=0;g<4;++g)
        invalid|=!sycl::isfinite(f32_from_f16(pools.k_scale[source_row*4+g])) ||
                 !sycl::isfinite(f32_from_f16(pools.v_scale[source_row*4+g]));
      packed.bad_rows[row]=invalid;
    }
    for (size_t d=lane;d<Dim;d+=64) {
      float k=0.f,v=0.f;
      if (valid) {
        k=float(pools.k_q[source_row*Dim+d])*f32_from_f16(pools.k_scale[source_row*4+d/64]);
        v=float(pools.v_q[source_row*Dim+d])*f32_from_f16(pools.v_scale[source_row*4+d/64]);
      }
      packed.k[row*Dim+d]=sycl::bit_cast<uint16_t>(BFloat(k));
      packed.v[row*Dim+d]=sycl::bit_cast<uint16_t>(BFloat(v));
    }
  });
  finish(stream,prepared);
  qsa_pair_union_prefill(ids,steps,queries,cap,pair.unions,pair.membership,pair.counts,stream);
  return qsa_pair_attention_prefill(query,pools,ids,steps,cap,shape,output,queries,
      bounds,live_cells,packed,pair.unions,pair.membership,pair.counts,stream);
}
} // namespace strata::kernels
