// Main-authored Strata experiment; repository LICENSE (MIT) applies.
// Independent stable merge-path construction over this fork's sorted selected
// IDs. No external engine/algorithm implementation was copied or adapted.
#include "strata/kernels/qsa_pair_union.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>

namespace strata::kernels {
namespace {
constexpr int WG=128, SG=32, MaxCap=2051, MaxContext=262144;
constexpr int MaxItems=(2*MaxCap+WG-1)/WG;
static_assert(MaxItems==33);
}

void qsa_pair_union_prefill(const int32_t *ids, const int32_t *steps,
    int64_t queries, int64_t cap, int32_t *unions, uint8_t *membership,
    int32_t *counts, void *stream) {
  using namespace sycl_backend;
  if (queries<=0 || queries>MaxContext || cap<=0 || cap>MaxCap)
    throw std::invalid_argument("invalid pair-union geometry");
  const int64_t pairs=(queries+1)/2, stride=2*cap;
  validate_spans({{unions,checked_count(pairs,stride)*sizeof(int32_t)},
                  {membership,checked_count(pairs,stride)*sizeof(uint8_t)},
                  {counts,checked_count(pairs,1)*sizeof(int32_t)}},
                 {{ids,checked_count(queries,cap)*sizeof(int32_t)},
                  {steps,checked_count(queries,kStepCount)*sizeof(int32_t)}});
  auto &queue=queue_for(stream);
  const auto device=queue.get_device();
  const auto sizes=device.get_info<sycl::info::device::sub_group_sizes>();
  if (device.get_info<sycl::info::device::max_work_group_size>()<WG ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]<WG ||
      std::find(sizes.begin(),sizes.end(),SG)==sizes.end())
    throw std::invalid_argument("pair union requires WG128/SG32");
  auto event=queue.submit([&](sycl::handler &handler) {
    handler.parallel_for(sycl::nd_range<1>(size_t(pairs)*WG,WG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
      const int tid=int(item.get_local_linear_id());
      const int64_t pair=item.get_group_linear_id(), qa=pair*2;
      const bool has_b=qa+1<queries;
      const auto sa=steps+qa*kStepCount;
      // A missing final query never forms or dereferences its steps pointer.
      const int na=sa[kStepNKv], wa=sa[kStepWidth];
      int nb=0, wb=0, bidb=0;
      if (has_b) {
        const auto sb=steps+(qa+1)*kStepCount;
        nb=sb[kStepNKv]; wb=sb[kStepWidth]; bidb=sb[kStepNBid];
      }
      const bool invalid=na<0 || na>MaxContext || wa<0 || wa>cap || wa>na ||
          sa[kStepNBid]!=na/4 || nb<0 || nb>MaxContext || wb<0 ||
          wb>cap || wb>nb || (has_b && bidb!=nb/4);
      // All shape/empty branches are workgroup-uniform, before collectives.
      if (invalid) { if (!tid) counts[pair]=-1; return; }
      const auto a=ids+qa*cap;
      // Use the existing A row address for an absent, never-read B row.
      const auto b=has_b ? ids+(qa+1)*cap : a;
      bool bad=false;
      for (int i=tid;i<wa;i+=WG)
        bad |= a[i]<0 || a[i]>=na || (i>0 && a[i]<=a[i-1]);
      for (int j=tid;j<wb;j+=WG)
        bad |= b[j]<0 || b[j]>=nb || (j>0 && b[j]<=b[j-1]);
      if (sycl::any_of_group(item.get_group(),bad)) {
        if (!tid) counts[pair]=-1;
        return;
      }
      const int total=wa+wb;
      if (!total) { if (!tid) counts[pair]=0; return; }
      const int chunk=(total+WG-1)/WG;
      const int start=sycl::min(total,tid*chunk);
      const int end=sycl::min(total,start+chunk);
      // Partition the stable A-before-B merged stream at `start`.
      // A[i-1]<=B[j] and B[j-1]<A[i] when those elements exist.
      int lo=sycl::max(0,start-wb), hi=sycl::min(start,wa);
      while (lo<hi) {
        const int mid=lo+(hi-lo)/2, j=start-mid;
        if (mid<wa && j>0 && b[j-1]>=a[mid]) lo=mid+1;
        else hi=mid;
      }
      int i=lo, j=start-lo;
      int previous=-1;
      if (i>0) previous=a[i-1];
      if (j>0) previous=sycl::max(previous,b[j-1]);
      int values[MaxItems]; uint8_t flags[MaxItems]; int count=0;
      for (int rank=start;rank<end;++rank) {
        const bool from_a=i<wa && (j>=wb || a[i]<=b[j]);
        const int value=from_a ? a[i] : b[j];
        const uint8_t mask=from_a ? uint8_t(1|((j<wb && b[j]==value)?2:0)) : uint8_t(2);
        if (value!=previous) {
          values[count]=value; flags[count]=mask; ++count;
        }
        previous=value;
        if (from_a) ++i; else ++j;
      }
      const int offset=sycl::exclusive_scan_over_group(item.get_group(),count,sycl::plus<int>());
      const int64_t base=pair*stride+offset;
      for (int k=0;k<count;++k) {
        unions[base+k]=values[k]; membership[base+k]=flags[k];
      }
      // Empty trailing lanes still scan; the last lane sees the total prefix.
      if (tid==WG-1) counts[pair]=offset+count;
    });
  });
  finish(stream,event);
}
} // namespace strata::kernels
