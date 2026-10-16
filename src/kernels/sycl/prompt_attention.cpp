// Selected-cell prompt attention on Intel joint_matrix units. Queries and
// probabilities retain hi/lo FP16 parts; KV codes enter exactly, with FP32
// per-group scales. Online softmax keeps memory independent of the context.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace mi = sycl::ext::intel::experimental::matrix;
constexpr int HD = 256, HEADS = 12, CH = 32, WG = 128;
// Set by the owning process before exec; only the exact value "1" opts in.
bool prompt_softmax_subgroup_enabled() {
  const char *value = std::getenv("STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP");
  return value && value[0] == '1' && value[1] == '\0';
}
// One subgroup owns one head row at a time: lane c covers cells c and c+16.
// Every lane must call this helper, including padded heads and missing rows.
// Bf16 retains the typed route's infinity rules and score-NaN diagnostic.
template <bool Bf16>
int prompt_softmax_row(sycl::sub_group sg, int head,
                       sycl::local_accessor<float, 1> scores,
                       sycl::local_accessor<float, 1> maxima,
                       sycl::local_accessor<float, 1> sums,
                       sycl::local_accessor<float, 1> alpha,
                       sycl::local_accessor<int64_t, 1> rows) {
  static_assert(CH == 32 && WG == 8 * 16);
  const int c = sg.get_local_linear_id();
  const bool valid0 = rows[c] >= 0, valid1 = rows[c + 16] >= 0;
  const float score0 = scores[head * CH + c],
              score1 = scores[head * CH + c + 16];
  const float old_max = maxima[head];
  // fmax with the old maximum also suppresses NaNs exactly as the old scan.
  // Thus maximum<float> never receives a NaN from a score.
  float m = sycl::fmax(old_max, valid0 ? score0 : -INFINITY);
  m = sycl::fmax(m, valid1 ? score1 : -INFINITY);
  m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
  int bad = 0;
  if constexpr (Bf16) {
    bad = (valid0 && sycl::isnan(score0)) || (valid1 && sycl::isnan(score1));
    bad = sycl::reduce_over_group(sg, bad, sycl::maximum<int>());
  }
  const auto probability = [=](float score, bool valid) {
    // Test validity before exp: first-tile all-missing must not evaluate -inf-(-inf).
    if (!valid) return 0.f;
    if constexpr (Bf16)
      return sycl::isinf(m) ? (score == m ? 1.f : 0.f) : sycl::exp(score - m);
    else
      return sycl::exp(score - m);
  };
  const float p0 = probability(score0, valid0), p1 = probability(score1, valid1);
  // A 16-lane reduction of pair sums changes only the FP32 summation order.
  const float total = sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
  scores[head * CH + c] = p0;
  scores[head * CH + c + 16] = p1;
  if (c == 0) {
    const float old_sum = sums[head];
    float a;
    if constexpr (Bf16)
      a = old_sum > 0.f
          ? (old_max == m ? 1.f : (m == INFINITY ? 0.f : sycl::exp(old_max - m)))
          : 0.f;
    else
      a = old_max == -INFINITY ? 0.f : sycl::exp(old_max - m);
    maxima[head] = m;
    sums[head] = sycl::fma(old_sum, a, total);
    alpha[head] = a;
  }
  return bad;
}
template <int Mode, bool Value> constexpr int scale_groups() {
  if constexpr (Mode == 0)
    return 1;
  if constexpr (Mode == 2 || (Mode == 3 && Value))
    return 8;
  return 4;
}
template <int Mode, bool Value>
float code(QsaAttnPools p, int64_t row, int dim) {
  if constexpr (Mode == 0)
    return f32_from_f16((Value ? p.v_pool : p.k_pool)[row * HD + dim]);
  if constexpr (Mode == 2 || (Mode == 3 && Value)) {
    const auto *b =
        reinterpret_cast<const block_q4_0 *>(Value ? p.v_q4 : p.k_q4);
    const auto &block = b[row * 8 + dim / 32];
    return float(int((block.qs[dim % 16] >> (dim % 32 < 16 ? 0 : 4)) & 15) - 8);
  }
  if constexpr (Mode == 1 || (Mode == 3 && !Value))
    return float((Value ? p.v_q : p.k_q)[row * HD + dim]);
  return 0;
}
template <int Mode, bool Value, bool Native>
sycl::half half_code(QsaAttnPools p, int64_t row, int dim) {
  if constexpr (Native && Mode == 0) {
    const uint16_t bits = (Value ? p.v_pool : p.k_pool)[row * HD + dim];
    // Retain the existing NaN conversion; finite values and infinities can
    // enter the half tile without a half -> float -> half round trip.
    if ((bits & 0x7fff) <= 0x7c00)
      return sycl::bit_cast<sycl::half>(bits);
    return sycl::half(f32_from_f16(bits));
  }
  return sycl::half(code<Mode, Value>(p, row, dim));
}
template <int Mode, bool Value>
float scale(QsaAttnPools p, int64_t row, int group) {
  if constexpr (Mode == 0)
    return 1.f;
  if constexpr (Mode == 2 || (Mode == 3 && Value)) {
    const auto *b =
        reinterpret_cast<const block_q4_0 *>(Value ? p.v_q4 : p.k_q4);
    return f32_from_f16(b[row * 8 + group].d);
  }
  if constexpr (Mode == 1 || (Mode == 3 && !Value))
    return f32_from_f16((Value ? p.v_scale : p.k_scale)[row * 4 + group]);
  return 0;
}
// One genuine half array: no integer alias or pointer arithmetic across rows.
struct alignas(4) PaddedK {
  sycl::half values[HD * 40];
};
static_assert(alignof(PaddedK) >= 4 && sizeof(PaddedK) == HD * 40 * sizeof(sycl::half));
static_assert(std::is_standard_layout_v<PaddedK> && std::is_trivially_copyable_v<PaddedK>);
static_assert(sycl::is_device_copyable_v<PaddedK>);
template <int Mode, bool Direct, bool Native = false, int KMapping = 0,
          bool SubgroupSoftmax = false>
void launch(const float *query, QsaAttnPools pools, const int32_t *ids,
            const int32_t *steps, int64_t cap, QsaShapes shapes, float *output,
            int64_t queries, void *stream) {
  static_assert(KMapping >= 0 && KMapping <= 2);
  static_assert(KMapping == 0 || (Mode == 1 && Direct && !Native));
  constexpr int KP = KMapping == 2 ? 40 : CH;
  constexpr int KG = scale_groups<Mode, false>(),
                VG = scale_groups<Mode, true>();
  auto event = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<sycl::half, 1> qhi(16 * HD, h), qlo(16 * HD, h);
    using KVStorage = std::conditional_t<KMapping == 2, PaddedK, sycl::half>;
    sycl::local_accessor<KVStorage, 1> kv(KMapping == 2 ? 1 : CH * HD, h);
    sycl::local_accessor<sycl::half, 1> phi(VG * 16 * CH, h),
        plo(VG * 16 * CH, h);
    sycl::local_accessor<float, 1> scores(16 * CH, h), ks(CH * KG, h),
        vs(CH * VG, h);
    sycl::local_accessor<float, 1> maxima(16, h), sums(16, h), alpha(16, h),
        factors(VG, h), final(Direct ? 1 : 16 * HD, h);
    sycl::local_accessor<int64_t, 1> rows(CH, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(queries) * shapes.n_head_kv * WG, WG),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          const size_t index = it.get_group_linear_id(),
                       batch = index / shapes.n_head_kv,
                       kvhead = index % shapes.n_head_kv;
          const int tid = it.get_local_linear_id(), subgroup = tid / 16;
          const int hb = subgroup / 4, db = subgroup % 4;
          auto sg = it.get_sub_group();
          const auto kvp = [&]() {
            if constexpr (KMapping == 2) {
              // This points into the sole local object's single big half array.
              // Explicitly retain local address space for joint_matrix_load.
              return sycl::address_space_cast<sycl::access::address_space::local_space,
                  sycl::access::decorated::no>(&kv[0].values[0]);
            } else {
              return kv.template get_multi_ptr<sycl::access::decorated::no>();
            }
          }();
          const float *q =
              query + (batch * shapes.n_head + kvhead * HEADS) * HD;
          float qm = 0;
          for (int i = tid; i < HEADS * HD; i += WG)
            qm = sycl::fmax(qm, sycl::fabs(q[i]));
          qm = sycl::reduce_over_group(it.get_group(), qm,
                                       sycl::maximum<float>());
          int qe = 0;
          if (qm > 0)
            (void)sycl::frexp(qm, &qe);
          const int shift = sycl::clamp(14 - qe, -126, 126);
          const float qup = sycl::ldexp(1.f, shift),
                      qdown = sycl::ldexp(.0625f, -shift);
          for (int i = tid; i < 16 * HD; i += WG) {
            const float x = i < HEADS * HD ? q[i] * qup : 0.f;
            const sycl::half hi = sycl::half(x);
            qhi[i] = hi;
            qlo[i] = sycl::half(x - float(hi));
          }
          if (tid < 16) {
            maxima[tid] = -INFINITY;
            sums[tid] = 0;
          }
          using Acc = mx::joint_matrix<sycl::sub_group, float,
                                       mx::use::accumulator, 8, 16>;
          using A = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8,
                                     16, mx::layout::row_major>;
          using B = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b,
                                     16, 16, mx::layout::row_major>;
          Acc acc[4];
          for (int j = 0; j < 4; ++j)
            mx::joint_matrix_fill(sg, acc[j], 0.f);
          const int count0 = steps[batch * kStepCount + kStepWidth];
          const int count = count0 >= 0 && count0 <= cap ? count0 : 0;
          it.barrier(sycl::access::fence_space::local_space);
          for (int c0 = 0; c0 < count; c0 += CH) {
            if (tid < CH) {
              int64_t row = -1;
              if (c0 + tid < count) {
                const int cell = ids[batch * cap + c0 + tid];
                const int page =
                    cell < 0 ? -1 : pools.page_table[cell / shapes.page_size];
                if (page >= 0)
                  row = (int64_t(page) * shapes.n_head_kv + kvhead) *
                            shapes.page_size +
                        cell % shapes.page_size;
              }
              rows[tid] = row;
            }
            for (int i = tid; i < 16 * CH; i += WG)
              scores[i] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = tid; i < HD * CH; i += WG) {
              if constexpr (KMapping != 0) {
                // Read consecutive INT8 dimensions; retain the exact QK tile.
                const int c = i / HD, dim = i % HD;
                kvp[dim * KP + c] = rows[c] >= 0
                    ? half_code<Mode, false, Native>(pools, rows[c], dim)
                    : sycl::half(0.f);
              } else {
                const int dim = i / CH, c = i % CH;
                kvp[i] = rows[c] >= 0
                            ? half_code<Mode, false, Native>(pools, rows[c], dim)
                            : sycl::half(0.f);
              }
            }
            for (int i = tid; i < CH * KG; i += WG)
              ks[i] = rows[i / KG] >= 0
                          ? scale<Mode, false>(pools, rows[i / KG], i % KG)
                          : 0.f;
            it.barrier(sycl::access::fence_space::local_space);
            if (subgroup < 4) {
              const int head = (subgroup / 2) * 8, cell = (subgroup % 2) * 16;
              for (int g = 0; g < KG; ++g) {
                Acc part;
                A a;
                B b;
                mx::joint_matrix_fill(sg, part, 0.f);
                for (int k = g * (HD / KG); k < (g + 1) * (HD / KG); k += 16) {
                  mx::joint_matrix_load(
                      sg, b,
                      kvp + k * KP + cell,
                      KP);
                  mx::joint_matrix_load(sg, a,
                                        qhi.template get_multi_ptr<
                                            sycl::access::decorated::no>() +
                                            head * HD + k,
                                        HD);
                  mx::joint_matrix_mad(sg, part, a, b, part);
                  mx::joint_matrix_load(sg, a,
                                        qlo.template get_multi_ptr<
                                            sycl::access::decorated::no>() +
                                            head * HD + k,
                                        HD);
                  mx::joint_matrix_mad(sg, part, a, b, part);
                }
                mi::joint_matrix_apply(
                    sg, part, [=](float &v, size_t r, size_t c) {
                      scores[(head + r) * CH + cell + c] +=
                          v * ks[(cell + c) * KG + g] * qdown;
                    });
              }
            }
            it.barrier(sycl::access::fence_space::local_space);
            if constexpr (SubgroupSoftmax) {
              for (int pass = 0; pass < 2; ++pass)
                (void)prompt_softmax_row<false>(
                    sg, sg.get_group_linear_id() + pass * 8,
                    scores, maxima, sums, alpha, rows);
            } else if (tid < 16) {
              float m = maxima[tid];
              for (int c = 0; c < CH; ++c)
                if (rows[c] >= 0)
                  m = sycl::fmax(m, scores[tid * CH + c]);
              const float a =
                  maxima[tid] == -INFINITY ? 0.f : sycl::exp(maxima[tid] - m);
              float l = 0;
              for (int c = 0; c < CH; ++c) {
                const float p =
                    rows[c] >= 0 ? sycl::exp(scores[tid * CH + c] - m) : 0.f;
                scores[tid * CH + c] = p;
                l += p;
              }
              maxima[tid] = m;
              sums[tid] = sycl::fma(sums[tid], a, l);
              alpha[tid] = a;
            }
            for (int i = tid; i < CH * VG; i += WG)
              vs[i] = rows[i / VG] >= 0
                          ? scale<Mode, true>(pools, rows[i / VG], i % VG)
                          : 0.f;
            for (int i = tid; i < CH * HD; i += WG) {
              const int c = i / HD, dim = i % HD;
              kvp[i] = rows[c] >= 0
                          ? half_code<Mode, true, Native>(pools, rows[c], dim)
                          : sycl::half(0.f);
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (tid < VG) {
              float m = 0;
              for (int c = 0; c < CH; ++c)
                m = sycl::fmax(m, sycl::fabs(vs[c * VG + tid]));
              factors[tid] = m;
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = tid; i < VG * 16 * CH; i += WG) {
              const int g = i / (16 * CH), r = (i / CH) % 16, c = i % CH;
              const float m = factors[g];
              const float p =
                  m > 0 ? scores[r * CH + c] * (vs[c * VG + g] * (16384.f / m))
                        : 0.f;
              const sycl::half hi = sycl::half(p);
              phi[i] = hi;
              plo[i] = sycl::half(p - float(hi));
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int j = 0; j < 4; ++j) {
              const int dim = (db * 4 + j) * 16, vg = dim / (HD / VG);
              Acc part;
              A a;
              B b;
              mx::joint_matrix_fill(sg, part, 0.f);
              for (int k = 0; k < CH; k += 16) {
                mx::joint_matrix_load(
                    sg, b,
                    kvp + k * HD + dim,
                    HD);
                mx::joint_matrix_load(
                    sg, a,
                    phi.template get_multi_ptr<sycl::access::decorated::no>() +
                        (vg * 16 + hb * 8) * CH + k,
                    CH);
                mx::joint_matrix_mad(sg, part, a, b, part);
                mx::joint_matrix_load(
                    sg, a,
                    plo.template get_multi_ptr<sycl::access::decorated::no>() +
                        (vg * 16 + hb * 8) * CH + k,
                    CH);
                mx::joint_matrix_mad(sg, part, a, b, part);
              }
              mi::joint_matrix_apply(
                  sg, acc[j],
                  [=](float &v, size_t r, size_t) { v *= alpha[hb * 8 + r]; });
              const float down = factors[vg] * (1.f / 16384.f);
              mx::joint_matrix_apply(
                  sg, acc[j], part, [=](float &v, float &u) { v += u * down; });
            }
            it.barrier(sycl::access::fence_space::local_space);
          }
          for (int j = 0; j < 4; ++j) {
            mi::joint_matrix_apply(sg, acc[j], [=](float &v, size_t r, size_t c) {
              const float sum = sums[hb * 8 + r];
              v = sum > 0 ? v / sum : 0.f;
              if constexpr (Direct) {
                if (hb * 8 + r < HEADS)
                  output[(batch * shapes.n_head + kvhead * HEADS + hb * 8 + r) *
                             HD + (db * 4 + j) * 16 + c] = v;
              }
            });
            if constexpr (!Direct)
              mx::joint_matrix_store(
                  sg, acc[j],
                  final.template get_multi_ptr<sycl::access::decorated::no>() +
                      hb * 8 * HD + (db * 4 + j) * 16,
                  HD, mx::layout::row_major);
          }
          if constexpr (!Direct) {
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = tid; i < HEADS * HD; i += WG)
              output[(batch * shapes.n_head + kvhead * HEADS) * HD + i] = final[i];
          }
        });
  });
  finish(stream, event);
}
// Select distinct kernels so the existing output path has no device branch.
template <int Mode, bool SubgroupSoftmax>
void dispatch_softmax(const float *q, QsaAttnPools p, const int32_t *ids,
                      const int32_t *steps, int64_t cap, QsaShapes s, float *out,
                      int64_t nq, void *stream) {
  const char *direct = std::getenv("STRATA_SYCL_PROMPT_DIRECT");
  if (direct && direct[0] == '1') {
    if constexpr (Mode == 1) {
      // Fixed by the owning process before exec; do not mutate runtime env.
      const char *cell_major = std::getenv("STRATA_SYCL_PROMPT_K_CELL_MAJOR");
      if (cell_major && cell_major[0] == '2' && cell_major[1] == '\0') {
        // Preserve existing matrix support gate; never silently relabel arm 0/1.
        if (queue_for(stream).get_device().get_info<sycl::info::device::local_mem_size>() <
            size_t(60880 + 4096))
          throw std::runtime_error("prompt K pitch40 requires local memory >=64976 bytes");
        launch<1, true, false, 2, SubgroupSoftmax>(
            q, p, ids, steps, cap, s, out, nq, stream);
        return;
      }
      if (cell_major && cell_major[0] == '1' && cell_major[1] == '\0') {
        launch<1, true, false, 1, SubgroupSoftmax>(
            q, p, ids, steps, cap, s, out, nq, stream);
        return;
      }
    }
    if constexpr (Mode == 0) {
      const char *native = std::getenv("STRATA_SYCL_PROMPT_NATIVE_KV_MIN");
      char *end = nullptr;
      const long long minimum = native ? std::strtoll(native, &end, 10) : 0;
      if (minimum > 0 && end != native && *end == '\0' && nq >= minimum) {
        launch<Mode, true, true, 0, SubgroupSoftmax>(
            q, p, ids, steps, cap, s, out, nq, stream);
        return;
      }
    }
    launch<Mode, true, false, 0, SubgroupSoftmax>(
        q, p, ids, steps, cap, s, out, nq, stream);
  } else
    launch<Mode, false, false, 0, SubgroupSoftmax>(
        q, p, ids, steps, cap, s, out, nq, stream);
}
template <int Mode>
void dispatch(const float *q, QsaAttnPools p, const int32_t *ids,
              const int32_t *steps, int64_t cap, QsaShapes s, float *out,
              int64_t nq, void *stream) {
  if (prompt_softmax_subgroup_enabled())
    dispatch_softmax<Mode, true>(q, p, ids, steps, cap, s, out, nq, stream);
  else
    dispatch_softmax<Mode, false>(q, p, ids, steps, cap, s, out, nq, stream);
}
bool matrix_available(const sycl::device &device, int mode) {
  const auto subgroups = device.get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(subgroups.begin(), subgroups.end(), 16) == subgroups.end() ||
      device.get_info<sycl::info::device::local_mem_size>() <
          size_t(51648 +
                 2180 * (mode == 2 || mode == 3 ? 8
                         : mode == 1            ? 4
                                                : 1) +
                 128 * (mode == 2   ? 8
                        : mode == 0 ? 1
                                    : 4)) ||
      device.get_info<sycl::info::device::max_work_group_size>() < WG)
    return false;
  try {
    for (const auto &c : device.get_info<sycl::ext::oneapi::experimental::info::
                                             device::matrix_combinations>())
      if (c.atype == mx::matrix_type::fp16 &&
          c.btype == mx::matrix_type::fp16 &&
          c.ctype == mx::matrix_type::fp32 &&
          (c.msize == 8 || c.max_msize >= 8) &&
          (c.nsize == 16 || c.max_nsize >= 16) &&
          (c.ksize == 16 || c.max_ksize >= 16))
        return true;
  } catch (const sycl::exception &) {
    return false;
  }
  return false;
}
} // namespace
bool qsa_prompt_attn_batch(const float *q, const QsaAttnPools &p,
                           const int32_t *ids, const int32_t *steps,
                           int64_t cap, const QsaShapes &s, float *out,
                           int64_t nq, void *stream) {
  if (nq == 0)
    return true;
  if (nq < 0 || nq > 65535 || cap <= 0 || cap > INT_MAX || s.head_dim != HD ||
      s.n_head_kv != 2 || s.n_head != HEADS * s.n_head_kv || s.page_size <= 0 ||
      s.page_size > INT_MAX)
    return false;
  const int mode = p.k_q4 ? 2 : p.k_q ? (p.v_q4 ? 3 : 1) : 0;
  if (mode == 2) {
    const char *off = std::getenv("STRATA_PROMPT_ATTN_Q4");
    if (off && off[0] == '0')
      return false;
  }
  auto &queue = queue_for(stream);
  if (!matrix_available(queue.get_device(), mode))
    return false;
  const size_t n = checked_count(checked_count(nq, s.n_head), HD);
  validate_spans({{out, checked_count(n, 4)}},
                 {{q, checked_count(n, 4)},
                  {ids, checked_count(checked_count(nq, cap), 4)},
                  {steps, checked_count(checked_count(nq, kStepCount), 4)},
                  {p.page_table, 4}});
  auto input = [&](const void *ptr) {
    validate_spans({{out, n * 4}}, {{ptr, 2}}, 2);
  };
  if (mode == 0) {
    input(p.k_pool);
    input(p.v_pool);
  }
  if (mode == 1 || mode == 3) {
    input(p.k_q);
    input(p.k_scale);
  }
  if (mode == 1) {
    input(p.v_q);
    input(p.v_scale);
  }
  if (mode == 2) {
    input(p.k_q4);
    input(p.v_q4);
  }
  if (mode == 3)
    input(p.v_q4);
  switch (mode) {
  case 0:
    dispatch<0>(q, p, ids, steps, cap, s, out, nq, stream);
    break;
  case 1:
    dispatch<1>(q, p, ids, steps, cap, s, out, nq, stream);
    break;
  case 2:
    dispatch<2>(q, p, ids, steps, cap, s, out, nq, stream);
    break;
  case 3:
    dispatch<3>(q, p, ids, steps, cap, s, out, nq, stream);
    break;
  }
  return true;
}
} // namespace strata::kernels

// Native BF16 I/O candidate. Own MIT implementation built on this file's
// existing head12+4 / WG128 Mapping0 algorithm; softmax opt-in is shared.
namespace strata::kernels {
namespace {
float prompt_bf16_float(uint16_t bits) {
  return sycl::bit_cast<float>(uint32_t(bits) << 16);
}
uint16_t prompt_bf16_bits(float value) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    return uint16_t((bits >> 16) | 0x0040U); // Quiet NaN; retain sign/payload.
  return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
// Exact declared local storage: Q 8192, KV 16384, P 4096, scores 2048,
// K/V scales 1024, max/sum/alpha 192, factors 16, rows 256, shifts/bad 128.
constexpr size_t BF16_SLM = 32336;
static_assert(BF16_SLM == (16 * HD + CH * HD + 4 * 16 * CH) * sizeof(sycl::half)
    + (16 * CH + 2 * CH * 4 + 3 * 16 + 4) * sizeof(float)
    + CH * sizeof(int64_t) + 2 * 16 * sizeof(int));
bool prompt_bf16_matrix_available(const sycl::device &device) {
  if (!device.is_gpu() || !device.has(sycl::aspect::fp16) ||
      !device.has(sycl::aspect::ext_intel_matrix) ||
      device.get_info<sycl::info::device::local_mem_size>() < BF16_SLM + 4096 ||
      device.get_info<sycl::info::device::max_work_group_size>() < WG ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < WG)
    return false;
  const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(sizes.begin(), sizes.end(), 16) == sizes.end())
    return false;
  try {
    for (const auto &c : device.get_info<sycl::ext::oneapi::experimental::info::
                                             device::matrix_combinations>())
      if (c.atype == mx::matrix_type::fp16 &&
          c.btype == mx::matrix_type::fp16 &&
          c.ctype == mx::matrix_type::fp32 &&
          c.dtype == mx::matrix_type::fp32 &&
          (c.msize == 8 || (c.msize == 0 && c.max_msize >= 8)) &&
          (c.nsize == 16 || (c.nsize == 0 && c.max_nsize >= 16)) &&
          (c.ksize == 16 || (c.ksize == 0 && c.max_ksize >= 16)))
        return true;
  } catch (const sycl::exception &) {
    return false;
  }
  return false;
}
void prompt_bf16_registered(sycl::queue &queue, Span span) {
  // Full range, live Strata device allocation, and exact queue context/device.
  // The caller still supplies true suballocation capacities from the owners.
  validate_device_span(span.pointer, span.bytes, queue);
}
template <bool SubgroupSoftmax>
void launch_prompt_bf16(const uint16_t *query, QsaAttnPools pools,
                       const int32_t *ids, const int32_t *steps, int64_t cap,
                       QsaShapes shapes, uint16_t *output, int64_t queries,
                       QsaPromptBf16Bounds bounds, void *stream) {
  constexpr int KG = 4, VG = 4;
  auto event = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<sycl::half, 1> qtile(16 * HD, h), kv(CH * HD, h),
        probability(VG * 16 * CH, h);
    sycl::local_accessor<float, 1> scores(16 * CH, h), ks(CH * KG, h),
        vs(CH * VG, h), maxima(16, h), sums(16, h), alpha(16, h), factors(VG, h);
    sycl::local_accessor<int64_t, 1> rows(CH, h);
    sycl::local_accessor<int, 1> shifts(16, h), query_bad(16, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(queries) * shapes.n_head_kv * WG, WG),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          const size_t index = it.get_group_linear_id(),
                       batch = index / shapes.n_head_kv,
                       kvhead = index % shapes.n_head_kv;
          const int tid = it.get_local_linear_id();
          auto sg = it.get_sub_group();
          const int subgroup = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
          const int hb = subgroup / 4, db = subgroup % 4;
          auto kvp = kv.template get_multi_ptr<sycl::access::decorated::no>();
          const uint16_t *q = query + (batch * shapes.n_head + kvhead * HEADS) * HD;
          // Per-head power-of-two scaling respects BF16's wide exponent range.
          // Only scalar register values widen; no global FP32 tensor exists.
          for (int head = subgroup; head < 16; head += 8) {
            float maximum = 0.f;
            int bad = 0;
            if (head < HEADS)
              for (int d = lane; d < HD; d += 16) {
                const float value = prompt_bf16_float(q[head * HD + d]);
                if (sycl::isfinite(value))
                  maximum = sycl::fmax(maximum, sycl::fabs(value));
                else
                  bad = 1;
              }
            maximum = sycl::reduce_over_group(sg, maximum, sycl::maximum<float>());
            bad = sycl::reduce_over_group(sg, bad, sycl::maximum<int>());
            if (lane == 0) {
              int exponent = 0;
              if (maximum > 0) (void)sycl::frexp(maximum, &exponent);
              shifts[head] = maximum > 0 ? sycl::clamp(14 - exponent, -126, 126) : 0;
              query_bad[head] = bad;
              maxima[head] = -INFINITY;
              sums[head] = 0.f;
            }
          }
          it.barrier(sycl::access::fence_space::local_space);
          for (int i = tid; i < 16 * HD; i += WG) {
            float value = i < HEADS * HD ? prompt_bf16_float(q[i]) : 0.f;
            if (!sycl::isfinite(value)) value = 0.f;
            qtile[i] = sycl::half(sycl::ldexp(value, shifts[i / HD]));
          }
          using Acc = mx::joint_matrix<sycl::sub_group, float,
                                       mx::use::accumulator, 8, 16>;
          using A = mx::joint_matrix<sycl::sub_group, sycl::half,
                                     mx::use::a, 8, 16, mx::layout::row_major>;
          using B = mx::joint_matrix<sycl::sub_group, sycl::half,
                                     mx::use::b, 16, 16, mx::layout::row_major>;
          Acc acc[4];
          for (int j = 0; j < 4; ++j) mx::joint_matrix_fill(sg, acc[j], 0.f);
          const int count0 = steps[batch * kStepCount + kStepWidth];
          const int count = count0 >= 0 && count0 <= cap ? count0 : 0;
          int scales_bad = 0; // Uniform work-group value after each reduction.
          it.barrier(sycl::access::fence_space::local_space);
          for (int64_t c0 = 0; c0 < count; c0 += CH) {
            if (tid < CH) {
              int64_t row = -1;
              if (c0 + tid < count) {
                const int cell = ids[batch * cap + c0 + tid];
                const int64_t logical = cell < 0 ? -1 : cell / shapes.page_size;
                const int page = logical >= 0 && logical < bounds.logical_pages
                                     ? pools.page_table[logical] : -1;
                if (page >= 0 && page < bounds.physical_pages)
                  row = (int64_t(page) * shapes.n_head_kv + kvhead) *
                            shapes.page_size + cell % shapes.page_size;
              }
              rows[tid] = row;
            }
            for (int i = tid; i < 16 * CH; i += WG) scores[i] = 0.f;
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = tid; i < HD * CH; i += WG) {
              const int d = i / CH, c = i % CH;
              kvp[i] = rows[c] >= 0 ? sycl::half(pools.k_q[rows[c] * HD + d])
                                    : sycl::half(0.f);
            }
            int bad = 0;
            for (int i = tid; i < CH * KG; i += WG) {
              const int64_t row = rows[i / KG];
              float k = row >= 0 ? scale<1, false>(pools, row, i % KG) : 0.f;
              float v = row >= 0 ? scale<1, true>(pools, row, i % VG) : 0.f;
              if (!sycl::isfinite(k)) { bad = 1; k = 0.f; }
              if (!sycl::isfinite(v)) { bad = 1; v = 0.f; }
              ks[i] = k; vs[i] = v;
            }
            scales_bad |= sycl::reduce_over_group(it.get_group(), bad, sycl::maximum<int>());
            it.barrier(sycl::access::fence_space::local_space);
            if (subgroup < 4) {
              const int head = (subgroup / 2) * 8, cell = (subgroup % 2) * 16;
              for (int g = 0; g < KG; ++g) {
                Acc part; A a; B b;
                mx::joint_matrix_fill(sg, part, 0.f);
                for (int k = g * 64; k < (g + 1) * 64; k += 16) {
                  mx::joint_matrix_load(sg, b, kvp + k * CH + cell, CH);
                  mx::joint_matrix_load(sg, a,
                      qtile.template get_multi_ptr<sycl::access::decorated::no>() +
                          head * HD + k, HD);
                  mx::joint_matrix_mad(sg, part, a, b, part);
                }
                mi::joint_matrix_apply(sg, part, [=](float &value, size_t r, size_t c) {
                  // Apply ldexp last: qdown need not itself be representable.
                  scores[(head + r) * CH + cell + c] += sycl::ldexp(
                      value * (ks[(cell + c) * KG + g] * .0625f), -shifts[head + r]);
                });
              }
            }
            it.barrier(sycl::access::fence_space::local_space);
            if constexpr (SubgroupSoftmax) {
              for (int pass = 0; pass < 2; ++pass) {
                const int head = subgroup + pass * 8;
                const int score_bad = prompt_softmax_row<true>(
                    sg, head, scores, maxima, sums, alpha, rows);
                if (lane == 0) query_bad[head] |= score_bad;
              }
            } else if (tid < 16) {
              const float old_max = maxima[tid];
              float m = old_max;
              for (int c = 0; c < CH; ++c)
                if (rows[c] >= 0) {
                  const float score = scores[tid * CH + c];
                  if (sycl::isnan(score)) query_bad[tid] = 1;
                  m = sycl::fmax(m, score);
                }
              // Equal infinite maxima share mass; missing chunks add no mass.
              // No exp(inf-inf) or exp(-inf-(-inf)) is evaluated.
              const float a = sums[tid] > 0.f
                  ? (old_max == m ? 1.f : (m == INFINITY ? 0.f : sycl::exp(old_max - m)))
                  : 0.f;
              float total = 0.f;
              for (int c = 0; c < CH; ++c) {
                const float score = scores[tid * CH + c];
                const float p = rows[c] < 0 ? 0.f
                    : (sycl::isinf(m) ? (score == m ? 1.f : 0.f) : sycl::exp(score - m));
                scores[tid * CH + c] = p;
                total += p;
              }
              maxima[tid] = m;
              sums[tid] = sycl::fma(sums[tid], a, total);
              alpha[tid] = a;
            }
            // The same bounded tile now holds V codes in cell-major order.
            for (int i = tid; i < CH * HD; i += WG) {
              const int c = i / HD, d = i % HD;
              kvp[i] = rows[c] >= 0 ? sycl::half(pools.v_q[rows[c] * HD + d])
                                    : sycl::half(0.f);
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (tid < VG) {
              float m = 0.f;
              for (int c = 0; c < CH; ++c) m = sycl::fmax(m, sycl::fabs(vs[c * VG + tid]));
              factors[tid] = m;
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = tid; i < VG * 16 * CH; i += WG) {
              const int g = i / (16 * CH), r = (i / CH) % 16, c = i % CH;
              const float m = factors[g];
              // Divide FIRST: |vs/m|<=1 for finite scales, including 2^-24.
              // P is one FP16 part, bounded by 16384, never a hi+lo emulation.
              const float p = m > 0.f
                  ? (scores[r * CH + c] * (vs[c * VG + g] / m)) * 16384.f : 0.f;
              probability[i] = sycl::half(p);
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int j = 0; j < 4; ++j) {
              const int dim = (db * 4 + j) * 16, vg = dim / 64;
              Acc part; A a; B b;
              mx::joint_matrix_fill(sg, part, 0.f);
              for (int k = 0; k < CH; k += 16) {
                mx::joint_matrix_load(sg, b, kvp + k * HD + dim, HD);
                mx::joint_matrix_load(sg, a,
                    probability.template get_multi_ptr<sycl::access::decorated::no>() +
                        (vg * 16 + hb * 8) * CH + k, CH);
                mx::joint_matrix_mad(sg, part, a, b, part);
              }
              mi::joint_matrix_apply(sg, acc[j],
                  [=](float &value, size_t r, size_t) { value *= alpha[hb * 8 + r]; });
              const float down = factors[vg] * (1.f / 16384.f);
              mx::joint_matrix_apply(sg, acc[j], part,
                  [=](float &value, float &partial) { value += partial * down; });
            }
            it.barrier(sycl::access::fence_space::local_space);
          }
          for (int j = 0; j < 4; ++j)
            mi::joint_matrix_apply(sg, acc[j], [=](float &value, size_t r, size_t c) {
              const int head = hb * 8 + r;
              if (head < HEADS) {
                const float sum = sums[head];
                output[(batch * shapes.n_head + kvhead * HEADS + head) * HD +
                       (db * 4 + j) * 16 + c] = sum == 0.f ? uint16_t(0)
                    : (query_bad[head] || scales_bad ? uint16_t(0x7fc0)
                                                    : prompt_bf16_bits(value / sum));
              }
            });
        });
  });
  finish(stream, event);
}
} // namespace

bool qsa_prompt_attn_bf16_validate(const uint16_t *q, const QsaAttnPools &p,
                                const int32_t *ids, const int32_t *steps,
                                int64_t cap, const QsaShapes &s, uint16_t *out,
                                int64_t nq, const QsaPromptBf16Bounds &bounds,
                                void *stream) {
  if (nq == 0) return true;
  // Explicit Mode1 only. Do not inspect old prompt environment switches.
  if (nq < 0 || nq > 65535 || cap <= 0 || cap > INT_MAX ||
      s.head_dim != HD || s.n_head_kv != 2 || s.n_head != HEADS * 2 ||
      s.page_size <= 0 || s.page_size > INT_MAX ||
      bounds.logical_pages <= 0 || bounds.logical_pages > INT_MAX ||
      bounds.physical_pages <= 0 || bounds.physical_pages > INT_MAX ||
      !p.k_q || !p.v_q || !p.k_scale || !p.v_scale || p.k_pool || p.v_pool || p.k_q4 || p.v_q4)
    return false;
  auto &queue = queue_for(stream);
  if (!prompt_bf16_matrix_available(queue.get_device())) return false;
  const size_t n = checked_count(checked_count(nq, s.n_head), HD);
  const size_t rows = checked_count(checked_count(bounds.physical_pages, 2), s.page_size);
  const Span output{out, checked_count(n, 2)};
  const Span inputs[] = {
      {q, checked_count(n, 2)},
      {ids, checked_count(checked_count(nq, cap), 4)},
      {steps, checked_count(checked_count(nq, kStepCount), 4)},
      {p.page_table, checked_count(bounds.logical_pages, 4)},
      {p.k_q, checked_count(rows, HD)}, {p.v_q, checked_count(rows, HD)},
      {p.k_scale, checked_count(checked_count(rows, 4), 2)},
      {p.v_scale, checked_count(checked_count(rows, 4), 2)}};
  // Byte-aligned INT8 pools are legal; scalar I/O/scales are 2-byte aligned,
  // indices/table 4-byte aligned. Every full input is disjoint from output.
  for (const auto &input : inputs) validate_spans({output}, {input}, 1);
  validate_spans({output}, {{q, inputs[0].bytes},
                           {p.k_scale, inputs[6].bytes}, {p.v_scale, inputs[7].bytes}}, 2);
  validate_spans({}, {{ids, inputs[1].bytes}, {steps, inputs[2].bytes},
                      {p.page_table, inputs[3].bytes}}, 4);
  prompt_bf16_registered(queue, output);
  for (const auto &input : inputs) prompt_bf16_registered(queue, input);
  return true;
}

bool qsa_prompt_attn_bf16_batch(const uint16_t *q, const QsaAttnPools &p,
                                const int32_t *ids, const int32_t *steps,
                                int64_t cap, const QsaShapes &s, uint16_t *out,
                                int64_t nq, const QsaPromptBf16Bounds &bounds,
                                void *stream) {
  if (!qsa_prompt_attn_bf16_validate(q, p, ids, steps, cap, s, out, nq, bounds, stream))
    return false;
  if (nq != 0) {
    if (prompt_softmax_subgroup_enabled())
      launch_prompt_bf16<true>(q, p, ids, steps, cap, s, out, nq, bounds, stream);
    else
      launch_prompt_bf16<false>(q, p, ids, steps, cap, s, out, nq, bounds, stream);
  }
  return true;
}
} // namespace strata::kernels
