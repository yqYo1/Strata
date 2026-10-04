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
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace mi = sycl::ext::intel::experimental::matrix;
constexpr int HD = 256, HEADS = 12, CH = 32, WG = 128;
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
template <int Mode, bool Direct, bool Native = false>
void launch(const float *query, QsaAttnPools pools, const int32_t *ids,
            const int32_t *steps, int64_t cap, QsaShapes shapes, float *output,
            int64_t queries, void *stream) {
  constexpr int KG = scale_groups<Mode, false>(),
                VG = scale_groups<Mode, true>();
  auto event = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<sycl::half, 1> qhi(16 * HD, h), qlo(16 * HD, h),
        kv(CH * HD, h);
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
              const int dim = i / CH, c = i % CH;
              kv[i] = rows[c] >= 0
                          ? half_code<Mode, false, Native>(pools, rows[c], dim)
                          : sycl::half(0.f);
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
                      kv.template get_multi_ptr<sycl::access::decorated::no>() +
                          k * CH + cell,
                      CH);
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
            if (tid < 16) {
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
              kv[i] = rows[c] >= 0
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
                    kv.template get_multi_ptr<sycl::access::decorated::no>() +
                        k * HD + dim,
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
template <int Mode>
void dispatch(const float *q, QsaAttnPools p, const int32_t *ids,
              const int32_t *steps, int64_t cap, QsaShapes s, float *out,
              int64_t nq, void *stream) {
  const char *direct = std::getenv("STRATA_SYCL_PROMPT_DIRECT");
  if (direct && direct[0] == '1') {
    if constexpr (Mode == 0) {
      const char *native = std::getenv("STRATA_SYCL_PROMPT_NATIVE_KV_MIN");
      char *end = nullptr;
      const long long minimum = native ? std::strtoll(native, &end, 10) : 0;
      if (minimum > 0 && end != native && *end == '\0' && nq >= minimum) {
        launch<Mode, true, true>(q, p, ids, steps, cap, s, out, nq, stream);
        return;
      }
    }
    launch<Mode, true>(q, p, ids, steps, cap, s, out, nq, stream);
  } else
    launch<Mode, false>(q, p, ids, steps, cap, s, out, nq, stream);
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
