// Split attention over paged KV; 64 cells and 12 query heads per work group.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/sycl/launch.hpp"
#include <climits>
#include <cmath>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
constexpr int HD = 256, G = 12, CHUNK = 64, WG = 256;
float sum32(sycl::sub_group sg, float x) {
  for (int o = 16; o; o /= 2)
    x += sycl::permute_group_by_xor(sg, x, o);
  return x;
}
float max32(sycl::sub_group sg, float x) {
  for (int o = 16; o; o /= 2)
    x = sycl::fmax(x, sycl::permute_group_by_xor(sg, x, o));
  return x;
}
void geometry(int64_t cap, const QsaShapes &s) {
  if (cap <= 0 || cap > INT_MAX || s.head_dim != HD || s.n_head_kv <= 0 ||
      s.n_head_kv > INT_MAX / G || s.n_head != G * s.n_head_kv ||
      s.page_size <= 0 || s.page_size > INT_MAX)
    throw std::invalid_argument("unsupported SYCL split attention geometry");
}
template <int Format>
float load(const QsaAttnPools &p, bool value, size_t row, int dim) {
  if constexpr (Format == 0)
    return f32_from_f16((value ? p.v_pool : p.k_pool)[row * HD + dim]);
  if constexpr (Format == 1 || Format == 3) {
    if (Format == 1 || !value) {
      const float scale =
          f32_from_f16((value ? p.v_scale : p.k_scale)[row * 4 + dim / 64]);
      return float((value ? p.v_q : p.k_q)[row * HD + dim]) * scale;
    }
  }
  if constexpr (Format == 2 || Format == 3) {
    const auto &b = reinterpret_cast<const block_q4_0 *>(
        value ? p.v_q4 : p.k_q4)[row * 8 + dim / 32];
    const uint8_t v = b.qs[dim % 16];
    const int q = int(dim % 32 < 16 ? v & 15 : v >> 4) - 8;
    return float(q) * f32_from_f16(b.d);
  }
  return 0;
}
template <int Format>
void launch(const float *query, QsaAttnPools p, const int32_t *ids,
            const int32_t *steps, int64_t cap, QsaShapes s, float *scratch,
            float *attn, int64_t nq, void *stream) {
  const size_t chunks = (cap + CHUNK - 1) / CHUNK,
               stride = qsa_decode_attn_scratch_floats(cap, s);
  const size_t partials = chunks * s.n_head,
               groups = checked_count(nq, checked_count(chunks, s.n_head_kv));
  auto &q = queue_for(stream);
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> sq(G * HD, cgh), sp(G * CHUNK, cgh);
    sycl::local_accessor<int64_t, 1> rows(CHUNK, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(groups * WG, WG),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const size_t group = it.get_group_linear_id(), chunk = group % chunks,
                       kvh = (group / chunks) % s.n_head_kv,
                       batch = group / (chunks * s.n_head_kv);
          const int t = it.get_local_linear_id(), lane = t % 32, warp = t / 32;
          const int64_t count = steps[batch * kStepCount + kStepWidth];
          const int here =
              count >= 0 && count <= cap
                  ? int(sycl::clamp(int64_t(count) - int64_t(chunk * CHUNK),
                                    int64_t(0), int64_t(CHUNK)))
                  : 0;
          const size_t slot = (kvh * chunks + chunk) * G;
          float *pa = scratch + batch * stride;
          float *pm = pa + partials * HD;
          float *pl = pm + partials;
          if (!here) {
            if (t < G) {
              pm[slot + t] = -INFINITY;
              pl[slot + t] = 0;
            }
            for (int h = 0; h < G; ++h)
              pa[(slot + h) * HD + t] = 0;
            return;
          }
          for (int i = t; i < G * HD; i += WG)
            sq[i] = query[(batch * s.n_head + kvh * G) * HD + i];
          if (t < CHUNK) {
            int64_t row = -1;
            if (t < here) {
              const int cell = ids[batch * cap + chunk * CHUNK + t];
              const int page = cell < 0 ? -1 : p.page_table[cell / s.page_size];
              if (page >= 0)
                row = (int64_t(page) * s.n_head_kv + kvh) * s.page_size +
                      cell % s.page_size;
            }
            rows[t] = row;
          }
          it.barrier(sycl::access::fence_space::local_space);
          auto sg = it.get_sub_group();
          for (int cell = warp; cell < CHUNK; cell += 8) {
            if (cell >= here || rows[cell] < 0) {
              if (lane < G)
                sp[lane * CHUNK + cell] = -INFINITY;
              continue;
            }
            float k[8];
            for (int j = 0; j < 8; ++j)
              k[j] = load<Format>(p, false, size_t(rows[cell]), lane * 8 + j);
            for (int h = 0; h < G; ++h) {
              float dot = k[0] * sq[h * HD + lane * 8];
              for (int j = 1; j < 8; ++j)
                dot = sycl::fma(k[j], sq[h * HD + lane * 8 + j], dot);
              dot = sum32(sg, dot);
              if (!lane)
                sp[h * CHUNK + cell] = dot * .0625f;
            }
          }
          it.barrier(sycl::access::fence_space::local_space);
          for (int h = warp; h < G; h += 8) {
            const float a = sp[h * CHUNK + lane], b = sp[h * CHUNK + lane + 32],
                        m = max32(sg, sycl::fmax(a, b));
            const float ea = lane < here && rows[lane] >= 0 ? sycl::exp(a - m)
                                                            : 0.f,
                        eb = lane + 32 < here && rows[lane + 32] >= 0
                                 ? sycl::exp(b - m)
                                 : 0.f;
            sp[h * CHUNK + lane] = ea;
            sp[h * CHUNK + lane + 32] = eb;
            const float l = sum32(sg, ea + eb);
            if (!lane) {
              pm[slot + h] = m;
              pl[slot + h] = l;
            }
          }
          it.barrier(sycl::access::fence_space::local_space);
          float acc[G] = {};
          for (int cell = 0; cell < here; ++cell) {
            if (rows[cell] < 0)
              continue;
            const float v = load<Format>(p, true, size_t(rows[cell]), t);
            for (int h = 0; h < G; ++h)
              acc[h] = sycl::fma(sp[h * CHUNK + cell], v, acc[h]);
          }
          for (int h = 0; h < G; ++h)
            pa[(slot + h) * HD + t] = acc[h];
        });
  });
  auto e = q.parallel_for(
      sycl::range<1>(size_t(nq) * s.n_head * HD), [=](sycl::id<1> id) {
        const size_t i = id[0], d = i % HD, h = (i / HD) % s.n_head,
                     batch = i / (s.n_head * HD), kvh = h / G, hl = h % G;
        const float *pa = scratch + batch * stride;
        const float *pm = pa + partials * HD;
        const float *pl = pm + partials;
        float m = -INFINITY;
        for (size_t c = 0; c < chunks; ++c)
          m = sycl::fmax(m, pm[(kvh * chunks + c) * G + hl]);
        float sum = 0, acc = 0;
        for (size_t c = 0; c < chunks; ++c) {
          const size_t slot = (kvh * chunks + c) * G + hl;
          if (pl[slot] <= 0)
            continue;
          const float factor = sycl::exp(pm[slot] - m);
          sum = sycl::fma(pl[slot], factor, sum);
          acc = sycl::fma(pa[slot * HD + d], factor, acc);
        }
        attn[i] = sum > 0 ? acc / sum : 0.f;
      });
  finish(stream, e);
}
} // namespace
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes &s) {
  geometry(cap, s);
  const size_t n =
      sycl_backend::checked_count((cap + CHUNK - 1) / CHUNK, s.n_head);
  if (n > (SIZE_MAX / sizeof(float) - 64) / (HD + 2))
    throw std::invalid_argument("split attention scratch overflow");
  return n * (HD + 2) + 64;
}
void qsa_decode_attn_batch(const float *q, const QsaAttnPools &p,
                           const int32_t *ids, const int32_t *steps,
                           int64_t cap, const QsaShapes &s, float *scratch,
                           float *attn, int64_t nq, void *stream) {
  geometry(cap, s);
  if (nq < 0 || nq > 65535)
    throw std::invalid_argument("invalid split attention batch");
  if (!nq)
    return;
  const size_t n = sycl_backend::checked_count(
                   nq, sycl_backend::checked_count(s.n_head, HD)),
               sn = sycl_backend::checked_count(
                   nq, qsa_decode_attn_scratch_floats(cap, s));
  if (n > SIZE_MAX / 4 || sn > SIZE_MAX / 4)
    throw std::invalid_argument("split attention buffer overflow");
  sycl_backend::validate_spans({{scratch, sn * 4}, {attn, n * 4}},
                               {{q, n * 4},
                                {ids, sycl_backend::checked_count(nq, cap) * 4},
                                {steps, size_t(nq) * kStepCount * 4},
                                {p.page_table, 4}});
  const int mode = p.k_q4 ? 2 : p.k_q ? (p.v_q4 ? 3 : 1) : 0;
  auto input = [&](const void *ptr) {
    sycl_backend::validate_spans({{scratch, sn * 4}, {attn, n * 4}},
                                 {{ptr, 4}});
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
    launch<0>(q, p, ids, steps, cap, s, scratch, attn, nq, stream);
    break;
  case 1:
    launch<1>(q, p, ids, steps, cap, s, scratch, attn, nq, stream);
    break;
  case 2:
    launch<2>(q, p, ids, steps, cap, s, scratch, attn, nq, stream);
    break;
  case 3:
    launch<3>(q, p, ids, steps, cap, s, scratch, attn, nq, stream);
    break;
  }
}
void qsa_decode_attn_step(const float *q, const QsaAttnPools &p,
                          const int32_t *ids, const int32_t *step, int64_t cap,
                          const QsaShapes &s, float *scratch, float *attn,
                          void *stream) {
  qsa_decode_attn_batch(q, p, ids, step, cap, s, scratch, attn, 1, stream);
}
} // namespace strata::kernels
