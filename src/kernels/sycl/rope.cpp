// SYCL NEOX rotations: host float64 tables and analytic float32 angles.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/sycl/launch.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>

namespace strata::kernels {

namespace {
// The process's rope config (rope_scaling.hpp).  One writer - the engine's
// startup thread, before session_init builds any table or captures any graph -
// and readers after it.
RopeScaling g_rope_scaling;
} // namespace

void rope_scaling_set(const RopeScaling &scaling) { g_rope_scaling = scaling; }
const RopeScaling &rope_scaling() { return g_rope_scaling; }

void build_rope_table(int n_rot, double theta, int max_pos, float *cos_tab,
                      float *sin_tab) {
  if (n_rot <= 0 || n_rot % 2 || max_pos <= 0 || !std::isfinite(theta) ||
      theta <= 1.)
    throw std::invalid_argument("invalid SYCL RoPE table geometry/base");
  const size_t count = sycl_backend::checked_count(max_pos, n_rot / 2);
  if (count > SIZE_MAX / sizeof(float))
    throw std::invalid_argument("RoPE table overflow");
  sycl_backend::validate_spans({{cos_tab, count * 4}, {sin_tab, count * 4}},
                               {});
  const int half = n_rot / 2;
  for (int p = 0; p < max_pos; ++p) {
    for (int i = 0; i < half; ++i) {
      // float64 throughout, in the reference's order: inv, then ang, then
      // cos/sin
      const double inv = std::pow(theta, -2.0 * (double)i / (double)n_rot);
      const double ang = (double)p * inv;
      cos_tab[(size_t)p * half + i] = (float)std::cos(ang);
      sin_tab[(size_t)p * half + i] = (float)std::sin(ang);
    }
  }
}

void build_rope_table(int n_rot, const RopeScaling &sc, int max_pos,
                      float *cos_tab, float *sin_tab) {
  if (rope_scaling_invalid(sc) || n_rot <= 0 || n_rot % 2 || max_pos <= 0)
    throw std::invalid_argument("invalid SYCL RoPE table scaling/geometry");
  const size_t count = sycl_backend::checked_count(max_pos, n_rot / 2);
  if (count > SIZE_MAX / sizeof(float))
    throw std::invalid_argument("RoPE table overflow");
  sycl_backend::validate_spans({{cos_tab, count * 4}, {sin_tab, count * 4}},
                               {});
  if (sc.type == RopeScalingType::None) {
    build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab,
                     sin_tab); // the original loop, verbatim
    return;
  }
  const int half = n_rot / 2;
  const double fs = sc.freq_scale();
  const double ms = sc.mscale();
  double cd[2];
  sc.corr_dims(n_rot, cd);
  const bool correct =
      sc.ext_factor !=
      0; // ggml: the correction rides on ext_factor, not the type
  for (int p = 0; p < max_pos; ++p) {
    for (int i = 0; i < half; ++i) {
      const double inv =
          std::pow(sc.freq_base, -2.0 * (double)i / (double)n_rot);
      const double extrap =
          (double)p * inv; // the trained angle, ggml's theta_extrap
      const double interp = fs * extrap; // ggml's theta_interp
      double ang = interp;
      if (correct) {
        const double ramp =
            (double)rope_yarn_ramp((float)cd[0], (float)cd[1], i) *
            sc.ext_factor;
        ang = interp * (1.0 - ramp) + extrap * ramp;
      }
      cos_tab[(size_t)p * half + i] = (float)(std::cos(ang) * ms);
      sin_tab[(size_t)p * half + i] = (float)(std::sin(ang) * ms);
    }
  }
}

namespace {
std::atomic<bool> enabled{false};
std::atomic<const int32_t *> mtab{nullptr};
std::mutex registry_mutex;
RopeTab registered;
RopeScaling registered_scaling;
bool same_scaling(const RopeScaling &a, const RopeScaling &b) {
  return a.type == b.type && a.freq_base == b.freq_base &&
         a.factor == b.factor && a.freq_scale_in == b.freq_scale_in &&
         a.orig_ctx == b.orig_ctx && a.ext_factor == b.ext_factor &&
         a.attn_factor == b.attn_factor && a.beta_fast == b.beta_fast &&
         a.beta_slow == b.beta_slow;
}
// The initial SYCL engine owns device 0. Never pass its registered pointers to
// a queue in another context. Ordinary rotations without registrations can
// still run on any explicitly supplied in-order queue.
void validate_registry_queue(sycl::queue &q, const int32_t *m, RopeTab tab) {
  if ((m || tab.cos) &&
      q.get_context() != sycl_backend::runtime_for()->context())
    throw std::invalid_argument(
        "SYCL RoPE registered tables belong to runtime device 0");
}
size_t validate_rotation(const float *x, float *out, int64_t rows, int width,
                         int nrot, const int *positions) {
  if (rows <= 0 || width <= 0 || nrot <= 0 || nrot % 2 || nrot > width)
    throw std::invalid_argument("invalid SYCL RoPE geometry");
  const size_t count = sycl_backend::checked_count(rows, width);
  if (count > SIZE_MAX / sizeof(float))
    throw std::invalid_argument("SYCL RoPE size overflow");
  const size_t bytes = count * sizeof(float);
  sycl_backend::validate_spans({{out, bytes}},
                               {{positions, size_t(rows) * sizeof(int)}});
  sycl_backend::validate_spans({}, {{x, bytes}});
  if (x != out)
    sycl_backend::validate_spans({{out, bytes}}, {{x, bytes}});
  return bytes;
}
int position(const int32_t *table, int cell, int pair) {
  return table && cell >= 0 ? table[size_t(cell) * 3 + pair % 3] : cell;
}
} // namespace
void native_rope_set_enabled(bool value) { enabled.store(value); }
bool native_rope_enabled() { return enabled.load(); }
void mrope_table_set(const int32_t *table) {
  if (table)
    sycl_backend::validate_spans({}, {{table, 3 * sizeof(int32_t)}});
  mtab.store(table);
}
const int32_t *mrope_table() { return mtab.load(); }
void rope_table_set(const float *c, const float *s, int maxpos,
                    const RopeScaling &sc) {
  if (maxpos <= 0 || rope_scaling_invalid(sc))
    throw std::invalid_argument("invalid SYCL RoPE table");
  const size_t bytes = size_t(maxpos) * 32 * sizeof(float);
  sycl_backend::validate_spans({}, {{c, bytes}, {s, bytes}});
  std::lock_guard lock(registry_mutex);
  registered = {c, s, maxpos};
  registered_scaling = sc;
}
void rope_table_release(const float *c) {
  std::lock_guard lock(registry_mutex);
  if (c && registered.cos == c)
    registered = {};
}
RopeTab rope_table_for(const RopeScaling &sc) {
  static const bool use = [] {
    const char *e = std::getenv("STRATA_ROPE_TABLE");
    return e && e[0] == '1';
  }();
  if (!use)
    return {};
  std::lock_guard lock(registry_mutex);
  return same_scaling(sc, registered_scaling) ? registered : RopeTab{};
}
void rope_neox_apply(const float *x, float *out, int64_t rows, int width,
                     int nrot, const float *c, const float *s,
                     const int *positions, void *stream) {
  const size_t bytes = validate_rotation(x, out, rows, width, nrot, positions);
  // This API carries no table length: the caller guarantees each position is
  // in range. Validate the known minimum span, as for the CUDA entry point.
  sycl_backend::validate_spans(
      {{out, bytes}}, {{c, size_t(nrot / 2) * 4}, {s, size_t(nrot / 2) * 4}});
  const auto m = mrope_table();
  auto &q = sycl_backend::queue_for(stream);
  validate_registry_queue(q, m, {});
  const int half = nrot / 2;
  auto event =
      q.parallel_for(sycl::range<1>(size_t(rows)), [=](sycl::id<1> id) {
        const size_t row = id[0], start = row * width;
        for (int d = nrot; d < width; ++d)
          out[start + d] = x[start + d];
        for (int i = 0; i < half; ++i) {
          const int p = position(m, positions[row], i);
          const float co = p < 0 ? sycl::nan(0u) : c[size_t(p) * half + i];
          const float si = p < 0 ? sycl::nan(0u) : s[size_t(p) * half + i];
          const float a = x[start + i], b = x[start + half + i];
          rope_neox_pair(a, b, co, si, out[start + i], out[start + half + i]);
        }
      });
  sycl_backend::finish(stream, event);
}
void native_rope_apply(const float *x, float *out, int rows, int width,
                       int nrot, const RopeScaling &sc, const int *positions,
                       void *stream) {
  if (!stream || rows > 65535 || (width != 128 && width != 256) || nrot != 64 ||
      rope_scaling_invalid(sc))
    throw std::invalid_argument(
        "native SYCL RoPE requires width 128/256, rotation 64, valid scaling "
        "and explicit queue");
  const size_t bytes = validate_rotation(x, out, rows, width, nrot, positions);
  const auto m = mrope_table();
  const auto tab = rope_table_for(sc);
  if (tab.cos)
    sycl_backend::validate_spans({{out, bytes}},
                                 {{tab.cos, size_t(tab.max_pos) * 128},
                                  {tab.sin, size_t(tab.max_pos) * 128}});
  auto &q = sycl_backend::queue_for(stream);
  validate_registry_queue(q, m, tab);
  const float theta_scale = std::pow(float(sc.freq_base), -2.f / nrot);
  // A one-ULP frequency error grows with the position. Compute the 32
  // float32 frequencies with host libm, then capture 128 bytes by value;
  // no allocation or per-token table transfer is needed.
  std::array<float, 32> frequencies;
  for (int i = 0; i < 32; ++i)
    frequencies[i] = std::pow(theta_scale, float(i));
  const auto k = sc.kernel_args(nrot);
  q.parallel_for(
      sycl::range<2>(size_t(rows), size_t(width / 2)), [=](sycl::id<2> id) {
        const size_t start = id[0] * width;
        const int pair = int(id[1]);
        if (pair >= nrot / 2) {
          if (x != out) {
            out[start + 2 * pair] = x[start + 2 * pair];
            out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
          }
          return;
        }
        const int p = position(m, positions[id[0]], pair);
        float c, s;
        if (tab.cos && p >= 0 && p < tab.max_pos) {
          c = tab.cos[size_t(p) * 32 + pair];
          s = tab.sin[size_t(p) * 32 + pair];
        } else {
          const float extrap = float(p) * frequencies[pair];
          float angle = k.freq_scale * extrap, magnitude = k.attn_factor;
          if (k.ext_factor != 0.f) {
            const float ramp =
                rope_yarn_ramp(k.corr_low, k.corr_high, pair) * k.ext_factor;
            angle = angle * (1.f - ramp) + extrap * ramp;
            magnitude *= 1.f + .1f * sycl::log(1.f / k.freq_scale);
          }
          c = sycl::cos(angle) * magnitude;
          s = sycl::sin(angle) * magnitude;
        }
        const float a = x[start + pair], b = x[start + pair + nrot / 2];
        rope_neox_pair(a, b, c, s, out[start + pair],
                       out[start + pair + nrot / 2]);
      });
}
} // namespace strata::kernels

namespace strata::prefill {
void rope(float *x, int64_t T, int64_t heads, int64_t dim, int64_t ld,
          int64_t pos0, const strata::kernels::RopeScaling &sc, void *stream) {
  using namespace strata::kernels;
  using namespace strata::sycl_backend;
  if (T <= 0 || heads <= 0 || dim < 64 ||
      ld < int64_t(checked_count(heads, dim)) || pos0 < 0 || T > INT32_MAX ||
      pos0 > INT32_MAX - T || rope_scaling_invalid(sc))
    throw std::invalid_argument("invalid SYCL prompt RoPE shape or scaling");
  size_t count = checked_count(T, ld);
  if (count > SIZE_MAX / 4)
    throw std::invalid_argument("SYCL prompt RoPE span overflow");
  validate_spans({{x, count * 4}}, {});
  const auto m = mrope_table();
  const auto tab = rope_table_for(sc);
  auto &q = queue_for(stream);
  validate_registry_queue(q, m, tab);
  if (tab.cos)
    validate_spans({{x, count * 4}}, {{tab.cos, size_t(tab.max_pos) * 128},
                                      {tab.sin, size_t(tab.max_pos) * 128}});
  const float theta_scale = std::pow(float(sc.freq_base), -2.f / 64.f);
  std::array<float, 32> frequencies;
  for (int i = 0; i < 32; ++i)
    frequencies[i] = std::pow(theta_scale, float(i));
  const auto k = sc.kernel_args(64);
  auto event = q.parallel_for(
      sycl::range<2>(size_t(T) * heads, 32), [=](sycl::id<2> id) {
        const size_t t = id[0] / heads, head = id[0] % heads,
                     start = t * ld + head * dim;
        const int pair = id[1], p = position(m, int(pos0 + t), pair);
        float c, s;
        if (tab.cos && p >= 0 && p < tab.max_pos) {
          c = tab.cos[size_t(p) * 32 + pair];
          s = tab.sin[size_t(p) * 32 + pair];
        } else {
          float extrap = float(p) * frequencies[pair],
                angle = k.freq_scale * extrap, magnitude = k.attn_factor;
          if (k.ext_factor != 0.f) {
            float ramp =
                rope_yarn_ramp(k.corr_low, k.corr_high, pair) * k.ext_factor;
            angle = angle * (1.f - ramp) + extrap * ramp;
            magnitude *= 1.f + .1f * sycl::log(1.f / k.freq_scale);
          }
          c = sycl::cos(angle) * magnitude;
          s = sycl::sin(angle) * magnitude;
        }
        const float a = x[start + pair], b = x[start + pair + 32];
        rope_neox_pair(a, b, c, s, x[start + pair], x[start + pair + 32]);
      });
  finish(stream, event);
}
} // namespace strata::prefill
