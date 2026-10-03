#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
template <class T> struct Buffer {
  sycl_backend::Allocation mem;
  Buffer(size_t n)
      : mem(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device) {}
  T *data() { return mem.as<T>(); }
  void put(const std::vector<T> &v) {
    runtime->wait(
        runtime->compute().memcpy(data(), v.data(), v.size() * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(mem.size() / sizeof(T));
    runtime->wait(runtime->compute().memcpy(v.data(), data(), mem.size()));
    return v;
  }
};
void check(bool c, const char *what) {
  if (!c)
    throw std::runtime_error(what);
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
float decode(uint16_t x) { return float(std::bit_cast<_Float16>(x)); }
void attention(const float *q, const QsaAttnPools &p, const int32_t *ids,
               const int32_t *steps, int64_t cap, const QsaShapes &s,
               float *scratch, float *out, int64_t nq) {
#ifdef STRATA_TEST_PROMPT_MATRIX
  check(qsa_prompt_attn_batch(q, p, ids, steps, cap, s, out, nq,
                              &runtime->compute()),
        "matrix attention unavailable");
#else
  qsa_decode_attn_batch(q, p, ids, steps, cap, s, scratch, out, nq,
                        &runtime->compute());
#endif
}
void run(int format, int page_size, bool masked, bool scaled = false) {
  constexpr int cells = 141, cap = 133, nq = 3, heads = 24, kvheads = 2,
                dim = 256;
  auto s = qsa_real_shapes();
  s.page_size = page_size;
  const int pages = (cells + page_size - 1) / page_size,
            rows = pages * page_size * kvheads;
  std::vector<int32_t> table(pages);
  for (int i = 0; i < pages; ++i)
    table[i] = pages - 1 - i;
  if (masked)
    for (int i = 0; i < pages; i += 3)
      table[i] = -1;
  std::vector<uint16_t> k16(rows * dim), v16(k16.size()), ks(rows * 4),
      vs(ks.size());
  std::vector<int8_t> k8(rows * dim), v8(k8.size());
  std::vector<uint8_t> k4(rows * 144), v4(k4.size());
  for (size_t i = 0; i < k16.size(); ++i) {
    k16[i] = half(float(std::sin(i * .073)));
    v16[i] = half(float(std::cos(i * .017)));
    k8[i] = int8_t(int(i * 7 % 255) - 127);
    v8[i] = int8_t(int(i * 11 % 255) - 127);
  }
  for (size_t i = 0; i < ks.size(); ++i) {
    ks[i] = half(.007f + float(i % 7) * .0001f);
    vs[i] = half(.008f + float(i % 11) * .0002f);
  }
  for (int i = 0; i < rows * 8; ++i) {
    uint16_t a = half(.11f + float(i % 7) * .002f),
             b = half(-.09f - float(i % 9) * .003f);
    std::memcpy(k4.data() + i * 18, &a, 2);
    std::memcpy(v4.data() + i * 18, &b, 2);
    for (int j = 0; j < 16; ++j) {
      k4[i * 18 + 2 + j] = uint8_t((i * 7 + j * 13) % 256);
      v4[i * 18 + 2 + j] = uint8_t((i * 11 + j * 19) % 256);
    }
  }
  auto load = [&](bool value, int row, int d) {
    if (format == 0)
      return decode((value ? v16 : k16)[row * dim + d]);
    if (format == 1 || (format == 3 && !value))
      return float((value ? v8 : k8)[row * dim + d]) *
             decode((value ? vs : ks)[row * 4 + d / 64]);
    const auto &data = value ? v4 : k4;
    const int block = row * 8 + d / 32;
    uint16_t scale;
    std::memcpy(&scale, data.data() + block * 18, 2);
    uint8_t b = data[block * 18 + 2 + d % 16];
    return float(int(d % 32 < 16 ? b & 15 : b >> 4) - 8) * decode(scale);
  };
  std::vector<float> query(nq * heads * dim);
  for (size_t i = 0; i < query.size(); ++i)
    query[i] = float(std::sin(i * .019)) *
               (scaled ? (i / (heads * dim) == 0 ? 64.f : 0x1p-20f) : 1.f);
  std::vector<int32_t> ids(nq * cap), steps(nq * kStepCount);
  const int counts[nq] = {13, 67, 129};
  for (int b = 0; b < nq; ++b) {
    steps[b * 4 + 3] = counts[b];
    for (int i = 0; i < cap; ++i)
      ids[b * cap + i] = (i * 17 + b * 3) % cells;
    if (scaled)
      ids[b * cap + 5] = -1;
  }
  Buffer<uint16_t> dk16(k16.size()), dv16(v16.size()), dks(ks.size()),
      dvs(vs.size());
  Buffer<int8_t> dk8(k8.size()), dv8(v8.size());
  Buffer<uint8_t> dk4(k4.size()), dv4(v4.size());
  Buffer<int32_t> dt(table.size()), di(ids.size()), ds(steps.size());
  Buffer<float> q(query.size());
  dk16.put(k16);
  dv16.put(v16);
  dks.put(ks);
  dvs.put(vs);
  dk8.put(k8);
  dv8.put(v8);
  dk4.put(k4);
  dv4.put(v4);
  dt.put(table);
  di.put(ids);
  ds.put(steps);
  q.put(query);
  QsaAttnPools p;
  p.page_table = dt.data();
  if (format == 0) {
    p.k_pool = dk16.data();
    p.v_pool = dv16.data();
  }
  if (format == 1 || format == 3) {
    p.k_q = dk8.data();
    p.k_scale = dks.data();
  }
  if (format == 1) {
    p.v_q = dv8.data();
    p.v_scale = dvs.data();
  }
  if (format == 2) {
    p.k_q4 = dk4.data();
    p.v_q4 = dv4.data();
  }
  if (format == 3)
    p.v_q4 = dv4.data();
  const size_t stride = qsa_decode_attn_scratch_floats(cap, s);
  Buffer<float> scratch(nq * stride + 16), out(query.size() + 16),
      single(query.size() + 16), ss(stride + 16);
  scratch.put(std::vector<float>(nq * stride + 16, 1234.f));
  out.put(std::vector<float>(query.size() + 16, 1234.f));
  single.put(std::vector<float>(query.size() + 16, 1234.f));
  attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
            out.data(), nq);
  auto y = out.get();
  for (int b = 0; b < nq; ++b)
    attention(q.data() + b * heads * dim, p, di.data() + b * cap,
              ds.data() + b * kStepCount, cap, s, ss.data(),
              single.data() + b * heads * dim, 1);
  check(y == single.get(), "split attention batch/single exact");
  double worst = 0;
  for (int b = 0; b < nq; ++b)
    for (int h = 0; h < heads; ++h) {
      std::vector<double> weights(counts[b]);
      double mx = -INFINITY, sum = 0;
      for (int j = 0; j < counts[b]; ++j) {
        int cell = ids[b * cap + j],
            page = cell < 0 ? -1 : table[cell / page_size];
        if (page < 0) {
          weights[j] = -INFINITY;
          continue;
        }
        int row = (page * kvheads + h / 12) * page_size + cell % page_size;
        double dot = 0;
        for (int d = 0; d < dim; ++d)
          dot += double(query[(b * heads + h) * dim + d]) * load(false, row, d);
        weights[j] = dot / 16.;
        mx = std::max(mx, weights[j]);
      }
      for (auto &w : weights) {
        w = std::isfinite(w) ? std::exp(w - mx) : 0.;
        sum += w;
      }
      for (int d = 0; d < dim; ++d) {
        double ref = 0;
        for (int j = 0; j < counts[b]; ++j) {
          int cell = ids[b * cap + j],
              page = cell < 0 ? -1 : table[cell / page_size];
          if (page < 0)
            continue;
          int row = (page * kvheads + h / 12) * page_size + cell % page_size;
          ref += weights[j] * load(true, row, d);
        }
        if (sum)
          ref /= sum;
        float actual = y[(b * heads + h) * dim + d];
        double err = std::abs(actual - ref);
        worst = std::max(worst, err);
#ifdef STRATA_TEST_PROMPT_MATRIX
        constexpr double tolerance = 1e-5;
#else
        const double tolerance = scaled ? 1e-5 : 2e-6;
#endif
        check(std::isfinite(actual) && err < tolerance * (1 + std::abs(ref)),
              "split attention double oracle");
      }
    }
  for (size_t i = query.size(); i < y.size(); ++i)
    check(y[i] == 1234.f, "output guard");
  const auto work = scratch.get();
  for (size_t i = nq * stride; i < work.size(); ++i)
    check(work[i] == 1234.f, "scratch guard");
  // Reuse the same buffers with all pages missing, then with zero widths.
  dt.put(std::vector<int32_t>(pages, -1));
  attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
            out.data(), nq);
  auto zero = out.get();
  for (size_t i = 0; i < query.size(); ++i)
    check(zero[i] == 0, "all-masked attention");
  ds.put(std::vector<int32_t>(steps.size(), 0));
  attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
            out.data(), nq);
  check(out.get() == zero, "empty attention");
  for (int invalid : {-1, cap + 1}) {
    for (int b = 0; b < nq; ++b)
      steps[b * kStepCount + kStepWidth] = invalid;
    ds.put(steps);
    attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
              out.data(), nq);
    check(out.get() == zero, "invalid selected-cell width");
  }
  std::cout << "split format=" << format << " page=" << page_size
            << " masked=" << masked << " max_abs=" << worst << "\n";
}
} // namespace
int main() {
  try {
    runtime = sycl_backend::runtime_for();
    for (int fmt = 0; fmt < 4; ++fmt)
      for (int page : {1, 4, 16})
        run(fmt, page, page != 1);
    for (int fmt = 0; fmt < 4; ++fmt)
      run(fmt, 16, true, true);
    auto s = qsa_real_shapes();
    auto reject = [](auto f) {
      bool caught = false;
      try {
        f();
      } catch (const std::invalid_argument &) {
        caught = true;
      }
      check(caught, "invalid split attention accepted");
    };
    reject([&] { qsa_decode_attn_scratch_floats(0, s); });
    s.head_dim = 128;
    reject([&] { qsa_decode_attn_scratch_floats(10, s); });
#ifdef STRATA_TEST_PROMPT_MATRIX
    check(!qsa_prompt_attn_batch(nullptr, {}, nullptr, nullptr, 10, s, nullptr,
                                 1, nullptr),
          "unsupported matrix shape");
    check(qsa_prompt_attn_batch(nullptr, {}, nullptr, nullptr, 0, s, nullptr, 0,
                                nullptr),
          "empty matrix batch");
#endif
    s = qsa_real_shapes();
    Buffer<float> q(24 * 256), out(24 * 256),
        scratch(qsa_decode_attn_scratch_floats(1, s));
    Buffer<int32_t> ids(1), steps(4), table(1);
    QsaAttnPools p;
    p.page_table = table.data();
    reject([&] {
      qsa_decode_attn_step(q.data(), p, ids.data(), steps.data(), 1, s,
                           scratch.data(), out.data(), nullptr);
    });
    runtime->wait();
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
