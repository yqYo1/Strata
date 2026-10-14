#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <climits>
#include <utility>
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
               float *scratch, float *out, int64_t nq, int layout = -1) {
  if (layout < 0)
    qsa_decode_attn_batch(q, p, ids, steps, cap, s, scratch, out, nq,
                          &runtime->compute());
  else
    qsa_decode_attn_batch_layout(q, p, ids, steps, cap, s, scratch, out,
                                 nq, layout, &runtime->compute());
}
void run(int format, int nq, int cap, bool benchmark = false) {
  constexpr int page_size = 4, heads = 24, kvheads = 2, dim = 256;
  constexpr int pages = 65536;
  const int dense_pages = (cap + page_size - 1) / page_size;
  const int physical_pages = benchmark ? dense_pages + 1 : 5;
  const int rows = physical_pages * page_size * kvheads;
  std::cout << "BEGIN format=" << format << " nq=" << nq << " cap=" << cap
            << " physical_pages=" << physical_pages << std::endl;
  auto s = qsa_real_shapes();
  std::vector<int32_t> table(pages, -1);
  // Sparse nonidentity map; the highest logical page needs no large KV pool.
  if (benchmark) {
    // Every active benchmark cell is resident. Keep a nonidentity map and the
    // last logical cell, without diluting the score work with mostly empty pages.
    for (int i = 0; i < dense_pages; ++i)
      table[i] = (i + 7) % dense_pages;
    table[65535] = dense_pages;
  } else {
    table[0] = 3; table[1] = 1; table[17] = 4;
    table[65534] = 0; table[65535] = 2;
  }
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
    query[i] = float(std::sin(i * .019));
  std::vector<int32_t> ids(nq * cap), steps(nq * kStepCount);
  std::vector<int> counts(nq);
  for (int b = 0; b < nq; ++b) {
    const int widths[] = {cap, 0, std::min(cap, 63), std::min(cap, 64),
                          std::min(cap, 65), cap - 1, -1, cap + 1};
    counts[b] = benchmark ? cap - (b % 4) * 16 : widths[b % 8];
    steps[b * kStepCount + kStepWidth] = counts[b];
    steps[b * kStepCount + kStepPos] = 262143;
    steps[b * kStepCount + kStepNKv] = 262144;
    for (int i = 0; i < cap; ++i) {
      // Distinct ascending positive IDs, with explicit negative mask entries.
      // Place the high logical boundary at the end of each valid selection.
      const int width = counts[b] >= 0 && counts[b] <= cap ? counts[b] : 0;
      int cell = benchmark ? (i == width - 1 ? 262143 : i)
                           : (i < width / 2 ? i * 2 + b % 2 : 262144 - width + i);
      if (i >= width) cell = -1;
      else if (!benchmark && i % 11 == 9 && i != width - 1) cell = -7;
      ids[b * cap + i] = cell;
    }
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
  if (benchmark) {
    // Separate process mode: no CPU oracle, readback, or diagnostics in timings.
    auto &queue = runtime->compute();
    for (int i = 0; i < 10; ++i)
      attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
                out.data(), nq, i % 2);
    runtime->wait();
    for (int sample = 0; sample < 40; ++sample) {
      const int layout = sample % 4 == 0 || sample % 4 == 3 ? 0 : 1;
      auto begin = queue.single_task([] {});
      for (int repeat = 0; repeat < 8; ++repeat)
        attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
                  out.data(), nq, layout);
      auto end = queue.single_task([] {});
      runtime->wait(end);
      const auto ns = end.get_profiling_info<sycl::info::event_profiling::command_start>() -
                      begin.get_profiling_info<sycl::info::event_profiling::command_end>();
      std::cout << "device_envelope sample=" << sample << " layout=" << layout
                << " repeats=8 ns=" << ns << " ns_per_batch=" << double(ns) / 8
                << " (includes chunk, merge and submission gaps; not exclusive kernel time)"
                << std::endl;
    }
    return;
  }
  std::cout << "candidate pointer/overlap/format validation" << std::endl;
  for (int layout : {0, 1}) {
    auto reject = [&](const float *query_ptr, QsaAttnPools pools,
                      float *scratch_ptr) {
      bool caught = false;
      try {
        qsa_decode_attn_batch_layout(query_ptr, pools, di.data(), ds.data(),
                                     cap, s, scratch_ptr, out.data(), nq, layout,
                                     reinterpret_cast<void *>(uintptr_t(1)));
      } catch (const std::invalid_argument &) { caught = true; }
      check(caught, "candidate failed pre-submission pointer validation");
    };
    reject(nullptr, p, scratch.data());
    reject(q.data(), p, out.data());
    QsaAttnPools missing = p;
    missing.page_table = nullptr;
    reject(q.data(), missing, scratch.data());
    missing = p;
    if (format == 0) missing.v_pool = nullptr;
    if (format == 1) missing.v_q = nullptr;
    if (format == 2 || format == 3) missing.v_q4 = nullptr;
    reject(q.data(), missing, scratch.data());
  }
  std::cout << "natural batch" << std::endl;
  attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
            out.data(), nq);
  auto y = out.get();
  const auto original_scratch = scratch.get();
  for (int layout : {0, 1}) {
    std::cout << "candidate layout=" << layout << std::endl;
    attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
              out.data(), nq, layout);
    const auto candidate = out.get();
    check(std::memcmp(y.data(), candidate.data(), y.size() * sizeof(float)) == 0,
          "natural/layout output bitwise");
    const auto candidate_scratch = scratch.get();
    check(std::memcmp(original_scratch.data(), candidate_scratch.data(),
                      original_scratch.size() * sizeof(float)) == 0,
          "natural/layout partial scratch bitwise");
  }
  std::cout << "natural step API" << std::endl;
  for (int b = 0; b < nq; ++b)
    qsa_decode_attn_step(q.data() + b * heads * dim, p, di.data() + b * cap,
                         ds.data() + b * kStepCount, cap, s, ss.data(),
                         single.data() + b * heads * dim, &runtime->compute());
  check(y == single.get(), "split attention batch/single exact");
  std::cout << "independent double CPU reference" << std::endl;
  double worst = 0;
  for (int b = 0; b < nq; ++b)
    for (int h = 0; h < heads; ++h) {
      const int width = counts[b] >= 0 && counts[b] <= cap ? counts[b] : 0;
      std::vector<double> weights(width);
      double mx = -INFINITY, sum = 0;
      for (int j = 0; j < width; ++j) {
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
        for (int j = 0; j < width; ++j) {
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
        constexpr double tolerance = 2e-6;
        check(std::isfinite(actual) && err < tolerance * (1 + std::abs(ref)),
              "split attention double oracle");
      }
    }
  for (size_t i = query.size(); i < y.size(); ++i)
    check(y[i] == 1234.f, "output guard");
  const auto work = scratch.get();
  for (size_t i = nq * stride; i < work.size(); ++i)
    check(work[i] == 1234.f, "scratch guard");
  // Reuse identical allocations while changing metadata: both candidate arms
  // must consume current device contents, rather than cached host metadata.
  std::cout << "all pages nonresident and metadata reuse" << std::endl;
  dt.put(std::vector<int32_t>(pages, -1));
  for (int layout : {0, 1}) {
    attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
              out.data(), nq, layout);
    auto zero = out.get();
    for (size_t i = 0; i < query.size(); ++i)
      check(std::isfinite(zero[i]) && zero[i] == 0, "all-masked attention");
  }
  dt.put(table);
  for (int invalid : {0, -1, cap + 1}) {
    for (int b = 0; b < nq; ++b)
      steps[b * kStepCount + kStepWidth] = invalid;
    ds.put(steps);
    for (int layout : {0, 1}) {
      attention(q.data(), p, di.data(), ds.data(), cap, s, scratch.data(),
                out.data(), nq, layout);
      auto zero = out.get();
      for (size_t i = 0; i < query.size(); ++i)
        check(std::isfinite(zero[i]) && zero[i] == 0, "empty/invalid width");
    }
  }
  std::cout << "PASS format=" << format << " nq=" << nq << " cap=" << cap
            << " max_abs=" << worst << std::endl;
}
} // namespace
// These gates deliberately use no GPU allocations or submissions.
void host_contracts() {
  auto s = qsa_real_shapes();
  auto reject = [](auto f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument &) { caught = true; }
    check(caught, "invalid split attention accepted");
  };
  for (int layout : {-1, 2, INT_MAX})
    reject([&] {
      // nq=0 and a bogus stream pin validation before queue/submission access.
      qsa_decode_attn_batch_layout(nullptr, {}, nullptr, nullptr, 1, s,
                                   nullptr, nullptr, 0, layout,
                                   reinterpret_cast<void *>(uintptr_t(1)));
    });
  for (int layout : {0, 1}) {
    qsa_decode_attn_batch_layout(nullptr, {}, nullptr, nullptr, 1, s,
                                 nullptr, nullptr, 0, layout, nullptr);
    for (int64_t nq : {int64_t(-1), int64_t(65536)})
      reject([&] { qsa_decode_attn_batch_layout(nullptr, {}, nullptr, nullptr,
                  1, s, nullptr, nullptr, nq, layout, nullptr); });
  }
  for (int64_t cap : {int64_t(0), int64_t(-1), int64_t(INT_MAX) + 1})
    reject([&] { qsa_decode_attn_scratch_floats(cap, s); });
  // Largest accepted capacity/geometry has checked arithmetic only; never
  // allocate its enormous scratch. Expected formula computed independently.
  const uint64_t chunks = (uint64_t(INT_MAX) + 63) / 64;
  check(qsa_decode_attn_scratch_floats(INT_MAX, s) == chunks * 24 * 258 + 64,
        "maximum capacity scratch formula");
  s.n_head_kv = INT_MAX / 12;
  s.n_head = s.n_head_kv * 12;
  s.page_size = INT_MAX;
  check(qsa_decode_attn_scratch_floats(1, s) == uint64_t(s.n_head) * 258 + 64,
        "maximum accepted geometry scratch formula");
  reject([&] { qsa_decode_attn_scratch_floats(INT_MAX, s); });
  // Per-batch checked product refusal precedes pointer/queue use.
  reject([&] { qsa_decode_attn_batch_layout(nullptr, {}, nullptr, nullptr,
               100000000, s, nullptr, nullptr, 65535, 1, nullptr); });
  s = qsa_real_shapes(); s.head_dim = 128;
  reject([&] { qsa_decode_attn_scratch_floats(1, s); });
  s = qsa_real_shapes(); s.n_head_kv = int64_t(INT_MAX) / 12 + 1;
  reject([&] { qsa_decode_attn_scratch_floats(1, s); });
  std::cout << "PASS host geometry/layout/overflow contracts" << std::endl;
}
int main(int argc, char **argv) {
  try {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
    const bool host_only = argc == 2 && std::strcmp(argv[1], "--host-only") == 0;
    check(argc == 1 || benchmark || host_only,
          "usage: sycl_qsa_fallback_layout_test [--host-only|--benchmark]");
    if (!benchmark) host_contracts();
    if (host_only) return 0;
    runtime = sycl_backend::runtime_for();
    std::cout << "device=" << runtime->device().get_info<sycl::info::device::name>()
              << " mode=" << (benchmark ? "benchmark" : "correctness") << std::endl;
    if (benchmark) {
      run(1, 32, 2051, true);
    } else {
      for (int format = 0; format < 4; ++format)
        for (auto shape : {std::pair{1, 63}, std::pair{17, 64},
                           std::pair{32, 65}, std::pair{17, 129}, std::pair{1, 129}})
          run(format, shape.first, shape.second);
    }
    runtime->wait();
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << std::endl;
    return 1;
  }
}
