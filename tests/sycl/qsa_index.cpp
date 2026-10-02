#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <numeric>
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
void close(float a, double b, double tol, const char *what) {
  if (!std::isfinite(a) || std::abs(a - b) > tol * (1 + std::abs(b)))
    throw std::runtime_error(std::string(what) + ": " + std::to_string(a) +
                             " vs " + std::to_string(b));
}
void native_pool() {
  constexpr int N = 23, D = 128, P = (N / 4 + 1) * D;
  auto shape = qsa_real_shapes();
  Buffer<float> raw(N * D), gamma(D), tail(3 * D), dead(D), pooled(P),
      tail2(3 * D), dead2(D), pooled2(P);
  Buffer<int32_t> position(1), block(1), block2(1);
  QsaIndexerBuffers a{tail.data(), dead.data(), pooled.data(), block.data()};
  QsaIndexerBuffers b{tail2.data(), dead2.data(), pooled2.data(),
                      block2.data()};
  std::vector<float> input(N * D), weight(D), rounded(N * D);
  for (int i = 0; i < N * D; ++i) {
    input[i] = float(std::sin(i * .713) * (1 + (i % 11)));
    rounded[i] = float(sycl::half(input[i]));
  }
  for (int d = 0; d < D; ++d)
    weight[d] = .5f + float(d % 9) / 13;
  raw.put(input);
  gamma.put(weight);
  auto *q = &runtime->compute();
  for (int mode = 0; mode < 3; ++mode) {
    RopeScaling sc;
    if (mode) {
      sc.type = mode == 1 ? RopeScalingType::Linear : RopeScalingType::YaRN;
      sc.factor = 4;
    }
    tail.put(std::vector<float>(3 * D));
    dead.put(std::vector<float>(D));
    pooled.put(std::vector<float>(P));
    block.put({-1});
    tail2.put(std::vector<float>(3 * D));
    dead2.put(std::vector<float>(D));
    pooled2.put(std::vector<float>(P));
    block2.put({-1});
    for (int i = 0; i < N; ++i) {
      position.put({i});
      native_qsa_indexer_append(raw.data() + i * D, position.data(), 100,
                                gamma.data(), 1e-6f, a, shape, N, sc, q);
    }
    int p0 = 0;
    for (int count : {1, 2, 6, 1, 8, 5}) {
      native_qsa_indexer_append_batch(raw.data() + p0 * D, count, p0, 100,
                                      gamma.data(), 1e-6f, b, shape, N, sc, q);
      p0 += count;
    }
    check(tail.get() == tail2.get() && dead.get() == dead2.get() &&
              pooled.get() == pooled2.get() && block.get() == block2.get(),
          "native batch state differs");
    check(block.get()[0] == 116, "native block position");
    auto out = pooled.get();
    const auto args = sc.kernel_args(64);
    const float theta = std::pow(float(sc.freq_base), -2.f / 64);
    for (int bidx = 0; bidx < 6; ++bidx) {
      const bool spare = bidx == 5;
      std::vector<float> means(D), ref(D);
      double ss = 0;
      for (int d = 0; d < D; ++d) {
        float sum = rounded[(spare ? 0 : bidx * 4) * D + d];
        for (int j = 1; j < 4; ++j)
          sum += rounded[(spare ? 0 : bidx * 4 + j) * D + d];
        means[d] = .25f * sum;
        ss += double(means[d]) * means[d];
      }
      for (int d = 0; d < D; ++d)
        ref[d] = float(means[d] / std::sqrt(ss / D + 1e-6) * weight[d]);
      for (int pair = 0; pair < 32; ++pair) {
        float c, s;
        rope_scaled_angle(float(spare ? 0 : 100 + 4 * bidx) *
                              std::pow(theta, float(pair)),
                          args.freq_scale, args.corr_low, args.corr_high,
                          args.ext_factor, args.attn_factor, pair, c, s);
        const float x = ref[pair], z = ref[pair + 32];
        ref[pair] = x * c - z * s;
        ref[pair + 32] = x * s + z * c;
      }
      for (int d = 0; d < D; ++d)
        close(out[bidx * D + d], ref[d], 2e-6, "native pool oracle");
    }
    const auto before = pooled.get();
    for (int p : {-1, N}) {
      position.put({p});
      native_qsa_indexer_append(raw.data(), position.data(), 100, gamma.data(),
                                1e-6f, a, shape, N, sc, q);
    }
    check(before == pooled.get(), "invalid native position modified state");
  }
  std::cout << "Native indexer: 3 scaling modes, chunked/step equality and CPU "
               "oracle passed\n";
}
void block_selection() {
  auto shape = qsa_real_shapes();
  constexpr int cap = 2051, nq = 6;
  const int counts[nq] = {0, 1, 2051, 2052, 32769, 262147};
  const int blocks = counts[nq - 1] / 4 + 1;
  Buffer<float> scores(nq * blocks);
  Buffer<int32_t> steps(nq * kStepCount), ids(nq * cap), refids(nq * cap);
  std::vector<int32_t> st(nq * kStepCount), expected(nq * cap, -77);
  for (int q = 0; q < nq; ++q) {
    st[q * kStepCount + kStepPos] = counts[q] - 1;
    st[q * kStepCount + kStepNKv] = counts[q];
    st[q * kStepCount + kStepNBid] = counts[q] / 4;
    st[q * kStepCount + kStepWidth] = std::min(counts[q], cap);
  }
  steps.put(st);
  for (int pattern = 0; pattern < 4; ++pattern) {
    std::vector<float> sc(nq * blocks);
    std::fill(expected.begin(), expected.end(), -77);
    for (int q = 0; q < nq; ++q) {
      for (int b = 0; b < blocks; ++b) {
        float v = pattern == 0   ? float(std::sin(b * .79 + q))
                  : pattern == 1 ? float(b % 13)
                  : pattern == 2 ? (b % 2 ? -0.f : 0.f)
                  : b % 13 == 0  ? NAN
                  : b % 7 == 0   ? INFINITY
                  : b % 3 == 0   ? -INFINITY
                                 : float(b % 17);
        sc[q * blocks + b] = v;
      }
      std::vector<int> all(counts[q]);
      std::iota(all.begin(), all.end(), 0);
      std::sort(all.begin(), all.end(), [&](int a, int b) {
        float x = sc[q * blocks + a / 4], y = sc[q * blocks + b / 4];
        if (std::isnan(x) != std::isnan(y))
          return !std::isnan(x);
        if (x > y)
          return true;
        if (x < y)
          return false;
        return a < b;
      });
      all.resize(std::min(counts[q], cap));
      std::sort(all.begin(), all.end());
      std::copy(all.begin(), all.end(), expected.begin() + q * cap);
    }
    scores.put(sc);
    ids.put(std::vector<int32_t>(nq * cap, -77));
    refids.put(std::vector<int32_t>(nq * cap, -77));
    qsa_block_topk(scores.data(), steps.data(), nq, blocks, cap, shape,
                   ids.data(), nullptr, blocks);
    qsa_block_topk_ref(scores.data(), steps.data(), nq, blocks, cap, shape,
                       refids.data(), nullptr);
    check(ids.get() == expected && refids.get() == expected,
          "long block selection oracle");
  }
  constexpr int B = 19, Q = 3;
  Buffer<float> pooled(B * 128), dead(128), queries(Q * 512), result(Q * B);
  Buffer<int32_t> smallsteps(Q * kStepCount);
  std::vector<float> p(B * 128), d(128), qu(Q * 512);
  for (int i = 0; i < B * 128; ++i)
    p[i] = float(std::sin(i * .134));
  for (int i = 0; i < 128; ++i)
    d[i] = float(std::cos(i * .71) * 2);
  for (int i = 0; i < Q * 512; ++i)
    qu[i] = float(std::sin(i * .017));
  std::vector<int32_t> ss(Q * kStepCount);
  const int ns[Q] = {7, 32, 70};
  for (int q = 0; q < Q; ++q) {
    ss[q * kStepCount + kStepNKv] = ns[q];
    ss[q * kStepCount + kStepNBid] = ns[q] / 4;
  }
  pooled.put(p);
  dead.put(d);
  queries.put(qu);
  smallsteps.put(ss);
  result.put(std::vector<float>(Q * B, -77));
  qsa_block_scores(pooled.data(), dead.data(), queries.data(),
                   smallsteps.data(), Q, B, shape, result.data(), nullptr, 18);
  const auto actual = result.get();
  for (int q = 0; q < Q; ++q)
    for (int b = 0; b < B; ++b) {
      if (b > ns[q] / 4) {
        check(actual[q * B + b] == -77, "block score canary");
        continue;
      }
      double total = 0;
      for (int h = 0; h < 4; ++h) {
        double dot = 0;
        for (int j = 0; j < 128; ++j)
          dot += double(b == ns[q] / 4 ? d[j] : p[b * 128 + j]) *
                 qu[q * 512 + h * 128 + j];
        total += std::max(0., dot);
      }
      if (b == ns[q] / 4 && ns[q] % 4)
        total += 1e9;
      close(actual[q * B + b], total, 3e-6, "block scores oracle");
    }
  std::cout << "24 long-context selections through 262147 cells and batched "
               "block scores passed\n";
}
void selection(int n, int pattern) {
  QsaShapes s = qsa_real_shapes();
  const int cap = int(qsa_selection_width(kTopkMaxCells, s)),
            width = int(qsa_selection_width(n, s));
  std::vector<float> scores(kTopkMaxCells, 0);
  for (int i = 0; i < n; ++i) {
    if (pattern == 0)
      scores[i] = float(std::sin(i * .137) + std::cos(i * .931));
    if (pattern == 1)
      scores[i] = float((i / 4) % 19 - 9);
    if (pattern == 2)
      scores[i] = i % 2 ? -0.f : 0.f;
    if (pattern == 3)
      scores[i] = i % 13 == 0   ? NAN
                  : i % 17 == 0 ? INFINITY
                  : i % 19 == 0 ? -INFINITY
                                : float(i % 7);
  }
  std::vector<int> ref(n);
  std::iota(ref.begin(), ref.end(), 0);
  std::sort(ref.begin(), ref.end(), [&](int a, int b) {
    float x = scores[a], y = scores[b];
    if (std::isnan(x))
      return std::isnan(y) && a < b;
    if (std::isnan(y))
      return true;
    return x == y ? a < b : x > y;
  });
  ref.resize(width);
  std::sort(ref.begin(), ref.end());
  Buffer<float> sc(scores.size());
  Buffer<int32_t> step(4), ids(cap + 16), scalar(cap + 16);
  sc.put(scores);
  step.put({n - 1, n, n / 4, width});
  ids.put(std::vector<int32_t>(cap + 16, -987));
  scalar.put(std::vector<int32_t>(cap + 16, -987));
  topk_512_step(sc.data(), s, cap, step.data(), ids.data(),
                &runtime->compute());
  auto a = ids.get();
  topk_512(sc.data(), n, s, cap, scalar.data(), &runtime->compute());
  check(a == scalar.get(), "scalar/step selection");
  for (int i = 0; i < width; ++i)
    if (a[i] != ref[i])
      throw std::runtime_error("top-k n=" + std::to_string(n) +
                               " pattern=" + std::to_string(pattern) +
                               " index=" + std::to_string(i) +
                               " actual=" + std::to_string(a[i]) +
                               " expected=" + std::to_string(ref[i]));
  for (size_t i = width; i < a.size(); ++i)
    check(a[i] == -987, "selection guard");
}
void scoring(int cells, int dim, int heads) {
  QsaShapes s = qsa_real_shapes();
  s.idx_dim = dim;
  s.idx_n_head = heads;
  const int blocks = cells / 4 + 3, n = cells / 4;
  std::vector<float> pooled(blocks * dim), query(heads * dim), bias(blocks);
  for (size_t i = 0; i < pooled.size(); ++i)
    pooled[i] = float(std::sin(i * .019));
  for (int h = 0; h < heads; ++h)
    for (int d = 0; d < dim; ++d)
      query[h * dim + d] = float(std::cos(d * .05) * (h % 2 ? -1 : 1));
  for (int b = 0; b < blocks; ++b)
    bias[b] = float(std::sin(b * .3));
  Buffer<float> p(pooled.size()), q(query.size()), b(bias.size()),
      out(blocks * 4 + 16), scalar(blocks * 4 + 16);
  Buffer<int32_t> step(4);
  p.put(pooled);
  q.put(query);
  b.put(bias);
  step.put({cells - 1, cells, n, std::min(cells, 2051)});
  for (bool biased : {false, true}) {
    out.put(std::vector<float>(blocks * 4 + 16, 1234.f));
    scalar.put(std::vector<float>(blocks * 4 + 16, 1234.f));
    qsa_index_step(p.data(), q.data(), biased ? b.data() : nullptr, s,
                   step.data(), blocks, out.data(), &runtime->compute());
    qsa_index(p.data(), n, q.data(), biased ? b.data() : nullptr, s, cells,
              scalar.data(), &runtime->compute());
    auto y = out.get();
    check(y == scalar.get(), "scalar/step scores");
    for (int block = 0; block <= n; ++block) {
      double score = biased ? bias[block] : 0.;
      for (int h = 0; h < heads; ++h) {
        double dot = 0;
        for (int d = 0; d < dim; ++d)
          dot += double(pooled[block * dim + d]) * query[h * dim + d];
        score += std::max(0., dot);
      }
      float ref = float(score);
      if (block == n && cells % 4)
        ref += 1e9f;
      for (int j = block * 4; j < std::min(cells, (block + 1) * 4); ++j)
        close(y[j], ref, 1e-6, "index score/per-head relu/tail bias");
    }
    for (size_t i = cells; i < y.size(); ++i)
      check(y[i] == 1234.f, "score guard");
  }
}
void pooling(int dim, int block, bool multi, bool scaled) {
  constexpr int cells = 19, base = 100, maxpos = 200;
  const int nrot = 64;
  QsaShapes s = qsa_real_shapes();
  s.idx_dim = dim;
  s.idx_block = block;
  RopeScaling scale;
  if (scaled) {
    scale.type = RopeScalingType::YaRN;
    scale.factor = 8;
    scale.ext_factor = 1;
  }
  std::vector<float> ct(maxpos * 32), st(ct.size()), raw(cells * dim),
      gamma(dim);
  build_rope_table(nrot, scale, maxpos, ct.data(), st.data());
  for (size_t i = 0; i < raw.size(); ++i)
    raw[i] = float(std::sin(i * .031) + .2 * std::cos(i * .17));
  for (int d = 0; d < dim; ++d)
    gamma[d] = .8f + .003f * d;
  std::vector<int32_t> mt(maxpos * 3);
  for (int p = 0; p < maxpos; ++p)
    for (int j = 0; j < 3; ++j)
      mt[p * 3 + j] = std::max(0, p - j * 3);
  Buffer<float> dc(ct.size()), ds(st.size()), x(raw.size()), g(gamma.size());
  Buffer<int32_t> pos(1), bp(1), m(mt.size());
  dc.put(ct);
  ds.put(st);
  x.put(raw);
  g.put(gamma);
  m.put(mt);
  mrope_table_set(multi ? m.data() : nullptr);
  const int tailcount = std::max(1, (block - 1) * dim),
            poolcount = (cells / block + 2) * dim;
  Buffer<float> tail(tailcount + 16), dead(dim + 16), pooled(poolcount + 16);
  tail.put(std::vector<float>(tailcount + 16, 1234.f));
  dead.put(std::vector<float>(dim + 16, 1234.f));
  pooled.put(std::vector<float>(poolcount + 16, 1234.f));
  bp.put({-1});
  QsaIndexerBuffers buffers{tail.data(), dead.data(), pooled.data(), bp.data()};
  std::vector<float> expected(poolcount + 16, 1234.f),
      expectedtail(tailcount + 16, 1234.f), spare(dim);
  auto norm = [&](std::vector<double> row) {
    double ss = 0;
    for (double a : row)
      ss += a * a;
    double inv = 1. / std::sqrt(ss / dim + double(qsa_rms_eps()));
    std::vector<float> y(dim);
    for (int d = 0; d < dim; ++d)
      y[d] = float(row[d] * inv * gamma[d]);
    return y;
  };
  double worst = 0;
  for (int t = 0; t < cells; ++t) {
    pos.put({t});
    indexer_key_append(x.data() + t * dim, pos.data(), base, g.data(),
                       qsa_rms_eps(), buffers, s, dc.data(), ds.data(),
                       &runtime->compute());
    if (t % block < block - 1)
      std::copy(raw.begin() + t * dim, raw.begin() + (t + 1) * dim,
                expectedtail.begin() + (t % block) * dim);
    if (t == 0) {
      std::vector<double> row(raw.begin(), raw.begin() + dim);
      spare = norm(row);
      for (int d = 0; d < nrot; ++d)
        spare[d] *= ct[d % 32];
      std::copy(spare.begin(), spare.end(), expected.begin());
    }
    if (t % block == block - 1) {
      int b = t / block;
      std::vector<double> row(dim, 0);
      for (int j = 0; j < block; ++j)
        for (int d = 0; d < dim; ++d)
          row[d] += raw[(b * block + j) * dim + d];
      for (auto &a : row)
        a /= block;
      auto y = norm(row);
      for (int d = 0; d < 32; ++d) {
        int p = multi ? mt[(base + b * block) * 3 + d % 3] : base + b * block;
        float a = y[d], bval = y[d + 32], c = ct[p * 32 + d],
              si = st[p * 32 + d];
        y[d] = a * c - bval * si;
        y[d + 32] = a * si + bval * c;
      }
      std::copy(y.begin(), y.end(), expected.begin() + b * dim);
      std::copy(spare.begin(), spare.end(), expected.begin() + (b + 1) * dim);
      check(bp.get()[0] == base + b * block, "pooled first-cell position");
    }
    auto actual = pooled.get();
    for (size_t i = 0; i < actual.size(); ++i) {
      worst = std::max(worst, double(std::abs(actual[i] - expected[i])));
      close(actual[i], expected[i], 2e-7, "pooled keys/spare/guards");
    }
    check(tail.get() == expectedtail, "raw tail slots/guards");
    auto dd = dead.get();
    for (int d = 0; d < dim; ++d)
      close(dd[d], spare[d], 2e-7, "dead key");
    for (size_t i = dim; i < dd.size(); ++i)
      check(dd[i] == 1234.f, "dead guard");
  }
  mrope_table_set(nullptr);
  std::cout << "pool dim=" << dim << " block=" << block << " mrope=" << multi
            << " scaled=" << scaled << " max_abs=" << worst << "\n";
}
} // namespace
int main() {
  try {
    runtime = sycl_backend::runtime_for();
    native_pool();
    block_selection();
    for (int n : {0, 1, 2051, 2052, 8193, 32768})
      for (int pattern = 0; pattern < 4; ++pattern)
        selection(n, pattern);
    for (int n : {1, 4, 7, 8193}) {
      scoring(n, 128, 4);
      scoring(n, 67, 3);
    }
    for (bool m : {false, true})
      for (bool scale : {false, true}) {
        pooling(128, 4, m, scale);
        pooling(67, 1, m, scale);
      }
    Buffer<float> scores(kTopkMaxCells), pooled(1024), query(512);
    Buffer<int32_t> ids(2051), step(4);
    auto reject = [](auto f) {
      bool caught = false;
      try {
        f();
      } catch (const std::invalid_argument &) {
        caught = true;
      }
      check(caught, "invalid indexer accepted");
    };
    auto shape = qsa_real_shapes();
    reject([&] {
      topk_512_step(scores.data(), shape, 2048, step.data(), ids.data(),
                    nullptr);
    });
    reject([&] {
      topk_512(scores.data(), 32769, shape, 2051, ids.data(), nullptr);
    });
    reject([&] {
      qsa_index(pooled.data(), 0, query.data(), nullptr, shape, 8,
                scores.data(), nullptr);
    });
    reject([&] {
      qsa_index_step(pooled.data(), query.data(), nullptr, shape, nullptr, 8,
                     scores.data(), nullptr);
    });
    runtime->wait();
    std::cout << "24 stable selections and 16 score cases passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
