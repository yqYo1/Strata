#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
constexpr int N = NG_N_EMBD, H = NG_HC, D = NG_HC_DIM, HIST = NG_HIST;
std::shared_ptr<sycl_backend::Runtime> rt;
template <class T> struct Buffer {
  sycl_backend::Allocation storage;
  explicit Buffer(size_t count)
      : storage(rt, count * sizeof(T), sycl_backend::MemoryKind::Device) {}
  T *data() { return storage.as<T>(); }
  void put(const std::vector<T> &v) {
    rt->wait(rt->compute().memcpy(data(), v.data(), v.size() * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(storage.size() / sizeof(T));
    rt->wait(rt->compute().memcpy(v.data(), data(), storage.size()));
    return v;
  }
};
void check(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
void close(const std::vector<float> &a, const std::vector<float> &b,
           const char *msg, double tol = 8e-6) {
  check(a.size() >= b.size(), "reference size");
  for (size_t i = 0; i < b.size(); ++i)
    if (!std::isfinite(a[i]) ||
        std::abs(double(a[i]) - b[i]) > tol * (1 + std::abs(b[i])))
      throw std::runtime_error(std::string(msg) + " at " + std::to_string(i) +
                               ": " + std::to_string(a[i]) + " vs " +
                               std::to_string(b[i]));
}
uint16_t half(float v) { return std::bit_cast<uint16_t>(_Float16(v)); }
float fromhalf(uint16_t v) { return float(std::bit_cast<_Float16>(v)); }
uint16_t bf(float v) {
  auto b = std::bit_cast<uint32_t>(v);
  return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
float frombf(uint16_t b) { return std::bit_cast<float>(uint32_t(b) << 16); }
std::vector<float> norm(const std::vector<float> &x,
                        const std::vector<float> &w) {
  std::vector<float> out(D);
  for (int h = 0; h < H; ++h) {
    double ss = 0;
    for (int d = 0; d < N; ++d) {
      const float v = x[h * N + d];
      ss += double(v * v);
    }
    const float scale = 1.f / std::sqrt(float(ss / N) + NG_RMS_EPS);
    for (int d = 0; d < N; ++d)
      out[h * N + d] = (x[h * N + d] * scale) * w[h * N + d];
  }
  return out;
}
struct HostWeights {
  std::vector<float> nk, nq, nc;
  std::vector<uint16_t> taps;
  Buffer<float> dk{D}, dq{D}, dc{D};
  Buffer<uint16_t> dt{4 * D};
  PleWeights w;
  HostWeights() : nk(D), nq(D), nc(D), taps(4 * D) {
    for (int i = 0; i < D; ++i) {
      nk[i] = .8f + float(i % 13) / 31;
      nq[i] = .9f + float(i % 7) / 29;
      nc[i] = .7f + float(i % 17) / 41;
      for (int k = 0; k < 4; ++k)
        taps[i * 4 + k] = half(float(std::sin(i * .023 + k) * .2));
    }
    dk.put(nk);
    dq.put(nq);
    dc.put(nc);
    dt.put(taps);
    w.norm_key = dk.data();
    w.norm_query = dq.data();
    w.norm_conv = dc.data();
    w.conv1d_f16 = dt.data();
  }
};
struct Reference {
  std::vector<float> key, gate, gated, normalized, conv, result;
};
Reference reference(const std::vector<float> &key,
                    const std::vector<float> &hidden,
                    const std::vector<float> &value,
                    const std::vector<float> &history, const HostWeights &w,
                    bool native) {
  Reference r;
  r.key = norm(key, w.nk);
  const auto q = norm(hidden, w.nq);
  r.gate.resize(H);
  r.gated.resize(D);
  r.conv.resize(D);
  r.result.resize(D);
  for (int h = 0; h < H; ++h) {
    double dot = 0;
    for (int d = 0; d < N; ++d)
      dot += double(r.key[h * N + d] * q[h * N + d]);
    float s = float(dot) * (1.f / std::sqrt(float(N)));
    const float sign = float((s > 0) - (s < 0));
    r.gate[h] =
        1.f / (1.f + std::exp(-sign * std::sqrt(std::max(std::abs(s), 1e-6f))));
    for (int d = 0; d < N; ++d)
      r.gated[h * N + d] = value[d] * r.gate[h];
  }
  r.normalized = norm(r.gated, w.nc);
  for (int d = 0; d < D; ++d) {
    float sum = 0;
    for (int k = 0; k < 4; ++k) {
      const float term = fromhalf(w.taps[d * 4 + k]) *
                         (k == 3 ? r.normalized[d] : history[d * HIST + 3 * k]);
      sum = k == 0 ? term : sum + term;
    }
    r.conv[d] = sum / (1.f + std::exp(-sum));
    r.result[d] = native ? hidden[d] + (r.gated[d] + r.conv[d])
                         : (hidden[d] + r.gated[d]) + r.conv[d];
  }
  return r;
}
void postops(int T) {
  HostWeights w;
  std::vector<float> key(T * D), hidden(T * D), value(T * N), history(HIST * D);
  for (int i = 0; i < T * D; ++i) {
    key[i] = float(std::sin(i * .117) + std::cos(i * .031));
    hidden[i] = i / N % H == 2 ? 0.f : key[i] * (i / N % 2 ? -1.f : 1.f);
  }
  for (int i = 0; i < T * N; ++i)
    value[i] = float(std::sin(i * .177));
  for (int i = 0; i < HIST * D; ++i)
    history[i] = float(std::cos(i * .351));
  Buffer<float> k(T * D), x(T * D), v(T * N), hist(HIST * D), hist2(HIST * D),
      query(T * D), gated(T * D), gate(T * H), conv(D), normalized(D), nk(D),
      nq(D), sg(H), sv(D), out(D);
  k.put(key);
  x.put(hidden);
  v.put(value);
  hist.put(history);
  hist2.put(history);
  auto *q = &rt->compute();
  native_ple_postops_batch(k.data(), x.data(), v.data(), hist.data(), w.w,
                           query.data(), gated.data(), gate.data(), T, q);
  const auto batchk = k.get(), batchx = x.get(), batchnorm = query.get(),
             batchgate = gate.get(), batchgated = gated.get();
  k.put(key);
  std::vector<float> initial = history;
  for (int t = 0; t < T; ++t) {
    Buffer<float> h(D);
    h.put(std::vector<float>(hidden.begin() + t * D,
                             hidden.begin() + (t + 1) * D));
    // Exercise both documented aliases at once.
    NativePlePostopsBuffers b{nk.data(), nq.data(),   sg.data(), sv.data(),
                              nq.data(), conv.data(), h.data()};
    native_ple_postops(k.data() + t * D, h.data(), v.data() + t * N,
                       hist2.data(), w.w, b, q);
    const auto a = nk.get(), bnorm = nq.get(), g = sg.get(), y = h.get(),
               gatedone = sv.get();
    check(std::equal(a.begin(), a.end(), batchk.begin() + t * D),
          "PLE batch key differs");
    check(std::equal(bnorm.begin(), bnorm.end(), batchnorm.begin() + t * D),
          "PLE batch norm differs");
    check(std::equal(g.begin(), g.end(), batchgate.begin() + t * H),
          "PLE batch gate differs");
    check(std::equal(y.begin(), y.end(), batchx.begin() + t * D),
          "PLE batch result differs");
    check(std::equal(gatedone.begin(), gatedone.end(),
                     batchgated.begin() + t * D),
          "PLE batch gated differs");
    const auto ref = reference(
        std::vector<float>(key.begin() + t * D, key.begin() + (t + 1) * D),
        std::vector<float>(hidden.begin() + t * D,
                           hidden.begin() + (t + 1) * D),
        std::vector<float>(value.begin() + t * N, value.begin() + (t + 1) * N),
        initial, w, true);
    close(a, ref.key, "PLE key");
    close(g, ref.gate, "PLE gate");
    close(gatedone, ref.gated, "PLE gated");
    close(bnorm, ref.normalized, "PLE normalized");
    close(conv.get(), ref.conv, "PLE conv");
    close(y, ref.result, "PLE result");
    for (int c = 0; c < D; ++c) {
      for (int j = 0; j < HIST - 1; ++j)
        initial[c * HIST + j] = initial[c * HIST + j + 1];
      initial[c * HIST + HIST - 1] = ref.normalized[c];
    }
    ple_history_advance(hist2.data(), nq.data(), q);
  }
  check(hist.get() == hist2.get(), "PLE batch history differs");
  close(hist.get(), initial, "PLE CPU history");
  std::cout << "PLE postops T=" << T
            << ": CPU stages and batch/step bit equality passed\n";
}
void block(int keymode, bool nativevalue, bool nativeops) {
  HostWeights w;
  std::vector<float> emb(N), hidden(D), history(HIST * D), keyref(D), valref(N);
  for (int i = 0; i < N; ++i)
    emb[i] = float(std::sin(i * .12) + .2);
  for (int i = 0; i < D; ++i)
    hidden[i] = float(std::sin(i * .037));
  for (int i = 0; i < HIST * D; ++i)
    history[i] = float(std::cos(i * .063) * .1);
  Buffer<float> e(N), x(D), hist(HIST * D), k(D), v(N), g(H), gated(D),
      normed(D), conv(D), y(D);
  Buffer<uint8_t> scratch(ple_block_scratch_bytes());
  Buffer<uint8_t> nativeweights(size_t(D) * (N / 64) * 18),
      nativeact(native_q8_1_bytes(N));
  Buffer<uint16_t> kw(size_t(D) * N), vw(size_t(N) * N);
  Buffer<uint8_t> codes(size_t(D) * N / 4);
  Buffer<float> scales(size_t(D) * (N / 64));
  std::vector<uint16_t> keyweights(size_t(D) * N), valueweights(size_t(N) * N);
  for (int row = 0; row < D; ++row) {
    const int j = (row * 7) % N;
    keyweights[size_t(row) * N + j] = bf(.125f);
    keyref[row] = emb[j] * .125f;
  }
  for (int row = 0; row < N; ++row) {
    const int j = (row * 3) % N;
    valueweights[size_t(row) * N + j] = bf(.25f);
    valref[row] = (nativevalue ? emb[j] : frombf(bf(emb[j]))) * .25f;
  }
  e.put(emb);
  x.put(hidden);
  hist.put(history);
  kw.put(keyweights);
  vw.put(valueweights);
  w.w.value_bf16 = vw.data();
  if (keymode == 1)
    w.w.key_bf16 = kw.data();
  else {
    std::vector<uint8_t> c(size_t(D) * N / 4, 0x55);
    std::vector<float> sc(size_t(D) * (N / 64), .125f);
    std::vector<float> deq(N);
    for (int b = 0; b < N / 32; ++b) {
      float a = 0;
      for (int j = 0; j < 32; ++j)
        a = std::max(a, std::abs(emb[b * 32 + j]));
      const float d = a / 127.f;
      for (int j = 0; j < 32; ++j)
        deq[b * 32 + j] = std::round(emb[b * 32 + j] / d) * fromhalf(half(d));
    }
    for (int row = 0; row < D; ++row) {
      const int j = (row * 7) % N;
      const int shift = (j % 4) * 2;
      c[size_t(row) * (N / 4) + j / 4] =
          uint8_t((0x55 & ~(3 << shift)) | (2 << shift));
      keyref[row] = .125f * deq[j];
    }
    codes.put(c);
    scales.put(sc);
    w.w.key_codes = codes.data();
    w.w.key_scales = scales.data();
    if (keymode == 2) {
      std::vector<uint8_t> packed(size_t(D) * (N / 64) * 18, 0x55);
      for (int row = 0; row < D; ++row)
        for (int group = 0; group < N / 64; ++group) {
          const size_t base = (size_t(row) * (N / 64) + group) * 18;
          const auto scale = half(.125f);
          packed[base] = uint8_t(scale);
          packed[base + 1] = uint8_t(scale >> 8);
          std::copy_n(c.begin() + size_t(row) * (N / 4) + group * 16, 16,
                      packed.begin() + base + 2);
        }
      nativeweights.put(packed);
      w.w.key_codes = nullptr;
      w.w.key_scales = nullptr;
      w.w.key_native_data = nativeweights.data();
      w.w.key_native_type = 42;
      w.w.key_native_q8_1 = nativeact.data();
    }
  }
  PleOut o{k.data(),      v.data(),    g.data(), gated.data(),
           normed.data(), conv.data(), y.data()};
  ple_set_native_bf16(nativevalue);
  ple_set_native_postops(nativeops);
  ple_block(e.data(), x.data(), hist.data(), w.w, o, scratch.data(),
            keymode == 2 ? &rt->compute() : nullptr);
  const auto ref = reference(keyref, hidden, valref, history, w, nativeops);
  close(k.get(), ref.key, "block key");
  close(v.get(), valref, "block value");
  close(g.get(), ref.gate, "block gate");
  close(gated.get(), ref.gated, "block gated");
  close(normed.get(), ref.normalized, "block norm");
  close(conv.get(), ref.conv, "block conv");
  close(y.get(), ref.result, "block result");
  bool rejected = false;
  o.key = reinterpret_cast<float *>(scratch.data());
  try {
    ple_block(e.data(), x.data(), hist.data(), w.w, o, scratch.data(),
              keymode == 2 ? &rt->compute() : nullptr);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "PLE scratch export overlap accepted");
  std::cout << "PLE block " << keymode << nativevalue << nativeops
            << ": projections and CPU stages passed\n";
}
} // namespace
int main() {
  try {
    rt = sycl_backend::runtime_for();
    for (int t : {1, 3, 9, 12})
      postops(t);
    for (int key : {0, 1, 2})
      for (bool value : {false, true})
        for (bool post : {false, true})
          block(key, value, post);
    rt->wait();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
