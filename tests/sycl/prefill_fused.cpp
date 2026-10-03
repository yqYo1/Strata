#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/sycl/gguf_decode.hpp"
#include "strata/sycl/runtime.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <ggml.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace strata;
namespace fused = strata::prefill::fused;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
void check(bool v, const char *s) {
  if (!v)
    throw std::runtime_error(s);
}
template <class T> struct Buffer {
  sycl_backend::Allocation a;
  size_t n;
  Buffer(size_t n)
      : a(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device), n(n) {}
  T *data() { return a.as<T>(); }
  void put(const std::vector<T> &v) {
    check(v.size() == n, "upload size");
    runtime->wait(runtime->compute().memcpy(data(), v.data(), a.size()));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    runtime->wait(runtime->compute().memcpy(v.data(), data(), a.size()));
    return v;
  }
};
int perm(int k) { return 8 * (k & 3) + 4 * (k >> 4) + ((k & 15) >> 2); }
uint16_t half(float x) {
  _Float16 v = _Float16(x);
  uint16_t b;
  std::memcpy(&b, &v, 2);
  return b;
}
std::vector<uint8_t> quant(const std::vector<float> &x, bool native) {
  std::vector<uint8_t> q(x.size() / 64 * 80, 0);
  for (size_t i = 0; i < x.size(); i += 32) {
    float m = 0;
    for (int j = 0; j < 32; ++j)
      m = std::max(m, std::abs(x[i + j]));
    const int h = (i / 32) % 2;
    auto *b = q.data() + i / 64 * 80;
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
      int c = m ? int(std::nearbyint(x[i + j] * (127.f / m))) : 0;
      b[h * 32 + (native ? j : perm(j))] = uint8_t(int8_t(c));
      sum += c;
    }
    float d = m / 127.f;
    std::memcpy(b + 64 + (native ? h : h * 2) * 4, &d, 4);
    if (!native) {
      float c = -(12582912.f + float(sum));
      std::memcpy(b + 68 + h * 8, &c, 4);
    }
  }
  return q;
}
std::vector<float> expand(const uint8_t *q, int cols, bool native) {
  std::vector<float> x(cols);
  for (int i = 0; i < cols; ++i) {
    auto *b = q + i / 64 * 80;
    float d;
    int h = i % 64 / 32;
    std::memcpy(&d, b + 64 + (native ? h : 2 * h) * 4, 4);
    x[i] = d * int8_t(b[h * 32 + (native ? i % 32 : perm(i % 32))]);
  }
  return x;
}
std::vector<uint8_t> matrix(int type, int rows, int cols, int seed) {
  const auto f = sycl_backend::block_format(type);
  std::vector<uint8_t> w(size_t(rows) * cols / f.width * f.bytes);
  std::mt19937 rnd(seed);
  for (auto &v : w)
    v = uint8_t(rnd());
  for (size_t i = 0; i < w.size(); i += f.bytes) {
    const uint16_t d =
        half((type == 42 ? .008f : .00015f) * (int(rnd() % 11) - 5));
    std::memcpy(w.data() + i, &d, 2);
  }
  return w;
}
std::vector<float> decoded(const std::vector<uint8_t> &w, int type) {
  auto f = sycl_backend::block_format(type);
  std::vector<float> v(w.size() / f.bytes * f.width);
  ggml_get_type_traits(ggml_type(type))->to_float(w.data(), v.data(), v.size());
  return v;
}
void run(int gt, int dt) {
  constexpr int N = 74, T = N / 2, E = 4, D = 2560, F = 640;
  const bool native = gt != -1;
  std::vector<float> x(T * D);
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < D; ++k)
      x[t * D + k] =
          k / 32 == 0 ? 0.f : float(std::sin(k * .031 + (t % 2) * .73));
  auto xq = quant(x, native);
  Buffer<float> dx(x.size()), output(N * D + 16);
  Buffer<uint8_t> xa(xq.size() + 16), ha(fused::act_bytes(N, F) + 16),
      scratch(fused::group_bytes(N, E) + 16);
  Buffer<int32_t> ids(N), slot(N + 4), source(N + 4);
  std::vector<int32_t> routing(N, 1);
  for (int i = 65; i < 73; ++i)
    routing[i] = 3;
  routing[73] = 2;
  dx.put(x);
  ids.put(routing);
  slot.put(std::vector<int32_t>(slot.n, -777));
  source.put(std::vector<int32_t>(source.n, -777));
  xa.put(std::vector<uint8_t>(xa.n, 0xa5));
  ha.put(std::vector<uint8_t>(ha.n, 0xa5));
  scratch.put(std::vector<uint8_t>(scratch.n, 0xa5));
  output.put(std::vector<float>(output.n, -777));
  void *stream = &runtime->compute();
  if (native)
    fused::quantize_act_native(dx.data(), T, D, xa.data(), stream);
  else
    fused::quantize_act(dx.data(), T, D, xa.data(), stream);
  auto actualq = xa.get();
  for (size_t i = 0; i < xq.size(); ++i) {
    if (xq[i] != actualq[i])
      std::cerr << "activation byte " << i << " expected=" << int(xq[i])
                << " actual=" << int(actualq[i]) << '\n';
    check(xq[i] == actualq[i], "fused activation ABI independent reference");
  }
  fused::group(ids.data(), N, 2, E, scratch.data(), slot.data(), source.data(),
               stream);
  // A second layer reuses these tables; counts and cursors must be reset.
  fused::group(ids.data(), N, 2, E, scratch.data(), slot.data(), source.data(),
               stream);
  auto slots = slot.get(), sources = source.get();
  std::vector<bool> seen(N);
  for (int i = 0; i < N; ++i) {
    check(slots[i] >= 0 && slots[i] < N && !seen[slots[i]],
          "GPU routing permutation");
    seen[slots[i]] = true;
    check(sources[slots[i]] == i / 2, "GPU routing source");
  }
  for (int i = N; i < N + 4; ++i)
    check(slots[i] == -777 && sources[i] == -777, "routing guard");
  std::array<std::vector<uint8_t>, E> blobs;
  std::array<std::vector<float>, E> gw, uw, dw;
  std::array<std::unique_ptr<Buffer<uint8_t>>, E> device;
  fused::NativeGeom geo;
  for (int e = 1; e < E; ++e) {
    auto gate = matrix(native ? gt : 42, F, D, 700 + e),
         up = matrix(native ? gt : 42, F, D, 900 + e),
         down = matrix(native ? dt : 42, D, F, 1100 + e);
    gw[e] = decoded(gate, native ? gt : 42);
    uw[e] = decoded(up, native ? gt : 42);
    dw[e] = decoded(down, native ? dt : 42);
    if (native) {
      geo = {gt,
             dt,
             gate.size() / F,
             down.size() / D,
             gate.size(),
             gate.size() + up.size()};
      blobs[e] = gate;
      blobs[e].insert(blobs[e].end(), up.begin(), up.end());
      blobs[e].insert(blobs[e].end(), down.begin(), down.end());
    } else {
      constexpr size_t GC = 1280 * 640, DC = 2560 * 160, GS = GC + DC,
                       DS = GS + 1280 * 80;
      blobs[e].resize(DS + 2560 * 20);
      auto copyrow = [&](const std::vector<uint8_t> &w, int rows, int blocks,
                         size_t co, size_t so, bool up) {
        for (int r = 0; r < rows; ++r)
          for (int k = 0; k < blocks; ++k) {
            int row = rows == F ? 2 * r + int(up) : r;
            const uint8_t *b = w.data() + (size_t(r) * blocks + k) * 18;
            std::memcpy(blobs[e].data() + co + (size_t(row) * blocks + k) * 16,
                        b + 2, 16);
            std::memcpy(blobs[e].data() + so + (size_t(row) * blocks + k) * 2,
                        b, 2);
          }
      };
      copyrow(gate, F, 40, 0, GS, false);
      copyrow(up, F, 40, 0, GS, true);
      copyrow(down, D, 10, GC, DS, false);
    }
    device[e] = std::make_unique<Buffer<uint8_t>>(blobs[e].size());
    device[e]->put(blobs[e]);
  }
  for (int first : {0, 2}) {
    fused::Batch b;
    b.e0 = first;
    b.e1 = first + 2;
    for (int e = first; e < first + 2; ++e)
      b.blob[e - first] = e ? device[e]->data() : nullptr;
    if (native)
      fused::experts_native(b, geo, E, N, scratch.data(), xa.data(),
                            source.data(), ha.data(), output.data(), stream);
    else
      fused::experts(b, E, N, scratch.data(), xa.data(), source.data(),
                     ha.data(), output.data(), stream);
  }
  auto hidden = ha.get();
  auto got = output.get();
  double worst = 0;
  for (int e = 1; e < E; ++e)
    for (int pattern = 0; pattern < 2; ++pattern) {
      auto input = expand(xq.data() + pattern * 40 * 80, D, native);
      std::vector<float> h(F);
      for (int f = 0; f < F; ++f) {
        double gate = 0, up = 0;
        for (int k = 0; k < D; ++k) {
          gate += double(gw[e][f * D + k]) * input[k];
          up += double(uw[e][f * D + k]) * input[k];
        }
        h[f] = float(gate / (1 + std::exp(-gate)) * up);
      }
      auto hq = quant(h, native);
      auto hd = expand(hq.data(), F, native);
      std::vector<double> reference(D), magnitude(D);
      for (int d = 0; d < D; ++d)
        for (int k = 0; k < F; ++k) {
          const double term = double(dw[e][d * F + k]) * hd[k];
          reference[d] += term;
          magnitude[d] += std::abs(term);
        }
      for (int i = 0; i < N; ++i)
        if (routing[i] == e && (i / 2) % 2 == pattern) {
          int r = slots[i];
          auto actualh = expand(hidden.data() + size_t(r) * 10 * 80, F, native);
          for (int f = 0; f < F; ++f)
            check(std::abs(actualh[f] - hd[f]) < 2e-5f * (1 + std::abs(hd[f])),
                  "fused SwiGLU/int8 independent reference");
          for (int d = 0; d < D; ++d) {
            double err = std::abs(got[r * D + d] - reference[d]);
            worst = std::max(worst, err);
            // Bound FP32 rounding by the actual products, including outputs
            // near zero after cancellation of large IQ4_XS hidden values.
            check(std::isfinite(got[r * D + d]) &&
                      err < 2e-6 * (1 + magnitude[d]),
                  "fused down independent reference");
          }
        }
    }
  for (size_t i = xq.size(); i < actualq.size(); ++i)
    check(actualq[i] == 0xa5, "activation guard");
  for (size_t i = fused::act_bytes(N, F); i < hidden.size(); ++i)
    check(hidden[i] == 0xa5, "hidden guard");
  for (size_t i = N * D; i < got.size(); ++i)
    check(got[i] == -777, "output guard");
  auto table = scratch.get();
  for (size_t i = fused::group_bytes(N, E); i < table.size(); ++i)
    check(table[i] == 0xa5, "group table guard");
  std::cout << "fused gu=" << gt << " down=" << dt << " max_abs=" << worst
            << '\n';
}
} // namespace
int main() {
  try {
    runtime = sycl_backend::runtime_for();
    check(fused::built() && fused::available(), "fused unavailable");
    run(-1, 42);
    for (int gu : {16, 17, 18, 21, 22, 23})
      for (int down : {20, 42})
        run(gu, down);
    runtime->wait();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
