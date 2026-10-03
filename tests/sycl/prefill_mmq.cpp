#include "strata/prefill/moe_mmq.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/sycl/gguf_decode.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef STRATA_SYCL_GGML_ORACLE
#include <ggml.h>
#endif
namespace mmq = strata::prefill::mmq;
using namespace strata;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
void check(bool yes, const char *what) { if (!yes) throw std::runtime_error(what); }
template <class T> struct Buffer {
  sycl_backend::Allocation memory;
  size_t n;
  explicit Buffer(size_t size) : memory(runtime, size * sizeof(T), sycl_backend::MemoryKind::Device), n(size) {}
  T *data() { return memory.as<T>(); }
  void put(const std::vector<T> &v) {
    check(v.size() == n, "upload size");
    runtime->wait(runtime->compute().memcpy(data(), v.data(), memory.size()));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    runtime->wait(runtime->compute().memcpy(v.data(), data(), memory.size()));
    return v;
  }
};
uint16_t half(float v) { _Float16 h = _Float16(v); uint16_t b; std::memcpy(&b, &h, 2); return b; }
float unhalf(const uint8_t *v) { _Float16 h; std::memcpy(&h, v, 2); return float(h); }
void decode(int type, const uint8_t *b, float *out) {
#ifdef STRATA_SYCL_GGML_ORACLE
  ggml_get_type_traits(ggml_type(type))->to_float(b, out, sycl_backend::block_format(type).width);
#else
  sycl_backend::decode_block(type, b, out);
#endif
}
std::vector<uint8_t> weights(int type, int cols, int rows, int experts) {
  auto f = sycl_backend::block_format(type);
  std::vector<uint8_t> w(mmq::matrix_bytes(type, rows, cols) * experts);
  std::mt19937 rnd(777 + type);
  for (auto &b : w) b = uint8_t(rnd());
  for (size_t i = 0; i < w.size(); i += f.bytes) {
    const uint16_t d = half((int(rnd() % 31) - 15) * .002f);
    if (type == 29) {
      for (int sc = 0; sc < 4; ++sc)
        w[i + 49 + 2 * sc] = uint8_t((w[i + 49 + 2 * sc] & 15) | (((d >> (4 * sc)) & 15) << 4));
    } else {
      const int off = type == 11 ? 108 : type == 14 ? 208 : 0;
      std::memcpy(w.data() + i + off, &d, 2);
      if (type == 12 || type == 13 || type == 7) {
        const auto m = half(.003f); std::memcpy(w.data() + i + 2, &m, 2);
      }
    }
  }
  return w;
}
void product(int type, int cols) {
  constexpr int R = 7, E = 3, T = 37, LD = 11, XS = 49;
  const int xld = cols + 5, padded = (cols + 511) / 512 * 512;
  auto w = weights(type, cols, R, E);
  std::vector<float> x(size_t(XS) * xld, 777.f);
  for (int r = 0; r < XS; ++r)
    for (int k = 0; k < cols; ++k)
      x[size_t(r) * xld + k] = float(std::sin(r * .13 + k * .19) + .05 * std::cos(k * .37));
  std::vector<int32_t> source(T), map(T);
  for (int r = 0; r < T; ++r) { source[r] = (r * 13) % XS; map[r] = (r * 7) % T; }
  source[7] = -1; map[11] = -1;
  Buffer<uint8_t> dw(w.size()), xq(mmq::q8_bytes(T, cols) + 16);
  Buffer<float> dx(x.size()), dy(T * LD + 4);
  Buffer<int32_t> ds(T), dm(T), bounds(4);
  dw.put(w); dx.put(x); ds.put(source); dm.put(map); bounds.put({2, 5, 5, 36});
  xq.put(std::vector<uint8_t>(xq.n, 0xa5)); dy.put(std::vector<float>(dy.n, -123.f));
  auto *stream = &runtime->compute();
  mmq::quantize(dx.data(), ds.data(), xq.data(), type, cols, xld, T, stream);
  auto q = xq.get();
  for (int r = 0; r < T; ++r)
    for (int b = 0; b < padded / 32; ++b) {
      std::array<float, 32> sums;
      float maximum = 0;
      for (int j = 0; j < 32; ++j) {
        sums[j] = source[r] >= 0 && b * 32 + j < cols ? x[size_t(source[r]) * xld + b * 32 + j] : 0;
        maximum = std::max(maximum, std::abs(sums[j]));
      }
      for (int offset = 16; offset; offset >>= 1) {
        const auto prev = sums;
        for (int j = 0; j < 32; ++j) sums[j] = prev[j] + prev[j ^ offset];
      }
      const float scale = maximum / 127.f;
      const auto *qb = q.data() + (size_t(r) * padded / 32 + b) * 36;
      check(iq_read_u16(qb) == half(scale) && iq_read_u16(qb + 2) == half(sums[0]), "Q8_1 scale/sum or row padding");
      for (int j = 0; j < 32; ++j) {
        const float v = source[r] >= 0 && b * 32 + j < cols ? x[size_t(source[r]) * xld + b * 32 + j] : 0;
        const int code = maximum == 0 ? 0 : int(std::round(v / scale));
        check(int8_t(qb[4 + j]) == code, "Q8_1 gathered code");
      }
    }
  for (size_t i = mmq::q8_bytes(T, cols); i < q.size(); ++i) check(q[i] == 0xa5, "quantization canary");
  mmq::Context context;
  mmq::Product p;
  p.w = dw.data(); p.type = type; p.w_rows = R; p.w_cols = cols;
  p.expert_bytes = w.size() / E; p.n = E; p.xq = xq.data();
  p.bounds = bounds.data(); p.ids = dm.data(); p.total_rows = T; p.max_rows = 31;
  p.dst = dy.data(); p.ld_dst = LD;
  context.run(p, stream);
  const auto got = dy.get();
  std::vector<float> expected(dy.n, -123.f);
  const auto f = sycl_backend::block_format(type);
  double worst = 0;
  for (int r = 2; r < 36; ++r) {
    if (map[r] < 0) continue;
    const int e = r < 5 ? 0 : 2;
    for (int row = 0; row < R; ++row) {
      double sum = 0, magnitude = 0;
      for (int b = 0; b < cols / f.width; ++b) {
        const auto *wb = w.data() + e * p.expert_bytes + (size_t(row) * cols / f.width + b) * f.bytes;
        float decoded[256]; decode(type, wb, decoded);
        for (int j = 0; j < f.width; ++j) {
          const int dim = b * f.width + j;
          const auto *qb = q.data() + (size_t(r) * padded / 32 + dim / 32) * 36;
          const double term = double(decoded[j]) * unhalf(qb) * int8_t(qb[4 + dim % 32]);
          sum += term; magnitude += std::abs(term);
        }
        if (type == 2 || type == 6 || type == 7) {
          const auto *qb = q.data() + (size_t(r) * padded / 32 + b) * 36;
          int qsum = 0; for (int j = 0; j < 32; ++j) qsum += int8_t(qb[4 + j]);
          const double factor = type == 7 ? -double(unhalf(wb + 2)) : (type == 2 ? 8. : 16.) * unhalf(wb);
          const double correction = factor * (double(unhalf(qb)) * qsum - unhalf(qb + 2));
          sum += correction; magnitude += std::abs(correction);
        }
      }
      const float actual = got[size_t(map[r]) * LD + row];
      const double error = std::abs(actual - sum) / (1 + magnitude);
      if (!std::isfinite(actual) || error > 3e-6)
        throw std::runtime_error("routed product type " + std::to_string(type) + " row " + std::to_string(r) + " error " + std::to_string(error));
      worst = std::max(worst, error);
      expected[size_t(map[r]) * LD + row] = actual;
    }
  }
  check(got == expected, "routed row map, untouched rows, stride or output canary");
  bounds.put({37, 2, 2, 2}); dy.put(std::vector<float>(dy.n, -123.f));
  context.run(p, nullptr);
  check(dy.get() == std::vector<float>(dy.n, -123.f), "invalid or empty bounds wrote output");
  std::cout << "MMQ type=" << type << " K=" << cols << " routed rows=37 max_scaled=" << worst << '\n';
}
void gather() {
  constexpr int E = 4, B = 128;
  Buffer<uint8_t> source(E * B), gu(E * 48 + 16), down(E * 32 + 16);
  std::vector<uint8_t> input(source.n);
  for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i * 17 + i / B);
  source.put(input); gu.put(std::vector<uint8_t>(gu.n, 0xa5)); down.put(std::vector<uint8_t>(down.n, 0xa5));
  mmq::GatherGroup group; group.first = 1; group.n = 4;
  for (int e = 1; e < E; ++e) group.blob[e] = source.data() + e * B;
  check(mmq::gather_native_group(group, 48, 16, 96, 16, gu.data(), 48, down.data(), 32, nullptr), "group gather refused");
  auto g = gu.get(), d = down.get();
  auto eg = std::vector<uint8_t>(gu.n, 0xa5), ed = std::vector<uint8_t>(down.n, 0xa5);
  for (int e = 1; e < E; ++e) {
    std::copy_n(input.begin() + e * B, 16, eg.begin() + e * 48);
    std::copy_n(input.begin() + e * B + 48, 16, eg.begin() + e * 48 + 16);
    std::copy_n(input.begin() + e * B + 96, 16, ed.begin() + e * 32);
  }
  check(g == eg && d == ed, "group gathered bytes or padding");
  check(!mmq::gather_native_group(group, 49, 16, 96, 16, gu.data(), 48, down.data(), 32, nullptr), "unaligned group accepted");
  check(gu.get() == eg && down.get() == ed, "declined group launched a write");
  mmq::gather_native(source.data(), source.data() + 48, 16, source.data() + 96, 16, gu.data(), down.data(), nullptr);
  std::copy_n(input.begin(), 16, eg.begin());
  std::copy_n(input.begin() + 48, 16, eg.begin() + 16);
  std::copy_n(input.begin() + 96, 16, ed.begin());
  check(gu.get() == eg && down.get() == ed, "single aligned gather or untouched padding");
  mmq::gather_native(source.data(), source.data() + 48, 14, source.data() + 96, 14, gu.data(), down.data(), nullptr);
  auto oneg = gu.get(), oned = down.get();
  check(std::equal(input.begin(), input.begin() + 14, oneg.begin()) &&
        std::equal(input.begin() + 48, input.begin() + 62, oneg.begin() + 14) &&
        std::equal(input.begin() + 96, input.begin() + 110, oned.begin()), "single unaligned-size gather");
  constexpr size_t GC = 1280 * 640, DC = 2560 * 160, GS = GC + DC, GB = 1280 * 40, DB = 2560 * 10, DS = GS + GB * 2;
  Buffer<uint8_t> blob(DS + DB * 2), cg(GB * 18 + 16), cd(DB * 18 + 16);
  std::vector<uint8_t> raw(blob.n);
  for (size_t i = 0; i < raw.size(); ++i) raw[i] = uint8_t(i * 29 + i / 17);
  blob.put(raw); cg.put(std::vector<uint8_t>(cg.n, 0xa5)); cd.put(std::vector<uint8_t>(cd.n, 0xa5));
  mmq::gather_strata_q2(blob.data(), cg.data(), cd.data(), nullptr);
  const auto gv = cg.get(), dv = cd.get();
  for (size_t b = 0; b < GB + DB; ++b) {
    const bool gate = b < GB; const size_t k = gate ? b : b - GB;
    const auto *dst = (gate ? gv.data() : dv.data()) + k * 18;
    check(dst[0] == raw[(gate ? GS : DS) + k * 2] && dst[1] == raw[(gate ? GS : DS) + k * 2 + 1], "Q2 plane scale");
    check(std::equal(dst + 2, dst + 18, raw.begin() + (gate ? 0 : GC) + k * 16), "Q2 plane codes");
  }
  check(std::all_of(gv.end() - 16, gv.end(), [](uint8_t v) { return v == 0xa5; }) &&
        std::all_of(dv.end() - 16, dv.end(), [](uint8_t v) { return v == 0xa5; }), "Q2 gather canary");
}
void postops() {
  constexpr int T = 13, FF = 17;
  std::vector<float> gu(T * FF * 2);
  for (size_t i = 0; i < gu.size(); ++i) gu[i] = float(std::sin(i * .17) * 3);
  Buffer<float> g(gu.size()), h(T * FF + 1); Buffer<int32_t> ids(T + 1);
  g.put(gu); ids.put(std::vector<int32_t>(T + 1, -7)); mmq::iota(ids.data(), T, nullptr);
  auto iv = ids.get(); for (int i = 0; i < T; ++i) check(iv[i] == i, "identity map"); check(iv.back() == -7, "identity canary");
  for (bool interleaved : {false, true}) {
    h.put(std::vector<float>(h.n, -777)); mmq::swiglu(g.data(), h.data(), T, FF, interleaved, nullptr);
    auto hv = h.get();
    for (int r = 0; r < T; ++r) for (int f = 0; f < FF; ++f) {
      const int at = r * FF * 2 + (interleaved ? 2 * f : f);
      const float expected = (gu[at] / (1.f + std::exp(-gu[at]))) * gu[at + (interleaved ? 1 : FF)];
      check(std::abs(hv[r * FF + f] - expected) < 2e-6, "prompt SwiGLU");
    }
    check(hv.back() == -777, "SwiGLU canary");
  }
}
} // namespace
int main() {
  try {
    runtime = sycl_backend::runtime_for(); check(mmq::built(), "quantized prefill unavailable");
    for (int type : {2, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 42})
      product(type, type == 20 || type == 42 || type == 7 ? 640 : 2560);
    gather(); postops(); runtime->wait(); std::cout << "SYCL quantized prefill PASS\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
