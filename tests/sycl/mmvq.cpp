#include "strata/artifact/dequant.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/esimd_half.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef STRATA_SYCL_GGML_ORACLE
#include <ggml.h>
#endif
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
template <typename T> struct Buffer {
  sycl_backend::Allocation storage;
  size_t count;
  explicit Buffer(size_t n)
      : storage(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device),
        count(n) {}
  T *data() { return storage.as<T>(); }
  void upload(const std::vector<T> &v) {
    if (v.size() != count)
      throw std::logic_error("upload size");
    runtime->wait(runtime->compute().memcpy(data(), v.data(), storage.size()));
  }
  std::vector<T> read() {
    std::vector<T> v(count);
    runtime->wait(runtime->compute().memcpy(v.data(), data(), storage.size()));
    return v;
  }
};
uint16_t half(float value) {
  const _Float16 h = _Float16(value);
  uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}
float unhalf(const uint8_t *p) {
  _Float16 h;
  std::memcpy(&h, p, 2);
  return float(h);
}
void check(bool condition, const char *label) {
  if (!condition)
    throw std::runtime_error(label);
}
void half_conversion() {
  namespace e = sycl::ext::intel::esimd;
  constexpr int count = 65536;
  Buffer<float> output(count);
  auto *out = output.data();
  runtime->compute().parallel_for(
      sycl::nd_range<1>(count / 16, 16),
      [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const uint32_t first = uint32_t(it.get_global_linear_id()) * 16;
        const e::simd<uint32_t, 16> bits(first, 1);
        sycl_backend::esimd_half_decode<16>(bits).copy_to(out + first);
      });
  const auto values = output.read();
  for (uint32_t code = 0; code < count; ++code) {
    const uint16_t bits = uint16_t(code);
    _Float16 half_value;
    std::memcpy(&half_value, &bits, 2);
    float expected = float(half_value);
    if ((bits & 0x7c00u) == 0x7c00u && (bits & 1023u)) {
      // Preserve the old decoder's payload, including signaling NaNs.
      const uint32_t payload = ((code & 0x8000u) << 16) |
                               0x7f800000u | ((code & 1023u) << 13);
      std::memcpy(&expected, &payload, 4);
    }
    check(std::memcmp(&values[code], &expected, 4) == 0,
          "ESIMD FP16 conversion bits");
  }
  std::cout << "ESIMD FP16 conversion: all 65536 encodings match\n";
}
struct Format {
  int type, width, bytes, scale;
};
constexpr Format formats[] = {{2, 32, 18, 0},      {6, 32, 22, 0},
                              {8, 32, 34, 0},      {11, 256, 110, 108},
                              {12, 256, 144, 0},   {13, 256, 176, 0},
                              {14, 256, 210, 208}, {20, 32, 18, 0},
                              {23, 256, 136, 0},   {42, 64, 18, 0},
                              {21, 256, 110, 0},   {18, 256, 98, 0},
                              {22, 256, 82, 0},    {16, 256, 66, 0},
                              {17, 256, 74, 0},    {29, 256, 56, 48},
                              {7, 32, 24, 0}};
void decode(int type, const uint8_t *p, float *out) {
  switch (type) {
  case 2:
    dequantize_q4_0(p, out);
    break;
  case 6:
    dequantize_q5_0(p, out);
    break;
  case 7:
#ifdef STRATA_SYCL_GGML_ORACLE
    ggml_get_type_traits(GGML_TYPE_Q5_1)->to_float(p, out, 32);
#else
    dequantize_q5_1(p, out);
#endif
    break;
  case 16:
  case 17:
  case 29:
#ifdef STRATA_SYCL_GGML_ORACLE
    ggml_get_type_traits(ggml_type(type))->to_float(p, out, 256);
#else
    if (type == 16) dequantize_iq2_xxs(p, out);
    if (type == 17) dequantize_iq2_xs(p, out);
    if (type == 29) dequantize_iq1_m(p, out);
#endif
    break;
  case 8:
    dequantize_q8_0(p, out);
    break;
  case 11:
    dequantize_q3_K(p, out);
    break;
  case 12:
    dequantize_q4_K(p, out);
    break;
  case 13:
    dequantize_q5_K(p, out);
    break;
  case 14:
    dequantize_q6_K(p, out);
    break;
  case 20:
    dequantize_iq4_nl(p, out);
    break;
  case 21:
#ifdef STRATA_SYCL_GGML_ORACLE
    ggml_get_type_traits(GGML_TYPE_IQ3_S)->to_float(p, out, 256);
#else
    dequantize_iq3_s(p, out);
#endif
    break;
  case 18:
#ifdef STRATA_SYCL_GGML_ORACLE
    ggml_get_type_traits(GGML_TYPE_IQ3_XXS)->to_float(p, out, 256);
#else
    dequantize_iq3_xxs(p, out);
#endif
    break;
  case 22:
#ifdef STRATA_SYCL_GGML_ORACLE
    ggml_get_type_traits(GGML_TYPE_IQ2_S)->to_float(p, out, 256);
#else
    dequantize_iq2_s(p, out);
#endif
    break;
  case 23:
    dequantize_iq4_xs(p, out);
    break;
  case 30:
    dequantize_bf16(p, out, 1);
    break;
  case 42:
    dequantize_q2_0(p, out);
    break;
  default:
    throw std::logic_error("test decode type");
  }
}
std::vector<uint8_t> weights(Format f, int n, int rows) {
  std::vector<uint8_t> bytes(size_t(n / f.width) * rows * f.bytes);
  std::mt19937 random(876 + f.type);
  for (auto &v : bytes)
    v = uint8_t(random());
  for (size_t i = 0; i < bytes.size(); i += f.bytes) {
    const float scale = (int(random() % 1025) - 512) / 131072.f;
    const uint16_t bits = half(scale);
    std::memcpy(bytes.data() + i + f.scale, &bits, 2);
    if (f.type == 29) {
      for (int j = 0; j < 4; ++j) {
        auto *sc = bytes.data() + i + 48 + 2 * j;
        const auto low = uint16_t(sc[0]) | (uint16_t(sc[1] & 15) << 8);
        const auto full = uint16_t(low | (((bits >> (4 * j)) & 15) << 12));
        std::memcpy(sc, &full, 2);
      }
    }
    if (f.type == 12 || f.type == 13 || f.type == 7) {
      const uint16_t min_bits = half(float(random() % 512) / 131072.f);
      std::memcpy(bytes.data() + i + 2, &min_bits, 2);
    }
  }
  // A subnormal block scale exercises two-byte loads and bit conversion.
  const uint16_t smallest = 1;
  if (f.type == 29) {
    for (int j = 0; j < 4; ++j) {
      auto *sc = bytes.data() + 48 + 2 * j;
      sc[1] = uint8_t((sc[1] & 15) | (j == 0 ? 16 : 0));
    }
  } else
    std::memcpy(bytes.data() + f.scale, &smallest, 2);
  return bytes;
}
std::vector<uint8_t> quantize(const std::vector<float> &x) {
  std::vector<uint8_t> out(x.size() / 32 * 36);
  for (size_t b = 0; b < x.size() / 32; ++b) {
    std::array<float, 32> sum;
    float maximum = 0;
    for (int i = 0; i < 32; ++i) {
      sum[i] = x[b * 32 + i];
      maximum = std::max(maximum, std::abs(sum[i]));
    }
    for (int off = 16; off; off /= 2) {
      const auto old = sum;
      for (int i = 0; i < 32; ++i)
        sum[i] = old[i] + old[i ^ off];
    }
    const float d = maximum / 127.f;
    const auto db = half(d), sb = half(sum[0]);
    std::memcpy(out.data() + b * 36, &db, 2);
    std::memcpy(out.data() + b * 36 + 2, &sb, 2);
    for (int i = 0; i < 32; ++i) {
      const float quotient = maximum == 0 ? 0 : x[b * 32 + i] / d;
      out[b * 36 + 4 + i] = uint8_t(int8_t(std::round(quotient)));
    }
  }
  return out;
}
void grouped(Format f, int n, int ff, bool scale_edges = false,
             const Format *down_format = nullptr) {
  constexpr int G = 3, E = 15, T = 3;
  const Format down = down_format ? *down_format : f;
  const auto L = native_expert_layout(f.type, down.type, n, ff);
  auto gw = weights(f, n, ff), dw = weights(down, ff, n);
  if (scale_edges) {
    check(f.type == 42, "scale-edge fixture requires Q2_0");
    constexpr uint16_t edges[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x0400,
                                  0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    // One edge scale per row; small inputs keep hidden Q8_1 scales finite
    // even for the maximum finite FP16 weight scale.
    for (int row = 0; row < ff; ++row) {
      const auto bits = edges[row % 10];
      gw[size_t(row) * L.gu_row] = uint8_t(bits);
      gw[size_t(row) * L.gu_row + 1] = uint8_t(bits >> 8);
    }
    for (int row = 0; row < n; ++row) {
      const auto bits = edges[(row + 3) % 10];
      dw[size_t(row) * L.d_row] = uint8_t(bits);
      dw[size_t(row) * L.d_row + 1] = uint8_t(bits >> 8);
    }
  }
  std::vector<uint8_t> blob(L.bytes), other(L.bytes);
  std::copy(gw.begin(), gw.end(), blob.begin());
  for (int row = 0; row < ff; ++row)
    std::copy_n(gw.begin() + size_t(ff - 1 - row) * L.gu_row, L.gu_row,
                blob.begin() + L.up_off + size_t(row) * L.gu_row);
  std::copy(dw.begin(), dw.end(), blob.begin() + L.down_off);
  std::copy(blob.begin() + L.up_off, blob.begin() + L.down_off, other.begin());
  std::copy(blob.begin(), blob.begin() + L.up_off, other.begin() + L.up_off);
  for (int row = 0; row < n; ++row)
    std::copy_n(dw.begin() + size_t(n - 1 - row) * L.d_row, L.d_row,
                other.begin() + L.down_off + size_t(row) * L.d_row);
  Buffer<uint8_t> b0(L.bytes);
  sycl_backend::Allocation b1(runtime, L.bytes, sycl_backend::MemoryKind::Host);
  b0.upload(blob);
  std::memcpy(b1.data(), other.data(), other.size());
  Buffer<unsigned long long> pointers(G);
  pointers.upload({reinterpret_cast<unsigned long long>(b0.data()),
                   reinterpret_cast<unsigned long long>(b1.data()), 0});
  Buffer<int32_t> starts(G + 1), count(1), dst(E), token(E);
  starts.upload({0, 2, 13, 13});
  count.upload({2});
  std::vector<int32_t> destinations(E, -1), tokens(E, -1);
  for (int e = 0; e < 13; ++e) {
    destinations[e] = (e * 7) % E;
    tokens[e] = e % T;
  }
  dst.upload(destinations);
  token.upload(tokens);
  std::vector<float> inputs(T * n);
  for (size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = float(std::sin(i * .139) + .03 * std::cos(i * .071)) *
                (scale_edges ? 1e-4f : 1.f);
  Buffer<float> input(inputs.size());
  input.upload(inputs);
  Buffer<uint8_t> xq(native_q8_1_bytes(n, T)),
      scratch(native_expert_scratch_bytes(E, ff) + 16),
      hq(native_q8_1_bytes(ff));
  Buffer<float> output(size_t(E) * n + 1), gate(ff), up(ff), h(ff), one(n);
  auto *q = &runtime->compute();
  native_quantize_q8_1(input.data(), xq.data(), n, T, q);
  std::vector<float> expected(size_t(E) * n + 1, -777);
  for (int entry = 0; entry < 13; ++entry) {
    auto p = entry < 2 ? b0.data() : b1.as<uint8_t>();
    auto x = xq.data() + size_t(tokens[entry]) * native_q8_1_bytes(n);
    native_mmvq(f.type, p, x, gate.data(), n, ff, 1, q);
    native_mmvq(f.type, p + L.up_off, x, up.data(), n, ff, 1, q);
    auto gp = gate.data(), upp = up.data(), hp = h.data();
    q->parallel_for(sycl::range<1>(ff), [=](sycl::id<1> i) {
      const float v = gp[i];
      hp[i] = (v / (1.f + sycl::exp(-v))) * upp[i];
    });
    native_quantize_q8_1(h.data(), hq.data(), ff, 1, q);
    native_mmvq(down.type, p + L.down_off, hq.data(), one.data(), ff, n, 1, q);
    const auto ref = one.read();
    std::copy(ref.begin(), ref.end(),
              expected.begin() + size_t(destinations[entry]) * n);
  }
  for (bool old : {false, true}) {
    iq_set_old_kernels(old);
    native_grouped_set_v1(old);
    output.upload(std::vector<float>(size_t(E) * n + 1, -777));
    scratch.upload(std::vector<uint8_t>(scratch.count, 0x9a));
    native_expert_grouped(L, pointers.data(), starts.data(), count.data(),
                          dst.data(), token.data(), G, E, xq.data(),
                          scratch.data(), output.data(), q, old ? 1 : G);
    const auto result = output.read();
    check(scale_edges ? std::memcmp(result.data(), expected.data(),
                                    expected.size() * sizeof(float)) == 0
                      : result == expected,
          "native grouped expert differs from separate projections");
    const auto guard = scratch.read();
    check(std::all_of(guard.end() - 16, guard.end(),
                      [](uint8_t v) { return v == 0x9a; }),
          "group scratch canary");
  }
  auto invalid = tokens;
  invalid[1] = -1;
  invalid[5] = E;
  token.upload(invalid);
  auto zeroed = expected;
  for (int entry : {1, 5})
    std::fill_n(zeroed.begin() + size_t(destinations[entry]) * n, n, 0.f);
  for (bool old : {false, true}) {
    iq_set_old_kernels(old);
    output.upload(std::vector<float>(size_t(E) * n + 1, -777));
    native_expert_grouped(L, pointers.data(), starts.data(), count.data(),
                          dst.data(), token.data(), G, E, xq.data(),
                          scratch.data(), output.data(), q);
    const auto result = output.read();
    check(scale_edges ? std::memcmp(result.data(), zeroed.data(),
                                    zeroed.size() * sizeof(float)) == 0
                      : result == zeroed,
          "invalid token row must yield zero expert");
  }
  iq_set_old_kernels(false);
  native_grouped_set_v1(false);
  count.upload({0});
  output.upload(std::vector<float>(size_t(E) * n + 1, -777));
  native_expert_grouped(L, pointers.data(), starts.data(), count.data(),
                        dst.data(), token.data(), G, E, xq.data(),
                        scratch.data(), output.data(), nullptr);
  const auto empty = output.read();
  check(std::all_of(empty.begin(), empty.end(),
                    [](float v) { return v == -777; }),
        "zero group count wrote output");
  iq_set_old_kernels(false);
  std::cout << "native grouped type " << f.type << " " << n << "x" << ff
            << (scale_edges ? " FP16 scale edges" : "")
            << " passed both entry tiles\n";
}
void transfers(Format f, int row_count = 4,
               const std::vector<uint8_t> *fixture = nullptr) {
  constexpr int N = 256;
  const int R = row_count;
  const auto w = fixture ? *fixture : weights(f, N, R);
  std::vector<float> reference(N * R);
  for (int i = 0; i < N * R / f.width; ++i)
    decode(f.type, w.data() + size_t(i) * f.bytes,
           reference.data() + i * f.width);
  Buffer<uint8_t> packed(w.size());
  packed.upload(w);
  Buffer<float> out(N * R + 1), gathered(N * 3 + 1);
  Buffer<uint16_t> halfout(N * R + 2), interleaved(N * R * 2 + 2);
  Buffer<int32_t> ids(3);
  ids.upload({3, 0, 1});
  out.upload(std::vector<float>(out.count, -77));
  halfout.upload(std::vector<uint16_t>(halfout.count, 0x1234));
  interleaved.upload(std::vector<uint16_t>(interleaved.count, 0x1234));
  gathered.upload(std::vector<float>(gathered.count, -77));
  iq_dequant_f32(f.type, packed.data(), N * R, out.data(), nullptr);
  iq_dequant_f16(f.type, packed.data(), N * R, halfout.data(), nullptr);
  iq_dequant_gu_f16(f.type, packed.data(), packed.data(), R, N,
                    interleaved.data(), nullptr);
  iq_embed_rows(f.type, packed.data(), iq_row_bytes(f.type, N), ids.data(), 3,
                N, gathered.data(), nullptr);
  const auto got = out.read(), emb = gathered.read();
  const auto f16 = halfout.read(), gu = interleaved.read();
  if (f.type == 16 || f.type == 17 || f.type == 18 || f.type == 21 ||
      f.type == 22 || f.type == 29)
    check(std::memcmp(got.data(), reference.data(), N * R * 4) == 0,
          "IQ expert dequantization bits against ggml");
  for (int i = 0; i < N * R; ++i) {
    check(got[i] == reference[i], "GGUF f32 transfer");
    check(f16[i] == half(reference[i]), "GGUF f16 transfer");
    check(gu[(i / N) * 2 * N + i % N] == half(reference[i]) &&
              gu[(i / N) * 2 * N + N + i % N] == half(reference[i]),
          "interleaved GU transfer");
  }
  const int rows[3] = {3, 0, 1};
  for (int t = 0; t < 3; ++t)
    for (int i = 0; i < N; ++i)
      check(emb[t * N + i] == reference[rows[t] * N + i],
            "GGUF embedding rows");
  check(got.back() == -77 && emb.back() == -77 && f16.back() == 0x1234 &&
            gu.back() == 0x1234,
        "GGUF transfer canaries");
}
size_t tested_cases = 0;
double max_scaled_error = 0;
void run(Format f, int n, int rows, int cols, bool exact,
         const std::vector<uint8_t> &w) {
  check(native_mmvq_weight_bytes(f.type, n, rows) == w.size(), "weight bytes");
  const auto scratch_bytes = native_q8_1_bytes(n, cols);
  check(scratch_bytes == size_t(cols) * n / 32 * 36, "Q8_1 bytes");
  std::vector<float> x(size_t(n) * cols);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = float(.37 + std::sin(i * .73) * (1. + (i % 3) * .31));
  std::fill_n(x.begin(), 32, 0.f);
  if (x.size() >= 64) {
    for (int i = 32; i < 64; ++i)
      x[i] = float(i - 47) + .5f;
    x[32] = 127.f;
  }
  const auto q8 = quantize(x);
  Buffer<float> input(x.size()), output(size_t(rows) * cols + 4),
      single(rows + 4);
  Buffer<uint8_t> weight(w.size()), scratch(scratch_bytes + 16);
  input.upload(x);
  weight.upload(w);
  scratch.upload(std::vector<uint8_t>(scratch_bytes + 16, 0xa5));
  output.upload(std::vector<float>(size_t(rows) * cols + 4, 12345.f));
  single.upload(std::vector<float>(rows + 4, 12345.f));
  auto *stream = &runtime->compute();
  native_mmvq_set_multi_exact(exact);
  native_quantize_q8_1(input.data(), scratch.data(), n, cols, stream);
  native_mmvq(f.type, weight.data(), scratch.data(), output.data(), n, rows,
              cols, stream);
  const auto got_quant = scratch.read();
  check(std::equal(q8.begin(), q8.end(), got_quant.begin()),
        "native Q8_1 bytes");
  for (size_t i = scratch_bytes; i < got_quant.size(); ++i)
    check(got_quant[i] == 0xa5, "Q8_1 canary");
  const auto got = output.read();
  if (exact && ((f.type == 12 || f.type == 23) ||
                ((f.type == 11 || f.type == 42) && cols > 1 && cols <= 4))) {
    // Preserve the original four-warp arithmetic in the SPMD oracle.
    // Small projection errors can change later Q8_1 activations in the model.
    const bool old = iq_old_kernels();
    if (f.type == 12 || f.type == 23)
      iq_set_old_kernels(true);
    native_mmvq_set_multi_exact(f.type == 12 || f.type == 23);
    native_mmvq(f.type, weight.data(), scratch.data(), output.data(), n, rows,
                cols, stream);
    const auto original = output.read();
    if (std::memcmp(original.data(), got.data(), size_t(rows) * cols * 4) != 0) {
      for (int i = 0; i < rows * cols; ++i)
        if (std::memcmp(&original[i], &got[i], 4) != 0) {
          std::cerr << "native oracle type " << f.type << " width " << n
                    << " columns " << cols << " index " << i << " original "
                    << std::hexfloat << original[i] << " candidate " << got[i]
                    << std::defaultfloat << '\n';
          break;
        }
    }
    check(std::memcmp(original.data(), got.data(), size_t(rows) * cols * 4) == 0,
          "native ESIMD/SPMD bits");
    native_mmvq_set_multi_exact(exact);
    iq_set_old_kernels(old);
  }
  for (int col = 0; col < cols; ++col) {
    if (exact) {
      native_mmvq(f.type, weight.data(),
                  scratch.data() + size_t(col) * n / 32 * 36, single.data(), n,
                  rows, 1, stream);
      const auto one = single.read();
      if (std::memcmp(one.data(), got.data() + col * rows, rows * 4) != 0) {
        int different = 0;
        float largest = 0;
        for (int r = 0; r < rows; ++r) {
          const float expected = got[col * rows + r];
          if (one[r] != expected && different++ < 5)
            std::cerr << "single/multi type " << f.type << " row " << r
                      << " single " << one[r] << " multi " << expected << '\n';
          largest = std::max(largest, std::abs(one[r] - expected));
        }
        std::cerr << "different rows " << different << " maximum difference "
                  << largest << '\n';
      }
      check(std::memcmp(one.data(), got.data() + col * rows, rows * 4) == 0,
            "native multi/single bits");
      for (int i = rows; i < rows + 4; ++i)
        check(one[i] == 12345.f, "single canary");
    }
    for (int row = 0; row < rows; ++row) {
      double expected = 0, magnitude = 0;
      for (int b = 0; b < n / f.width; ++b) {
        const auto *wb = w.data() + (size_t(row) * (n / f.width) + b) * f.bytes;
        float decoded[256];
        decode(f.type, wb, decoded);
        for (int i = 0; i < f.width; ++i) {
          const size_t global = size_t(col) * n + b * f.width + i;
          const auto *qb = q8.data() + global / 32 * 36;
          const double term =
              double(decoded[i]) * unhalf(qb) * int8_t(qb[4 + global % 32]);
          expected += term;
          magnitude += std::abs(term);
        }
        if (f.type == 2 || f.type == 6 || f.type == 7) {
          const auto *qb = q8.data() + (size_t(col) * n / 32 + b) * 36;
          int qsum = 0;
          for (int i = 0; i < 32; ++i)
            qsum += int8_t(qb[4 + i]);
          // Native affine formats use the FP16 original-input sum, not
          // sum(q)*d.
          const double correction =
              (f.type == 7 ? -double(unhalf(wb + 2))
                           : (f.type == 2 ? 8. : 16.) * unhalf(wb)) *
              (double(unhalf(qb)) * qsum - unhalf(qb + 2));
          expected += correction;
          magnitude += std::abs(correction);
        }
      }
      const double error =
          std::abs(got[col * rows + row] - expected) / (1 + magnitude);
      max_scaled_error = std::max(max_scaled_error, error);
      if (!std::isfinite(got[col * rows + row]) || error > 3e-6)
        throw std::runtime_error(
            "MMVQ mismatch type=" + std::to_string(f.type) +
            " n=" + std::to_string(n) + " row=" + std::to_string(row) +
            " col=" + std::to_string(col) +
            " actual=" + std::to_string(got[col * rows + row]) +
            " expected=" + std::to_string(expected));
    }
  }
  for (size_t i = size_t(rows) * cols; i < got.size(); ++i)
    check(got[i] == 12345.f, "MMVQ output canary");
  ++tested_cases;
}
void real_model(const char *path) {
  const auto model = GgufModel::open(path);
  size_t tensors = 0;
  std::map<int, size_t> counts;
  for (size_t shard = 0; shard < model.size(); ++shard)
    for (const auto &t : model.shard(shard).tensors()) {
      if (!native_mmvq_supported(t.type))
        continue;
      check(model.in_bounds(t, shard), "model tensor bounds");
      check(!t.shape.empty() && t.shape[0] <= INT32_MAX, "model row width");
      uint64_t total_rows = 1;
      for (size_t i = 1; i < t.shape.size(); ++i)
        total_rows *= t.shape[i];
      const auto it =
          std::find_if(std::begin(formats), std::end(formats),
                       [&](Format f) { return f.type == int(t.type); });
      const auto f = *it;
      const int n = int(t.shape[0]);
      const size_t row_bytes = native_mmvq_weight_bytes(f.type, n, 1);
      const uint8_t *source = model.shard(shard).tensor_data(t);
      std::vector<uint8_t> sample(3 * row_bytes);
      const uint64_t indices[] = {0, total_rows / 2, total_rows - 1};
      for (int i = 0; i < 3; ++i)
        std::memcpy(sample.data() + i * row_bytes,
                    source + indices[i] * row_bytes, row_bytes);
      try {
        run(f, n, 3, 3, true, sample);
      } catch (const std::exception &e) {
        throw std::runtime_error(t.name + ": " + e.what());
      }
      ++tensors;
      ++counts[f.type];
    }
  check(tensors > 0, "no supported model tensors tested");
  std::cout << "Real GGUF: first/middle/last rows of " << tensors
            << " tensors, 3 activation columns\n";
  for (const auto &[type, count] : counts)
    std::cout << "  " << ggml_type_name(type) << ": " << count << '\n';
}
template <typename F> void rejects(F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid MMVQ input accepted");
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::fesetenv(FE_DFL_ENV);
#ifdef STRATA_SYCL_GGML_ORACLE
    // Initialize ggml's FP16 lookup table before calling its independent
    // IQ3_S dequantizer. No tensor or model allocation is needed.
    auto *context = ggml_init({16384, nullptr, true});
    check(context != nullptr, "ggml oracle initialization");
    ggml_free(context);
#endif
    runtime = sycl_backend::runtime_for();
    if (argc == 1) {
      half_conversion();
      for (auto f : formats) {
        transfers(f);
        grouped(f, 256, 256);
      }
      grouped(formats[9], 2560, 640);
      grouped(formats[9], 2560, 640, true);
      grouped(formats[10], 2560, 640, false, &formats[7]);
      for (const auto f : {formats[11], formats[12], formats[13],
                            formats[14], formats[15]}) {
        grouped(f, 2560, 640, false, &formats[7]);
        grouped(f, 2560, 640, false, &formats[9]);
      }
      {
        const auto f = formats[10];
        constexpr uint16_t scales[] = {0, 0x8000, 1, 0x8001, 0x3ff,
                                       0x400, 0x8400, 0x7bff};
        // Every nine-bit codebook index, low/high subscales, explicit sign
        // bytes and finite FP16 scale edge go through ggml's independent
        // decoder and both GPU transfer precisions.
        auto w = weights(f, 256, 512);
        for (int row = 0; row < 512; ++row) {
          auto *block = w.data() + row * 110;
          std::memcpy(block, &scales[row % 8], 2);
          for (int group = 0; group < 8; ++group) {
            for (int word = 0; word < 8; ++word)
              block[2 + group * 8 + word] = uint8_t(row);
            block[66 + group] = row >= 256 ? 255 : 0;
            for (int word = 0; word < 4; ++word)
              block[74 + group * 4 + word] = uint8_t(row + group * 4 + word);
          }
          std::fill_n(block + 106, 4,
                      uint8_t((row % 16) | ((15 - row % 16) << 4)));
        }
        transfers(f, 512, &w);
        for (int n : {2560, 6144}) {
          auto projection = weights(f, n, 17);
          for (int block = 0; block < 17 * (n / 256); ++block)
            std::memcpy(projection.data() + block * 110,
                        &scales[block % 8], 2);
          for (int cols : {1, 3, 8})
            run(f, n, 17, cols, true, projection);
        }
      }
      for (const auto f : {formats[11], formats[12]}) {
        constexpr uint16_t scales[] = {0, 0x8000, 1, 0x8001, 0x3ff,
                                       0x400, 0x8400, 0x7bff};
        const int rows = f.type == 18 ? 256 : 1024;
        auto w = weights(f, 256, rows);
        for (int row = 0; row < rows; ++row) {
          auto *block = w.data() + row * f.bytes;
          std::memcpy(block, &scales[row % 8], 2);
          for (int group = 0; group < 8; ++group) {
            if (f.type == 18) {
              std::fill_n(block + 2 + group * 8, 8, uint8_t(row));
              uint32_t aux = uint32_t(row % 16) << 28;
              for (int cell = 0; cell < 4; ++cell)
                aux |= uint32_t((row + group * 4 + cell) & 127) << (7 * cell);
              std::memcpy(block + 66 + group * 4, &aux, 4);
            } else {
              for (int cell = 0; cell < 4; ++cell) {
                block[2 + group * 4 + cell] = uint8_t(row);
                block[34 + group * 4 + cell] = uint8_t(row + group * 4 + cell);
              }
              block[66 + group] = uint8_t((row >> 8) * 0x55);
              block[74 + group] = uint8_t((row % 16) | ((15 - row % 16) << 4));
            }
          }
        }
        transfers(f, rows, &w);
        for (int n : {2560, 6144}) {
          auto projection = weights(f, n, 17);
          for (int block = 0; block < 17 * (n / 256); ++block)
            std::memcpy(projection.data() + block * f.bytes,
                        &scales[block % 8], 2);
          for (int cols : {1, 3, 8})
            run(f, n, 17, cols, true, projection);
        }
      }
      transfers({30, 1, 2, 0});
      for (const auto f : {formats[13], formats[14], formats[15]}) {
        const int rows = f.type == 16 ? 256 : f.type == 17 ? 512 : 2048;
        auto w = weights(f, 256, rows);
        const uint16_t edges[] = {0, 0x8000, 1, 0x8001, 0x3ff,
                                  0x400, 0x8400, 0x7bff};
        for (int row = 0; row < rows; ++row) {
          auto *b = w.data() + row * f.bytes;
          if (f.type == 29) {
            for (int sc = 0; sc < 4; ++sc)
              b[49 + 2 * sc] = uint8_t((b[49 + 2 * sc] & 15) |
                                (((edges[row % 8] >> (4 * sc)) & 15) << 4));
          } else
            std::memcpy(b, &edges[row % 8], 2);
          for (int group = 0; group < 8; ++group) {
            if (f.type == 16) {
              std::fill_n(b + 2 + 8 * group, 4, uint8_t(row));
              uint32_t aux = uint32_t(row % 16) << 28;
              for (int cell = 0; cell < 4; ++cell)
                aux |= uint32_t((row + group * 4 + cell) & 127) << (7 * cell);
              std::memcpy(b + 6 + 8 * group, &aux, 4);
            } else if (f.type == 17) {
              for (int cell = 0; cell < 4; ++cell) {
                const uint16_t qs = uint16_t(row | (((row + group + cell) & 127) << 9));
                std::memcpy(b + 2 + 8 * group + 2 * cell, &qs, 2);
              }
            } else {
              for (int cell = 0; cell < 4; ++cell)
                b[4 * group + cell] = uint8_t(row);
              const int high = (row >> 8) | (row % 2 ? 8 : 0);
              b[32 + 2 * group] = b[33 + 2 * group] = uint8_t(high | (high << 4));
            }
          }
        }
        transfers(f, rows, &w);
      }
    }
    if (argc == 3 && std::string(argv[1]) == "--gguf")
      real_model(argv[2]);
    else if (argc == 1) {
      for (const auto f : formats) {
        check(native_mmvq_supported(f.type), "supported type");
        for (int n : {f.width, 2560, 8192}) {
          const auto w = weights(f, n, 7);
          for (int cols : {1, 3, 8})
            run(f, n, 7, cols, true, w);
          run(f, n, 7, 3, false, w);
          run(f, n, 7, 8, false, w);
        }
      }
      {
        constexpr int rows = 17, n = 6144;
        auto w = weights(formats[3], n, rows);
        // Q3_K alternate 110-byte blocks are only two-byte aligned. Include
        // signed zero, subnormal and negative FP16 scales in changing columns.
        const uint16_t scales[] = {0, 0x8000, 1, 0x3ff, 0x400, 0x8400,
                                   0x8001, 0x7bff};
        for (int block = 0; block < rows * (n / 256); ++block)
          std::memcpy(w.data() + size_t(block) * 110 + 108,
                      &scales[block % 8], 2);
        for (int cols : {1, 3, 8})
          run(formats[3], n, rows, cols, true, w);
      }
      {
        constexpr int rows = 17;
        const uint16_t scales[] = {0, 0x8000, 1, 0x3ff, 0x400, 0x8400,
                                   0x8001, 0x7bff};
        // Affine Q4_K scales include two FP16 values. Cover a partial
        // ten-block iteration and a full second iteration with row tails.
        for (int n : {2560, 6144}) {
          auto w = weights(formats[4], n, rows);
          for (int block = 0; block < rows * (n / 256); ++block) {
            std::memcpy(w.data() + size_t(block) * 144,
                        &scales[block % 8], 2);
            std::memcpy(w.data() + size_t(block) * 144 + 2,
                        &scales[(block + 3) % 8], 2);
          }
          for (int cols : {1, 2, 3, 4, 8})
            run(formats[4], n, rows, cols, true, w);
        }
      }
      {
        constexpr int rows = 17;
        const uint16_t scales[] = {0, 0x8000, 1, 0x3ff, 0x400, 0x8400,
                                   0x8001, 0x7bff};
        // IQ4_XS uses 16 virtual blocks per iteration. Exercise the partial
        // ten-block row and a second iteration, with FP16 scale edge cases.
        for (int n : {2560, 6144}) {
          auto w = weights(formats[8], n, rows);
          for (int block = 0; block < rows * (n / 256); ++block)
            std::memcpy(w.data() + size_t(block) * 136,
                        &scales[block % 8], 2);
          for (int cols : {1, 3, 8})
            run(formats[8], n, rows, cols, true, w);
        }
      }
      {
        constexpr int rows = 17;
        const uint16_t scales[] = {0, 0x8000, 1, 0x3ff, 0x400, 0x8400,
                                   0x8001, 0x7bff};
        for (int n : {640, 2560, 6144}) {
          auto w = weights(formats[9], n, rows);
          for (int block = 0; block < rows * (n / 64); ++block)
            std::memcpy(w.data() + size_t(block) * 18,
                        &scales[block % 8], 2);
          for (int cols : {1, 3, 8})
            run(formats[9], n, rows, cols, true, w);
        }
      }
      {
        // The vocabulary path has a row tail and must retain the ordinary
        // multi-column reduction, including finite FP16 scale edge cases.
        constexpr int rows = 4097, n = 2560;
        auto w = weights(formats[5], n, rows);
        const uint16_t scales[] = {0, 1, 0x3ff, 0x400, 0x8400, 0x8001, 0x7bff};
        for (int i = 0; i < 7; ++i) {
          std::memcpy(w.data() + size_t(i) * 10 * 176, &scales[i], 2);
          std::memcpy(w.data() + size_t(i) * 10 * 176 + 2, &scales[6 - i], 2);
        }
        for (int cols : {1, 2, 3, 4, 8})
          run(formats[5], n, rows, cols, true, w);
      }
      Buffer<float> x(256), y(256);
      Buffer<uint8_t> w(1024), scratch(1024);
      auto *s = &runtime->compute();
      rejects([&] { native_q8_1_bytes(31, 1); });
      rejects([&] { native_q8_1_bytes(32, 9); });
      rejects([&] { native_mmvq_weight_bytes(42, 32, 1); });
      rejects([&] { native_mmvq_weight_bytes(99, 32, 1); });
      check(!native_mmvq_supported(99), "unsupported type");
      rejects([&] {
        native_quantize_q8_1(x.data(), scratch.data(), 32, 1, nullptr);
      });
      rejects([&] {
        native_mmvq(42, w.data(), scratch.data(),
                    reinterpret_cast<float *>(scratch.data()), 64, 1, 1, s);
      });
      rejects([&] {
        native_q2_0_f32(w.data(), x.data(), scratch.data(), x.data(), 64, 1, 1,
                        s);
      });
    } else
      throw std::invalid_argument(
          "usage: sycl_mmvq_test [--gguf first-shard.gguf]");
    native_mmvq_set_multi_exact(true);
    runtime->wait();
    std::cout << "SYCL native MMVQ PASS: " << tested_cases
              << " cases; max |error|/(1+sum|terms|)=" << max_scaled_error
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
