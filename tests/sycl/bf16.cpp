#include "strata/kernels/bf16_gemv.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
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

float value(uint16_t bits) {
  return std::bit_cast<float>(uint32_t(bits) << 16);
}
uint16_t bf16(float v) { return uint16_t(std::bit_cast<uint32_t>(v) >> 16); }
void check(bool ok, const char *what) {
  if (!ok)
    throw std::runtime_error(what);
}
float xor_sum(std::array<float, 32> v) {
  for (int off = 16; off; off /= 2) {
    const auto old = v;
    for (int i = 0; i < 32; ++i)
      v[i] = old[i] + old[i ^ off];
  }
  return v[0];
}
float native_oracle(const float *x, const uint16_t *w, int n) {
  int block = 32, best = (n + 63) / 64;
  for (int b = 64; b <= 256; b += 32) {
    const int loops = (n + 2 * b - 1) / (2 * b);
    if (loops < best) {
      best = loops;
      block = b;
    }
  }
  std::array<float, 32> warps{};
  for (int warp = 0; warp < block / 32; ++warp) {
    std::array<float, 32> lanes{};
    for (int lane = 0; lane < 32; ++lane)
      for (int pair = warp * 32 + lane; pair < n / 2; pair += block) {
        lanes[lane] = std::fma(value(w[pair * 2]), x[pair * 2], lanes[lane]);
        lanes[lane] =
            std::fma(value(w[pair * 2 + 1]), x[pair * 2 + 1], lanes[lane]);
      }
    warps[warp] = xor_sum(lanes);
  }
  return block == 32 ? warps[0] : xor_sum(warps);
}
void native_case(int n, int rows, int tokens) {
  const int ldx = n + 6, ldy = rows + 3;
  std::vector<float> x(tokens * ldx, 777.f);
  std::vector<uint16_t> w(n * rows);
  std::mt19937 random(35 + n);
  for (int t = 0; t < tokens; ++t)
    for (int i = 0; i < n; ++i)
      x[t * ldx + i] = float(int(random() % 20001) - 10000) / 7919.f;
  for (auto &v : w)
    v = bf16(float(int(random() % 20001) - 10000) / 10001.f);
  Buffer<float> input(x.size()), output(tokens * ldy + 4), single(rows + 4);
  Buffer<uint16_t> weights(w.size());
  input.upload(x);
  weights.upload(w);
  output.upload(std::vector<float>(tokens * ldy + 4, 12345.f));
  single.upload(std::vector<float>(rows + 4, 12345.f));
  bf16_gemv_fp32_mmvf_multi(input.data(), ldx, weights.data(), output.data(),
                            ldy, n, rows, tokens, &runtime->compute());
  const auto result = output.read();
  for (int t = 0; t < tokens; ++t) {
    bf16_gemv_fp32_mmvf(input.data() + t * ldx, weights.data(), single.data(),
                        n, rows, nullptr);
    const auto one = single.read();
    check(std::memcmp(one.data(), result.data() + t * ldy, rows * 4) == 0,
          "multi/single bits");
    for (int r = 0; r < rows; ++r) {
      const auto ref = native_oracle(x.data() + t * ldx, w.data() + r * n, n);
      if (std::bit_cast<uint32_t>(ref) != std::bit_cast<uint32_t>(one[r]))
        throw std::runtime_error("native BF16 oracle bits width=" +
                                 std::to_string(n));
      double sum = 0, abs_sum = 0;
      for (int i = 0; i < n; ++i) {
        const double term = double(x[t * ldx + i]) * value(w[r * n + i]);
        sum += term;
        abs_sum += std::abs(term);
      }
      check(std::abs(one[r] - sum) < 2e-6 * (1 + abs_sum),
            "native double oracle");
    }
    for (int i = rows; i < ldy; ++i)
      check(result[t * ldy + i] == 12345.f, "output stride padding");
    for (int i = rows; i < rows + 4; ++i)
      check(one[i] == 12345.f, "single canary");
  }
  for (int i = tokens * ldy; i < tokens * ldy + 4; ++i)
    check(result[i] == 12345.f, "multi canary");
}
void legacy_case(int n, int rows, int threads) {
  std::vector<uint16_t> x(n), w(n * rows);
  for (int i = 0; i < n; ++i)
    x[i] = bf16(float(std::sin(i * .7)));
  for (int i = 0; i < n * rows; ++i)
    w[i] = bf16(float(std::cos(i * .11)));
  Buffer<uint16_t> input(n), weights(w.size());
  Buffer<float> output(rows + 4);
  input.upload(x);
  weights.upload(w);
  output.upload(std::vector<float>(rows + 4, 12345.f));
  if (threads)
    bf16_gemv_split(input.data(), weights.data(), output.data(), n, rows,
                    threads, &runtime->compute());
  else
    bf16_gemv(input.data(), weights.data(), output.data(), n, rows,
              &runtime->compute());
  const auto actual = output.read();
  for (int r = 0; r < rows; ++r) {
    double sum = 0, magnitude = 0;
    for (int i = 0; i < n; ++i) {
      const double product = double(value(x[i])) * value(w[r * n + i]);
      sum += product;
      magnitude += std::abs(product);
    }
    check(std::abs(actual[r] - sum) < 2e-6 * (1 + magnitude),
          "legacy BF16 sum");
  }
  for (int i = rows; i < rows + 4; ++i)
    check(actual[i] == 12345.f, "legacy canary");
}
template <typename F> void rejects(F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid BF16 geometry accepted");
}
} // namespace
int main() {
  try {
    std::fesetenv(FE_DFL_ENV);
    runtime = sycl_backend::runtime_for();
    for (int n : {2, 64, 66, 130, 194, 258, 322, 386, 450, 514, 2560, 10240})
      for (int tokens : {1, 2, 3, 4, 5, 8})
        native_case(n, 7, tokens);
    native_case(2560, 48, 8);
    for (int threads : {0, 1, 16, 32, 64, 128, 256}) {
      legacy_case(259, 7, threads);
      legacy_case(2560, 67, threads);
    }
    Buffer<float> x(512), y(512);
    Buffer<uint16_t> w(512);
    rejects([&] {
      bf16_gemv_fp32_mmvf_multi(x.data(), 30, w.data(), y.data(), 8, 32, 8, 2,
                                nullptr);
    });
    rejects([&] {
      bf16_gemv_fp32_mmvf_multi(x.data(), 32, w.data(), y.data(), 7, 32, 8, 2,
                                nullptr);
    });
    rejects([&] {
      bf16_gemv_fp32_mmvf(x.data(), w.data(), y.data(), 31, 8, nullptr);
    });
    rejects([&] {
      bf16_gemv_fp32_mmvf(x.data(), w.data(), x.data(), 32, 8, nullptr);
    });
    rejects([&] {
      bf16_gemv_split(w.data(), w.data(), y.data(), 32, 8, 33, nullptr);
    });
    rejects([&] {
      bf16_gemv_fp32_mmvf_multi(x.data(), INT64_MAX - 1, w.data(), y.data(), 8,
                                32, 8, 8, nullptr);
    });
    runtime->wait();
    std::cout << "SYCL BF16 GEMV: 73 native shapes bit-exact vs CPU ordered "
                 "FMA; 14 legacy shapes passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
