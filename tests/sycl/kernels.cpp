#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/sycl/runtime.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace strata;
using namespace strata::kernels;

namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;

template <typename T> struct Buffer {
  sycl_backend::Allocation memory;
  size_t count;
  explicit Buffer(size_t n)
      : memory(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device),
        count(n) {}
  T *data() { return memory.as<T>(); }
  void upload(const std::vector<T> &values) {
    if (values.size() != count)
      throw std::logic_error("test upload length");
    runtime->wait(
        runtime->compute().memcpy(data(), values.data(), memory.size()));
  }
  std::vector<T> read() {
    std::vector<T> values(count);
    runtime->wait(
        runtime->compute().memcpy(values.data(), data(), memory.size()));
    return values;
  }
};

void close(float actual, double expected, double tolerance, const char *label) {
  if (!std::isfinite(actual) || std::abs(double(actual) - expected) >
                                    tolerance * (1 + std::abs(expected)))
    throw std::runtime_error(std::string(label) +
                             ": actual=" + std::to_string(actual) +
                             ", expected=" + std::to_string(expected));
}

void conversions() {
  std::vector<float> values;
  for (unsigned bits = 0; bits < 65536; ++bits)
    values.push_back(float(std::bit_cast<_Float16>(uint16_t(bits))));
  for (uint32_t bits :
       {0x3f801000u, 0x3f803000u, 0x33000000u, 0x33000001u, 0x387fffffu,
        0x477fefffu, 0x477ff000u, 0x3f808000u, 0x3f818000u, 0x7fffffffu}) {
    values.push_back(std::bit_cast<float>(bits));
    values.push_back(std::bit_cast<float>(bits | 0x80000000u));
  }
  std::mt19937 random(17);
  for (int i = 0; i < 8192; ++i)
    values.push_back(std::bit_cast<float>(uint32_t(random())));
  Buffer<float> input(values.size());
  Buffer<uint16_t> half(values.size()), brain(values.size());
  input.upload(values);
  f32_to_f16_bulk(input.data(), half.data(), values.size(),
                  &runtime->compute());
  f32_to_bf16_bulk(input.data(), brain.data(), values.size(),
                   &runtime->compute());
  auto h = half.read(), b = brain.read();
  for (size_t i = 0; i < values.size(); ++i) {
    if (std::isnan(values[i])) {
      if ((h[i] & 0x7c00) != 0x7c00 || !(h[i] & 0x3ff) ||
          (b[i] & 0x7f80) != 0x7f80 || !(b[i] & 0x7f))
        throw std::runtime_error("conversion lost a NaN");
    } else {
      const uint16_t expected_half =
          std::bit_cast<uint16_t>(_Float16(values[i]));
      const uint16_t expected_brain =
          std::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(values[i]));
      if (h[i] != expected_half || b[i] != expected_brain)
        throw std::runtime_error("conversion bit mismatch at " +
                                 std::to_string(i));
    }
  }
  std::cout << "FP16/BF16 conversion: " << values.size() << " inputs PASS\n";
}

void elementwise() {
  for (int cols : {1, 17, 128, 257, 4096}) {
    const int rows = 3;
    std::vector<float> input(rows * cols + 1), weights(cols);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = float(std::sin(double(i) * .31) * 2);
    input.back() = 123456.f;
    for (int c = 0; c < cols; ++c)
      weights[c] = float(.3 + c * .001);
    Buffer<float> x(input.size()), w(weights.size());
    x.upload(input);
    w.upload(weights);
    rms_norm_weighted(x.data(), w.data(), rows, cols, 1.e-6f,
                      &runtime->compute());
    auto output = x.read();
    for (int row = 0; row < rows; ++row) {
      double sum = 0;
      for (int c = 0; c < cols; ++c)
        sum += double(input[row * cols + c]) * input[row * cols + c];
      for (int c = 0; c < cols; ++c)
        close(output[row * cols + c],
              input[row * cols + c] * double(weights[c]) /
                  std::sqrt(sum / cols + 1.e-6),
              2.e-6, "weighted RMS");
    }
    if (output.back() != input.back())
      throw std::runtime_error("RMS wrote past final row");
  }
  const std::vector<float> input{-90, -20,    -1, 0,  .1f,  1,  20, 21,
                                 90,  -.001f, 7,  -8, 1.25, 10, 100};
  const std::vector<float> dt{.1f, -.5f, 3, -4, 0},
      a{-.5f, -1, -2, -.25f, -.75f};
  Buffer<float> x(input.size()), y(input.size()), d(dt.size()), s(a.size());
  x.upload(input);
  d.upload(dt);
  s.upload(a);
  gdn_gate(x.data(), d.data(), s.data(), y.data(), 3, 5, &runtime->compute());
  auto output = y.read();
  for (size_t i = 0; i < input.size(); ++i) {
    const float value = input[i] + dt[i % 5];
    close(output[i],
          (value > 20 ? double(value) : std::log1p(std::exp(double(value)))) *
              a[i % 5],
          2.e-6, "batched GDN gate");
  }
  silu_inplace(x.data(), input.size(), nullptr);
  output = x.read();
  for (size_t i = 0; i < input.size(); ++i)
    close(output[i], double(input[i]) / (1 + std::exp(-double(input[i]))),
          2.e-7, "SiLU");
  scale_inplace(x.data(), input.size(), 2, &runtime->compute());
  add_inplace(x.data(), y.data(), input.size(), &runtime->compute());
  auto combined = x.read();
  for (size_t i = 0; i < input.size(); ++i)
    close(combined[i],
          2.f * output[i] +
              (input[i] + dt[i % 5] > 20.f
                   ? input[i] + dt[i % 5]
                   : std::log1p(std::exp(double(input[i] + dt[i % 5])))) *
                  a[i % 5],
          3.e-6, "add/scale");
  std::cout << "RMS tails/canary, batched GDN gate, SiLU and elementwise "
               "arithmetic: PASS\n";
}

void routing() {
  for (int experts : {17, 129, 512, 1000}) {
    constexpr int tokens = 4;
    const int k = std::min(experts, 10);
    std::vector<float> logits(tokens * experts);
    for (int t = 0; t < tokens; ++t)
      for (int e = 0; e < experts; ++e)
        logits[t * experts + e] = t == 0   ? 0.f
                                  : t == 1 ? float((e * 17) % 23) * .25f
                                  : t == 2 ? float(e % 19 - 9) * 100.f
                                           : float(std::sin(e * 1.21) * 7);
    Buffer<float> input(logits.size()), weight(tokens * k);
    Buffer<int> index(tokens * k);
    input.upload(logits);
    for (bool native : {false, true}) {
      if (native && experts != 512)
        continue;
      if (native)
        native_router_top10_multi(input.data(), index.data(), weight.data(),
                                  tokens, &runtime->compute());
      else
        router_top10(input.data(), tokens, experts, k, index.data(),
                     weight.data(), &runtime->compute());
      auto ids = index.read();
      auto weights = weight.read();
      for (int t = 0; t < tokens; ++t) {
        const auto begin = logits.begin() + t * experts;
        const double maximum = *std::max_element(begin, begin + experts);
        std::vector<double> exponentials(experts);
        for (int e = 0; e < experts; ++e)
          exponentials[e] = std::exp(double(begin[e]) - maximum);
        const float inv = float(
            1. / std::accumulate(exponentials.begin(), exponentials.end(), 0.));
        std::vector<float> probabilities(experts);
        for (int e = 0; e < experts; ++e)
          probabilities[e] = float(exponentials[e] * inv);
        std::vector<int> order(experts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
          return probabilities[a] > probabilities[b];
        });
        double sum = 0;
        for (int j = 0; j < k; ++j)
          sum += probabilities[order[j]];
        for (int j = 0; j < k; ++j) {
          if (ids[t * k + j] != order[j])
            throw std::runtime_error(
                "router top-k mismatch: native=" + std::to_string(native) +
                " experts=" + std::to_string(experts) +
                " token=" + std::to_string(t) + " rank=" + std::to_string(j) +
                " got=" + std::to_string(ids[t * k + j]) +
                " expected=" + std::to_string(order[j]));
          close(weights[t * k + j],
                probabilities[order[j]] / std::max(sum, 0x1p-14), 2.e-6,
                "router weight");
        }
      }
    }
  }
  std::cout << "generic/native batched routing, stable ties, large logits and "
               "weights: PASS\n";
}

void movement() {
  for (int bits : {2, 4, 8}) {
    constexpr int n = 96, group = 32;
    const int bias = -(1 << (bits - 1));
    std::vector<uint8_t> codes(n * bits / 8);
    std::vector<float> scales{.33333334f, -.14285715f, 1.00000012f},
        offsets{-.25f, 1.f, -17.f};
    std::vector<float> expected(n);
    for (int i = 0; i < n; ++i) {
      const int code = (i * 3 + 7) & ((1 << bits) - 1);
      codes[i / (8 / bits)] |= uint8_t(code << ((i % (8 / bits)) * bits));
      const float product = float(code + bias) * scales[i / group];
      expected[i] = product + offsets[i / group];
    }
    Buffer<uint8_t> packed(codes.size());
    Buffer<float> scale(3), offset(3), result(n);
    packed.upload(codes);
    scale.upload(scales);
    offset.upload(offsets);
    embedding_gather(packed.data(), scale.data(), offset.data(), n, bits, bias,
                     group, result.data(), &runtime->compute());
    if (result.read() != expected)
      throw std::runtime_error("embedding separate-rounding mismatch");
  }
  std::vector<float> values(35);
  std::iota(values.begin(), values.end(), 1.f);
  Buffer<float> input(35), output(35);
  Buffer<int32_t> hits(2), count(1);
  input.upload(values);
  hits.upload({1, 3});
  count.upload({2});
  copy_rows_from_mapped(output.data(), input.data(), 5, 7, hits.data(),
                        count.data(), &runtime->compute());
  auto rows = output.read();
  for (size_t i = 0; i < rows.size(); ++i)
    if (rows[i] != (i / 7 == 1 || i / 7 == 3 ? 0.f : values[i]))
      throw std::runtime_error("expert row exclusion mismatch");
  Buffer<float> weights(2), weights_out(2);
  Buffer<int32_t> ids_out(2);
  Buffer<uint32_t> sequence(1);
  weights.upload({.25f, .75f});
  sequence.upload({0});
  for (int i = 0; i < 5; ++i)
    doorbell_publish(input.data(), hits.data(), weights.data(), 35, 2,
                     output.data(), ids_out.data(), weights_out.data(),
                     sequence.data(), &runtime->compute());
  runtime->wait();
  if (output.read() != values || ids_out.read() != std::vector<int32_t>{1, 3} ||
      weights_out.read() != std::vector<float>{.25f, .75f} ||
      sequence.read()[0] != 5)
    throw std::runtime_error("event-completed payload publication mismatch");
  bool refused = false;
  try {
    doorbell_wait(nullptr, nullptr, &runtime->compute());
  } catch (const std::logic_error &) {
    refused = true;
  }
  if (!refused)
    throw std::runtime_error("unsupported GPU spin-wait accepted");
  std::cout << "embedding packing/rounding, row exclusions and event-completed "
               "publication: PASS\n";
}
} // namespace

int main() {
  try {
    // The CPU oracle must retain subnormals: treating exp(-100) as zero
    // changes the stable top-k order even when its weight is very small.
    std::fesetenv(FE_DFL_ENV);
    runtime = sycl_backend::runtime_for();
    conversions();
    elementwise();
    routing();
    movement();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
