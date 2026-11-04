// Main-authored Q6_K column-specialization comparison; repository LICENSE.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace k = strata::kernels;
using Clock = std::chrono::steady_clock;
constexpr size_t Pad = 32;
std::shared_ptr<sb::Runtime> rt;
void need(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template <class T> T sentinel();
template <> float sentinel<float>() { return -731.25f; }
template <> uint8_t sentinel<uint8_t>() { return 0xa5; }
template <class T> struct Buffer {
  sb::Allocation owner;
  size_t n;
  explicit Buffer(size_t count)
      : owner(rt, (count + 2 * Pad) * sizeof(T), sb::MemoryKind::Device), n(count) {
    std::vector<T> initial(n + 2 * Pad, sentinel<T>());
    rt->wait(rt->compute().memcpy(owner.data(), initial.data(), initial.size() * sizeof(T)));
  }
  T *data() { return owner.template as<T>() + Pad; }
  void set(const std::vector<T> &values) {
    need(values.size() == n, "input extent");
    rt->wait(rt->compute().memcpy(data(), values.data(), n * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> result(n + 2 * Pad);
    rt->wait(rt->compute().memcpy(result.data(), owner.data(), result.size() * sizeof(T)));
    for (size_t i = 0; i < Pad; ++i)
      need(result[i] == sentinel<T>() && result[Pad + n + i] == sentinel<T>(),
           "buffer guard changed");
    return result;
  }
  void unchanged(const std::vector<T> &expected) {
    auto values = get();
    need(expected.size() == n &&
             std::memcmp(values.data() + Pad, expected.data(), n * sizeof(T)) == 0,
         "read-only input changed");
  }
};
using Kernel = void (*)(const void *, const void *, float *, int, int, int, void *);
constexpr std::array<Kernel, 2> Arms{k::native_q6_k_mmvq, k::native_q6_k_columns_mmvq};
struct Shape {
  int ni, no, iterations;
};
// Actual Q6_K tensor geometries in the selected two-shard IQ3_S model.
constexpr std::array<Shape, 7> Shapes{{{2560, 10240, 256}, {6144, 2560, 256},
    {2560, 6144, 256}, {2560, 640, 1024}, {2560, 512, 1024},
    {2560, 12288, 256}, {2560, 248320, 8}}};

// Decode scalar Q6_K/Q8_1 values rather than reproducing the GPU's packed-dot
// lane decomposition. Share only the already tested half-bit converter. The
// GPU quantizer is not independently checked by this oracle.
void scalar_reference(const Shape &shape, int columns,
                      const std::vector<uint8_t> &weights,
                      const std::vector<uint8_t> &packed,
                      const std::vector<float> &output) {
  const auto half_at = [](const uint8_t *p) {
    return double(k::f32_from_f16(uint16_t(p[0]) | (uint16_t(p[1]) << 8)));
  };
  const auto signed_byte = [](uint8_t v) { return v < 128 ? int(v) : int(v) - 256; };
  std::vector<int> rows{0, shape.no / 2, shape.no - 1};
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  for (int column = 0; column < columns; ++column) for (int row : rows) {
    double expected = 0, magnitude = 0;
    for (int block = 0; block < shape.ni / 256; ++block) {
      const auto *w = weights.data() + (size_t(row) * (shape.ni / 256) + block) * 210;
      const double scale = half_at(w + 208);
      for (int index = 0; index < 256; ++index) {
        const int half = index / 128, part = (index % 128) / 32, lane = index % 32;
        const int low = (w[half * 64 + (part % 2) * 32 + lane] >> (part >= 2 ? 4 : 0)) & 15;
        const int high = (w[128 + half * 32 + lane] >> (2 * part)) & 3;
        const int quant = (low | (high << 4)) - 32;
        const int subscale = signed_byte(w[192 + half * 8 + part * 2 + lane / 16]);
        const int input_index = block * 256 + index;
        const auto *x = packed.data() + Pad +
            (size_t(column) * (shape.ni / 32) + input_index / 32) * 36;
        const double term = scale * subscale * quant * half_at(x) * signed_byte(x[4 + input_index % 32]);
        expected += term;
        magnitude += std::abs(term);
      }
    }
    const double actual = output[Pad + size_t(column) * shape.no + row];
    need(std::isfinite(expected) && std::abs(actual - expected) <= 2e-5 + 2e-6 * magnitude,
         "Q6 scalar dequantized-dot oracle mismatch");
  }
}

void one_shape(Shape shape, bool smoke) {
  const int ni = shape.ni, no = shape.no;
  std::cout << "Q6_SHAPE_BEGIN ni=" << ni << " no=" << no << '\n' << std::flush;
  std::vector<uint8_t> weights(k::native_mmvq_weight_bytes(14, ni, no));
  uint32_t random = 0x173594ab;
  for (auto &value : weights) {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    value = uint8_t(random);
  }
  for (size_t block = 0; block < weights.size() / 210; ++block) {
    const uint16_t scale = block % 3 ? 0x2400 : 0x9800;
    std::memcpy(weights.data() + block * 210 + 208, &scale, sizeof(scale));
  }
  Buffer<uint8_t> w(weights.size());
  w.set(weights);
  const std::vector<int> columns = smoke ? std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}
                                          : std::vector<int>{1, 2, 3, 4, 8};
  for (int nc : columns) {
    std::vector<float> input(size_t(ni) * nc);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = float(int((i * 37 + i / 101 + 17) % 127) - 63) / 32.f;
    Buffer<float> x(input.size()), reference(size_t(no) * nc), candidate(size_t(no) * nc);
    Buffer<uint8_t> packed(k::native_q8_1_bytes(ni, nc));
    x.set(input);
    k::native_quantize_q8_1(x.data(), packed.data(), ni, nc, &rt->compute());
    rt->wait();
    auto packed_before = packed.get();
    auto call = [&](int arm) {
      Arms[arm](w.data(), packed.data(), arm ? candidate.data() : reference.data(),
                ni, no, nc, &rt->compute());
    };
    auto compare = [&](bool oracle = false) {
      auto a = reference.get(), b = candidate.get();
      need(a.size() == b.size() &&
               std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
           "Q6 specialization differs from the production reference");
      for (size_t i = 0; i < size_t(no) * nc; ++i)
        need(std::isfinite(a[Pad + i]), "nonfinite reference output");
      if (oracle) scalar_reference(shape, nc, weights, packed_before, a);
    };
    for (int arm : {0, 1}) {
      call(arm);
      rt->wait();
    }
    compare(true);
    if (!smoke) {
      for (int warmup = 0; warmup < 4; ++warmup) {
        call(0);
        call(1);
      }
      rt->wait();
      for (int batch = 0; batch < 5; ++batch) {
        for (int position = 0; position < 2; ++position) {
          const int arm = (batch + nc + position) % 2;
          rt->wait();
          const auto begin = Clock::now();
          for (int j = 0; j < shape.iterations; ++j) call(arm);
          rt->wait();
          const double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
          need(elapsed >= .01, "timing batch below declared 10ms floor");
          std::cout << "Q6_SAMPLE ni=" << ni << " no=" << no << " nc=" << nc
                    << " arm=" << arm << " batch=" << batch << " position=" << position
                    << " calls=" << shape.iterations << " seconds=" << std::setprecision(12)
                    << elapsed << '\n' << std::flush;
        }
        compare();
      }
    }
    x.unchanged(input);
    need(packed.get() == packed_before, "quantized activation/guard changed");
    std::cout << "Q6_CASE_PASS ni=" << ni << " no=" << no << " nc=" << nc
              << " scalar_oracle=PASS" << '\n' << std::flush;
  }
  w.unchanged(weights);
}
} // namespace

int main(int argc, char **argv) {
  try {
    need(argc == 2 && (std::string(argv[1]) == "--smoke" ||
                          std::string(argv[1]) == "--speed"), "expected --smoke or --speed");
    const bool smoke = std::string(argv[1]) == "--smoke";
    rt = sb::runtime_for();
    // Check the representation on the actual device; SYCL2020 has no
    // info::device::endian_little query in the pinned headers.
    {
      Buffer<uint8_t> representation(4);
      auto *bytes = representation.data();
      rt->wait(rt->compute().single_task([=] {
        const uint32_t word = 0x04030201u;
        __builtin_memcpy(bytes, &word, sizeof(word));
      }));
      const auto observed = representation.get();
      for (unsigned i = 0; i < 4; ++i)
        need(observed[Pad + i] == i + 1, "Q6 device word representation is not little endian");
    }
    std::cout << "Q6_TARGET_ENDIAN_LITTLE=1\n";
    need(k::native_mmvq_multi_exact(), "fixed exact-layout production reference");
    if (smoke) {
      for (auto shape : {Shape{256, 1, 0}, Shape{512, 3, 0}, Shape{1024, 13, 0},
                          Shape{2560, 129, 0}})
        one_shape(shape, true);
      // Refuse invalid geometry before submitting any candidate work.
      for (int columns : {0, 9}) {
        bool rejected = false;
        try {
          k::native_q6_k_columns_mmvq(nullptr, nullptr, nullptr, 256, 1, columns,
                                     &rt->compute());
        } catch (const std::invalid_argument &) { rejected = true; }
        need(rejected, "invalid column count accepted");
      }
    } else {
      for (auto shape : Shapes) one_shape(shape, false);
    }
    rt->wait();
    std::cout << "Q6_COLUMNS_PASS mode=" << (smoke ? "smoke" : "speed") << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Q6_COLUMNS_FAIL " << error.what() << '\n';
    return 1;
  }
}
