// Main-authored; repository LICENSE (MIT). UNRUN component correctness fixture.
// The controller must require host D-copy ownership before any device launch.
// No model dispatch, timing, decode or physical-context qualification here.
#include "native16.hpp"
#include <oneapi/mkl/blas.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace s = strata::sycl_backend;
namespace x = strata::experimental::vllm_xe2_native16;
namespace {
void need(bool pass, const std::string &message) {
  if (!pass) throw std::runtime_error(message);
}
struct CheckedDrain {
  s::Runtime &runtime;
  sycl::queue &queue;
  bool armed = true;
  void finish() { queue.wait_and_throw(); runtime.check(); armed = false; }
  ~CheckedDrain() noexcept {
    if (!armed) return;
    try { finish(); }
    catch (...) {
      std::cerr << "checked queue retirement failed before owner release\n" << std::flush;
      std::_Exit(86);
    }
  }
};
struct Buffer {
  s::Allocation allocation;
  std::vector<uint32_t> initial, observed;
  size_t begin, payload;
  Buffer(const std::shared_ptr<s::Runtime> &runtime, size_t bytes)
      : allocation(runtime, ((bytes + 255) / 64) * 64, s::MemoryKind::Device),
        initial(allocation.size() / 4, 0x7fc055a5u), observed(initial.size()),
        begin(64 + (64 - uintptr_t(allocation.data()) % 64) % 64), payload(bytes) {
    need(allocation.size() % 4 == 0 && begin % 4 == 0 && begin + payload + 64 <= allocation.size(),
         "guarded allocation extent/alignment");
  }
  void *data() { return static_cast<unsigned char *>(allocation.data()) + begin; }
  void half(size_t i, float value) {
    const uint16_t bits = sycl::bit_cast<uint16_t>(sycl::half(value));
    need(i * 2 + 2 <= payload, "host input extent");
    std::memcpy(reinterpret_cast<unsigned char *>(initial.data()) + begin + i * 2, &bits, 2);
  }
  void upload(sycl::queue &q) { q.memcpy(allocation.data(), initial.data(), allocation.size()); }
  void download(sycl::queue &q) { q.memcpy(observed.data(), allocation.data(), allocation.size()); }
  void immutable() const { need(initial == observed, "input or input guard changed"); }
  uint64_t output(int rows, int n, int ld) const {
    uint64_t not_half = 0;
    for (size_t i = 0; i < observed.size(); ++i) {
      const bool after_begin = i >= begin / 4;
      const size_t position = after_begin ? i - begin / 4 : 0;
      const bool valid = after_begin && position / size_t(ld) < size_t(rows) && position % size_t(ld) < size_t(n);
      if (!valid) { need(observed[i] == initial[i], "output prefix/tail/row padding guard changed"); continue; }
      const int row = int(position / size_t(ld)), col = int(position % size_t(ld));
      const int a = row % 7 - 3, b = col % 11 - 5;
      // Sum over all2560 K values: sum(u)=0,sum(v)=-5,sum(u*v)=10,
      // where u=k%5-2,v=k%7-3. Every product/sum is exactly representable
      // in FP32. Many final values cannot be represented in FP16.
      const float expected = float(10 * a * b - 5 * b + 2560) / 8192.f;
      need(observed[i] == std::bit_cast<uint32_t>(expected),
           "FP32 dyadic oracle mismatch at row=" + std::to_string(row) + " col=" + std::to_string(col));
      not_half += float(sycl::half(expected)) != expected;
    }
    need(not_half > 0, "oracle does not detect a hidden FP16 round trip");
    return not_half;
  }
};

void test_case(const std::shared_ptr<s::Runtime> &runtime, x::Gemm &native,
               int rows, int n, int pad) {
  constexpr int k = 2560;
  const int ld = n + pad;
  auto &queue = runtime->compute();
  Buffer input(runtime, size_t(rows) * k * 2), weight(runtime, size_t(n) * k * 2),
         output(runtime, size_t(rows) * ld * 4), reference(runtime, size_t(rows) * ld * 4);
  for (int row = 0; row < rows; ++row) for (int col = 0; col < k; ++col)
    input.half(size_t(row) * k + col, float((row % 7 - 3) * (col % 5 - 2) + 1) / 64.f);
  for (int row = 0; row < n; ++row) for (int col = 0; col < k; ++col)
    weight.half(size_t(row) * k + col, float((row % 11 - 5) * (col % 7 - 3) + 1) / 128.f);
  // All host vectors, USM owners and the same-Runtime in-order queue remain
  // alive through event checks and this drain, including exception unwinding.
  CheckedDrain drain{*runtime, queue};
  std::cerr << "case-start T=" << rows << " N=" << n << " ld=" << ld << std::endl;
  input.upload(queue); weight.upload(queue); output.upload(queue); reference.upload(queue);
  x::GdnProjectionFp32Problem p;
  p.x = static_cast<const uint16_t *>(input.data()); p.w = static_cast<const uint16_t *>(weight.data());
  p.y = static_cast<float *>(output.data()); p.t = rows; p.n = n; p.k = k; p.ldy = ld;
  native.gdn_projection_fp32(p).wait_and_throw(); runtime->check();
  std::cerr << "native-complete" << std::endl;
  // Same unmodified oneMKL beta-zero producer math as Gemm::f16. Setup and
  // validation are not timed; this fixture makes no speed claim.
  oneapi::mkl::blas::row_major::gemm(queue, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans, rows, n, k, 1.f,
      static_cast<const sycl::half *>(input.data()), k,
      static_cast<const sycl::half *>(weight.data()), k, 0.f,
      static_cast<float *>(reference.data()), ld).wait_and_throw();
  runtime->check(); std::cerr << "reference-complete" << std::endl;
  input.download(queue); weight.download(queue); output.download(queue); reference.download(queue);
  drain.finish();
  input.immutable(); weight.immutable();
  const auto sensitive = output.output(rows, n, ld);
  need(reference.output(rows, n, ld) == sensitive, "reference sensitivity mismatch");
  std::cout << "{\"T\":" << rows << ",\"N\":" << n << ",\"ldy\":" << ld
            << ",\"dyadic_ULP\":0,\"guards\":\"passed\",\"oneMKL\":\"passed\","
            << "\"not_FP16_representable_outputs\":" << sensitive << ",\"owner_retirement\":\"passed\"}\n" << std::flush;
}
}
int main(int argc, char **argv) {
  try {
    need(argc == 2 && (std::string(argv[1]) == "--smoke" || std::string(argv[1]) == "--validate"),
         "use --smoke or --validate;controller must qualify host ownership first");
    int su = 0, sv = 0, suv = 0;
    for (int i = 0; i < 2560; ++i) { const int u = i % 5 - 2, v = i % 7 - 3; su += u; sv += v; suv += u * v; }
    need(su == 0 && sv == -5 && suv == 10, "independent dyadic oracle coefficients");
    auto runtime = std::make_shared<s::Runtime>(0);
    x::Gemm native(*runtime, runtime->compute());
    if (std::string(argv[1]) == "--smoke") {
      test_case(runtime, native, 1, 6144, 0); test_case(runtime, native, 129, 10240, 16);
    } else {
      for (int rows : {1, 127, 128, 129, 255, 256, 257, 8191, 8192})
        for (int n : {6144, 10240}) test_case(runtime, native, rows, n, rows % 2 ? 4 : 16);
    }
    runtime->wait(); std::cout << "fixture_status=PASS;no model/speed/physical262144 qualification\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << std::endl; return 1; }
}
