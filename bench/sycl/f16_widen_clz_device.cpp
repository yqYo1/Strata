// Main-authored binary16 device semantic screen; repository LICENSE.
// No inference caller or timing. Run only after the frozen centering comparison.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/q6_f16_widen_clz.hpp"
#include "strata/sycl/runtime.hpp"
#include <sycl/ext/intel/info/device.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace k = strata::kernels;
constexpr std::size_t Patterns = 65536, Guard = 32;
constexpr std::uint32_t Mark = 0x4a55a55a, Poison = 0xdeadbeef;
constexpr std::array<const char *, 3> Names{{"reference", "clz", "native"}};
template <unsigned Arm> class Widen;

void need(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// Host-only independent dyadic oracle. Never capture or call it on the GPU.
std::uint32_t oracle(std::uint16_t h) {
  const unsigned exponent = (h >> 10) & 31u, mantissa = h & 1023u;
  if (exponent == 31)
    return (std::uint32_t(h & 0x8000u) << 16) | 0x7f800000u | (mantissa << 13);
  const double magnitude = std::ldexp(
      double(mantissa + (exponent == 0 ? 0u : 1024u)),
      exponent == 0 ? -24 : int(exponent) - 25);
  const float value = static_cast<float>((h & 0x8000u) ? -magnitude : magnitude);
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <unsigned Arm> std::uint32_t convert(std::uint16_t h) {
  if constexpr (Arm == 0)
    return sycl::bit_cast<std::uint32_t>(k::f32_from_f16(h));
  else if constexpr (Arm == 1)
    return k::q6_f32_bits_from_f16_clz(h);
  else {
    static_assert(Arm == 2);
    // Representation transfer into a real half object; no half pointer cast.
    const auto value = sycl::bit_cast<sycl::half>(h);
    return sycl::bit_cast<std::uint32_t>(static_cast<float>(value));
  }
}

template <unsigned Arm>
void execute(const std::shared_ptr<sb::Runtime> &rt, const unsigned char *input,
             std::uint32_t *output) {
  rt->wait(rt->compute().parallel_for<Widen<Arm>>(sycl::range<1>(Patterns),
      [=](sycl::id<1> id) {
        // The production Q6 seam also uses byte-safe two-byte copies.
        std::uint16_t h;
        __builtin_memcpy(&h, input + 2 * id[0], sizeof(h));
        output[id[0]] = convert<Arm>(h);
      }));
}

struct Counts {
  unsigned finite_mismatches = 0, subnormal_mismatches = 0;
  unsigned zero_mismatches = 0, infinity_mismatches = 0;
  unsigned nan_class_mismatches = 0, nan_sign_mismatches = 0;
  unsigned nan_payload_changes = 0, nan_quiet_changes = 0;
  unsigned signaling_quieted = 0, signaling_preserved = 0;
  unsigned bit_mismatches = 0;
  std::uint64_t checksum = 14695981039346656037ull;

  void add(unsigned h, std::uint32_t expected, std::uint32_t actual) {
    const unsigned exponent = (h >> 10) & 31u, mantissa = h & 1023u;
    const bool mismatch = expected != actual;
    bit_mismatches += mismatch;
    if (exponent != 31) {
      finite_mismatches += mismatch;
      subnormal_mismatches += exponent == 0 && mantissa != 0 && mismatch;
      zero_mismatches += exponent == 0 && mantissa == 0 && mismatch;
    } else if (mantissa == 0) {
      infinity_mismatches += mismatch;
    } else {
      const bool is_nan = (actual & 0x7f800000u) == 0x7f800000u &&
                          (actual & 0x007fffffu) != 0;
      nan_class_mismatches += !is_nan;
      nan_sign_mismatches += (actual ^ expected) >> 31;
      // Payload excludes the quiet bit, which is recorded independently.
      nan_payload_changes += (actual & 0x003fffffu) != (expected & 0x003fffffu);
      nan_quiet_changes += ((actual ^ expected) & 0x00400000u) != 0;
      if (is_nan && !(mantissa & 0x200u)) {
        signaling_quieted += (actual & 0x00400000u) != 0;
        signaling_preserved += (actual & 0x00400000u) == 0;
      }
    }
    checksum = (checksum ^ actual) * 1099511628211ull;
  }

  bool passed(unsigned arm) const {
    // Integer arms preserve every bit. Native may quiet/canonicalize NaNs,
    // while preserving NaN classification/sign and every non-NaN value.
    return arm < 2 ? bit_mismatches == 0 :
        finite_mismatches == 0 && infinity_mismatches == 0 &&
        nan_class_mismatches == 0 && nan_sign_mismatches == 0;
  }
};

bool one_offset(const std::shared_ptr<sb::Runtime> &rt, unsigned offset) {
  const std::size_t start = Guard + offset;
  std::vector<unsigned char> input(start + 2 * Patterns + Guard, 0xa5);
  for (unsigned h = 0; h < Patterns; ++h) {
    input[start + 2 * h] = static_cast<unsigned char>(h);
    input[start + 2 * h + 1] = static_cast<unsigned char>(h >> 8);
  }
  constexpr std::size_t Stride = Patterns + 2 * Guard;
  std::vector<std::uint32_t> initial(3 * Stride, Mark);
  for (unsigned arm = 0; arm < 3; ++arm)
    for (std::size_t i = 0; i < Patterns; ++i)
      initial[arm * Stride + Guard + i] = Poison;
  sb::Allocation in(rt, input.size(), sb::MemoryKind::Device);
  sb::Allocation out(rt, initial.size() * sizeof(initial[0]), sb::MemoryKind::Device);
  rt->wait(rt->compute().memcpy(in.data(), input.data(), input.size()));
  rt->wait(rt->compute().memcpy(out.data(), initial.data(), out.size()));
  const auto *bytes = in.as<unsigned char>() + start;
  auto *words = out.as<std::uint32_t>();
  execute<0>(rt, bytes, words + Guard);
  execute<1>(rt, bytes, words + Stride + Guard);
  execute<2>(rt, bytes, words + 2 * Stride + Guard);
  std::vector<std::uint32_t> observed(initial.size());
  std::vector<unsigned char> unchanged(input.size());
  rt->wait(rt->compute().memcpy(observed.data(), out.data(), out.size()));
  rt->wait(rt->compute().memcpy(unchanged.data(), in.data(), input.size()));
  need(input == unchanged, "input or input guards changed");
  bool passed = true;
  for (unsigned arm = 0; arm < 3; ++arm) {
    for (std::size_t i = 0; i < Guard; ++i)
      need(observed[arm * Stride + i] == Mark &&
               observed[arm * Stride + Guard + Patterns + i] == Mark,
           "output guard changed");
    Counts counts;
    for (unsigned h = 0; h < Patterns; ++h)
      counts.add(h, oracle(static_cast<std::uint16_t>(h)),
                 observed[arm * Stride + Guard + h]);
    const bool eligible = counts.passed(arm);
    passed = passed && eligible;
    std::printf("F16_WIDEN_ARM arm=%s offset=%u patterns=65536 status=%s "
        "finite_mismatches=%u subnormal_mismatches=%u zero_mismatches=%u "
        "infinity_mismatches=%u nan_class_mismatches=%u nan_sign_mismatches=%u "
        "nan_payload_changes=%u nan_quiet_changes=%u signaling_quieted=%u "
        "signaling_preserved=%u bit_mismatches=%u checksum=%016llx\n",
        Names[arm], offset, eligible ? "PASS" : "FAIL", counts.finite_mismatches,
        counts.subnormal_mismatches, counts.zero_mismatches, counts.infinity_mismatches,
        counts.nan_class_mismatches, counts.nan_sign_mismatches, counts.nan_payload_changes,
        counts.nan_quiet_changes, counts.signaling_quieted, counts.signaling_preserved,
        counts.bit_mismatches, static_cast<unsigned long long>(counts.checksum));
    std::fflush(stdout);
  }
  return passed;
}
} // namespace

int main() {
  try {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(std::numeric_limits<double>::digits >= 53);
    static_assert(sizeof(sycl::half) == 2 &&
                  std::is_trivially_copyable_v<sycl::half>);
    const auto rt = sb::runtime_for();
    const auto &device = rt->device();
    need(device.get_backend() == sycl::backend::ext_oneapi_level_zero &&
             device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
             device.get_info<sycl::ext::intel::info::device::device_id>() == 0xe20c &&
             device.get_info<sycl::ext::intel::info::device::pci_address>() == "0000:05:00.0",
         "requires the exact B570 Level Zero device");
    need(device.has(sycl::aspect::fp16), "native arm requires fp16");
    need(rt->compute().has_property<sycl::property::queue::in_order>(),
         "requires the qualified in-order queue");
    std::puts("F16_WIDEN_TARGET_B570=1 timing=0 offsets=0,1");
    std::fflush(stdout);
    const bool aligned = one_offset(rt, 0);
    const bool odd = one_offset(rt, 1);
    rt->wait();
    std::printf("F16_WIDEN_DEVICE_STATUS=%s patterns_per_arm_offset=65536 "
                "arms=3 offsets=2 readonly_and_guards=PASS timing=0\n",
                aligned && odd ? "PASS" : "FAIL");
    return aligned && odd ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "F16_WIDEN_DEVICE_ERROR %s\n", error.what());
    return 1;
  }
}
