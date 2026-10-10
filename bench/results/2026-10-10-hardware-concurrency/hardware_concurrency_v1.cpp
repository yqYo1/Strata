#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <oneapi/mkl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

// An independent-capacity control: next-slot H2D does not touch GEMM operands.
// Timings include submission and both existing completion waits. No profiling
// property, guessed host/device clock conversion, or production-speed claim.
using Clock = std::chrono::steady_clock;
constexpr size_t copy_bytes = 8ull << 20, words = copy_bytes / 4;
constexpr int64_t max_rows = 8192, hidden = 2560, weight_stride = hidden * 1280;
constexpr int rotations = 8;
[[noreturn]] static void fail(const char* reason) {
    std::fprintf(stderr, "FAIL %s\n", reason); std::fflush(nullptr); std::_Exit(70);
}
static void need(bool ok, const char* reason) { if (!ok) fail(reason); }
static uint32_t pattern(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16);
}
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    need(argc == 2 && (!std::strcmp(argv[1], "--qualify") || !std::strcmp(argv[1], "--measure")), "explicit_mode");
    const bool qualify = !std::strcmp(argv[1], "--qualify");
    try {
        auto handler = [](sycl::exception_list) { fail("asynchronous_error"); };
        auto* compute = new sycl::queue(sycl::gpu_selector_v, handler, sycl::property::queue::in_order{});
        auto* copy = new sycl::queue(compute->get_context(), compute->get_device(), handler, sycl::property::queue::in_order{});
        need(compute->get_backend() == sycl::backend::ext_oneapi_level_zero, "LevelZero_required");
        auto device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(compute->get_device());
        ze_device_properties_t properties{}; properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        need(zeDeviceGetProperties(device, &properties) == ZE_RESULT_SUCCESS, "device_query");
        need(properties.vendorId == 0x8086 && properties.deviceId == 0xe20c && !(properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE), "B570_root");
        ze_pci_ext_properties_t pci{}; pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        need(zeDevicePciGetPropertiesExt(device, &pci) == ZE_RESULT_SUCCESS, "pci_query");
        need(pci.address.domain == 0 && pci.address.bus == 5 && pci.address.device == 0 && pci.address.function == 0, "pci_05_00_0");
        need(compute->get_device().has(sycl::aspect::fp16) && compute->get_device().has(sycl::aspect::usm_device_allocations) && compute->get_device().has(sycl::aspect::usm_host_allocations), "aspects");
        std::printf("{\"kind\":\"configuration\",\"device\":\"B570\",\"pci\":\"0000:05:00.0\",\"profiling\":false,\"in_order\":true,\"same_context\":true,\"independent_operands\":true,\"mode\":\"%s\",\"copy_bytes_per_call\":%zu,\"copy_calls\":4}\n", qualify ? "qualify" : "measure", copy_bytes);
        auto* host = sycl::malloc_host<uint32_t>(2 * words, *copy);
        auto* next = sycl::malloc_device<uint32_t>(2 * words, *copy);
        auto* readback = static_cast<uint32_t*>(std::malloc(2 * copy_bytes));
        auto* w = sycl::malloc_device<sycl::half>(weight_stride * rotations, *compute);
        auto* x = sycl::malloc_device<sycl::half>(max_rows * hidden, *compute);
        auto* y = sycl::malloc_device<float>(max_rows * hidden, *compute);
        auto* yh = static_cast<float*>(std::malloc(max_rows * hidden * sizeof(float)));
        need(host && next && readback && w && x && y && yh, "allocation");
        for (size_t i = 0; i < 2 * words; ++i) host[i] = pattern(static_cast<uint32_t>(i) + 1);
        compute->parallel_for(sycl::range<1>(weight_stride * rotations), [=](sycl::id<1> i) { w[i] = sycl::half((pattern(static_cast<uint32_t>(i) + 17) & 1) ? 0.125f : -0.125f); });
        compute->parallel_for(sycl::range<1>(max_rows * hidden), [=](sycl::id<1> i) { x[i] = sycl::half(1.0f); });
        compute->wait_and_throw();
        copy->memcpy(next, host, 2 * copy_bytes).wait_and_throw();
        const char* names[] = {"copy_only", "compute_only", "serialized", "concurrent"};
        for (bool down : {false, true}) for (int64_t rows : {80, 160, 8192}) {
            const int64_t n = down ? hidden : 1280, k = down ? 640 : hidden;
            const int gemms = rows == 8192 ? 8 : 128;
            auto submit_copies = [&] { for (int i = 0; i < 4; ++i) copy->memcpy(next + (i % 2) * words, host + (i % 2) * words, copy_bytes); };
            auto submit_products = [&] { for (int i = 0; i < gemms; ++i) oneapi::mkl::blas::column_major::gemm(*compute, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, n, rows, k, 1.0f, w + (i % rotations) * weight_stride, k, x, k, 0.0f, y, n); };
            auto drain = [&] { copy->wait_and_throw(); compute->wait_and_throw(); };
            auto batch = [&](int mode) {
                drain();
                const auto begin = Clock::now();
                if (mode != 1) submit_copies();
                if (mode == 2) copy->wait_and_throw();
                if (mode != 0) submit_products();
                drain();
                return std::chrono::duration<double>(Clock::now() - begin).count();
            };
            for (int mode = 0; mode < 4; ++mode) batch(mode); // Fixed warm-up, excluded.
            const int repetitions = qualify ? 1 : 7;
            for (int sample = 0; sample < repetitions; ++sample) for (int position = 0; position < 4; ++position) {
                const int mode = (position + sample) % 4;
                const double seconds = batch(mode);
                std::printf("{\"kind\":\"sample\",\"role\":\"%s\",\"M\":%lld,\"N\":%lld,\"K\":%lld,\"case\":\"%s\",\"sample\":%d,\"position\":%d,\"seconds\":%.9f,\"copy_calls\":%d,\"logical_copy_bytes\":%llu,\"gemm_calls\":%d,\"dense_flops\":%llu}\n", down ? "Down" : "GU", static_cast<long long>(rows), static_cast<long long>(n), static_cast<long long>(k), names[mode], sample, position, seconds, mode == 1 ? 0 : 4, static_cast<unsigned long long>(mode == 1 ? 0 : 4 * copy_bytes), mode == 0 ? 0 : gemms, static_cast<unsigned long long>(mode == 0 ? 0 : 2ull * n * rows * k * gemms));
            }
            copy->memcpy(readback, next, 2 * copy_bytes).wait_and_throw();
            for (size_t i = 0; i < 2 * words; ++i) need(readback[i] == pattern(static_cast<uint32_t>(i) + 1), "copy_full_validation");
            compute->memcpy(yh, y, rows * n * sizeof(float)).wait_and_throw();
            const int last = (gemms - 1) % rotations;
            for (int64_t j = 0; j < n; ++j) {
                int sum = 0;
                for (int64_t kk = 0; kk < k; ++kk) sum += (pattern(static_cast<uint32_t>(last * weight_stride + j * k + kk) + 17) & 1) ? 1 : -1;
                for (int64_t t = 0; t < rows; ++t) need(yh[t * n + j] == sum * 0.125f, "gemm_full_validation");
            }
            std::printf("{\"kind\":\"validation\",\"role\":\"%s\",\"M\":%lld,\"passed\":true,\"scope\":\"all copy words and final GEMM outputs; overwritten products not observed\"}\n", down ? "Down" : "GU", static_cast<long long>(rows));
        }
        copy->wait_and_throw(); compute->wait_and_throw();
        for (void* p : {static_cast<void*>(host), static_cast<void*>(next)}) sycl::free(p, *copy);
        for (void* p : {static_cast<void*>(w), static_cast<void*>(x), static_cast<void*>(y)}) sycl::free(p, *compute);
        std::free(readback); std::free(yh); delete copy; delete compute;
        std::puts("{\"kind\":\"PASS\",\"normal_release\":true}");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "synchronous %s\n", e.what()); fail("synchronous_error"); }
    catch (...) { fail("unknown_error"); }
}
