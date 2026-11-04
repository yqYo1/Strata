// SPDX-License-Identifier: MIT
// Main-authored exact-device diagnostic, not a sampler or throughput benchmark.
// The owned controller must set the qualified environment before exec.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/info/device.hpp>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t Count = 64, Guard = 4, Total = Count + 2 * Guard;
constexpr double Sentinel = -12345678.25;
constexpr double Increment = 0x1p-40;
using Values = std::array<double, Total>;

void need(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void environment() {
    constexpr const char* required[][2] = {
        {"ONEAPI_DEVICE_SELECTOR", "level_zero:gpu"},
        {"UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "1"},
        {"NEOReadDebugKeys", "1"},
        {"EnableDirectSubmission", "0"},
        {"ZES_ENABLE_SYSMAN", "0"},
        {"SYCL_CACHE_PERSISTENT", "0"}};
    for (const auto& pair : required) {
        const char* actual = std::getenv(pair[0]);
        need(actual && std::string(actual) == pair[1], pair[0]);
    }
}

sycl::device device() {
    std::vector<sycl::device> devices;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
            devices.push_back(d);
    need(devices.size() == 1, "one unambiguous Level Zero GPU required");
    const auto d = devices.front();
    const auto vendor = d.get_info<sycl::info::device::vendor_id>();
    const auto id = d.get_info<sycl::ext::intel::info::device::device_id>();
    const auto pci = d.get_info<sycl::ext::intel::info::device::pci_address>();
    const auto name = d.get_info<sycl::info::device::name>();
    std::cout << "DEVICE backend=level_zero name=" << std::quoted(name)
              << " vendor=" << vendor << " device_id=" << id
              << " pci=" << std::quoted(pci)
              << " driver=" << std::quoted(d.get_info<sycl::info::device::driver_version>())
              << " fp64=" << d.has(sycl::aspect::fp64)
              << " fp16=" << d.has(sycl::aspect::fp16)
              << " usm_device=" << d.has(sycl::aspect::usm_device_allocations)
              << " global_bytes=" << d.get_info<sycl::info::device::global_mem_size>()
              << " max_alloc_bytes=" << d.get_info<sycl::info::device::max_mem_alloc_size>() << '\n';
    need(vendor == 0x8086 && id == 0xe20c && pci == "0000:05:00.0" &&
             name == "Intel Arc B570 Graphics", "unexpected exact device");
    return d;
}

void guards(const Values& values) {
    for (size_t i = 0; i < Guard; ++i)
        need(values[i] == Sentinel && values[Total - 1 - i] == Sentinel,
             "device allocation guard changed");
}

// The finite owned child deliberately skips provider teardown after checked
// completion/free. A queue destructor is outside this capability diagnostic.
// In particular, an asynchronous error must not unwind borrowed host storage.
[[noreturn]] void finish(int status) {
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(status);
}

void arithmetic(const sycl::device& d, bool exponential) {
    need(d.has(sycl::aspect::fp64), "FP64 kernel refused: aspect unavailable");
    need(d.has(sycl::aspect::usm_device_allocations), "device USM unavailable");
    // Keep the queue alive through process exit; see finish(). No mutable env.
    auto* q = new sycl::queue(d, [](sycl::exception_list errors) {
        for (auto error : errors) {
            try { std::rethrow_exception(error); }
            catch (const std::exception& e) { std::cerr << "ASYNC_ERROR " << e.what() << '\n'; }
            catch (...) { std::cerr << "ASYNC_ERROR unknown\n"; }
        }
        finish(1);
    }, sycl::property::queue::in_order{});
    Values input, output, source_check;
    input.fill(Sentinel);
    output.fill(Sentinel);
    for (size_t i = 0; i < Count; ++i) {
        input[Guard + i] = exponential ? -double(i) : 1.0 + double(i) / 64.0;
        output[Guard + i] = std::numeric_limits<double>::quiet_NaN();
    }
    std::cout << "STAGE allocation elements=" << Total << " arrays=2\n";
    auto* in = sycl::malloc_device<double>(Total, *q);
    auto* out = sycl::malloc_device<double>(Total, *q);
    need(in && out, "device allocation failed");
    // Catch synchronous submission errors before unwinding the host arrays:
    // another copy may still borrow them when a later submission fails.
    try {
        q->memcpy(in, input.data(), sizeof(input));
        q->memcpy(out, output.data(), sizeof(output));
        q->wait_and_throw();
        std::cout << "STAGE submit operation=" << (exponential ? "exp" : "arithmetic")
                  << " elements=" << Count << " in_order=1\n";
        if (exponential) {
            q->parallel_for(sycl::range<1>(Count), [=](sycl::id<1> at) {
                const size_t i = at[0] + Guard;
                out[i] = sycl::exp(in[i]);
            });
        } else {
            q->parallel_for(sycl::range<1>(Count), [=](sycl::id<1> at) {
                const size_t i = at[0] + Guard;
                const double shifted = in[i] + Increment;
                out[i] = shifted - in[i];
            });
        }
        q->wait_and_throw();
        std::cout << "STAGE kernel_complete\n";
        q->memcpy(output.data(), out, sizeof(output));
        q->memcpy(source_check.data(), in, sizeof(source_check));
        q->wait_and_throw();
        guards(output);
        need(source_check == input, "device input changed");
        double largest_relative_error = 0;
        for (size_t i = 0; i < Count; ++i) {
            const double expected = exponential ? std::exp(input[Guard + i]) : Increment;
            const double value = output[Guard + i];
            need(std::isfinite(value), "missing or nonfinite device output");
            const double error = std::abs((value - expected) / expected);
            largest_relative_error = std::fmax(largest_relative_error, error);
            need(exponential ? error <= 1e-12 : value == expected,
                 "device FP64 numerical check failed");
        }
        sycl::free(out, *q);
        sycl::free(in, *q);
        q->wait_and_throw();
        std::cout << "PASS operation=" << (exponential ? "exp" : "arithmetic")
                  << " checked=" << Count << " guards=pass source=unchanged"
                  << " max_relative_error=" << largest_relative_error
                  << " ordinary_provider_teardown=not_exercised\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR device operation " << e.what() << '\n';
        finish(1);
    } catch (...) {
        std::cerr << "ERROR device operation unknown\n";
        finish(1);
    }
}
} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf << std::setprecision(17);
    try {
        need(argc == 3 && std::string(argv[1]) == "--operation",
             "usage: device_fp64_probe --operation query|arithmetic|exp");
        const std::string operation = argv[2];
        need(operation == "query" || operation == "arithmetic" || operation == "exp",
             "unknown operation");
        environment();
        std::cout << "STAGE query\n";
        const auto d = device();
        if (operation == "query") {
            std::cout << "PASS operation=query kernels_submitted=0\n";
        } else {
            if (!d.has(sycl::aspect::fp64)) {
                std::cout << "UNSUPPORTED operation=" << operation
                          << " aspect_fp64=0 kernels_submitted=0\n";
                finish(77);
            }
            arithmetic(d, operation == "exp");
        }
        finish(0);
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << '\n';
        finish(1);
    } catch (...) {
        std::cerr << "ERROR unknown\n";
        finish(1);
    }
}
