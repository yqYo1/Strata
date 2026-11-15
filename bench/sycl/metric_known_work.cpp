// Independently authored Strata diagnostic; repository LICENSE applies.
// One exact integer kernel or a matched-copy control, not a saturation benchmark.
// Main must commit a bounded build/run plan and qualify collector coexistence
// before execution. All runtime/collector settings are fixed before exec.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

class StrataMetricKnownIntegerWork;
static_assert(sizeof(int) == 4, "32-bit exact integer control required");

namespace {
void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

bool environment(const char* name, const char* expected) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, expected) == 0;
}

void gate(const char* stage) {
    std::cout << "{\"kind\":\"gate\",\"stage\":\"" << stage << "\",\"pid\":"
              << getpid() << "}\n" << std::flush;
    std::string line;
    require(bool(std::getline(std::cin, line)) && line == "continue", "gate acknowledgement required");
}

// Drain before freeing USM on both success and exceptions. An uncertain drain
// aborts instead of freeing storage still referenced by device work.
struct Allocation {
    sycl::queue& queue;
    int* data = nullptr;

    void close() {
        queue.wait_and_throw();
        if (data) {
            sycl::free(data, queue);
            data = nullptr;
        }
    }

    ~Allocation() {
        try { close(); }
        catch (...) {
            std::fputs("metric known work: uncertain drain/free during teardown\n", stderr);
            std::fflush(stderr);
            std::abort();
        }
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2 && (std::strcmp(argv[1], "--work") == 0 ||
                             std::strcmp(argv[1], "--copy-only") == 0), "choose --work or --copy-only");
        const bool work = std::strcmp(argv[1], "--work") == 0;
        require(environment("STRATA_METRIC_WORK_GATES", "1") &&
                environment("ZET_ENABLE_METRICS", "1") && environment("ZES_ENABLE_SYSMAN", "0") &&
                environment("NEOReadDebugKeys", "1") && environment("EnableDirectSubmission", "0") &&
                environment("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "1"), "required pre-exec safety environment");
        std::cout << std::unitbuf;
        std::cout << "{\"kind\":\"schema\",\"version\":1,\"scope\":\"tiny_known_work\","
                     "\"saturation_claim\":false}\n";
        gate("before-runtime");
        sycl::queue queue(sycl::gpu_selector_v,
                          sycl::property_list{sycl::property::queue::in_order{},
                                              sycl::property::queue::enable_profiling{}});
        const auto device = queue.get_device();
        require(device.get_backend() == sycl::backend::ext_oneapi_level_zero &&
                device.get_info<sycl::info::device::vendor_id>() == 0x8086,
                "Intel Level Zero GPU required");
        const auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
        ze_device_properties_t properties{};
        properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        require(zeDeviceGetProperties(native, &properties) == ZE_RESULT_SUCCESS &&
                properties.deviceId == 0xe20c && !(properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE),
                "root B570 required");
        ze_pci_ext_properties_t pci{};
        pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        require(zeDevicePciGetPropertiesExt(native, &pci) == ZE_RESULT_SUCCESS &&
                pci.address.domain == 0 && pci.address.bus == 5 &&
                pci.address.device == 0 && pci.address.function == 0, "exact PCI0000:05:00.0 required");
        std::cout << "{\"kind\":\"device\",\"pci\":\"0000:05:00.0\","
                     "\"in_order\":true,\"profiling\":true}\n";
        gate("runtime-ready");

        constexpr size_t n = 65536, guard = 64, total = n + 2 * guard;
        constexpr int sentinel = -1234567;
        std::vector<int> input(total, sentinel), output(total);
        for (size_t i = 0; i < n; ++i) input[guard + i] = int(i % 1024);
        Allocation owner{queue};
        owner.data = sycl::malloc_device<int>(total, queue);
        require(owner.data != nullptr, "device allocation refused");
        queue.memcpy(owner.data, input.data(), total * sizeof(int)).wait_and_throw();
        sycl::event arithmetic;
        if (work) {
            int* data = owner.data;
            arithmetic = queue.parallel_for<StrataMetricKnownIntegerWork>(sycl::range<1>(n),
                [=](sycl::id<1> i) { data[guard + i[0]] = data[guard + i[0]] * 3 + 7; });
            arithmetic.wait_and_throw();
        }
        queue.memcpy(output.data(), owner.data, total * sizeof(int)).wait_and_throw();
        for (size_t i = 0; i < total; ++i) {
            const bool payload = i >= guard && i < guard + n;
            const int expected = work && payload ? input[i] * 3 + 7 : input[i];
            require(output[i] == expected, "integer result or allocation guard mismatch");
        }
        uint64_t start = 0, end = 0;
        if (work) {
            start = arithmetic.get_profiling_info<sycl::info::event_profiling::command_start>();
            end = arithmetic.get_profiling_info<sycl::info::event_profiling::command_end>();
            require(end > start, "known kernel device timestamps unavailable");
        }
        owner.close();
        std::cout << "{\"kind\":\"complete\",\"kernel_count\":" << (work ? 1 : 0)
                  << ",\"elements\":" << n << ",\"integer_results_and_guards\":true,"
                  << "\"requested_kernel_read_bytes\":" << (work ? n * sizeof(int) : 0)
                  << ",\"requested_kernel_write_bytes\":" << (work ? n * sizeof(int) : 0)
                  << ",\"kernel_start\":" << start << ",\"kernel_end\":" << end
                  << ",\"physical_traffic_measured\":false,\"saturation_claim\":false}\n";
        gate("before-exit");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metric known work: FAIL " << error.what() << '\n';
        return 1;
    }
}
