#include "strata/core/device.hpp"
#include "strata/sycl_upstream/device_metadata.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

namespace native = strata::sycl_upstream;
namespace core = strata::core;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void ck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(cudaGetErrorName(status)) + ": " + native::cuda::backend_error_detail());
}
template<class F> void rejected(F action, const char* message) {
    bool caught = false;
    try { action(); } catch (const core::CudaError&) { caught = true; }
    require(caught, message);
}

int main() try {
    const int count = core::device_count();
    require(count > 0, "no visible native GPU");
    int cases = 0;
    size_t poison_words = 0;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        ck(cudaSetDevice(ordinal));
        native::DeviceFacts facts;
        ck(native::query_device_facts(ordinal, &facts));
        sycl::device device;
        ck(native::cuda::inspect_device(ordinal, [&](const sycl::device& d) { device = d; }));
        require(facts.name == device.get_info<sycl::info::device::name>() &&
                facts.architecture_id == uint64_t(device.get_info<sycl::ext::oneapi::experimental::info::device::architecture>()) &&
                facts.driver_version == device.get_info<sycl::info::device::driver_version>() &&
                facts.device_version == device.get_info<sycl::info::device::version>() &&
                facts.platform_version == device.get_platform().get_info<sycl::info::platform::version>() &&
                facts.total_memory_bytes == device.get_info<sycl::info::device::global_mem_size>() &&
                facts.local_memory_bytes == device.get_info<sycl::info::device::local_mem_size>() &&
                facts.compute_units == device.get_info<sycl::info::device::max_compute_units>() &&
                facts.max_clock_mhz == device.get_info<sycl::info::device::max_clock_frequency>() &&
                facts.vendor_id == device.get_info<sycl::info::device::vendor_id>() &&
                facts.max_workgroup_size == device.get_info<sycl::info::device::max_work_group_size>() &&
                facts.subgroup_sizes == device.get_info<sycl::info::device::sub_group_sizes>(), "native metadata differs from SYCL");
        require(facts.gpu == device.is_gpu() && facts.level_zero == (device.get_backend() == sycl::backend::ext_oneapi_level_zero) &&
                facts.fp16 == device.has(sycl::aspect::fp16) && facts.matrix == device.has(sycl::aspect::ext_intel_matrix) &&
                facts.usm_device == device.has(sycl::aspect::usm_device_allocations) &&
                facts.usm_host == device.has(sycl::aspect::usm_host_allocations) &&
                facts.usm_shared == device.has(sycl::aspect::usm_shared_allocations), "native aspects differ from SYCL");
        std::string name, detail;
        require(core::device_summary(ordinal, name, detail) && name == facts.name &&
                detail == native::device_summary_detail(facts), "device summary differs");
        require(core::gpu_arch_problem(ordinal).empty(), "actual GPU rejected by native admission");
        const auto info = core::device_info(ordinal);
        require(info.ordinal == ordinal && info.name == facts.name && info.arch == facts.architecture &&
                info.total_bytes == facts.total_memory_bytes && info.free_bytes > 0 && info.free_bytes <= info.total_bytes,
                "original public device information differs");
        require(info.cc_major == 0 && info.cc_minor == 0 && info.multi_processor_count == 0 &&
                info.driver_version == 0 && info.runtime_version == 0, "CUDA-only metadata was fabricated");
        int current = -1; ck(cudaGetDevice(&current)); require(current == ordinal, "device info did not select device");
        sycl::context context;
        ck(native::cuda::device_context(ordinal, &context));
        const auto context_devices = context.get_devices();
        require(std::find(context_devices.begin(), context_devices.end(), device) != context_devices.end(),
                "context does not contain selected GPU");
        ++cases;

        // Query the actual named executable image during capture. Admission
        // must not record a poison launch or modify an existing allocation.
        core::DeviceArena untouched(4096, ordinal, false);
        std::vector<uint32_t> marker(1024, 0x01234567), readback(1024);
        ck(cudaMemcpy(untouched.base(), marker.data(), 4096, cudaMemcpyHostToDevice));
        cudaGraph_t graph{};
        cudaStream_t capture_stream{};
        ck(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking));
        ck(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));
        require(core::device_code_error().empty(), "original named executable image unavailable");
        ck(cudaStreamEndCapture(capture_stream, &graph));
        size_t nodes = 123; ck(cudaGraphGetNodes(graph, nullptr, &nodes));
        require(nodes == 0, "device admission recorded GPU work");
        ck(cudaGraphDestroy(graph));
        ck(cudaStreamDestroy(capture_stream));
        ck(cudaMemcpy(readback.data(), untouched.base(), 4096, cudaMemcpyDeviceToHost));
        require(marker == readback, "image admission changed device memory");
        ++cases;

        // Include partial workgroups and a size not divisible by a float.
        // The original initializer covers floor(bytes / 4) float words.
        for (const uint64_t bytes : {uint64_t(4), uint64_t(1028), uint64_t(65539), uint64_t(4 * 1024 * 1024)}) {
            core::DeviceArena arena(bytes, ordinal, true);
            require(arena.capacity() == bytes && arena.used() == 0 && arena.peak() == 0 && arena.ordinal() == ordinal,
                    "original arena metadata");
            std::vector<uint32_t> words(bytes / 4);
            ck(cudaMemcpy(words.data(), arena.base(), words.size() * 4, cudaMemcpyDeviceToHost));
            require(std::all_of(words.begin(), words.end(), [](uint32_t x) { return x == 0x7fc00000; }), "original NaN poison bits");
            poison_words += words.size(); ++cases;
        }
        {
            core::DeviceArena arena(8192, ordinal, true);
            auto first = arena.alloc(3, 1), second = arena.alloc(17, 256), third = arena.alloc(13, 16);
            require(first == arena.base() && second == static_cast<char*>(arena.base()) + 256 &&
                    third == static_cast<char*>(arena.base()) + 288 && arena.used() == 301 && arena.peak() == 301,
                    "original bump allocator offsets");
            require(arena.alloc(0, 0) == nullptr && arena.used() == 301, "zero allocation changed arena");
            rejected([&] { arena.alloc(1, 0); }, "zero alignment accepted");
            rejected([&] { arena.alloc(1, 3); }, "non-power-of-two alignment accepted");
            rejected([&] { arena.alloc(8192); }, "arena over-allocation accepted");
            require(arena.used() == 301, "failed allocation consumed capacity");
            std::vector<unsigned char> bytes(17, 0xa7), result(17);
            ck(cudaMemcpy(second, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
            ck(cudaMemcpy(result.data(), second, result.size(), cudaMemcpyDeviceToHost));
            require(bytes == result, "suballocation copy differs");
            ++cases;
        }
        std::cout << "device=" << ordinal << " name=" << facts.name << " architecture=" << facts.architecture
                  << " target=" << core::compiled_gpu_archs() << " compute_units=" << facts.compute_units
                  << " clock_mhz=" << facts.max_clock_mhz << " named_image=present\n";
    }
    native::DeviceFacts sentinel; sentinel.name = "untouched"; sentinel.total_memory_bytes = 123;
    require(native::query_device_facts(-1, &sentinel) == cudaErrorInvalidDevice && sentinel.name == "untouched" &&
            sentinel.total_memory_bytes == 123, "failed native query modified output"); cudaGetLastError();
    require(native::query_device_facts(0, nullptr) == cudaErrorInvalidValue, "null metadata accepted"); cudaGetLastError();
    bool called = false;
    require(native::cuda::inspect_device(count, [&](const sycl::device&) { called = true; }) == cudaErrorInvalidDevice &&
            !called, "invalid ordinal invoked device callback"); cudaGetLastError();
    require(native::cuda::inspect_device(0, {}) == cudaErrorInvalidValue, "empty inspection accepted"); cudaGetLastError();
    require(native::cuda::device_context(0, nullptr) == cudaErrorInvalidValue, "null context output accepted"); cudaGetLastError();
    sycl::context sentinel_context;
    const auto original_context = sentinel_context;
    require(native::cuda::device_context(-1, &sentinel_context) == cudaErrorInvalidDevice &&
            sentinel_context == original_context, "failed context query changed output"); cudaGetLastError();
    std::string name = "untouched", detail = "untouched";
    require(!core::device_summary(count, name, detail) && name == "untouched" && detail == "untouched",
            "failed summary changed output");
    rejected([&] { core::device_info(count); }, "invalid device info accepted");
    rejected([&] { core::DeviceArena arena(0); }, "zero-byte arena accepted");
    rejected([&] { core::DeviceArena arena(4, count); }, "invalid arena device accepted"); cudaGetLastError();
    std::cout << "PASS native_device_cases=" << cases << " poison_words=" << poison_words << " invalid_queries=10\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
