// Main-authored diagnostic; repository LICENSE. No external implementation.
// API contracts: installed Level Zero1.17 headers and Intel SYCL interop.
// No inference, kernel, overlap, imported event, or environment mutation.
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace prefill_copy_probe {
constexpr std::size_t guard_words = 64;
constexpr std::uint32_t guard = 0x7162abcd;

void need(bool yes, const char* message) {
    if (!yes) throw std::runtime_error(message);
}
[[noreturn]] void fail_stop(const char* message) noexcept {
    std::cerr << "copy probe fail-stop: " << message << std::endl;
    std::_Exit(70); // No potentially in-flight buffer/context destruction.
}
void checked(ze_result_t result, const char* operation) {
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "LevelZero failure " << operation << " result="
                  << unsigned(result) << std::endl;
        throw std::runtime_error(operation);
    }
}
void gate(const char* name) {
    std::cout << "{\"kind\":\"gate\",\"name\":\"" << name << "\"}" << std::endl;
    std::string response;
    need(bool(std::getline(std::cin, response)) && response == "continue", "gate refused");
}
struct Options {
    bool copy = false;
    std::size_t bytes = 0, iterations = 0, batches = 0;
};
Options options(int argc, char** argv) {
    need(argc == 9, "use --policy sycl|copy --bytes N --iterations N --batches N");
    need(std::string(argv[1]) == "--policy" && std::string(argv[3]) == "--bytes"
         && std::string(argv[5]) == "--iterations" && std::string(argv[7]) == "--batches",
         "argument order");
    const std::string policy = argv[2];
    need(policy == "sycl" || policy == "copy", "policy");
    Options o{policy == "copy", std::stoull(argv[4]), std::stoull(argv[6]), std::stoull(argv[8])};
    need(o.bytes <= (256u << 20) && o.bytes % 4 == 0 && o.iterations <= 16
         && o.batches <= 4, "bounded aligned probe extent");
    need((o.bytes == 0 && o.iterations == 0 && o.batches == 0)
         || (o.bytes >= 4 && o.iterations >= 1 && o.batches >= 1), "extent agreement");
    return o;
}

// All handles borrow the same SYCL context/device. Only raw_list is owned.
// This owner drains both lists BEFORE freeing any USM, also on exceptions.
struct Owner {
    sycl::device device{sycl::gpu_selector_v};
    sycl::context context{device};
    sycl::queue queue{context, device,
        [](sycl::exception_list) { fail_stop("asynchronous SYCL exception"); },
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}}};
    ze_command_list_handle_t raw_list = nullptr;
    std::vector<void*> memory;

    Owner() {
        need(device.get_backend() == sycl::backend::ext_oneapi_level_zero, "LevelZero required");
        need(device.get_info<sycl::info::device::name>().find("B570") != std::string::npos,
             "B570 required");
        need(device.has(sycl::aspect::usm_host_allocations)
             && device.has(sycl::aspect::usm_device_allocations), "USM required");
        memory.reserve(3);
    }
    Owner(const Owner&) = delete;
    ~Owner() noexcept {
        try {
            drain();
            if (raw_list && zeCommandListDestroy(raw_list) != ZE_RESULT_SUCCESS)
                fail_stop("raw immediate list destruction");
            raw_list = nullptr;
            for (auto* p : memory) sycl::free(p, context);
        } catch (...) { fail_stop("checked retirement"); }
    }
    void drain() {
        // A timeout/lost device is an uncertain lifetime: never unwind/free.
        if (raw_list && zeCommandListHostSynchronize(raw_list, 5000000000ull) != ZE_RESULT_SUCCESS)
            fail_stop("raw list did not drain within5s");
        queue.wait_and_throw();
    }
    std::uint32_t* allocate(std::size_t words, bool host) {
        auto* p = host ? sycl::malloc_host<std::uint32_t>(words, context)
                       : sycl::malloc_device<std::uint32_t>(words, device, context);
        need(p != nullptr, "USM allocation");
        memory.push_back(p);
        return p;
    }
    void configure(bool copy) {
        auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
        need(std::holds_alternative<ze_command_list_handle_t>(native), "qualified immediate queue required");
        auto base = std::get<ze_command_list_handle_t>(native);
        std::uint32_t ordinal = ~0u, index = ~0u;
        checked(zeCommandListGetOrdinal(base, &ordinal), "SYCL ordinal");
        checked(zeCommandListImmediateGetIndex(base, &index), "SYCL index");
        need(ordinal == 0 && index == 0, "SYCL ordinal0/index0 required");
        auto native_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
        std::uint32_t count = 0;
        checked(zeDeviceGetCommandQueueGroupProperties(native_device, &count, nullptr), "group count");
        need(count >= 2 && count < 64, "bounded queue groups");
        std::vector<ze_command_queue_group_properties_t> groups(count);
        for (auto& g : groups) g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        checked(zeDeviceGetCommandQueueGroupProperties(native_device, &count, groups.data()), "groups");
        need((groups[0].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
             && (groups[0].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)
             && groups[0].numQueues == 1 && groups[1].numQueues == 1
             && groups[1].flags == ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY, "expected engine groups");
        if (copy) {
            ze_command_queue_desc_t desc{};
            desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
            desc.ordinal = 1;
            desc.index = 0;
            desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER | ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
            desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
            desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
            checked(zeCommandListCreateImmediate(
                sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context),
                native_device, &desc, &raw_list), "copy list creation");
            checked(zeCommandListGetOrdinal(raw_list, &ordinal), "copy ordinal");
            checked(zeCommandListImmediateGetIndex(raw_list, &index), "copy index");
            need(ordinal == 1 && index == 0, "copy ordinal1/index0 required");
        }
        std::cout << "{\"kind\":\"engine\",\"ordinal\":" << ordinal
                  << ",\"index\":" << index << ",\"in_order\":true}" << std::endl;
    }
    void transfer(void* dst, const void* src, std::size_t bytes, bool copy) {
        if (copy) {
            checked(zeCommandListAppendMemoryCopy(raw_list, dst, src, bytes, nullptr, 0, nullptr),
                    "copy append");
            if (zeCommandListHostSynchronize(raw_list, 5000000000ull) != ZE_RESULT_SUCCESS)
                fail_stop("copy completion");
        } else queue.memcpy(dst, src, bytes).wait_and_throw();
    }
};
std::uint32_t payload(std::size_t i, unsigned batch) {
    return std::uint32_t(i * 2654435761ull) ^ (0x5ac31e07u + batch * 7919u);
}
int run(const Options& o) {
    gate("before-runtime");
    {
        Owner owner;
        owner.configure(o.copy);
        gate("runtime-ready"); // Main checks all actual providers before work.
        if (o.bytes) {
            const auto words = o.bytes / 4, total = words + 2 * guard_words;
            auto* source = owner.allocate(total, true);
            auto* result = owner.allocate(total, true);
            auto* device = owner.allocate(total, false);
            for (std::size_t i = 0; i < total; ++i) source[i] = guard;
            // Common drained setup, outside both timed arms. No other queue
            // can access these bytes until each transfer is host-complete.
            owner.queue.memcpy(device, source, total * 4).wait_and_throw();
            gate("buffers-ready");
            for (unsigned batch = 0; batch <= o.batches; ++batch) {
                for (std::size_t i = 0; i < words; ++i) source[i + guard_words] = payload(i, batch);
                const auto start = std::chrono::steady_clock::now();
                const auto cpu = std::clock();
                const auto copies = batch == 0 ? 1 : o.iterations;
                for (std::size_t n = 0; n < copies; ++n)
                    owner.transfer(device + guard_words, source + guard_words, o.bytes, o.copy);
                const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const auto cpu_seconds = double(std::clock() - cpu) / CLOCKS_PER_SEC;
                owner.queue.memcpy(result, device, total * 4).wait_and_throw();
                for (std::size_t i = 0; i < total; ++i) {
                    const auto expected = i < guard_words || i >= guard_words + words
                                          ? guard : payload(i - guard_words, batch);
                    need(source[i] == expected && result[i] == expected, "payload/source/edge guard mismatch");
                }
                std::cout.precision(17);
                std::cout << "{\"kind\":\"batch\",\"warmup\":" << (batch == 0 ? "true" : "false")
                          << ",\"bytes\":" << o.bytes << ",\"copies\":" << copies
                          << ",\"wall_s\":" << seconds << ",\"cpu_s\":" << cpu_seconds
                          << ",\"full_readback_and_guards\":true}" << std::endl;
            }
        }
        owner.drain();
    } // Successful list destruction and USM retirement precede the final gate.
    gate("before-exit");
    std::cout << "{\"kind\":\"done\",\"passed\":true}" << std::endl;
    return 0;
}
} // namespace prefill_copy_probe
int main(int argc, char** argv) {
    try { return prefill_copy_probe::run(prefill_copy_probe::options(argc, argv)); }
    catch (const std::exception& e) { std::cerr << "copy probe: " << e.what() << std::endl; return 1; }
}
