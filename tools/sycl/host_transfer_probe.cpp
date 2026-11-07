// SPDX-License-Identifier: MIT
// Compare caller time in production-compatible copy submission and completion.
// This component has no model, math kernels, event profiling or environment changes.
#include "strata/sycl/cpu_wait_profile.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using strata::sycl_backend::Runtime;
namespace clock = strata::sycl_backend::cpu_wait_profile;
constexpr size_t Guard = 16, Alignment = 4096;
constexpr uint64_t Sentinel = 0xa5963c5af00dcafeULL;
void need(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
// Optional owned-controller handshake; environment remains immutable in child.
void gate(const char* stage) {
    const char* enabled = std::getenv("STRATA_TRANSFER_PROBE_GATES");
    if (!enabled || std::string(enabled) != "1") return;
    std::cout << "GATE stage=" << stage << '\n';
    std::string response;
    need(bool(std::getline(std::cin, response)) && response == "continue", "controller acknowledgement");
}
size_t number(const char* text) {
    need(text && text[0] >= '0' && text[0] <= '9', "unsigned decimal option required");
    errno = 0;
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    need(!errno && *end == '\0' && value <= (512u << 20), "invalid or excessive option");
    return size_t(value);
}
uint64_t pattern(size_t index, uint64_t seed) {
    return (seed ^ (uint64_t(index) * 0x9e3779b97f4a7c15ULL + 0x13579bdf2468ace0ULL)) | 1ULL;
}
void drain(const std::shared_ptr<Runtime>& rt) noexcept {
    // Never unwind borrowed host storage if completion is uncertain.
    try { rt->wait(); } catch (...) { std::abort(); }
}
struct Drain {
    std::shared_ptr<Runtime> rt;
    ~Drain() noexcept { drain(rt); }
};
void copy(const std::shared_ptr<Runtime>& rt, void* dst, const void* src, size_t bytes) {
    try { rt->wait(rt->compute().memcpy(dst, src, bytes)); }
    catch (...) { std::abort(); }
}
uint64_t* aligned(void* base, size_t capacity, size_t bytes) {
    need(std::align(Alignment, bytes, base, capacity) != nullptr, "host alignment bounds");
    return static_cast<uint64_t*>(base);
}
void guards(uint64_t* p, size_t words) {
    for (size_t i = 0; i < Guard; ++i) p[i] = p[Guard + words + i] = Sentinel;
}
void check(const uint64_t* p, size_t words, uint64_t seed) {
    for (size_t i = 0; i < Guard; ++i)
        need(p[i] == Sentinel && p[Guard + words + i] == Sentinel, "end guard changed");
    for (size_t i = 0; i < words; ++i)
        need(p[Guard + i] == pattern(i, seed), "full pattern mismatch");
}
void timing(const char* phase, const clock::Stamp& a, const clock::Stamp& b) {
    need(a.valid && b.valid && clock::duration(b.cpu-a.cpu) && clock::duration(b.wall-a.wall), "invalid clock");
    std::cout << ' ' << phase << "_cpu_s=" << b.cpu-a.cpu << ' ' << phase << "_wall_s=" << b.wall-a.wall;
}
void transfer(const std::shared_ptr<Runtime>& rt, void* dst, const void* src, size_t bytes,
              cudaMemcpyKind direction, const char* host, int repetition) {
    const auto begin = clock::stamp();
    const auto submit = cudaMemcpyAsync(dst, src, bytes, direction, &rt->compute());
    const auto submitted = clock::stamp();
    // Retire even if submission returned an error. Owners stay live on failure.
    const auto wait = cudaStreamSynchronize(&rt->compute());
    const auto end = clock::stamp();
    need(submit == cudaSuccess && wait == cudaSuccess, "compat copy or completion failed");
    rt->check();
    need(end.wall > begin.wall, "zero elapsed copy time");
    std::cout << "COPY host=" << host << " direction="
              << (direction == cudaMemcpyHostToDevice ? "H2D" : "D2H")
              << " repetition=" << repetition << " warmup=" << int(repetition < 0) << " bytes=" << bytes;
    timing("submit", begin, submitted);
    timing("wait", submitted, end);
    timing("total", begin, end);
    std::cout << " total_GBps=" << double(bytes)/1e9/(end.wall-begin.wall) << '\n';
}
void run(bool usm, size_t bytes, int repeats) {
    const char* kind = usm ? "usm" : "ordinary";
    const size_t words = bytes / sizeof(uint64_t), active_bytes = bytes + 2*Guard*sizeof(uint64_t);
    const size_t host_bytes = active_bytes + Alignment;
    // Ordinary storage precedes the final drain guard and every GPU borrower.
    std::unique_ptr<uint8_t[]> ordinary;
    auto rt = strata::sycl_backend::runtime_for();
    need(rt->device().is_gpu() && rt->device().has(sycl::aspect::usm_host_allocations), "GPU with host USM required");
    need(Runtime::devices().size() == 1, "one unambiguous GPU required for transfer probe");
    std::cout << "DEVICE name=" << rt->device().get_info<sycl::info::device::name>()
              << " driver=" << rt->device().get_info<sycl::info::device::driver_version>() << '\n';
    gate("runtime-ready");
    Allocation host;
    const auto allocation_begin = clock::stamp();
    if (usm) host = Allocation(rt, host_bytes, MemoryKind::Host);
    else ordinary.reset(new uint8_t[host_bytes]);
    const auto allocation_end = clock::stamp();
    auto* h = aligned(usm ? host.data() : ordinary.get(), host_bytes, active_bytes);
    Allocation device(rt, active_bytes, MemoryKind::Device), verification(rt, host_bytes, MemoryKind::Host);
    auto* v = aligned(verification.data(), host_bytes, active_bytes);
    Drain lease{rt}; // Completes before all USM owners and ordinary host storage.
    const auto type = sycl::get_pointer_type(h, rt->context());
    need(type == (usm ? sycl::usm::alloc::host : sycl::usm::alloc::unknown), "unexpected host allocation category");
    std::cout << "SETUP host=" << kind << " bytes=" << bytes << " host_allocation_bytes=" << host_bytes
              << " device_bytes=" << device.size() << " verification_allocation_bytes=" << verification.size()
              << " body_alignment_mod4096=" << (reinterpret_cast<uintptr_t>(h+Guard) % Alignment);
    timing("allocation", allocation_begin, allocation_end);
    const auto touch_begin = clock::stamp();
    guards(h, words); guards(v, words);
    for (size_t i = 0; i < words; ++i) { h[Guard+i] = pattern(i, 0); v[Guard+i] = Sentinel; }
    const auto touch_end = clock::stamp();
    timing("pretouch", touch_begin, touch_end);
    std::cout << '\n';
    copy(rt, device.data(), h, active_bytes); // Initialize device guards outside timing.
    for (int repetition = -1; repetition < repeats; ++repetition) {
        const uint64_t seed = 0x123456789abcdef0ULL + uint64_t(repetition+2)*0xfedcba987654321ULL;
        for (size_t i = 0; i < words; ++i) h[Guard+i] = pattern(i, seed);
        transfer(rt, device.as<uint64_t>()+Guard, h+Guard, bytes, cudaMemcpyHostToDevice, kind, repetition);
        copy(rt, v, device.data(), active_bytes);
        check(v, words, seed); check(h, words, seed);
        std::cout << "CHECK direction=H2D repetition=" << repetition << " full=PASS guards=PASS source=PASS\n";
        // A missing D2H copy must not pass from the previous H2D source contents.
        for (size_t i = 0; i < words; ++i) h[Guard+i] = ~pattern(i, seed);
        transfer(rt, h+Guard, device.as<uint64_t>()+Guard, bytes, cudaMemcpyDeviceToHost, kind, repetition);
        check(h, words, seed);
        copy(rt, v, device.data(), active_bytes);
        check(v, words, seed);
        std::cout << "CHECK direction=D2H repetition=" << repetition << " full=PASS guards=PASS source=PASS\n";
    }
    rt->wait();
    std::cout << "DONE host=" << kind << " repeats=" << repeats << " warmups=1 scope=transfer_component_not_model\n";
}
} // namespace
int main(int argc, char** argv) {
    std::cout << std::unitbuf << std::fixed << std::setprecision(9);
    try {
        need(argc == 7 && std::string(argv[1]) == "--host" && std::string(argv[3]) == "--bytes" &&
             std::string(argv[5]) == "--repeats", "usage: sycl_host_transfer_probe --host ordinary|usm --bytes N --repeats N");
        const std::string host = argv[2];
        const auto bytes = number(argv[4]), repeats = number(argv[6]);
        need((host == "ordinary" || host == "usm") && bytes >= 4096 && bytes % 8 == 0 && repeats >= 1 && repeats <= 8,
             "host/byte/repetition bounds");
        gate("before-runtime");
        run(host == "usm", bytes, int(repeats));
        gate("before-exit");
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
