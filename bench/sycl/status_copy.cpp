// Authored for Strata; repository LICENSE applies. Diagnostic only.
// Finite producer/copy/drain path; no pre-copy status query or queue policy change.
#include <cuda_runtime.h>
#include "strata/sycl/cpu_wait_profile.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>

namespace hp = strata::sycl_backend::cpu_wait_profile;
using strata::sycl_backend::runtime_for;
constexpr int rounds = 6, fills = 8, clock_pairs = 128;
constexpr uint32_t left = 0x17ab8932, right = 0x695de410;
struct Envelope { uint32_t before, value, after; };
static_assert(sizeof(Envelope) == 12);

[[noreturn]] void fail(const char* reason) noexcept {
    std::fprintf(stderr, "status query fail-stop: %s\n", reason);
    std::fflush(stderr);
    std::_Exit(1);
}
void need(bool value, const char* reason) { if (!value) fail(reason); }
void ck(cudaError_t error) { if (error != cudaSuccess) fail(cudaGetErrorString(error)); }
template<class Function> auto device_call(Function function) {
    // Fail at the call boundary, before caller-owned device/host data unwind.
    try { return function(); }
    catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown SYCL exception"); }
}
void elapsed(hp::Stamp begin, hp::Stamp end) {
    need(begin.valid && end.valid && hp::duration(end.cpu - begin.cpu) &&
        hp::duration(end.wall - begin.wall), "clock unavailable");
}
void loaded_paths() {
    std::ifstream maps("/proc/self/maps");
    need(bool(maps), "loaded maps unavailable");
    std::set<std::string> paths;
    std::string line;
    while (std::getline(maps, line)) {
        auto offset = line.find('/');
        if (offset == std::string::npos) continue;
        auto path = line.substr(offset);
        if (path.find("/libsycl.so") != std::string::npos ||
            path.find("/libur_loader.so") != std::string::npos ||
            path.find("/libur_adapter_level_zero_v2.so") != std::string::npos ||
            path.find("/libze_loader.so") != std::string::npos ||
            path.find("/libze_intel_gpu.so") != std::string::npos) paths.insert(path);
    }
    need(!maps.bad() && paths.size() == 5, "missing/ambiguous runtime providers");
    for (const auto& path : paths)
        std::fprintf(stderr, "status query loaded: %s\n", path.c_str());
    std::fflush(stderr);
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const bool profiling = std::string(argv[1]) == "profiled";
    const bool usm = std::string(argv[2]) == "usm";
    const bool smoke = std::string(argv[3]) == "smoke";
    if ((!profiling && std::string(argv[1]) != "plain") ||
        (!usm && std::string(argv[2]) != "ordinary") ||
        (!smoke && std::string(argv[3]) != "measure")) return 2;
    try {
        auto runtime = device_call([] { return runtime_for(); });
        auto queue = device_call([&] { return runtime->make_queue(profiling); });
        need(queue.get_context() == runtime->context() && queue.get_device() == runtime->device() &&
            queue.has_property<sycl::property::queue::in_order>() &&
            queue.has_property<sycl::property::queue::enable_profiling>() == profiling,
            "queue domain/properties");
        // Locally owned queue, not a cudaStreamCreate registry member.
        // Always drain it explicitly; never cudaStreamDestroy(&queue).
        const size_t bytes = smoke ? 64ull << 10 : 128ull << 20;
        Envelope ordinary{left, 0, right}, *host = &ordinary, *device{};
        if (usm) ck(cudaMallocHost(reinterpret_cast<void**>(&host), sizeof(Envelope)));
        if (usm) need(device_call([&] { return sycl::get_pointer_type(host, runtime->context()); }) ==
            sycl::usm::alloc::host, "USM-host domain");
        ck(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(Envelope)));
        uint8_t *backlog{};
        ck(cudaMalloc(reinterpret_cast<void**>(&backlog), bytes + 128));
        device_call([&] { queue.memset(backlog, 0x57, bytes + 128); });
        ck(cudaStreamSynchronize(&queue));
        loaded_paths();
        const auto resolution = device_call([&] {
            return runtime->device().get_info<sycl::info::device::profiling_timer_resolution>(); });
        need(resolution > 0, "profiling timer unavailable");
        std::array<double, clock_pairs> cpu{}, wall{};
        for (int i = 0; i < clock_pairs; ++i) {
            const auto begin = hp::stamp(), end = hp::stamp();
            elapsed(begin, end);cpu[i] = end.cpu - begin.cpu;wall[i] = end.wall - begin.wall;
        }
        std::sort(cpu.begin(), cpu.end());std::sort(wall.begin(), wall.end());
        const double clock_cpu = (cpu[63] + cpu[64]) / 2, clock_wall = (wall[63] + wall[64]) / 2;
        std::printf("{\"type\":\"begin\",\"schema\":4,\"profiling\":%s,\"host\":\"%s\",\"mode\":\"%s\","
            "\"samples\":%d,\"warmups\":%d,\"fills\":8,\"fill_bytes\":%zu,\"copy_bytes\":4,"
            "\"precopy_status_query\":false,\"clock_pair_cpu_s\":%.12g,\"clock_pair_wall_s\":%.12g,"
            "\"timer_resolution_ns\":%llu}\n", profiling ? "true" : "false", argv[2], argv[3],
            smoke ? 1 : rounds, smoke ? 0 : 1, bytes, clock_cpu, clock_wall, (unsigned long long)resolution);
        std::fflush(stdout);
        for (int rep = smoke ? 0 : -1; rep < (smoke ? 1 : rounds); ++rep) {
            const Envelope input{left, 0x21000000u + uint32_t(rep + 1), right};
            *host = Envelope{left, ~input.value, right};
            ck(cudaMemcpyAsync(device, &input, sizeof input, cudaMemcpyHostToDevice, &queue));
            ck(cudaStreamSynchronize(&queue));
            std::array<sycl::event, fills> retained;
            std::fprintf(stderr, "strata finite copy: sample=%d begin\n", rep);std::fflush(stderr);
            // The fixture performs no explicit event/profiling query before
            // the timed copy; internal runtime queries remain observable.
            // All eight producer events survive through the checked drain.
            const auto begin = hp::stamp();
            for (int f = 0; f < fills; ++f)
                retained[f] = device_call([&] { return queue.memset(backlog + 64, 0x30 + rep + 1, bytes); });
            const auto produced = hp::stamp();
            ck(cudaMemcpyAsync(&host->value, &device->value, sizeof(uint32_t), cudaMemcpyDeviceToHost, &queue));
            const auto copied = hp::stamp();
            ck(cudaStreamSynchronize(&queue));
            const auto drained = hp::stamp();
            std::fprintf(stderr, "strata finite copy: sample=%d end\n", rep);std::fflush(stderr);
            elapsed(begin, produced);elapsed(produced, copied);elapsed(copied, drained);elapsed(begin, drained);
            need(host->before == left && host->after == right && host->value == input.value, "host payload/guards");
            // These checks are after checked completion and outside all timers.
            uint64_t service = 0, span = 0, first = 0, last = 0;
            bool device_clock_valid = profiling, device_clock_eligible = profiling;
            for (int f = 0; f < fills; ++f) {
                need(device_call([&] { return retained[f].get_info<sycl::info::event::command_execution_status>(); }) ==
                    sycl::info::event_command_status::complete, "producer incomplete after checked drain");
                if (!profiling) continue;
                const auto start = device_call([&] { return retained[f].get_profiling_info<sycl::info::event_profiling::command_start>(); });
                const auto end = device_call([&] { return retained[f].get_profiling_info<sycl::info::event_profiling::command_end>(); });
                if (end <= start || (f && start < last)) device_clock_valid = device_clock_eligible = false;
                if (end > start) {
                    const uint64_t duration = end - start;
                    need(duration <= UINT64_MAX - service, "device duration overflow");service += duration;
                    if (duration < resolution) device_clock_eligible = false;
                }
                if (!f) first = start;
                last = end;
            }
            if (profiling && device_clock_valid && last >= first) span = last - first;
            if (profiling && (span < service || double(span) * 1e-9 >
                drained.wall - begin.wall + std::max(1e-6, 8e-9 * double(resolution)))) device_clock_eligible = false;
            Envelope check{};
            std::array<uint8_t, 64> before{}, after{};
            uint8_t first_byte = 0, final_byte = 0;
            ck(cudaMemcpyAsync(&check, device, sizeof check, cudaMemcpyDeviceToHost, &queue));
            ck(cudaMemcpyAsync(before.data(), backlog, before.size(), cudaMemcpyDeviceToHost, &queue));
            ck(cudaMemcpyAsync(after.data(), backlog + 64 + bytes, after.size(), cudaMemcpyDeviceToHost, &queue));
            ck(cudaMemcpyAsync(&first_byte, backlog + 64, 1, cudaMemcpyDeviceToHost, &queue));
            ck(cudaMemcpyAsync(&final_byte, backlog + 63 + bytes, 1, cudaMemcpyDeviceToHost, &queue));
            ck(cudaStreamSynchronize(&queue));
            need(check.before == left && check.after == right && check.value == input.value, "device envelope");
            for (auto value : before) need(value == 0x57, "producer prefix guard");
            for (auto value : after) need(value == 0x57, "producer suffix guard");
            need(first_byte == uint8_t(0x30 + rep + 1) && final_byte == first_byte, "producer endpoints");
            const auto eligible = [&](hp::Stamp a, hp::Stamp b) {
                return b.wall - a.wall >= 20e-6 && b.wall - a.wall >= 100 * clock_wall &&
                    b.cpu - a.cpu >= 100 * clock_cpu;
            };
            std::printf("{\"type\":\"%s\",\"rep\":%d,\"producer_cpu_s\":%.12g,\"producer_wall_s\":%.12g,"
                "\"enqueue_cpu_s\":%.12g,\"enqueue_wall_s\":%.12g,\"drain_cpu_s\":%.12g,\"drain_wall_s\":%.12g,"
                "\"whole_cpu_s\":%.12g,\"whole_wall_s\":%.12g,\"timing_eligible\":%s,"
                "\"producer_timing_eligible\":%s,\"enqueue_timing_eligible\":%s,\"drain_timing_eligible\":%s,"
                "\"gpu_fill_service_ns\":%llu,\"gpu_fill_span_ns\":%llu,\"gpu_clock_eligible\":%s,"
                "\"completed_events\":8,\"payload_and_guards_passed\":true}\n", rep < 0 ? "warmup" : "sample", rep,
                produced.cpu - begin.cpu, produced.wall - begin.wall, copied.cpu - produced.cpu, copied.wall - produced.wall,
                drained.cpu - copied.cpu, drained.wall - copied.wall, drained.cpu - begin.cpu, drained.wall - begin.wall,
                eligible(begin, drained) ? "true" : "false", eligible(begin, produced) ? "true" : "false",
                eligible(produced, copied) ? "true" : "false", eligible(copied, drained) ? "true" : "false",
                (unsigned long long)service, (unsigned long long)span, device_clock_eligible ? "true" : "false");
            std::fflush(stdout);
        }
        ck(cudaStreamSynchronize(&queue));
        ck(cudaFree(backlog));ck(cudaFree(device));if (usm) ck(cudaFreeHost(host));
        device_call([&] { runtime->wait(); });
        std::puts("{\"type\":\"end\",\"passed\":true,\"owners_closed\":true}");
        return 0;
    } catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown top-level exception"); }
}
