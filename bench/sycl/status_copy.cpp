// Authored for Strata; repository LICENSE applies. Diagnostic only.
// Compare completed-event queries; do not change production queue defaults.
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
constexpr int rounds = 4, queries = 4096, clock_pairs = 128;
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
    // No default GPU execution, runtime environment mutation or stalled gates.
    if (argc != 3) return 2;
    const bool profiling = std::string(argv[1]) == "profiled";
    const bool smoke = std::string(argv[2]) == "smoke";
    if ((!profiling && std::string(argv[1]) != "plain") ||
        (!smoke && std::string(argv[2]) != "measure")) return 2;
    try {
        auto runtime = device_call([] { return runtime_for(); });
        auto queue = device_call([&] { return runtime->make_queue(profiling); });
        need(queue.get_context() == runtime->context() && queue.get_device() == runtime->device() &&
            queue.has_property<sycl::property::queue::in_order>() &&
            queue.has_property<sycl::property::queue::enable_profiling>() == profiling,
            "queue domain/properties");
        // This queue is locally owned, not registered with cudaStreamCreate.
        // Use checked compat drain, but never cudaStreamDestroy(&queue).
        Envelope *device{}, input{left, 0, right}, output{};
        ck(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(Envelope)));
        ck(cudaMemcpyAsync(device, &input, sizeof input, cudaMemcpyHostToDevice, &queue));
        ck(cudaStreamSynchronize(&queue));
        loaded_paths();
        std::array<double, clock_pairs> cpu{}, wall{};
        for (int i = 0; i < clock_pairs; ++i) {
            const auto begin = hp::stamp(), end = hp::stamp();
            elapsed(begin, end);
            cpu[i] = end.cpu - begin.cpu; wall[i] = end.wall - begin.wall;
        }
        std::sort(cpu.begin(), cpu.end()); std::sort(wall.begin(), wall.end());
        const double clock_cpu = (cpu[63] + cpu[64]) / 2;
        const double clock_wall = (wall[63] + wall[64]) / 2;
        const int batch_rounds = smoke ? 1 : rounds, batch_queries = smoke ? 16 : queries;
        std::printf("{\"type\":\"begin\",\"schema\":3,\"profiling\":%s,\"mode\":\"%s\","
            "\"rounds\":%d,\"queries_per_batch\":%d,\"completed_before_batch\":true,"
            "\"clock_pair_cpu_s\":%.12g,\"clock_pair_wall_s\":%.12g}\n",
            profiling ? "true" : "false", argv[2], batch_rounds, batch_queries, clock_cpu, clock_wall);
        std::fflush(stdout);
        for (int rep = 0; rep < batch_rounds; ++rep) {
            // Finite 4-byte producer only. The event stays alive through query
            // batches and the final checked queue drain. No pending gate.
            auto event = device_call([&] { return queue.memset(&device->value, 0x31 + rep, sizeof(uint32_t)); });
            ck(cudaStreamSynchronize(&queue));
            need(device_call([&] { return event.get_info<sycl::info::event::command_execution_status>(); }) ==
                sycl::info::event_command_status::complete, "incomplete seed event");
            unsigned completed = 0;
            const auto begin = hp::stamp();
            for (int i = 0; i < batch_queries; ++i)
                completed += device_call([&] { return event.get_info<sycl::info::event::command_execution_status>(); }) ==
                    sycl::info::event_command_status::complete;
            const auto end = hp::stamp();
            elapsed(begin, end);
            need(completed == unsigned(batch_queries), "completed-event status regression");
            ck(cudaStreamSynchronize(&queue));
            ck(cudaMemcpyAsync(&output, device, sizeof output, cudaMemcpyDeviceToHost, &queue));
            ck(cudaStreamSynchronize(&queue));
            const auto byte = uint32_t(0x31 + rep);
            need(output.before == left && output.after == right && output.value == byte * 0x01010101u,
                "payload/guard verification");
            const bool eligible = !smoke && end.wall - begin.wall >= 20e-6 &&
                end.wall - begin.wall >= 100 * clock_wall && end.cpu - begin.cpu >= 100 * clock_cpu;
            std::printf("{\"type\":\"sample\",\"rep\":%d,\"queries\":%d,\"completed\":%u,"
                "\"cpu_s\":%.12g,\"wall_s\":%.12g,\"timing_eligible\":%s,\"payload_and_guards_passed\":true}\n",
                rep, batch_queries, completed, end.cpu - begin.cpu, end.wall - begin.wall,
                eligible ? "true" : "false");
            std::fflush(stdout);
        }
        ck(cudaStreamSynchronize(&queue));
        ck(cudaFree(device));
        // Explicitly finish the local queue before its destructor. Runtime
        // default queues and sticky errors are also drained before normal exit.
        device_call([&] { runtime->wait(); });
        std::puts("{\"type\":\"end\",\"passed\":true,\"owners_closed\":true}");
        return 0;
    } catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown top-level exception"); }
}
