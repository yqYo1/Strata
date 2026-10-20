// Authored for Strata; repository LICENSE applies. Component attribution only.
// Use the unchanged production compatibility library, not a second copy path.
#include <cuda_runtime.h>
#include "strata/sycl/cpu_wait_profile.hpp"
#include "strata/sycl/runtime.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace hp = strata::sycl_backend::cpu_wait_profile;
using strata::sycl_backend::runtime_for;
constexpr size_t bytes = 128ull << 20;
constexpr int fills = 8, samples = 12;
struct Envelope { uint32_t before, value, after; };
static_assert(sizeof(Envelope) == 12);
constexpr uint32_t left = 0x17ab8932, right = 0x695de410;

[[noreturn]] void fail(const char* reason) noexcept {
    std::fprintf(stderr, "status copy fail-stop: %s\n", reason);
    std::fflush(stderr);
    std::_Exit(1);
}
void need(bool value, const char* reason) {
    if (!value) fail(reason);
}
void ck(cudaError_t error) {
    if (error != cudaSuccess) fail(cudaGetErrorString(error));
}
template<class Function> auto device_call(Function function) {
    // Catch at the call boundary, before the caller's in-flight buffers unwind.
    try { return function(); }
    catch (const std::exception& error) { fail(error.what()); }
    catch (...) { fail("unknown SYCL exception"); }
}
void timing(const char* name, const hp::State& s, hp::StatusKind kind) {
    std::printf(",\"%s\":{\"calls\":%llu,\"cpu_s\":%.12g,\"wall_s\":%.12g,"
        "\"wait_cpu_s\":%.12g,\"wait_wall_s\":%.12g}", name,
        s.status_calls[kind], s.status_cpu[kind], s.status_wall[kind],
        s.status_wait_cpu[kind], s.status_wait_wall[kind]);
}
void validate(const hp::State& s) {
    need(!s.invalid && !s.depth && !s.host_depth && !s.status_depth && !s.status_part_depth &&
        s.status_scopes == 1 && s.status_completed == 1, "incomplete status accounting");
    double cpu = 0, wall = 0;
    for (int k = 0; k < hp::status_kinds; ++k) {
        const unsigned expected = k == hp::status_precheck || k == hp::status_queue ? 2 :
            k == hp::status_enqueue || k == hp::status_observer ||
            k == hp::status_drain || k == hp::status_postcheck ? 1 : 0;
        need(s.status_calls[k] == expected && !s.status_throws[k] &&
            hp::duration(s.status_cpu[k]) && hp::duration(s.status_wall[k]) &&
            hp::duration(s.status_wait_cpu[k]) && hp::duration(s.status_wait_wall[k]) &&
            s.status_wait_cpu[k] <= s.status_cpu[k] &&
            s.status_wait_wall[k] <= s.status_wall[k], "invalid status phase");
        cpu += s.status_cpu[k]; wall += s.status_wall[k];
    }
    need(hp::duration(s.status_scope_cpu) && hp::duration(s.status_scope_wall) &&
        cpu <= s.status_scope_cpu + 1e-7 && wall <= s.status_scope_wall + 1e-7,
        "status partition not conserved");
}

int main(int argc, char** argv) {
    // Explicit arguments prevent a default GPU run. Environment is never changed.
    if (argc != 3) return 2;
    const bool usm = std::string(argv[1]) == "usm";
    const bool pending = std::string(argv[2]) == "pending";
    if ((!usm && std::string(argv[1]) != "ordinary") ||
        (!pending && std::string(argv[2]) != "idle")) return 2;
    // Exception handling fail-stops while every owner is still alive. No buffer
    // destructor unwinds through an unproven device completion.
    try {
        auto runtime = device_call([] { return runtime_for(); });
        cudaStream_t stream{};
        ck(cudaStreamCreate(&stream));
        need(stream->get_context() == runtime->context() &&
            stream->get_device() == runtime->device() &&
            stream->has_property<sycl::property::queue::in_order>() &&
            stream->has_property<sycl::property::queue::enable_profiling>(), "queue domain/properties");
        auto ordinary = std::make_unique<Envelope>();
        Envelope *host = ordinary.get(), *device{};
        if (usm) ck(cudaMallocHost(reinterpret_cast<void**>(&host), sizeof(Envelope)));
        ck(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(Envelope)));
        uint8_t* backlog{};
        ck(cudaMalloc(reinterpret_cast<void**>(&backlog), bytes + 128));
        need(sycl::get_pointer_type(host, runtime->context()) ==
            (usm ? sycl::usm::alloc::host : sycl::usm::alloc::unknown), "host pointer domain");
        ck(cudaMemsetAsync(backlog, 0x57, bytes + 128, stream));
        const Envelope warm{left, 0xa5197b31, right};
        ck(cudaMemcpyAsync(device, &warm, sizeof warm, cudaMemcpyHostToDevice, stream));
        ck(cudaMemcpyAsync(host, device, sizeof warm, cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));
        need(host->before == left && host->value == warm.value && host->after == right, "warmup payload/guards");
        std::printf("{\"type\":\"begin\",\"schema\":1,\"host\":\"%s\",\"queue\":\"%s\","
            "\"samples\":%d,\"copy_bytes\":4,\"backlog_fills\":%d,\"fill_bytes\":%zu} \n",
            argv[1], argv[2], samples, pending ? fills : 0, bytes);
        std::fflush(stdout);
        for (int rep = 0; rep < samples; ++rep) {
            const Envelope input{left, uint32_t(0x417d9000 + rep), right};
            *host = Envelope{left, 0xdeadbeef, right};
            ck(cudaMemcpyAsync(device, &input, sizeof input, cudaMemcpyHostToDevice, stream));
            ck(cudaStreamSynchronize(stream)); // source remains alive through checked completion
            sycl::event last;
            if (pending)
                for (int f = 0; f < fills; ++f)
                    last = device_call([&] { return stream->memset(backlog + 64, 0x30 + rep, bytes); });
            hp::State state;
            need(hp::active == nullptr, "unexpected outer CPU profile");
            hp::active = &state;
            bool observed_pending = false, complete_after_copy = false;
            hp::Stamp queried, copy_begin;
            {
                hp::StatusScope scope;
                if (pending) observed_pending = device_call([&] { return last.get_info<sycl::info::event::command_execution_status>(); }) !=
                    sycl::info::event_command_status::complete;
                queried = hp::stamp();
                copy_begin = hp::stamp();
                ck(cudaMemcpyAsync(&host->value, &device->value, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
                if (pending) complete_after_copy = device_call([&] { return last.get_info<sycl::info::event::command_execution_status>(); }) ==
                    sycl::info::event_command_status::complete;
                ck(cudaStreamSynchronize(stream)); // preserves async error delivery and destination ownership
                scope.complete();
            }
            hp::active = nullptr;
            validate(state);
            need(queried.valid && copy_begin.valid && hp::duration(copy_begin.wall - queried.wall), "clock unavailable");
            need(host->before == left && host->value == input.value && host->after == right, "status payload/host guards");
            Envelope check{};
            std::array<uint8_t, 64> before{}, after{};
            ck(cudaMemcpyAsync(&check, device, sizeof check, cudaMemcpyDeviceToHost, stream));
            ck(cudaMemcpyAsync(before.data(), backlog, before.size(), cudaMemcpyDeviceToHost, stream));
            ck(cudaMemcpyAsync(after.data(), backlog + 64 + bytes, after.size(), cudaMemcpyDeviceToHost, stream));
            ck(cudaStreamSynchronize(stream));
            need(check.before == left && check.value == input.value && check.after == right, "device guards");
            for (auto value : before) need(value == 0x57, "backlog prefix guard");
            for (auto value : after) need(value == 0x57, "backlog suffix guard");
            // Also verify the final finite fill completed; these probes are not timed.
            if (pending) {
                uint8_t first = 0, final = 0;
                ck(cudaMemcpyAsync(&first, backlog + 64, 1, cudaMemcpyDeviceToHost, stream));
                ck(cudaMemcpyAsync(&final, backlog + 63 + bytes, 1, cudaMemcpyDeviceToHost, stream));
                ck(cudaStreamSynchronize(stream));
                need(first == uint8_t(0x30 + rep) && final == first, "backlog fill endpoints");
            }
            std::printf("{\"type\":\"sample\",\"rep\":%d,\"eligible\":%s,"
                "\"pending_observed\":%s,\"last_complete_after_copy\":%s,"
                "\"query_to_copy_wall_s\":%.12g,\"scope_cpu_s\":%.12g,\"scope_wall_s\":%.12g",
                rep, !pending || observed_pending ? "true" : "false", observed_pending ? "true" : "false",
                complete_after_copy ? "true" : "false", copy_begin.wall - queried.wall,
                state.status_scope_cpu, state.status_scope_wall);
            timing("enqueue", state, hp::status_enqueue);
            timing("observer", state, hp::status_observer);
            timing("drain", state, hp::status_drain);
            timing("precheck", state, hp::status_precheck);
            timing("queue_context", state, hp::status_queue);
            timing("postcheck", state, hp::status_postcheck);
            std::puts(",\"payload_and_guards_passed\":true}");
            std::fflush(stdout);
        }
        ck(cudaStreamSynchronize(stream));
        ck(cudaFree(backlog)); ck(cudaFree(device));
        if (usm) ck(cudaFreeHost(host));
        ck(cudaStreamDestroy(stream));
        std::puts("{\"type\":\"end\",\"passed\":true,\"samples\":12,\"owners_closed\":true}");
        return 0;
    } catch (const std::exception& error) {
        fail(error.what());
    }
}
