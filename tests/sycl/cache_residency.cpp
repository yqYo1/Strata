// Main-authored identity/payload oracle; repository LICENSE.
// Exercises ExpertCache's actual callback lookup and scratch lease, not model
// arithmetic. The legacy arm omits the new publication calls in the same test;
// it is a controlled old-sequence counterexample, not an old engine benchmark.
#include "strata/core/expert_cache.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sb = strata::sycl_backend;
using Cache = strata::core::ExpertCache;
constexpr int Missing = strata::core::kNotResident;
constexpr size_t Blob = 1024;
constexpr int Experts = 512, Slots = 130, Layer = 1, Victim = 64, Incoming = 400;

void need(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void need(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
void gate(const char* name) {
    std::cout << "COMPONENT_READY " << name << '\n' << std::flush;
    need(std::cin.get() == 'X', "owner gate");
}
std::vector<uint8_t> payload(int layer, int expert) {
    std::vector<uint8_t> result(Blob);
    for (size_t i = 0; i < Blob; ++i)
        result[i] = uint8_t((uint32_t(expert + 1) * 1177u + uint32_t(layer) * 271u + uint32_t(i) * 53u) ^
                            (uint32_t(expert) >> 3));
    return result;
}
struct BlockedCopy {
    struct State { std::mutex mutex; std::condition_variable cv; bool ready = false, timed_out = false; };
    std::shared_ptr<State> state = std::make_shared<State>();
    std::thread timer;
    explicit BlockedCopy(sycl::queue& queue) {
        const auto s = state;
        timer = std::thread([s] {
            std::unique_lock lock(s->mutex);
            if (!s->cv.wait_for(lock, std::chrono::seconds(2), [&] { return s->ready; })) {
                s->timed_out = true; s->ready = true; s->cv.notify_all();
            }
        });
        queue.submit([s](sycl::handler& handler) {
            handler.host_task([s] {
                std::unique_lock lock(s->mutex);
                need(s->cv.wait_for(lock, std::chrono::seconds(4), [&] { return s->ready; }), "copy release deadline");
            });
        });
    }
    void release() { std::lock_guard lock(state->mutex); state->ready = true; state->cv.notify_all(); }
    void finish() { release(); timer.join(); need(!state->timed_out, "copy gate timer fallback"); }
    ~BlockedCopy() { release(); if (timer.joinable()) timer.join(); }
};

void run(bool fixed) {
    auto runtime = sb::runtime_for();
    auto& queue = runtime->compute();
    need(queue.has_property<sycl::property::queue::in_order>(), "ordered fixture queue required");
    need(queue.get_device().get_info<sycl::info::device::name>().find("B570") != std::string::npos, "unexpected GPU");
    gate("runtime-ready");
    Cache cache;
    std::string error;
    need(cache.open(Slots, 2, Experts, Blob, error), error);
    cache.set_per_layer_admission(true);
    cache.enable_scratch_loan_tracking();
    std::vector<int32_t> host(2 * Experts, Missing);
    std::vector<std::vector<uint8_t>> initial;
    initial.reserve(Slots);
    for (int layer = 0; layer < 2; ++layer) for (int e = 0; e < Slots / 2; ++e) {
        const int slot = cache.admit(layer, e);
        need(slot == layer * (Slots / 2) + e, "initial admission");
        host[size_t(layer * Experts + e)] = slot;
        initial.push_back(payload(layer, e));
        need(cache.fill_slot_queued(slot, initial.back().data(), error), error);
    }
    need(cache.sync_queued(error), error);
    need(cache.resident() == Slots, "full initial cache");
    sb::Allocation table(runtime, host.size() * sizeof(int32_t), sb::MemoryKind::Device);
    auto* device = table.as<int32_t>();
    auto publish = [&] {
        need(cudaMemcpy(device, host.data(), host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess,
             "device residency publication");
    };
    auto check_table = [&] {
        std::vector<int32_t> actual(host.size());
        need(cudaMemcpy(actual.data(), device, actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) == cudaSuccess,
             "device residency readback");
        need(actual == host, "host/device residency mismatch");
    };
    auto check_callbacks = [&] {
        for (int layer = 0; layer < 2; ++layer) for (int e = 0; e < Experts; ++e)
            need(cache.slot_of(layer, e) == host[size_t(layer * Experts + e)], "callback map mismatch");
    };
    publish(); check_table(); check_callbacks();
    const size_t out = size_t(Layer * Experts + Victim), in = size_t(Layer * Experts + Incoming);
    const int slot = host[out];
    need(slot == Slots - 1 && host[in] == Missing, "tail replacement setup");
    host[out] = Missing;
    if (fixed) need(cache.replace_residency(host, error), error);
    publish(); check_table();
    const auto incoming = payload(Layer, Incoming);
    bool stale_while_pending = false;
    {
        BlockedCopy blocked(queue);
        auto copy = queue.memcpy(cache.device_slot(slot), incoming.data(), Blob);
        need(copy.get_info<sycl::info::event::command_execution_status>() !=
             sycl::info::event_command_status::complete, "copy was not pending");
        stale_while_pending = cache.slot_of(Layer, Victim) == slot;
        need(cache.slot_of(Layer, Incoming) == Missing, "incoming visible before copy completion");
        if (fixed) {
            need(!stale_while_pending, "victim visible during overwrite");
            need(cache.admit(Layer, Victim) == Missing && cache.admit(Layer, Incoming) == Missing,
                 "mask reopened claimed capacity");
        }
        blocked.finish(); runtime->wait();
    }
    host[in] = slot;
    if (fixed) need(cache.replace_residency(host, error), error);
    publish(); check_table();
    need(cache.verify_slot(slot, incoming.data(), error), error);
    if (!fixed) {
        need(stale_while_pending && cache.slot_of(Layer, Victim) == slot &&
             cache.slot_of(Layer, Incoming) == Missing, "old sequence counterexample absent");
        std::puts("legacy_stale_map=REPRODUCED overwritten_slot=129 incoming_payload=PASS");
        return;
    }
    check_callbacks();
    for (int test = 0; test < 5; ++test) {
        auto bad = host;
        if (test == 0) bad.pop_back();
        if (test == 1) bad[in] = Slots;
        if (test == 2) bad[in] = -2;
        if (test == 3) bad[in] = Slots - 2;
        if (test == 4) bad[in] = 0;
        std::string rejected;
        need(!cache.replace_residency(bad, rejected) && !rejected.empty(), "invalid map accepted");
        check_callbacks();
    }
    Cache partial;
    need(partial.open(2, 1, 3, Blob, error) && partial.admit(0, 0) == 0, "partial-cache setup");
    std::string rejected;
    need(!partial.replace_residency({0, 1, Missing}, rejected) && partial.slot_of(0, 1) == Missing,
         "unclaimed slot became visible");

    const auto stable = host;
    auto lease = cache.prepare_scratch_loan(128);
    std::vector<std::pair<int, int>> lent;
    for (size_t i = 0; i < host.size(); ++i) if (host[i] >= 128) {
        lent.emplace_back(int(i), host[i]); host[i] = Missing;
    }
    need(lent.size() == 2, "tail loan extent");
    need(cache.mask_scratch_loan(lease, [&] { publish(); return true; }), "scratch mask");
    need(cache.activate_scratch_loan(lease, [&] { runtime->wait(); return true; }), "scratch activation");
    need(!cache.decode_ready(), "decode admitted during scratch loan");
    rejected.clear();
    need(!cache.replace_residency(stable, rejected), "adaptive publication during scratch loan");
    queue.memset(cache.device_slot(128), 0xa5, 2 * Blob); runtime->wait();
    need(cache.retire_scratch_loan(lease, [&] { runtime->wait(); return true; }), "scratch retirement");
    cache.begin_scratch_refill(lease);
    std::vector<std::vector<uint8_t>> refill;
    refill.reserve(lent.size());
    for (const auto& [index, target] : lent) {
        refill.push_back(payload(index / Experts, index % Experts));
        queue.memcpy(cache.device_slot(target), refill.back().data(), Blob);
    }
    runtime->wait();
    host = stable;
    need(cache.publish_scratch_refill(lease, [&] { publish(); return true; }), "scratch refill publication");
    need(cache.decode_ready() && cache.resident() == Slots, "scratch handoff/capacity changed");
    check_callbacks(); check_table();
    for (size_t i = 0; i < lent.size(); ++i)
        need(cache.verify_slot(lent[i].second, refill[i].data(), error), "refilled payload differs");
    const auto prefix = payload(1, 62);
    need(cache.verify_slot(127, prefix.data(), error), "unlent prefix changed");
    std::puts("residency_map=PASS pending_exclusion=PASS invalid_maps=6 loan_refill=PASS payload=PASS prefix=PASS");
}

int main(int argc, char** argv) {
    try {
        need(argc == 2 && (std::string(argv[1]) == "legacy" || std::string(argv[1]) == "fixed"), "legacy|fixed");
        gate("before-runtime");
        run(std::string(argv[1]) == "fixed");
        gate("before-exit");
        std::puts("fixture_status=PASS mode=cache-residency");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cache residency fixture: %s\n", error.what()); return 1;
    }
}
