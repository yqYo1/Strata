// Compile the extracted production registration/callback/removal code against
// a CPU verifier with a heap payload. No SYCL or Level Zero headers/runtime.
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
using DiagFn = void (*)(std::FILE*);
std::atomic<DiagFn>& diag_verify_fn() { static std::atomic<DiagFn> f{}; return f; }
std::atomic<DiagFn>& release_gpu_fn() { static std::atomic<DiagFn> f{}; return f; }
std::atomic<uint64_t> g_diag_reads{}, g_release_reads{};

struct Gate {
    bool block = false;
    std::atomic<bool> entered{}, allow{}, destroy_started{}, retired{}, deleted{};
    std::atomic<uint64_t> calls{};
};
class Verifier {
    Gate* gate_;
    int* payload_ = nullptr;
    void read_payload() const {
        Gate* gate = gate_;
        gate->entered.store(true);
        if (gate->block)
            while (!gate->allow.load()) std::this_thread::yield();
        // Old callbacks retain only a raw pointer. This access occurs after
        // deletion in the deterministic control; ASan must diagnose it.
        assert(payload_ != nullptr && *payload_ == 4242);
        gate->calls.fetch_add(1);
    }
public:
    explicit Verifier(Gate& gate) : gate_(&gate) {}
    bool init(bool valid);
    ~Verifier();
    void diag(std::FILE*) const { read_payload(); g_diag_reads.fetch_add(1); }
    bool release_gpu_waits(int) { read_payload(); g_release_reads.fetch_add(1); return true; }
};

#include "registry-under-test.hpp"

template<class Ready> void wait(Ready ready) {
    auto deadline = Clock::now() + std::chrono::seconds(5);
    while (!ready()) {
        assert(Clock::now() < deadline);
        std::this_thread::yield();
    }
}

void race(bool release) {
    Gate gate; gate.block = true;
    auto* verifier = new Verifier(gate);
    assert(verifier->init(true));
    std::thread callback([&] {
        if (release) release_live_verifiers(nullptr);
        else diag_active_verifier(nullptr);
    });
    wait([&] { return gate.entered.load(); });
    std::thread owner([&] {
        gate.destroy_started.store(true);
        delete verifier;
        gate.deleted.store(true);
    });
    wait([&] { return gate.destroy_started.load(); });
#if STRATA_REGISTRY_CONTROL
    wait([&] { return gate.deleted.load(); });
    std::cout << "CONTROL: object deleted with callback in flight\n" << std::flush;
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    assert(!gate.retired.load() && !gate.deleted.load());
#endif
    gate.allow.store(true);
    callback.join(); owner.join();
    assert(gate.calls.load() == 1 && gate.deleted.load());
    diag_active_verifier(nullptr); release_live_verifiers(nullptr);
    assert(gate.calls.load() == 1);
}

#if !STRATA_REGISTRY_CONTROL
void failed_init() {
    Gate gate;
    auto verifier = std::make_unique<Verifier>(gate);
    assert(!verifier->init(false));
    diag_active_verifier(nullptr); release_live_verifiers(nullptr);
    assert(gate.calls.load() == 0);
    assert(verifier->init(true));
    assert(!verifier->init(false)); // Failed retries also remove prior publication.
    diag_active_verifier(nullptr); release_live_verifiers(nullptr);
    assert(gate.calls.load() == 0);
}

void capacity() {
    Gate gates[17];
    std::vector<std::unique_ptr<Verifier>> owners;
    for (int i = 0; i != 17; ++i) {
        owners.push_back(std::make_unique<Verifier>(gates[i]));
        assert(owners.back()->init(true) == (i < 16));
    }
    release_live_verifiers(nullptr);
    for (int i = 0; i != 17; ++i) assert(gates[i].calls.load() == (i < 16 ? 1 : 0));
    owners[0].reset();
    assert(register_live_verifier(owners[15].get())); // Must not duplicate into the gap.
    release_live_verifiers(nullptr);
    assert(gates[15].calls.load() == 2);
    assert(owners[16]->init(true)); // The freed slot can be reused.
    diag_active_verifier(nullptr);
    assert(gates[16].calls.load() == 1);
    owners.clear();
    diag_active_verifier(nullptr); release_live_verifiers(nullptr);
}

void stress() {
    std::atomic<bool> done{};
    std::thread diagnostic([&] { while (!done.load()) diag_active_verifier(nullptr); });
    std::thread releaser([&] { while (!done.load()) release_live_verifiers(nullptr); });
    std::vector<std::thread> owners;
    for (int thread = 0; thread != 4; ++thread) owners.emplace_back([&] {
        for (int i = 0; i != 512; ++i) {
            Gate gate;
            auto verifier = std::make_unique<Verifier>(gate);
            assert(verifier->init(true));
            if (i == 0) wait([] { return g_release_reads.load() != 0; });
            diag_active_verifier(nullptr);
        }
    });
    for (auto& owner : owners) owner.join();
    done.store(true); diagnostic.join(); releaser.join();
    diag_active_verifier(nullptr); release_live_verifiers(nullptr);
    assert(g_diag_reads.load() >= 2048 && g_release_reads.load() > 0);
    std::cout << "2048 ownership cycles, " << g_diag_reads.load() << " diagnostic reads, "
              << g_release_reads.load() << " release reads\n";
}
#endif
} // namespace strata::core

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    using namespace strata::core;
    if (mode == "diag-race") race(false);
    else if (mode == "release-race") race(true);
#if !STRATA_REGISTRY_CONTROL
    else if (mode == "failed-init") failed_init();
    else if (mode == "capacity") capacity();
    else if (mode == "stress") stress();
#endif
    else return 2;
    std::cout << "PASS: " << mode << '\n';
}
