// Original Strata fixture; repository LICENSE applies. No model inference.
#include "strata/prefill/prefill.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/mtp.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace core = strata::core;
using strata::prefill::Prefill;
std::shared_ptr<sb::Runtime> runtime;
constexpr size_t guard = 256;
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void check(cudaError_t status) { need(status == cudaSuccess, cudaGetErrorString(status)); }
void passed(const char* name) {
    std::cout << "prefill_scratch_device case=" << name << " status=PASS\n" << std::flush;
}
struct Session {
    core::ModelGeometry geometry;
    core::SessionState state;
    sb::Allocation allocation;
    void init() {
        const auto bytes = core::session_bytes(geometry, 1024, 10);
        need(bytes > 0 && bytes <= (512ull << 20), "fixed session budget");
        allocation = sb::Allocation(runtime, size_t(bytes), sb::MemoryKind::Device);
        need(core::session_init(geometry, 1024, 10, allocation.data(), state) != 0, "session init");
    }
    ~Session() noexcept {
        sb::checked_cleanup([&] { runtime->wait(); }, [&] {
            core::session_release(state);
            for (int64_t i = 0; state.qsa_states && i < geometry.n_qsa_layers(); ++i) {
                auto& q = state.qsa_states[i];
                if (q.host_step) check(cudaFreeHost(q.host_step));
                if (q.host_pos) check(cudaFreeHost(q.host_pos));
            }
            delete[] state.qsa_states;
            state.qsa_states = nullptr;
        }, "prefill fixture session retirement");
    }
};
struct Parent {
    sb::Allocation allocation;
    size_t bytes;
    explicit Parent(size_t n) : allocation(runtime, n + 2 * guard, sb::MemoryKind::Device), bytes(n) {
        need(n <= (512ull << 20) && uintptr_t(data()) % guard == 0, "parent size/alignment");
        runtime->compute().memset(allocation.data(), 0xa5, allocation.size());
        runtime->wait();
    }
    unsigned char* data() const { return allocation.as<unsigned char>() + guard; }
    void unchanged() const {
        // Every byte, including outer guards, must remain untouched by layout
        // construction. No tensor/session payload is retained outside this job.
        std::vector<unsigned char> observed(std::min<size_t>(allocation.size(), 8u << 20));
        for (size_t at = 0; at < allocation.size(); at += observed.size()) {
            const auto n = std::min(observed.size(), allocation.size() - at);
            try {
                runtime->compute().memcpy(observed.data(), allocation.as<unsigned char>() + at, n);
                runtime->wait();
            } catch (...) { sb::cleanup_fail_stop("prefill fixture readback did not retire"); }
            need(std::all_of(observed.begin(), observed.begin() + n,
                [](auto byte) { return byte == 0xa5; }), "parent or guard was written by layout");
        }
    }
};
size_t exact_bytes(const Session& s, int64_t rows) {
    // Alias1/MMQ0 uses the same production take sequence in bytes_needed/init.
    // Subtract ONLY the explicit sizing slack, retaining every take's guard.
    const auto need_bytes = Prefill::bytes_needed(s.geometry, s.state, rows);
    need(need_bytes > (8u << 20) && need_bytes <= (512ull << 20), "fixed prefill budget");
    return size_t(need_bytes - (8u << 20));
}
bool init(Prefill& p, Session& s, core::WeightTable& weights, sycl::queue& queue,
          void* base, size_t bytes, std::string& error) {
    return p.init(weights, s.geometry, s.state, nullptr, nullptr, nullptr, 256,
                  &queue, error, base, bytes);
}
void poisoned(Prefill& p) {
    std::string error;
    need(p.chunk() == 0, "poison reports a runnable chunk");
    need(!p.run(nullptr, 1, 0, error) && error.find("layout unavailable") != std::string::npos,
         "poison reached run");
    core::MtpDrafter mtp;
    error.clear();
    need(!p.draft_kv(mtp, nullptr, nullptr, 1, 0, error) &&
         error.find("layout unavailable") != std::string::npos, "poison reached draft KV");
    error.clear();
    need(!p.relayout(128, nullptr, 1, error) && error.find("layout unavailable") != std::string::npos,
         "poison reached relayout");
}
void run(const std::string& mode) {
    Session session; session.init();
    core::WeightTable weights;
    auto queue = runtime->make_queue(true);
    const size_t full = exact_bytes(session, 256), small = exact_bytes(session, 128);
    Parent a(full), b(full);
    std::string error;
    if (mode == "smoke") {
        Prefill p;
        need(!init(p,session,weights,queue,a.data(),(64u << 20)-1,error) &&
             error.find("GEMM scratch does not fit") != std::string::npos, "first take refusal");
        poisoned(p); passed("first_take_refusal");
        a.unchanged(); b.unchanged();
        std::cout << "prefill_scratch_device_fixture mode=smoke cases=1 status=PASS\n";
        return;
    }
    if (mode == "retirement") {
        Prefill p; need(init(p,session,weights,queue,a.data(),full,error), "retirement fixture init");
        check(cudaStreamSynchronize(&queue));
        // Controlled host-domain rejection, not a stalled/invalid GPU command.
        // The external queue object stays alive; no work remains on its old
        // value. Replacing it violates Prefill's queue-ownership contract.
        sycl::context other(runtime->device());
        queue = sycl::queue(other, runtime->device());
        std::cout << "prefill_scratch_device retirement=reset queue_domain_changed=1 owners_live=1\n" << std::flush;
        p.reset(); // must _Exit before releasing resources after failed sync
        std::cout << "prefill_scratch_device retirement=UNEXPECTED_RETURN\n" << std::flush;
        std::_Exit(99);
    }
    need(mode == "full", "unknown case");
    {
        Prefill p;
        need(init(p,session,weights,queue,a.data(),full,error) && p.chunk() == 256, "exact init");
        passed("exact_init");
        need(p.relayout(256,b.data(),full,error) && p.chunk() == 256, "moved layout");
        passed("moved_relayout");
        need(p.relayout(128,a.data(),small,error) && p.chunk() == 128, "smaller layout");
        passed("smaller_relayout");
        need(!p.relayout(257,b.data(),full,error) && p.chunk() == 128, "invalid growth changed valid layout");
        passed("growth_refusal");
    }
    {
        Prefill p;
        need(!init(p,session,weights,queue,a.data(),full-1,error) &&
            error.find("do not fit") != std::string::npos, "one byte short was admitted");
        passed("one_byte_short_poison"); poisoned(p); passed("poison_entries");
        need(!init(p,session,weights,queue,a.data(),full,error), "init bypassed reset");
        p.reset();
        need(init(p,session,weights,queue,a.data(),full,error), "reset failed after checked retirement");
        passed("reset_recovery");
        need(!p.relayout(128,b.data(),small-1,error), "short relayout was admitted");
        poisoned(p); passed("relayout_short_poison");
    }
    {
        Prefill p;
        need(!init(p,session,weights,queue,a.data()+2,full,error) &&
            error.find("GEMM") != std::string::npos, "GEMM init alignment refusal");
        poisoned(p); passed("gemm_init_refusal");
    }
    {
        Prefill p; need(init(p,session,weights,queue,a.data(),full,error), "rebind fixture init");
        bool threw = false;
        try { (void)p.relayout(128,b.data()+2,small,error); }
        catch (const std::invalid_argument&) { threw = true; }
        need(threw, "GEMM rebind did not refuse alignment"); poisoned(p); passed("gemm_rebind_refusal");
    }
    {
        auto bad = session.state; bad.qsa_states = nullptr;
        bool threw = false;
        try { (void)Prefill::bytes_needed(session.geometry,bad,256); }
        catch (const std::invalid_argument&) { threw = true; }
        need(threw, "planner accepted missing QSA state");
        Prefill p;
        need(!p.init(weights,session.geometry,bad,nullptr,nullptr,nullptr,256,&queue,error,a.data(),full),
            "init accepted missing QSA state");
        poisoned(p); passed("malformed_qsa_refusal");
    }
    {
        Prefill p;
        need(!init(p,session,weights,queue,a.data(),a.allocation.size(),error), "oversize parent span admitted");
        poisoned(p); passed("parent_bounds_refusal");
        sb::Allocation host(runtime,512,sb::MemoryKind::Host);
        p.reset();
        need(!init(p,session,weights,queue,host.data(),host.size(),error), "host USM admitted as scratch");
        poisoned(p); passed("host_usm_refusal");
        p.reset();
        sycl::context other(runtime->device()); sycl::queue foreign(other,runtime->device());
        need(!init(p,session,weights,foreign,a.data(),full,error) &&
             error.find("runtime domain") != std::string::npos, "foreign queue admitted");
        poisoned(p); passed("foreign_queue_refusal");
    }
    a.unchanged(); b.unchanged(); passed("canary_and_retirement");
    std::cout << "prefill_scratch_device_fixture mode=full cases=15 status=PASS\n";
}
}
int main(int argc, char** argv) try {
    need(argc == 3 && std::string(argv[1]) == "--case", "use --case smoke/full/retirement");
    runtime = sb::runtime_for();
    run(argv[2]);
    runtime->wait();
    std::cout << "prefill_scratch_device_complete=1 model_PP=UNRUN\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "prefill scratch fixture: " << error.what() << '\n'; return 1;
}
