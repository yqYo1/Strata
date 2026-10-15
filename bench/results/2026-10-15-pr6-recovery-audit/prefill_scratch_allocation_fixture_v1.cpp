// Actual production Prefill allocation/relayout only. Session metadata describes
// physical 262144-cell staging; no model/session payload or inference is run.
#include "strata/prefill/prefill.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Owners {
    std::shared_ptr<strata::sycl_backend::Runtime> runtime;
    cudaStream_t stream = nullptr;
    std::unique_ptr<sycl::queue> unordered;
    strata::core::ModelGeometry geometry;
    strata::core::WeightTable weights;
    strata::core::SessionState session;
    std::vector<strata::core::QsaState> states;
    std::unique_ptr<strata::prefill::Prefill> prefill;
    uint8_t* borrowed = nullptr;
    std::array<uint8_t, 256> marker{}, got{};
};
void need(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
void retire_prefill(Owners& o) {
    checked(cudaStreamSynchronize(o.stream), "prefill retirement");
    o.prefill.reset();
    checked(cudaGetLastError(), "prefill release");
}
void canary(Owners& o, uint64_t offset, bool write) {
    o.marker.fill(0xb7);
    if (write)
        checked(cudaMemcpy(o.borrowed+offset, o.marker.data(), o.marker.size(), cudaMemcpyHostToDevice), "canary write");
    else {
        o.got.fill(0);
        checked(cudaStreamSynchronize(o.stream), "canary drain");
        checked(cudaMemcpy(o.got.data(), o.borrowed+offset, o.got.size(), cudaMemcpyDeviceToHost), "canary read");
        need(o.got == o.marker, "borrowed end canary changed");
    }
}
bool initialize(Owners& o, int64_t chunk, void* stream, uint64_t bytes, std::string& err) {
    return o.prefill->init(o.weights, o.geometry, o.session, nullptr, nullptr, nullptr,
                          chunk, stream, err, o.borrowed, bytes);
}
void run(Owners& o, const std::string& pack) {
    std::string err;
    need(strata::kernels::cpu::expert_layout_load(pack, 48, 512, err), err);
    o.runtime = strata::sycl_backend::runtime_for();
    checked(cudaStreamCreate(&o.stream), "stream create");
    o.session.max_cells = 262144;
    o.session.layer_hi = 48;
    o.session.qsa_alloc = 12;
    o.session.gdn_alloc = 36;
    o.states.resize(12);
    for (auto& q : o.states) {
        q.max_cells = 262144;
        q.n_pages = 65536;
        q.n_slots = 8192;
        q.kv_mode = 1;
        q.kv_int8 = true;
        q.idx_pooled_rows = 65538;
    }
    o.session.qsa_states = o.states.data();
    const auto n8 = strata::prefill::Prefill::bytes_needed(o.geometry, o.session, 8192);
    const auto n4 = strata::prefill::Prefill::bytes_needed(o.geometry, o.session, 4096);
    need(n8 == 2588395008ull && n4 == 1503447296ull, "actual alias sizing differs from reviewed carve accounting");
    std::cout << "allocation_plan C8192=" << n8 << " C4096=" << n4 << '\n';
    size_t before = 0, total = 0, allocated = 0, retired = 0;
    checked(cudaMemGetInfo(&before, &total), "free before");
    o.prefill = std::make_unique<strata::prefill::Prefill>();
    need(initialize(o, 8192, o.stream, 0, err), "owned init: "+err);
    need(o.prefill->chunk() == 8192, "owned actual chunk");
    checked(cudaMemGetInfo(&allocated, &total), "free allocated");
    retire_prefill(o);
    checked(cudaMemGetInfo(&retired, &total), "free retired");
    std::cout << "owned_cycle chunk=8192 free_before=" << before << " free_allocated=" << allocated
              << " free_retired=" << retired << " total=" << total << '\n';

    checked(cudaMalloc(&o.borrowed, size_t(n8+256)), "borrowed slab");
    canary(o, n8, true);
    o.prefill = std::make_unique<strata::prefill::Prefill>();
    need(initialize(o, 8192, o.stream, n8, err), "borrowed exact init: "+err);
    need(o.prefill->chunk() == 8192, "borrowed actual chunk");
    canary(o, n8, false);
    canary(o, n4, true);
    need(o.prefill->relayout(4096, o.borrowed, n4, err), "borrowed relayout: "+err);
    need(o.prefill->chunk() == 4096, "relayout actual chunk");
    canary(o, n4, false); canary(o, n8, false);
    retire_prefill(o);
    std::cout << "borrowed_cycle init8192=PASS relayout4096=PASS both_end_canaries=PASS\n";

    o.prefill = std::make_unique<strata::prefill::Prefill>();
    need(!initialize(o, 4096, o.stream, n4-1, err) &&
         err.find("do not fit") != std::string::npos, "one-byte-short loan admitted or wrong failure");
    canary(o, n4, false); canary(o, n8, false);
    retire_prefill(o);
    std::cout << "borrowed_short_loan rejection=PASS both_end_canaries=PASS\n";
    checked(cudaFree(o.borrowed), "borrowed slab free"); o.borrowed = nullptr;

    o.prefill = std::make_unique<strata::prefill::Prefill>();
    need(!initialize(o, 4096, nullptr, 0, err) && err.find("explicit in-order queue") != std::string::npos,
         "null queue admitted");
    retire_prefill(o);
    o.unordered = std::make_unique<sycl::queue>(o.runtime->compute().get_context(), o.runtime->compute().get_device());
    need(!o.unordered->has_property<sycl::property::queue::in_order>(), "negative queue unexpectedly in order");
    o.prefill = std::make_unique<strata::prefill::Prefill>();
    need(!initialize(o, 4096, o.unordered.get(), 0, err) && err.find("in-order queue") != std::string::npos,
         "out-of-order queue admitted");
    retire_prefill(o);
    o.unordered->wait_and_throw(); o.unordered.reset();
    o.prefill = std::make_unique<strata::prefill::Prefill>();
    o.prefill->set_stage(0, 24, nullptr);
    need(!initialize(o, 4096, o.stream, 0, err) && err.find("full unsplit model") != std::string::npos,
         "split topology admitted");
    retire_prefill(o);
    checked(cudaStreamSynchronize(o.stream), "final drain");
    checked(cudaStreamDestroy(o.stream), "stream destroy"); o.stream = nullptr;
    std::cout << "scratch_allocation PASS owned borrowed relayout short_loan null_queue unordered_queue split no_inference\n";
}
} // namespace
int main(int argc, char** argv) {
    // All GPU/DMA owners live on this heap through any failed check. A failure
    // exits without RAII/atexit cleanup of potentially unretired resources.
    auto* owners = new Owners;
    try {
        need(argc == 2, "expected pack metadata directory");
        need(std::getenv("STRATA_SYCL_PREFILL_SCRATCH_ALIAS") &&
             std::string(std::getenv("STRATA_SYCL_PREFILL_SCRATCH_ALIAS")) == "1", "explicit alias arm required");
        run(*owners, argv[1]);
        delete owners;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "scratch allocation fail-stop: %s\n", e.what());
        std::fflush(stderr);
        std::_Exit(88);
    }
}
