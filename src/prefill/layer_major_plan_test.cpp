#include "strata/prefill/layer_major_plan.hpp"
#include "strata/prefill/draft_uploads.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#if defined(__unix__)
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::prefill::detail;

static void check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
template<class F> static void rejected(F f, const char* message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    check(threw, message);
}

int main() {
    constexpr uint64_t row_bytes = 10240 * sizeof(float);
    const auto full = LayerMajorResidualPlan::make(262144, 8192, 10240, 0, 0);
    check(full.total_bytes == 10737418240ull && full.host_bytes == full.total_bytes && !full.gpu_bytes,
          "full 256K uses 10 GiB of host residual with zero GPU budget");
    const auto mixed = LayerMajorResidualPlan::make(19, 4, 10240, 11 * row_bytes, -1);
    check(mixed.gpu_rows == 8 && mixed.host_bytes == 11 * row_bytes, "auto prefix rounds to whole chunks");
    check(mixed.span(4, 4).device && mixed.span(4, 4).offset_bytes == 4 * row_bytes,
          "GPU prefix has prompt-relative offsets");
    check(!mixed.span(8, 4).device && mixed.span(8, 4).offset_bytes == 0,
          "host tail begins at zero without unsigned offset underflow");
    check(mixed.span(16, 3).offset_bytes == 8 * row_bytes && mixed.span(16, 3).bytes == 3 * row_bytes,
          "partial final host chunk retains exact extent");
    const auto all_gpu = LayerMajorResidualPlan::make(19, 4, 10240, 19 * row_bytes, 19);
    check(all_gpu.gpu_rows == 19 && all_gpu.host_bytes == 0 && all_gpu.span(16, 3).device,
          "complete prompt prefix can include its partial final chunk");
    const auto one = LayerMajorResidualPlan::make(1, 8192, 10240, row_bytes, -1);
    check(one.gpu_rows == 1 && one.span(0, 1).bytes == row_bytes, "single short chunk fits exactly");
    check(LayerMajorResidualPlan::make(19, 4, 10240, 19 * row_bytes, 11).gpu_rows == 8,
          "explicit maximum rows round to a chunk boundary");
    rejected([&] { LayerMajorResidualPlan::make(19, 4, 10240, 7 * row_bytes, 8); }, "explicit prefix over budget fails");
    rejected([&] { LayerMajorResidualPlan::make(19, 0, 10240, 0, 0); }, "zero chunk fails");
    rejected([&] { LayerMajorResidualPlan::make(19, 4, 10240, 0, 20); }, "prefix beyond prompt fails");
    rejected([&] { LayerMajorResidualPlan::make(INT64_MAX, 4, 10240, 0, 0); }, "full residual multiplication overflow fails");
    rejected([&] { layer_major_add(INT64_MAX, 1); }, "signed pointer extent overflow fails");
    rejected([&] { mixed.span(-4, 4); }, "negative row fails");
    rejected([&] { mixed.span(6, 4); }, "nonchunk row fails");
    rejected([&] { mixed.span(16, 4); }, "tail overrun fails");
    rejected([&] { mixed.span(20, 1); }, "past-end row fails");
    check(layer_major_requested_rows(nullptr) == -1 && layer_major_requested_rows("auto") == -1 &&
          layer_major_requested_rows("0") == 0 && layer_major_requested_rows("262144") == 262144,
          "strict row option supports auto, host only, explicit decimal");
    for (const char* s : {"", "-1", "+1", " 1", "1 ", "1.0", "1x", "AUTO", "18446744073709551616"})
        rejected([&] { layer_major_requested_rows(s); }, "invalid row option fails");

    check(layer_major_residual_dtype(nullptr) == LayerMajorResidualDtype::fp32 &&
          layer_major_residual_dtype("fp32") == LayerMajorResidualDtype::fp32 &&
          layer_major_residual_dtype("fp16") == LayerMajorResidualDtype::fp16 &&
          layer_major_residual_dtype("bf16") == LayerMajorResidualDtype::bf16,
          "dtype defaults to FP32 and accepts explicit FP32/FP16/BF16");
    for (const char* value : {"", "FP16", "BF16", "fp8", " fp16", "fp16 "})
        rejected([&] { layer_major_residual_dtype(value); }, "invalid storage dtype rejected");
    rejected([&] { layer_major_storage_bytes(static_cast<LayerMajorResidualDtype>(99)); },
             "invalid typed format rejected");
    const auto bf = LayerMajorResidualPlan::make(262144, 8192, 10240, 8192ull * 10240 * 2,
                                               8192, LayerMajorResidualDtype::bf16);
    check(bf.total_bytes == 5368709120ull && bf.gpu_rows == 8192 && bf.staging_bytes == 0 &&
          bf.host_bytes == 5368709120ull - bf.gpu_bytes &&
          bf.span(253952, 8192).offset_bytes + bf.span(253952, 8192).bytes == bf.host_bytes,
          "native BF16 has direct typed chunk IO, no extra staging, and exact physical262144 last span");
    for (const auto dtype : {LayerMajorResidualDtype::fp32, LayerMajorResidualDtype::fp16}) {
        const bool half = dtype == LayerMajorResidualDtype::fp16;
        const uint64_t rb = 10240 * layer_major_storage_bytes(dtype);
        const uint64_t reserve = half ? 8192 * rb : 0;
        const uint64_t full_bytes = half ? 5368709120ull : 10737418240ull;
        const auto host = LayerMajorResidualPlan::make(262144, 8192, 10240, reserve, 0, dtype);
        check(host.total_bytes == full_bytes && host.host_bytes == full_bytes &&
              host.staging_bytes == reserve, "262144 physical rows: 10/5 GiB plus exact staging");
        const auto exact = LayerMajorResidualPlan::make(262144, 8192, 10240, full_bytes, -1, dtype);
        check(exact.gpu_rows == 262144 && !exact.host_bytes && !exact.staging_bytes,
              "exact full-store budget needs no host staging");
        const auto below = LayerMajorResidualPlan::make(262144, 8192, 10240, full_bytes - 1, -1, dtype);
        check(below.gpu_rows == (half ? 245760 : 253952) &&
              below.gpu_bytes + below.staging_bytes <= full_bytes - 1,
              "one byte below physical boundary reserves staging before auto prefix");
        const auto prefix = LayerMajorResidualPlan::make(262144, 8192, 10240,
            2 * 8192 * rb + reserve, -1, dtype);
        check(prefix.gpu_rows == 16384 && prefix.span(16384, 8192).offset_bytes == 0 &&
              prefix.span(253952, 8192).bytes == 8192 * rb,
              "physical boundary mixed store has exact first/last host spans");
        const uint64_t odd_stage = half ? 257 * rb : 0;
        const auto odd = LayerMajorResidualPlan::make(513, 257, 10240,
            257 * rb + odd_stage, 257, dtype);
        check(odd.gpu_rows == 257 && odd.host_bytes == 256 * rb &&
              odd.span(257, 256).offset_bytes == 0 && odd.span(257, 256).bytes == 256 * rb &&
              odd.staging_bytes == (half ? 256 * rb : 0), "257/256 split uses exact partial-tail staging");
        const auto main_fixture = LayerMajorResidualPlan::make(512, 257, 10240,
            512 * rb, 257, dtype);
        check(main_fixture.gpu_rows == 257 && main_fixture.host_bytes == 255 * rb &&
              main_fixture.span(257, 255).offset_bytes == 0 &&
              main_fixture.staging_bytes == (half ? 255 * rb : 0),
              "Main's 512 batched rows use exact 257/255 prefix/tail spans");
        const auto final_gpu = LayerMajorResidualPlan::make(19, 4, 10240, 19 * rb, 19, dtype);
        check(final_gpu.span(16, 3).device && final_gpu.span(16, 3).bytes == 3 * rb &&
              !final_gpu.host_bytes && !final_gpu.staging_bytes,
              "explicit all-GPU odd tail has no staging in either format");
        const uint64_t odd_budget = 4 * rb + (half ? 4 * rb : 0);
        const auto tail = LayerMajorResidualPlan::make(19, 4, 10240, odd_budget, 4, dtype);
        check(!tail.span(16, 3).device && tail.span(16, 3).offset_bytes == 12 * rb &&
              tail.span(16, 3).bytes == 3 * rb, "odd chunk count preserves host-relative final offset");
        rejected([&] { LayerMajorResidualPlan::make(19, 4, 10240, odd_budget - 1, 4, dtype); },
                 "explicit prefix budget includes full staging extent");
        rejected([&] { LayerMajorResidualPlan::make(INT64_MAX, 4, 10240, 0, 0, dtype); },
                 "total multiplication checked for either format");
    }
    rejected([&] { LayerMajorResidualPlan::make(19, 4, 10240, 4 * 10240 * 2 - 1, -1,
                                              LayerMajorResidualDtype::fp16); },
             "host-only FP16 still needs one complete conversion staging chunk");
    const auto partial = LayerMajorResidualPlan::make(19, 4, 10240, 19 * 10240 * 2, 16,
                                                     LayerMajorResidualDtype::fp16);
    check(partial.gpu_rows == 16 && partial.staging_bytes == 3 * 10240 * 2,
          "explicit final partial host chunk fits prefix plus exact small staging");

    LayerMajorChunkCursor cursor{19, 4};
    int64_t delivered = 0;
    for (int layer = 0; layer < 48; ++layer) {
        cursor = {19, 4};
        for (int64_t row = 0; row < 19; row += 4) {
            cursor.accept(row, std::min<int64_t>(4, 19 - row));
            if (layer == 47) ++delivered;
        }
        check(cursor.complete() && cursor.calls == 5, "each layer accepts five ordered chunks once");
    }
    check(delivered == 5, "final-layer delivery count is logical chunks, not layers times chunks");
    rejected([&] { cursor.accept(16, 3); }, "duplicate final chunk fails");
    cursor = {19, 4};
    rejected([&] { cursor.accept(4, 4); }, "out-of-order first chunk fails");

    std::vector<int> destruction;
    struct Owner { std::vector<int>& record; ~Owner() { record.push_back(2); } };
    try {
        Owner owner{destruction};
        DraftUploadGuard guard([&] { destruction.push_back(1); return true; }, [] { std::abort(); });
        guard.arm();
        throw std::runtime_error("simulated callback exception");
    } catch (const std::runtime_error&) {}
    check(destruction == std::vector<int>({1, 2}), "callback exception drains before owners are destroyed");
    int drained = 0;
    {
        DraftUploadGuard guard([&] { ++drained; return true; }, [] { std::abort(); });
        guard.arm(); guard.retire();
    }
    check(drained == 0, "explicit retirement requires no second destructor drain");
#if defined(__unix__)
    const pid_t child = fork();
    check(child >= 0, "fail-stop fixture fork succeeds");
    if (child == 0) {
        bool owner_live = true;
        struct LiveOwner { bool& live; ~LiveOwner() { live = false; } } owner{owner_live};
        DraftUploadGuard guard([] { return false; }, [&] { _Exit(owner_live ? 83 : 84); });
        guard.arm();
        // Destruction must enter the fatal path while the owner remains live.
        // Leaving this scope invokes guard before owner.
    } else {
        int status = 0;
        check(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 83,
              "failed drain fail-stops while owner is live");
    }
#endif
    std::cout << "layer-major host arithmetic/order/lease fixtures PASS (no kernels)\n";
}
