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
