#include "strata/prefill/draft_uploads.hpp"
#include <array>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <csignal>
#include <cstdio>
#include <numeric>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

using strata::prefill::detail::DraftHostUploads;
using strata::prefill::detail::DraftUploadGuard;

// Retain raw pointers and consume them only at drain, as an async upload does.
struct DeferredUpload {
    const int32_t* tokens;
    const int32_t* bounds;
    int64_t first, count;
};

void deferred_batches(int64_t n, int64_t first, int64_t batch, bool q8) {
    std::vector<int32_t> input(size_t(n), 0);
    std::iota(input.begin(), input.end(), 100);
    DraftHostUploads uploads(input.data(), n, first, batch, 4, q8);
    std::vector<DeferredUpload> pending;
    int drains = 0;
    auto drain = [&] {
        ++drains;
        for (const auto& p : pending) {
            for (int64_t i = 0; i < p.count; ++i) assert(p.tokens[i] == 100 + p.first + i);
            if (q8) {
                assert(p.bounds[0] == 0 && p.bounds[1] == p.count);
                assert(p.bounds[2] == 0 && p.bounds[3] == p.count * 4);
            }
        }
        return true;
    };
    {
        DraftUploadGuard lease(drain, [] { std::abort(); });
        for (int64_t b = first; b < n; b += batch) {
            lease.arm();
            pending.push_back({uploads.tokens(b), q8 ? uploads.bounds(b) : nullptr,
                               b, std::min(batch, n - b)});
        }
        std::fill(input.begin(), input.end(), -1);
        assert(drains == 0);
        assert(drain()); // The caller's existing successful final wait.
        lease.retire();
    }
    assert(drains == 1); // No additional healthy-path wait.
}

void early_exit_and_exception() {
    std::vector<int32_t> tokens(131, 17);
    DraftHostUploads uploads(tokens.data(), tokens.size(), 0, 64, 4, true);
    const int32_t* pending = uploads.tokens(0);
    int drains = 0;
    auto drain = [&] { ++drains; assert(pending[0] == 17); return true; };
    auto early_return = [&] { DraftUploadGuard lease(drain, [] { std::abort(); }); lease.arm(); };
    early_return();
    assert(drains == 1);
    try {
        DraftUploadGuard lease(drain, [] { std::abort(); });
        lease.arm();
        throw std::runtime_error("injected submission exception");
    } catch (const std::runtime_error&) {}
    assert(drains == 2);
    { DraftUploadGuard unarmed(drain, [] { std::abort(); }); }
    assert(drains == 2);
}

void failed_drain(bool throws, bool fatal_returns) {
    const pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        std::vector<int32_t> tokens(128, 23);
        DraftHostUploads uploads(tokens.data(), tokens.size(), 0, 64, 4, true);
        const int32_t* pending = uploads.tokens(0);
        DraftUploadGuard lease(
            [&]() -> bool { if (throws) throw std::runtime_error("sticky error"); return false; },
            [&] { assert(pending[0] == 23 && uploads.bounds(64)[3] == 256);
                  if (!fatal_returns) _exit(73); });
        lease.arm();
        return; // Guard runs before upload storage destruction.
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    if (fatal_returns) assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    else assert(WIFEXITED(status) && WEXITSTATUS(status) == 73);
}

int main() {
    const rlimit no_core{0, 0};
    assert(setrlimit(RLIMIT_CORE, &no_core) == 0);
    deferred_batches(4096, 0, 2304, true);
    deferred_batches(33214, 32768, 128, true);
    deferred_batches(262144, 0, 2304, true);
    deferred_batches(129, 0, 64, false);
    deferred_batches(64, 0, 64, true);
    early_exit_and_exception();
    failed_drain(false, false);
    failed_drain(true, false);
    failed_drain(false, true);
    int32_t token = 0;
    for (const auto& args : {std::array<int64_t, 4>{0, 0, 64, 4}, {1, 0, 0, 4},
                            {1, -1, 64, 4}, {1, 0, 64, 0}, {1, 0, 64, INT64_MAX}}) {
        bool rejected = false;
        try { DraftHostUploads bad(&token, args[0], args[1], args[2], args[3], true); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    DraftHostUploads one(&token, 1, 0, 64, 4, false);
    for (int64_t position : {-1, 1, 64}) {
        bool rejected = false;
        try { (void)one.tokens(position); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    bool rejected = false;
    try { (void)one.bounds(0); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::puts("draft upload ownership: deferred full/partial/256K, early exit/exception, failed drain and bounds passed");
}
