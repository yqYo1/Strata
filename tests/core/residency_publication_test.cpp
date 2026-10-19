#include "strata/core/residency_publication.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

// The two snapshots represent the same cache slot, first masked while its
// bytes are scratch, then restored only after its expert has been refilled.
// Failure on an early or later device must not authorize a dependent reader.
void transition(bool refill, int fail_at) {
    using Table = std::array<int32_t, 4>;
    const Table resident{0, -1, 7, 3}, lent{0, -1, -1, 3};
    const auto& before = refill ? lent : resident;
    const auto& after = refill ? resident : lent;
    std::array<Table, 3> devices{before, before, before};
    std::size_t attempted = 0;
    bool dependent_work = false;
    std::string err;
    const char* phase = refill ? "cache refill" : "cache lend";
    const bool ok = strata::core::publish_residency_tables(devices.size(), phase,
        [&](std::size_t target) {
            ++attempted;
            if (static_cast<int>(target) == fail_at) return 71;
            devices[target] = after;
            return 0;
        }, err);
    if (ok) dependent_work = true;
    require(ok == (fail_at < 0) && dependent_work == ok, "failed publication authorized a reader");
    const auto copied = fail_at < 0 ? devices.size() : static_cast<std::size_t>(fail_at);
    require(attempted == copied + (fail_at < 0 ? 0 : 1), "copy continued after failure");
    for (std::size_t i = 0; i < devices.size(); ++i)
        require(devices[i] == (i < copied ? after : before), "unexpected residency snapshot");
    if (ok) require(err.empty(), "successful publication returned an error");
    else require(err.find(phase) != std::string::npos &&
                 err.find("table " + std::to_string(fail_at)) != std::string::npos &&
                 err.find("status 71") != std::string::npos, "lost copy failure context");
    std::printf("residency_publication phase=%s fail_at=%d status=PASS\n", refill ? "refill" : "lend", fail_at);
}
} // namespace

int main() {
    for (const bool refill : {false, true})
        for (int fail_at = -1; fail_at < 3; ++fail_at) transition(refill, fail_at);
    std::string err;
    require(strata::core::publish_residency_tables(0, "no tables", [](std::size_t) {
        require(false, "no-table path attempted an upload"); return 1;
    }, err) && err.empty(), "empty publication failed");
    std::puts("residency_publication phase=empty status=PASS");
    std::puts("residency_publication_fixture cases=9 status=PASS");
}
