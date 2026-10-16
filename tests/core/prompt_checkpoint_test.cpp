#include "strata/program/prompt_checkpoint.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void check(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::abort(); }
}
} // namespace

int main() {
    constexpr int64_t turn = 100, system = 101, user = 102, assistant = 103;
    const std::vector<int64_t> ids = {
        turn, system, 4, turn, user, 5, turn, system, 6, turn, assistant};
    using strata::program::checkpoint_turn;
    check(checkpoint_turn(ids, 0, turn, -1) == 9, "default checkpoint changed");
    check(checkpoint_turn(ids, 0, turn, system) == 6, "effort turn entered reusable state");
    check(checkpoint_turn(ids, 0, turn, user) == 9, "an older role was incorrectly excluded");
    check(checkpoint_turn(ids, 6, turn, system) == 9, "resumed prefix was scanned backwards");
    check(checkpoint_turn(ids, 9, turn, system) == -1, "checkpoint repeated at resume boundary");
    check(checkpoint_turn(ids, 11, turn, system) == -1, "empty suffix created a checkpoint");
    check(checkpoint_turn(ids, 0, -1, system) == -1, "disabled turn marker selected state");
    check(checkpoint_turn({}, 0, turn, system) == -1, "empty prompt read out of bounds");
    check(ids[8] == 6 && ids.size() == 11, "checkpoint selection changed prompt tokens");
    for (const auto resume : {-1, 12}) {
        bool rejected = false;
        try { (void)checkpoint_turn(ids, resume, turn, system); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "invalid resume was accepted");
    }
    std::vector<int64_t> full(262144, 0);
    full[0] = turn;
    full[262140] = turn;
    full[262141] = system;
    full[262143] = turn;
    check(checkpoint_turn(full, 0, turn, system) == 262140,
          "full-context final turn accessed the wrong cell");
    check(checkpoint_turn(full, 262140, turn, system) == 262143,
          "near-boundary resume crossed the saved state");
    std::puts("effort checkpoint role, reuse boundary and 262144-token host indexing: PASS");
    return 0;
}
