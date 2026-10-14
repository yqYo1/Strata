// NOT RUN. CPU-only Linux fixture for Main after the active-run/review boundary.
// Build after applying the proposal, outside any GPU measurement:
// c++ -std=c++17 -pthread -Iinclude PATH/cpu_affinity_fixture.cpp -o PRIVATE/fixture
// Run normally and under a restricted taskset chosen from actually allowed CPUs.
#include "strata/kernels/cpu/stager_affinity.hpp"
#include <cstdio>
#include <thread>

using namespace strata::kernels::cpu::stager_affinity;

int main() {
    uint64_t original = 0;
    const int capture_rc = read_mask(original);
    if (capture_rc) {
        std::printf("SKIP bounded allowed-mask capture rc=%d\n", capture_rc);
        return 77;
    }
    // Exactly two CPUs form this fixture's permitted set; no host-wide assumption.
    uint64_t permitted = 0;
    for (int i = 0; i < 64; ++i) {
        if (!(original & (uint64_t{1} << i))) continue;
        permitted |= uint64_t{1} << i;
        if ((permitted & (permitted - 1)) != 0) break;
    }
    if ((permitted & (permitted - 1)) == 0) {
        std::puts("SKIP need two allowed CPUs");
        return 77;
    }
    struct Restore {
        uint64_t mask;
        ~Restore() {
            uint64_t effective = 0;
            if (apply_mask(mask, effective)) std::fputs("FAIL fixture teardown restore\n", stderr);
        }
    } restore{original};
    uint64_t effective = 0;
    if (apply_mask(permitted, effective) || effective != permitted) return 1;
    uint64_t before_pin = 0;
    if (read_mask(before_pin) || before_pin != permitted) return 2;
    const uint64_t pinned = permitted & (~permitted + 1);
    if (apply_mask(pinned, effective) || effective != pinned) return 3;

    uint64_t inherited_default = 0;
    int default_rc = -1;
    std::thread default_worker([&] { default_rc = read_mask(inherited_default); });
    default_worker.join();
    if (default_rc || inherited_default != pinned) return 4;

    uint64_t inherited_opt_in = 0, restored_worker = 0;
    int inherited_rc = -1, worker_rc = -1;
    std::thread opt_in_worker([&] {
        inherited_rc = read_mask(inherited_opt_in);
        worker_rc = apply_mask(before_pin, restored_worker);
    });
    opt_in_worker.join();
    if (inherited_rc || inherited_opt_in != pinned || worker_rc || restored_worker != permitted) return 5;
    if ((restored_worker & ~original) != 0) return 6;
    uint64_t parent_after = 0;
    if (read_mask(parent_after) || parent_after != pinned) return 7;
    if (apply_mask(0, effective) != EINVAL) return 8;
    if (read_mask(parent_after) || parent_after != pinned) return 9;
    // Restore explicitly so restoration errors affect exit status.
    if (apply_mask(original, effective) || effective != original) return 10;
    std::printf("PASS inheritance/worker-restore/permitted-mask/parent-unchanged original=0x%llx permitted=0x%llx pin=0x%llx\n",
                (unsigned long long) original, (unsigned long long) permitted, (unsigned long long) pinned);
    return 0;
}
