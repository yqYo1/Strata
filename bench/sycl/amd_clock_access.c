/* Main-authored host-only capability pilot; repository LICENSE.
 * AMD APM24594 revision3.36 (March2024), RDPRU entry; no Linux source copied.
 * Primary AMD manual mirrored at:
 * https://www.scs.stanford.edu/~zyedidia/docs/x86/amd-manual-v3.pdf
 * Consumer: bounded host qualification AFTER the current validation-cost cohort.
 * No GPU/runtime API. This source is unbuilt/unqualified until its pinned plan.
 * Report APERF/MPERF ratios only: P0 and SMT/counter scope remain unqualified.
 */
#define _GNU_SOURCE
#include <cpuid.h>
#include <inttypes.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

struct snapshot { uint64_t before, actual, after; int cpu; };

static void serialize(void) {
    unsigned a, b, c, d;
    __asm__ volatile("" ::: "memory");
    __cpuid_count(0, 0, a, b, c, d);
    __asm__ volatile("" ::: "memory");
}

static uint64_t read_register(unsigned index) {
    unsigned lo, hi;
    serialize();
    __asm__ volatile(".byte 0x0f,0x01,0xfd" : "=a"(lo), "=d"(hi) : "c"(index) : "memory");
    serialize();
    return ((uint64_t)hi << 32) | lo;
}

static struct snapshot snapshot(void) {
    struct snapshot value;
    value.cpu = sched_getcpu();
    value.before = read_register(0);
    value.actual = read_register(1);
    value.after = read_register(0);
    if (value.cpu != sched_getcpu()) value.cpu = -1;
    return value;
}

static uint64_t nanoseconds(clockid_t clock) {
    struct timespec value;
    if (clock_gettime(clock, &value)) return 0;
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

int main(void) {
    /* A fresh owner process must contain any SIGILL/SIGSEGV from inaccessible
     * RDPRU; do not install a handler or fall back to a guessed frequency.
     */
    setvbuf(stdout, NULL, _IOLBF, 0);
    unsigned a, b, c, d;
    char vendor[13] = {0};
    __cpuid_count(0, 0, a, b, c, d);
    memcpy(vendor, &b, 4); memcpy(vendor + 4, &d, 4); memcpy(vendor + 8, &c, 4);
    if (strcmp(vendor, "AuthenticAMD")) { puts("status=UNSUPPORTED vendor"); return 77; }
    __cpuid_count(1, 0, a, b, c, d);
    const unsigned signature = a, hypervisor = c >> 31;
    __cpuid_count(0x80000000, 0, a, b, c, d);
    if (a < 0x80000008) { puts("status=UNSUPPORTED CPUID-leaf"); return 77; }
    __cpuid_count(0x80000008, 0, a, b, c, d);
    printf("cpu_signature=%08x hypervisor=%u rdpru=%u max_index=%u\n",
        signature, hypervisor, (b >> 4) & 1u, d >> 16);
    if (hypervisor || !(b & (1u << 4)) || (d >> 16) < 1) {
        puts("status=UNQUALIFIED CPUID-contract"); return 77;
    }
    cpu_set_t allowed, selected;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) || !CPU_ISSET(0, &allowed)) {
        puts("status=UNQUALIFIED CPU0-affinity"); return 77;
    }
    CPU_ZERO(&selected); CPU_SET(0, &selected);
    if (sched_setaffinity(0, sizeof(selected), &selected)) return 1;
    printf("phase=before-RDPRU cpu=%d selected=0\n", sched_getcpu());
    const uint64_t overhead_begin = nanoseconds(CLOCK_MONOTONIC_RAW);
    for (unsigned i = 0; i < 1000; ++i) {
        const struct snapshot s = snapshot();
        if (s.cpu != 0 || s.after < s.before) return 1;
    }
    const uint64_t overhead_end = nanoseconds(CLOCK_MONOTONIC_RAW);
    if (!overhead_begin || overhead_end <= overhead_begin) return 1;
    printf("snapshot_count=1000 snapshot_batch_wall_ns=%" PRIu64 "\n", overhead_end - overhead_begin);
    for (unsigned sample = 0; sample < 3; ++sample) {
        const uint64_t begin = nanoseconds(CLOCK_MONOTONIC_RAW), cpu_begin = nanoseconds(CLOCK_THREAD_CPUTIME_ID);
        const struct snapshot first = snapshot();
        uint64_t work = 1, end = begin;
        while (end >= begin && end - begin < UINT64_C(32000000)) {
            for (unsigned i = 0; i < 4096; ++i) work = work * UINT64_C(6364136223846793005) + 1;
            __asm__ volatile("" : "+r"(work) :: "memory");
            end = nanoseconds(CLOCK_MONOTONIC_RAW);
        }
        const struct snapshot last = snapshot();
        const uint64_t cpu_end = nanoseconds(CLOCK_THREAD_CPUTIME_ID);
        if (!begin || !cpu_begin || end < begin || cpu_end <= cpu_begin || first.cpu != 0 || last.cpu != 0 ||
            last.before <= first.after || last.after < last.before || last.actual <= first.actual) return 1;
        const uint64_t lower = last.before - first.after, upper = last.after - first.before;
        const uint64_t delta = last.actual - first.actual;
        if (!lower || upper < lower) return 1;
        printf("sample=%u wall_ns=%" PRIu64 " thread_cpu_ns=%" PRIu64
            " delta_aperf=%" PRIu64 " delta_mperf_lower=%" PRIu64 " delta_mperf_upper=%" PRIu64
            " ratio_lower=%.9f ratio_upper=%.9f checksum=%" PRIu64 "\n",
            sample, end - begin, cpu_end - cpu_begin, delta, lower, upper,
            (double)delta / (double)upper, (double)delta / (double)lower, work);
    }
    puts("status=ACCESS_OBSERVED MHz=UNQUALIFIED SMT_scope=UNQUALIFIED model_clock=UNMEASURED");
    return 0;
}
