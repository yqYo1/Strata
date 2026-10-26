/* Main-authored host-only capability pilot; repository LICENSE.
 * AMD APM24594 revision3.36 (March2024), RDPRU entry; no Linux source copied.
 * Primary AMD manual mirrored at:
 * https://www.scs.stanford.edu/~zyedidia/docs/x86/amd-manual-v3.pdf
 * Consumer: CPU-only observer qualification before a separately planned model run.
 * No GPU/runtime API. The original access pilot is recorded at source303c1b06f.
 * Report APERF/MPERF ratios only: P0 and SMT/counter scope remain unqualified.
 */
#define _GNU_SOURCE
#include <cpuid.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/timerfd.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

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

static int native_mappings(void) {
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return 1;
    char line[8192];
    int failed = 0;
    while (fgets(line, sizeof(line), maps)) {
        char permissions[5];
        if (sscanf(line, "%*s %4s", permissions) != 1) { failed = 1; break; }
        char *path = strchr(line, '/');
        if (!path || permissions[2] != 'x') continue;
        char *end = strchr(path, '\n');
        if (!end) { failed = 1; break; }
        *end = '\0';
        if (printf("native_map_pid=%ld path=%s\n", (long)getpid(), path) < 0) { failed = 1; break; }
    }
    if (ferror(maps)) failed = 1;
    if (fclose(maps)) failed = 1;
    return failed;
}

static int initialize(void) {
    /* A fresh owner process must contain any SIGILL/SIGSEGV from inaccessible
     * RDPRU; do not install a handler or fall back to a guessed frequency.
     */
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (native_mappings()) return 1;
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
    const struct snapshot initial = snapshot();
    return initial.cpu != 0 || initial.after < initial.before;
}

static int access_pilot(void) {
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

static int emit_snapshot(unsigned sequence) {
    const uint64_t begin = nanoseconds(CLOCK_MONOTONIC_RAW);
    const struct snapshot s = snapshot();
    const uint64_t end = nanoseconds(CLOCK_MONOTONIC_RAW);
    if (!begin || end < begin || s.cpu != 0 || s.after < s.before) return 1;
    return printf("clock_tick=%u cpu=%d raw_before_ns=%" PRIu64 " raw_after_ns=%" PRIu64
        " mperf_before=%" PRIu64 " aperf=%" PRIu64 " mperf_after=%" PRIu64 "\n",
        sequence, s.cpu, begin, end, s.before, s.actual, s.after) < 0;
}

static int observe(void) {
    /* A private stdin pipe supplies EOF on owner exit. No signals or process
     * environment changes; poll also bounds an orphan to at most 900 ticks.
     */
    const int timer = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    if (timer < 0) return 1;
    int failed = 0;
    unsigned sequence = 0;
    const struct itimerspec interval = {.it_interval = {.tv_sec = 1}, .it_value = {.tv_sec = 1}};
    if (timerfd_settime(timer, 0, &interval, NULL) || emit_snapshot(sequence)) failed = 1;
    if (!failed && printf("observer_ready pid=%ld period_ns=1000000000 max_ticks=900\n", (long)getpid()) < 0) failed = 1;
    while (!failed) {
        struct pollfd fds[] = {{.fd = STDIN_FILENO, .events = POLLIN}, {.fd = timer, .events = POLLIN}};
        int code;
        do { code = poll(fds, 2, 2000); } while (code < 0 && errno == EINTR);
        if (code <= 0 || ((fds[0].revents | fds[1].revents) & (POLLERR | POLLNVAL))) { failed = 1; break; }
        if (fds[0].revents & (POLLIN | POLLHUP)) {
            char byte;
            ssize_t count;
            do { count = read(STDIN_FILENO, &byte, 1); } while (count < 0 && errno == EINTR);
            if (count != 0) failed = 1; /* Only EOF is a valid stop. */
            break;
        }
        if (!(fds[1].revents & POLLIN)) { failed = 1; break; }
        uint64_t expirations = 0;
        ssize_t bytes;
        do { bytes = read(timer, &expirations, sizeof(expirations)); } while (bytes < 0 && errno == EINTR);
        if (bytes != (ssize_t)sizeof(expirations) || expirations != 1 || sequence >= 900) { failed = 1; break; }
        if (emit_snapshot(++sequence)) failed = 1;
    }
    if (close(timer)) failed = 1;
    if (printf("observer_done pid=%ld ticks=%u failed=%d\n", (long)getpid(), sequence, failed) < 0) failed = 1;
    return failed;
}

static uint64_t fixed_work(uint64_t blocks) {
    uint64_t value = 1;
    for (uint64_t block = 0; block < blocks; ++block) {
        for (unsigned i = 0; i < 4096; ++i) value = value * UINT64_C(6364136223846793005) + 1;
        __asm__ volatile("" : "+r"(value) :: "memory");
    }
    return value;
}

static uint64_t usage_ns(struct timeval value) {
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_usec * 1000;
}

static int host_control(const char *self, int enabled) {
    /* Host-only fixture: fork is never used inside a SYCL/Level Zero process.
     * Both parent and observer run on CPU0. Charge fork/exec, initialization,
     * wakeups, snapshots, output, EOF shutdown and wait to the wall interval.
     */
    const uint64_t warm = fixed_work(UINT64_C(245760));
    const uint64_t begin = nanoseconds(CLOCK_MONOTONIC_RAW);
    const uint64_t cpu_begin = nanoseconds(CLOCK_PROCESS_CPUTIME_ID);
    int stops[2] = {-1, -1};
    pid_t child = -1;
    if (enabled) {
        if (pipe2(stops, O_CLOEXEC)) return 1;
        child = fork();
        if (child < 0) { close(stops[0]); close(stops[1]); return 1; }
        if (!child) {
            if (stops[0] != STDIN_FILENO) {
                if (dup2(stops[0], STDIN_FILENO) < 0) _exit(125);
                close(stops[0]);
            }
            close(stops[1]);
            execl(self, self, "--observe", (char *)NULL);
            _exit(125);
        }
        if (close(stops[0])) return 1;
    }
    const uint64_t checksum = fixed_work(UINT64_C(2457600));
    uint64_t observer_cpu = 0;
    if (enabled) {
        if (close(stops[1])) return 1;
        int status = 0;
        struct rusage usage;
        pid_t waited;
        do { waited = wait4(child, &status, 0, &usage); } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status)) return 1;
        observer_cpu = usage_ns(usage.ru_utime) + usage_ns(usage.ru_stime);
    }
    const uint64_t cpu_end = nanoseconds(CLOCK_PROCESS_CPUTIME_ID);
    const uint64_t end = nanoseconds(CLOCK_MONOTONIC_RAW);
    if (!begin || end <= begin || !cpu_begin || cpu_end <= cpu_begin || sched_getcpu() != 0) return 1;
    printf("control=%s wall_ns=%" PRIu64 " actor_cpu_ns=%" PRIu64
        " observer_cpu_ns=%" PRIu64 " iterations=10066329600 warm_iterations=1006632960"
        " checksum=%" PRIu64 " warm_checksum=%" PRIu64 " observer_pid=%ld\n",
        enabled ? "on" : "off", end - begin, cpu_end - cpu_begin, observer_cpu, checksum, warm, (long)child);
    return 0;
}

int main(int argc, char **argv) {
    if (!((argc == 1) || (argc == 2 && !strcmp(argv[1], "--observe")) ||
          (argc == 3 && !strcmp(argv[1], "--host-control") &&
           (!strcmp(argv[2], "on") || !strcmp(argv[2], "off"))))) return 125;
    const int ready = initialize();
    if (ready) return ready;
    if (argc == 1) return access_pilot();
    if (argc == 2) return observe();
    return host_control(argv[0], !strcmp(argv[2], "on"));
}
