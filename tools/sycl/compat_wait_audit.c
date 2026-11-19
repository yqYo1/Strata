/* Main-authored loader observer. Repository LICENSE applies. */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const char wait_symbol[] =
    "_ZN4sycl3_V15queue20wait_and_throw_proxyERKNS0_6detail13code_locationE";
static uintptr_t call_return;
static size_t parent_offset;
static int enabled;
static uint64_t entered, exited, rejected;

struct wait_sample {
    uintptr_t stack, parent;
    uint64_t wall, cpu;
};
static struct wait_sample sample;

static void emit(const char *line, int length, size_t capacity) {
    if (length < 0 || (size_t)length >= capacity ||
        write(STDERR_FILENO, line, (size_t)length) != (ssize_t)length)
        ++rejected;
}

static int clock_ns(clockid_t clock, uint64_t *out) {
    struct timespec t;
    if (clock_gettime(clock, &t) || t.tv_sec < 0 || t.tv_nsec < 0 || t.tv_nsec >= 1000000000L)
        return 0;
    *out = UINT64_C(1000000000) * (uint64_t)t.tv_sec + (uint64_t)t.tv_nsec;
    return 1;
}

unsigned int la_version(unsigned int version) {
    return version < LAV_CURRENT ? 0 : LAV_CURRENT;
}

unsigned int la_objopen(struct link_map *map, Lmid_t namespace_id, uintptr_t *cookie) {
    *cookie = (uintptr_t)map;
    if (namespace_id != LM_ID_BASE) return 0;
    if (!map->l_name[0] && map->l_addr == 0) return LA_FLG_BINDFROM;
    const char *basename = strrchr(map->l_name, '/');
    basename = basename ? basename + 1 : map->l_name;
    return !strncmp(basename, "libsycl.so.", 11) || !strcmp(basename, "libstrata_wait_canary.so")
        ? LA_FLG_BINDTO : 0;
}

uintptr_t la_symbind64(Elf64_Sym *symbol, unsigned int index, uintptr_t *from, uintptr_t *to,
                       unsigned int *flags, const char *name) {
    (void)index; (void)from; (void)to;
    if (strcmp(name, wait_symbol)) *flags |= LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
    return symbol->st_value;
}

static unsigned long long number(const char *name, int base) {
    const char *value = getenv(name);
    if (!value || !*value) return 0;
    errno = 0;
    char *end;
    unsigned long long n = strtoull(value, &end, base);
    return errno || *end ? 0 : n;
}

void la_preinit(uintptr_t *cookie) {
    const struct link_map *main_map = (const struct link_map *)*cookie;
    call_return = (uintptr_t)number("STRATA_WAIT_AUDIT_RETURN", 16);
    parent_offset = (size_t)number("STRATA_WAIT_AUDIT_PARENT", 10);
    enabled = main_map->l_addr == 0 && call_return && parent_offset >= 16 && parent_offset <= 2048
        && parent_offset % 8 == 0;
    char line[192];
    const int length = snprintf(line, sizeof(line), "strata wait audit: ready=%d return_pc=%" PRIxPTR
                                " parent_offset=%zu\n", enabled, call_return, parent_offset);
    emit(line, length, sizeof(line));
}

Elf64_Addr la_x86_64_gnu_pltenter(Elf64_Sym *symbol, unsigned int index, uintptr_t *from,
        uintptr_t *to, La_x86_64_regs *regs, unsigned int *flags, const char *name, long *frame_size) {
    (void)index; (void)from; (void)to; (void)flags;
    if (!enabled || strcmp(name, wait_symbol) || gettid() != getpid()) return symbol->st_value;
    uintptr_t return_pc;
    memcpy(&return_pc, (const void *)regs->lr_rsp, sizeof(return_pc));
    if (return_pc != call_return) return symbol->st_value;

    // The verified executable's call PC and unwind row determine this slot.
    memcpy(&sample.parent, (const void *)(regs->lr_rsp + parent_offset), sizeof(sample.parent));
    sample.stack = regs->lr_rsp;
    *frame_size = 0;  /* This entry has only register arguments. */
    ++entered;
    if (!clock_ns(CLOCK_MONOTONIC, &sample.wall) || !clock_ns(CLOCK_THREAD_CPUTIME_ID, &sample.cpu)) {
        sample.stack = 0;
        ++rejected;
    }
    return symbol->st_value;
}

unsigned int la_x86_64_gnu_pltexit(Elf64_Sym *symbol, unsigned int index, uintptr_t *from,
        uintptr_t *to, const La_x86_64_regs *regs, La_x86_64_retval *ret, const char *name) {
    (void)symbol; (void)index; (void)from; (void)to; (void)ret;
    if (!enabled || strcmp(name, wait_symbol) || gettid() != getpid() || regs->lr_rsp != sample.stack) return 0;
    uint64_t cpu, wall;
    sample.stack = 0;
    if (!clock_ns(CLOCK_THREAD_CPUTIME_ID, &cpu) || !clock_ns(CLOCK_MONOTONIC, &wall)
        || cpu < sample.cpu || wall < sample.wall) {
        ++rejected;
        return 0;
    }
    ++exited;
    char line[192];
    const int length = snprintf(line, sizeof(line), "strata wait site: ordinal=%" PRIu64
        " caller=%" PRIxPTR " wall_ns=%" PRIu64 " cpu_ns=%" PRIu64 "\n",
        entered, sample.parent, wall - sample.wall, cpu - sample.cpu);
    emit(line, length, sizeof(line));
    return 0;
}

__attribute__((destructor)) static void finish(void) {
    if (!enabled) return;
    char line[192];
    const int length = snprintf(line, sizeof(line), "strata wait audit: entered=%" PRIu64
        " exited=%" PRIu64 " rejected=%" PRIu64 "\n", entered, exited, rejected);
    emit(line, length, sizeof(line));
}
