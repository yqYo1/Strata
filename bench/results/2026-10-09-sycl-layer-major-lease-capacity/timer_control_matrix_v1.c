#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t observed = 0, timer_notifications = 0;

static void handler(int signo, siginfo_t *info, void *unused) {
    (void)signo;
    (void)unused;
    ++observed;
    if (info && info->si_code == SI_TIMER) ++timer_notifications;
}

static int seccomp_mode(void) {
    char line[256];
    int value = -1;
    FILE *stream = fopen("/proc/self/status", "r");
    if (!stream) return value;
    while (fgets(line, sizeof(line), stream))
        if (sscanf(line, "Seccomp: %d", &value) == 1) break;
    fclose(stream);
    return value;
}

static double elapsed(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) / 1e9;
}

static int burn(void) {
    struct timespec begin, now;
    if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0) return -1;
    volatile uint32_t counter = 1;
    do {
        for (int i = 0; i < 10000; ++i) counter = counter * 1664525u + 1013904223u;
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
    } while (elapsed(begin, now) < 0.2);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const int legacy = strcmp(argv[1], "legacy-prof") == 0;
    const int thread_clock = strncmp(argv[1], "thread-", 7) == 0;
    const int directed = strstr(argv[1], "-directed") != NULL;
    if (!legacy && strcmp(argv[1], "thread-directed") && strcmp(argv[1], "thread-signal") &&
        strcmp(argv[1], "monotonic-directed") && strcmp(argv[1], "monotonic-signal")) return 2;
    struct sigaction action = {0};
    action.sa_sigaction = handler;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    errno = 0;
    const int handler_rc = sigaction(SIGPROF, &action, NULL), handler_errno = errno;
    if (handler_rc != 0) {
        printf("{\"case\":\"%s\",\"handler_rc\":%d,\"handler_errno\":%d}\n", argv[1], handler_rc, handler_errno);
        return 1;
    }
    struct rlimit limit = {0};
    errno = 0;
    const int limit_rc = getrlimit(RLIMIT_SIGPENDING, &limit), limit_errno = errno;
    printf("{\"case\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"seccomp\":%d,\"signal\":%d,"
           "\"rlimit_sigpending_rc\":%d,\"rlimit_sigpending_errno\":%d,\"rlimit_sigpending_soft\":%llu,",
           argv[1], (long)getpid(), (long)syscall(SYS_gettid), seccomp_mode(), SIGPROF,
           limit_rc, limit_errno, (unsigned long long)limit.rlim_cur);
    if (legacy) {
        struct itimerval setting = {{0, 10000}, {0, 10000}}, zero = {{0, 0}, {0, 0}};
        errno = 0;
        const int arm_rc = setitimer(ITIMER_PROF, &setting, NULL), arm_errno = errno;
        const int burn_rc = arm_rc == 0 ? burn() : 0;
        errno = 0;
        const int disarm_rc = setitimer(ITIMER_PROF, &zero, NULL), disarm_errno = errno;
        printf("\"legacy_arm_rc\":%d,\"legacy_arm_errno\":%d,\"burn_rc\":%d,"
               "\"disarm_rc\":%d,\"disarm_errno\":%d,\"signals\":%d,\"SI_TIMER_signals\":%d}\n",
               arm_rc, arm_errno, burn_rc, disarm_rc, disarm_errno, (int)observed, (int)timer_notifications);
        return burn_rc == 0 && disarm_rc == 0 ? 0 : 1;
    }
    const clockid_t clock = thread_clock ? CLOCK_THREAD_CPUTIME_ID : CLOCK_MONOTONIC;
    struct timespec readout = {0};
    errno = 0;
    const int clock_rc = clock_gettime(clock, &readout), clock_errno = errno;
    struct sigevent event = {0};
    event.sigev_notify = directed ? SIGEV_THREAD_ID | SIGEV_SIGNAL : SIGEV_SIGNAL;
    event.sigev_signo = SIGPROF;
    event._sigev_un._tid = (pid_t)syscall(SYS_gettid);
    timer_t timer = {0};
    errno = 0;
    const int create_rc = timer_create(clock, &event, &timer), create_errno = errno;
    int arm_rc = -999, arm_errno = 0, get_rc = -999, get_errno = 0;
    int disarm_rc = -999, disarm_errno = 0, delete_rc = -999, delete_errno = 0, burn_rc = 0;
    struct itimerspec setting = {{0, 10000000}, {0, 10000000}}, actual = {0}, zero = {0};
    if (create_rc == 0) {
        errno = 0;
        arm_rc = timer_settime(timer, 0, &setting, NULL); arm_errno = errno;
        errno = 0;
        get_rc = timer_gettime(timer, &actual); get_errno = errno;
        if (arm_rc == 0) burn_rc = burn();
        errno = 0;
        disarm_rc = timer_settime(timer, 0, &zero, NULL); disarm_errno = errno;
        errno = 0;
        delete_rc = timer_delete(timer); delete_errno = errno;
    }
    printf("\"clock_id\":%d,\"notification\":%d,\"clock_gettime_rc\":%d,\"clock_gettime_errno\":%d,"
           "\"timer_create_rc\":%d,\"timer_create_errno\":%d,\"timer_settime_rc\":%d,\"timer_settime_errno\":%d,"
           "\"timer_gettime_rc\":%d,\"timer_gettime_errno\":%d,\"interval_ns\":%llu,\"burn_rc\":%d,"
           "\"disarm_rc\":%d,\"disarm_errno\":%d,\"timer_delete_rc\":%d,\"timer_delete_errno\":%d,"
           "\"signals\":%d,\"SI_TIMER_signals\":%d}\n",
           (int)clock, event.sigev_notify, clock_rc, clock_errno, create_rc, create_errno,
           arm_rc, arm_errno, get_rc, get_errno,
           (unsigned long long)actual.it_interval.tv_sec * 1000000000ULL + (unsigned long long)actual.it_interval.tv_nsec,
           burn_rc, disarm_rc, disarm_errno, delete_rc, delete_errno, (int)observed, (int)timer_notifications);
    return burn_rc == 0 && (create_rc != 0 || (disarm_rc == 0 && delete_rc == 0)) ? 0 : 1;
}
