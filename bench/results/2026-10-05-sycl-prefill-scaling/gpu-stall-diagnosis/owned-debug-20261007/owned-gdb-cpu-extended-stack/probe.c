#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
__attribute__((noinline)) static void waiting_probe(double seconds) {
    struct timespec start, now; clock_gettime(CLOCK_MONOTONIC, &start);
    for (;;) {
        usleep(10000); clock_gettime(CLOCK_MONOTONIC, &now);
        if (seconds > 0 && (now.tv_sec-start.tv_sec) +
            (now.tv_nsec-start.tv_nsec)*1e-9 >= seconds) return;
    }
}
int main(int n, char **v) {
    puts("READY"); fflush(stdout);
    if (!strcmp(v[1], "argv")) return n == 6 &&
        !strcmp(v[2], "two words") && !strcmp(v[3], "quote\"tail") &&
        !strcmp(v[4], "backslash\\tail") &&
        !strcmp(v[5], "literal $STRATA_TEST_LITERAL") ? 0 : 89;
    if (!strcmp(v[1], "crash")) { raise(SIGSEGV); return 99; }
    if (!strcmp(v[1], "resume")) { waiting_probe(3); return 0; }
    if (!strcmp(v[1], "blocked")) { waiting_probe(0); return 99; }
    return atoi(v[1]);
}
