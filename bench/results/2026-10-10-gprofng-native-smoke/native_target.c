/* Harmless CPU target. SIGPROF and SIGUSR2 belong exclusively to the collector. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double monotonic(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(2); }
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}
__attribute__((noinline)) uint64_t strata_native_busy(void) {
    volatile uint64_t x = 1;
    double end = monotonic() + 4.0;
    do {
        for (unsigned i = 0; i < 100000; ++i)
            x = x * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    } while (monotonic() < end);
    return x;
}
static void command(FILE *input, const char *expected) {
    char line[32];
    if (!fgets(line, sizeof line, input) || strcmp(line, expected)) {
        fprintf(stderr, "invalid or missing FIFO command\n"); exit(3);
    }
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *input = fopen(argv[1], "r");
    if (!input) { perror("request FIFO"); return 2; }
    FILE *output = fopen(argv[2], "w");
    if (!output) { perror("reply FIFO"); return 2; }
    setvbuf(output, NULL, _IOLBF, 0);
    if (fprintf(output, "READY %ld\n", (long)getpid()) < 0 || fflush(output)) return 2;
    command(input, "RUN\n");
    uint64_t result = strata_native_busy();
    if (fprintf(output, "DONE %" PRIu64 "\n", result) < 0 || fflush(output)) return 2;
    command(input, "QUIT\n");
    if (fprintf(output, "BYE\n") < 0 || fflush(output)) return 2;
    if (fclose(output) || fclose(input)) return 2;
    return 0;
}
