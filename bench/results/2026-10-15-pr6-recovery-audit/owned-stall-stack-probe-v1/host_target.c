/* Pure CPU preflight target: handshake, blocked stdin and a live worker. */
#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static _Atomic uint64_t ticks;
static _Atomic int done;
static void *worker(void *unused) {
    (void)unused;
    const struct timespec period = { .tv_sec = 0, .tv_nsec = 10000000 };
    while (!atomic_load(&done)) {
        atomic_fetch_add(&ticks, 1);
        nanosleep(&period, NULL);
    }
    return NULL;
}

int main(void) {
    pthread_t thread;
    if (pthread_create(&thread, NULL, worker, NULL)) return 20;
    printf("READY %ld %ld\n", (long)getpid(), (long)getppid());
    fflush(stdout);
    char line[32];
    int code = 21;
    while (fgets(line, sizeof(line), stdin)) {
        if (!strcmp(line, "PING\n")) {
            printf("PONG %" PRIu64 "\n", atomic_load(&ticks));
            fflush(stdout);
        } else if (!strcmp(line, "QUIT\n")) {
            code = 7; /* Expected original target exit; controller must preserve. */
            break;
        } else {
            code = 22;
            break;
        }
    }
    atomic_store(&done, 1);
    if (pthread_join(thread, NULL)) return 23;
    return code;
}
