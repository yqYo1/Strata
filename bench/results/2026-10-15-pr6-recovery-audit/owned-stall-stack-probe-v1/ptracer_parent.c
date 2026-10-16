/* Diagnostic-only preload. No environment changes and no stdout writes. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <unistd.h>

static void fail_grant(int error) {
    char text[128];
    int count = snprintf(text, sizeof(text),
                         "owned-ptracer grant FAILED errno=%d\n", error);
    if (count > 0 && count < (int)sizeof(text) &&
        write(STDERR_FILENO, text, (size_t)count) != count)
        _exit(125);
    _exit(125);
}

__attribute__((constructor)) static void grant_direct_parent(void) {
    pid_t parent = getppid();
    if (parent <= 1) fail_grant(ESRCH);
    if (prctl(PR_SET_PTRACER, (unsigned long)parent, 0UL, 0UL, 0UL) != 0)
        fail_grant(errno);
    if (getppid() != parent) fail_grant(ESRCH);
    char text[160];
    int count = snprintf(text, sizeof(text),
                         "owned-ptracer grant pid=%ld parent=%ld\n",
                         (long)getpid(), (long)parent);
    if (count <= 0 || count >= (int)sizeof(text)) fail_grant(EOVERFLOW);
    if (write(STDERR_FILENO, text, (size_t)count) != count) _exit(125);
}
