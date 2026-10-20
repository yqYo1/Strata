/* Main-authored host-only UR diagnostic; repository LICENSE applies. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <unified-runtime/ur_api.h>

int main(void) {
    Dl_info origin = {0};
    if (!dladdr((void *)urLoaderInit, &origin) || !origin.dli_fname) return 2;
    fprintf(stderr, "strata UR canary begin pid=%ld loader=%s\n",
            (long)getpid(), origin.dli_fname);
    ur_result_t initialized = urLoaderInit(0, NULL);
    ur_api_version_t version = UINT32_MAX;
    ur_result_t checked = UR_RESULT_ERROR_UNINITIALIZED;
    ur_result_t closed = UR_RESULT_ERROR_UNINITIALIZED;
    if (initialized == UR_RESULT_SUCCESS) {
        /* A valid output pointer; no platform/device/queue or enqueue call. */
        checked = urPlatformGetApiVersion(NULL, &version);
        closed = urLoaderTearDown();
    }
    printf("{\"pid\":%ld,\"initialized\":%u,\"checked\":%u,"
           "\"expected\":%u,\"version\":%u,\"closed\":%u}\n",
           (long)getpid(), (unsigned)initialized, (unsigned)checked,
           (unsigned)UR_RESULT_ERROR_INVALID_NULL_HANDLE, (unsigned)version,
           (unsigned)closed);
    fprintf(stderr, "strata UR canary end pid=%ld\n", (long)getpid());
    return initialized == UR_RESULT_SUCCESS &&
           checked == UR_RESULT_ERROR_INVALID_NULL_HANDLE &&
           version == UINT32_MAX && closed == UR_RESULT_SUCCESS ? 0 : 1;
}
