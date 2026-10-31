/* Main-authored process-local diagnostic; repository LICENSE applies.
 * No UR-reserved symbol definitions and no device/queue/enqueue operation.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <unified-runtime/ur_api.h>

static void fail(const char *reason) {
    dprintf(STDERR_FILENO, "strata UR validation failure: pid=%ld %s\n",
            (long)getpid(), reason);
    _Exit(125);
}

static void success(ur_result_t result, const char *operation) {
    dprintf(STDERR_FILENO, "strata UR validation API: pid=%ld operation=%s result=%u\n",
            (long)getpid(), operation, (unsigned)result);
    if (result != UR_RESULT_SUCCESS) fail(operation);
}

static void require_single_thread(void) {
    DIR *tasks = opendir("/proc/self/task");
    if (!tasks) fail("cannot inspect initial thread count");
    unsigned count = 0;
    struct dirent *entry;
    errno = 0;
    while ((entry = readdir(tasks)))
        if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') ++count;
    int error = errno;
    if (closedir(tasks) || error || count != 1) fail("concurrent initialization is ambiguous");
}

__attribute__((constructor)) static void check_before_main(void) {
    const char *expected = getenv("STRATA_UR_PROBE_EXPECTED_LOADER");
    Dl_info init_origin = {0}, api_origin = {0};
    char provider[PATH_MAX];
    if (!expected || !dladdr((void *)urLoaderInit, &init_origin) ||
        !dladdr((void *)urPlatformGetApiVersion, &api_origin) ||
        init_origin.dli_fbase != api_origin.dli_fbase ||
        !init_origin.dli_fname || !realpath(init_origin.dli_fname, provider) ||
        strcmp(provider, expected)) fail("unexpected loader provider");

    dprintf(STDERR_FILENO,
            "strata UR validation: pid=%ld phase=begin loader=%s\n",
            (long)getpid(), provider);
    require_single_thread();
    ur_api_version_t version = UINT32_MAX;
    ur_result_t before = urPlatformGetApiVersion(NULL, &version);
    dprintf(STDERR_FILENO, "strata UR validation API: pid=%ld operation=preinit result=%u\n",
            (long)getpid(), (unsigned)before);
    if (before != UR_RESULT_ERROR_UNINITIALIZED || version != UINT32_MAX)
        fail("loader was already initialized before this diagnostic");
    ur_loader_config_handle_t config = NULL;
    success(urLoaderConfigCreate(&config), "config-create");
    if (!config) fail("null created config");
    size_t bytes = 0;
    success(urLoaderConfigGetInfo(config, UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS,
                                  0, NULL, &bytes), "layers-size");
    if (!bytes || bytes > 4096) fail("available-layer extent");
    char layers[4096] = {0};
    success(urLoaderConfigGetInfo(config, UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS,
                                  bytes, layers, NULL), "layers-query");
    if (layers[bytes - 1] || !strstr(layers, "UR_LAYER_PARAMETER_VALIDATION"))
        fail("available validation layer");
    success(urLoaderConfigEnableLayer(config, "UR_LAYER_PARAMETER_VALIDATION"), "enable");
    success(urLoaderInit(0, config), "first-init");
    success(urLoaderConfigRelease(config), "config-release");
    ur_result_t result = urPlatformGetApiVersion(NULL, &version);
    dprintf(STDERR_FILENO, "strata UR validation API: pid=%ld operation=postinit result=%u\n",
            (long)getpid(), (unsigned)result);
    if (result != UR_RESULT_ERROR_INVALID_NULL_HANDLE || version != UINT32_MAX)
        fail("parameter validation dispatch");
    /* Deliberately no destructor/teardown: this extra reference is diagnostic
     * process-lifetime state and cannot qualify ordinary runtime teardown.
     */
    dprintf(STDERR_FILENO,
            "strata UR validation: pid=%ld phase=before-main threads_before=1 preinit=%u config_create=0 layers_query=0 enable=0 init=0 config_release=0 checked=%u untouched=1 retained_ref=1\n",
            (long)getpid(), (unsigned)before, (unsigned)result);
}
