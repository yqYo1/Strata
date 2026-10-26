/* Main-authored observer comparison; repository LICENSE (MIT) applies.
 * Based on our own f312b6691 preload lifecycle, not XeStrata source.
 * Both modes retain one loader reference until process exit. No teardown,
 * device enumeration, queue, enqueue, reserved UR interposition or setenv.
 * Compile against independently installed official Unified Runtime headers.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <unified-runtime/ur_api.h>

static void reject(const char *why) {
    dprintf(STDERR_FILENO, "strata UR hold-control failure: pid=%ld %s\n", (long)getpid(), why);
    _Exit(125);
}

static void success(ur_result_t status, const char *operation) {
    dprintf(STDERR_FILENO, "strata UR hold-control API: pid=%ld operation=%s result=%u\n",
            (long)getpid(), operation, (unsigned)status);
    if (status != UR_RESULT_SUCCESS) reject(operation);
}

static void single_thread(void) {
    DIR *tasks = opendir("/proc/self/task");
    if (!tasks) reject("cannot inspect initial threads");
    unsigned count = 0;
    struct dirent *entry;
    errno = 0;
    while ((entry = readdir(tasks)))
        if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') ++count;
    int error = errno;
    if (closedir(tasks) || error || count != 1) reject("concurrent initialization");
}

__attribute__((constructor)) static void before_main(void) {
    const char *mode = getenv("STRATA_UR_HOLD_VALIDATION");
    if (!mode || (strcmp(mode, "on") && strcmp(mode, "off"))) reject("explicit on/off mode required");
    const bool validate = !strcmp(mode, "on");
    const char *layers = getenv("UR_ENABLE_LAYERS");
    /* Empty config alone does not override environment-enabled layers. Restrict
     * both arms prospectively; TRACE alone is allowed for the small off smoke.
     */
    if (layers && *layers) {
        const bool trace = !strcmp(layers, "UR_LAYER_TRACING");
        const bool parameter = !strcmp(layers, "UR_LAYER_PARAMETER_VALIDATION");
        const bool both = !strcmp(layers, "UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION");
        if (validate ? !(parameter || both) : !trace) reject("unexpected enabled layers");
    }
    const char *expected = getenv("STRATA_UR_PROBE_EXPECTED_LOADER");
    Dl_info init = {0}, api = {0};
    char provider[PATH_MAX];
    if (!expected || !dladdr((void *)urLoaderInit, &init) ||
        !dladdr((void *)urAdapterGet, &api) || init.dli_fbase != api.dli_fbase ||
        !init.dli_fname || !realpath(init.dli_fname, provider) || strcmp(provider, expected))
        reject("unexpected loader provider");
    single_thread();
    dprintf(STDERR_FILENO, "strata UR hold-control: pid=%ld phase=begin mode=%s loader=%s\n",
            (long)getpid(), mode, provider);
    uint32_t adapters = UINT32_MAX;
    ur_result_t preinit = urAdapterGet(0, NULL, &adapters);
    if (preinit != UR_RESULT_ERROR_UNINITIALIZED || adapters != UINT32_MAX)
        reject("loader already initialized before control");
    ur_loader_config_handle_t config = NULL;
    success(urLoaderConfigCreate(&config), "config-create");
    if (!config) reject("null created config");
    size_t bytes = 0;
    success(urLoaderConfigGetInfo(config, UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS, 0, NULL, &bytes), "layers-size");
    if (!bytes || bytes > 4096) reject("layer-list extent");
    char available[4096] = {0};
    success(urLoaderConfigGetInfo(config, UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS, bytes, available, NULL), "layers-query");
    if (available[bytes - 1] || !strstr(available, "UR_LAYER_PARAMETER_VALIDATION")) reject("validation unavailable");
    if (validate) success(urLoaderConfigEnableLayer(config, "UR_LAYER_PARAMETER_VALIDATION"), "enable");
    success(urLoaderInit(0, config), "first-init");
    success(urLoaderConfigRelease(config), "config-release");
    /* Without validation a null handle can reach the adapter: NEVER probe it
     * in off mode. The on-only error check is outside model timing.
     */
    if (validate) {
        ur_api_version_t version = UINT32_MAX;
        const ur_result_t checked = urPlatformGetApiVersion(NULL, &version);
        if (checked != UR_RESULT_ERROR_INVALID_NULL_HANDLE || version != UINT32_MAX)
            reject("validation dispatch check");
    }
    dprintf(STDERR_FILENO,
            "strata UR hold-control: pid=%ld phase=before-main mode=%s preinit=%u enable_calls=%u postinit_probe=%s retained_ref=1\n",
            (long)getpid(), mode, (unsigned)preinit, validate ? 1u : 0u,
            validate ? "46" : "skipped");
    /* No destructor/urLoaderTearDown: preserve the already qualified retained
     * reference workaround in BOTH arms. Ordinary teardown remains unqualified.
     */
}
