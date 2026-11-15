/* Independent direct Level Zero C collection qualification; repository LICENSE.
 * API contracts: installed ze_api.h/zet_api.h and official Tools PROG.html.
 * No SYCL context borrowing, immediate lists, graphs, streamer or model work.
 * Source/module/recipe/runtime must be committed and supervised before use.
 */
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { ELEMENTS = 65536, GUARD = 64, WORDS = ELEMENTS + 2 * GUARD,
       MAX_DRIVERS = 8, MAX_DEVICES = 16, MAX_GROUPS = 128, METRICS = 83 };
static const uint32_t SENTINEL = UINT32_C(0xa7a7a7a7);
static const uint64_t DEADLINE_NS = UINT64_C(10000000000);

static void string(const char *s, size_t capacity) {
    putchar('"');
    for (size_t i = 0; i < capacity && s[i]; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c < 32 || c >= 127) printf("\\u%04x", (unsigned)c);
        else putchar(c);
    }
    putchar('"');
}

static int result(const char *name, ze_result_t code) {
    printf("{\"kind\":\"api\",\"name\":"); string(name, strlen(name));
    printf(",\"result\":%u}\n", (unsigned)code);
    return code == ZE_RESULT_SUCCESS;
}

#define API(expr) (fprintf(stderr, "metric query: before %s\n", #expr), result(#expr, (expr)))
#define TRY(expr) do { if (!API(expr)) { status = 10; goto cleanup; } } while (0)
#define CHECK(test, reason) do { if (!(test)) { refusal(reason); status = 11; goto cleanup; } } while (0)

static void refusal(const char *reason) {
    printf("{\"kind\":\"refusal\",\"reason\":"); string(reason, strlen(reason)); puts("}");
}

static int gate(const char *stage) {
    printf("{\"kind\":\"gate\",\"stage\":\"%s\",\"pid\":%ld}\n", stage, (long)getpid());
    char response[16];
    return fgets(response, sizeof(response), stdin) && !strcmp(response, "continue\n");
}

static int environment(const char *key, const char *expected) {
    const char *value = getenv(key);
    return value && !strcmp(value, expected);
}

/* A timeout or failed submit leaves completion unknown. Keep every device
 * owner alive until process exit instead of freeing possibly executing storage.
 */
static int execute(ze_command_queue_handle_t queue, ze_command_list_handle_t list) {
    if (!API(zeCommandListClose(list))) return 0;
    if (!API(zeCommandQueueExecuteCommandLists(queue, 1, &list, NULL)) ||
        !API(zeCommandQueueSynchronize(queue, DEADLINE_NS))) {
        refusal("completion unknown; no explicit owner destruction");
        fflush(NULL); _Exit(12);
    }
    return 1;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    if (argc != 3 || (strcmp(argv[1], "work-off") && strcmp(argv[1], "work-query") &&
                     strcmp(argv[1], "empty-query"))) return 2;
    if (!environment("STRATA_METRIC_QUERY_GATES", "1") || !environment("ZET_ENABLE_METRICS", "1") ||
        !environment("ZES_ENABLE_SYSMAN", "0") || !environment("NEOReadDebugKeys", "1") ||
        !environment("EnableDirectSubmission", "0")) { refusal("pre-exec environment"); return 11; }
    int status = 0, active = 0, query_on = strcmp(argv[1], "work-off") != 0;
    int work_on = strcmp(argv[1], "empty-query") != 0, completed = 0;
    ze_context_handle_t context = NULL;
    ze_command_queue_handle_t queue = NULL;
    ze_command_list_handle_t list = NULL;
    ze_module_handle_t module = NULL;
    ze_module_build_log_handle_t log = NULL;
    ze_kernel_handle_t kernel = NULL;
    zet_metric_query_pool_handle_t pool = NULL;
    zet_metric_query_handle_t query = NULL;
    uint32_t *host = NULL, *device_data = NULL;
    uint8_t *spirv = NULL, *raw = NULL;
    zet_typed_value_t *values = NULL;
    FILE *input = NULL;
    ze_device_handle_t device = NULL;
    ze_driver_handle_t driver = NULL;
    zet_metric_group_handle_t group = NULL;
    zet_metric_properties_t properties[METRICS] = {{0}};
    uint32_t matches = 0, count = 0, capacity = 0;
    printf("{\"kind\":\"schema\",\"version\":1,\"mode\":\"%s\",\"elements\":%u}\n", argv[1], ELEMENTS);
    CHECK(gate("before-runtime"), "before-runtime acknowledgement");
    TRY(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    TRY(zeDriverGet(&count, NULL));
    CHECK(count && count <= MAX_DRIVERS, "driver count");
    ze_driver_handle_t drivers[MAX_DRIVERS] = {0};
    capacity = count;
    TRY(zeDriverGet(&count, drivers));
    CHECK(count == capacity, "driver enumeration changed");
    for (uint32_t d = 0; d < count; ++d) {
        uint32_t n = 0;
        TRY(zeDeviceGet(drivers[d], &n, NULL));
        CHECK(n <= MAX_DEVICES, "device count");
        if (!n) continue;
        ze_device_handle_t devices[MAX_DEVICES] = {0};
        capacity = n;
        TRY(zeDeviceGet(drivers[d], &n, devices));
        CHECK(n == capacity, "device enumeration changed");
        for (uint32_t i = 0; i < n; ++i) {
            ze_device_properties_t p = {0}; p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            TRY(zeDeviceGetProperties(devices[i], &p));
            if (p.type != ZE_DEVICE_TYPE_GPU || p.vendorId != 0x8086 || p.deviceId != 0xe20c ||
                (p.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)) continue;
            ze_pci_ext_properties_t pci = {0}; pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
            TRY(zeDevicePciGetPropertiesExt(devices[i], &pci));
            if (pci.address.domain || pci.address.bus != 5 || pci.address.device || pci.address.function) continue;
            device = devices[i]; driver = drivers[d]; ++matches;
        }
    }
    CHECK(matches == 1 && device && driver, "unique B570 PCI0000:05:00.0");
    puts("{\"kind\":\"device\",\"pci\":\"0000:05:00.0\",\"vendor_id\":32902,\"device_id\":57868}");
    CHECK(gate("runtime-ready"), "runtime-ready acknowledgement");
    count = 0;
    TRY(zetMetricGroupGet(device, &count, NULL));
    CHECK(count && count <= MAX_GROUPS, "group count");
    zet_metric_group_handle_t groups[MAX_GROUPS] = {0};
    capacity = count;
    TRY(zetMetricGroupGet(device, &count, groups));
    CHECK(count == capacity, "group enumeration changed");
    matches = 0;
    for (uint32_t i = 0; i < count; ++i) {
        zet_metric_group_properties_t p = {0}; p.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
        TRY(zetMetricGroupGetProperties(groups[i], &p));
        if (!strcmp(p.name, "ComputeBasic") && p.samplingType == ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED &&
            p.domain == 3 && p.metricCount == METRICS) { group = groups[i]; ++matches; }
    }
    CHECK(matches == 1 && group, "unique ComputeBasic EVENT/domain3/83");
    zet_metric_handle_t metrics[METRICS] = {0}; count = METRICS;
    TRY(zetMetricGet(group, &count, metrics));
    CHECK(count == METRICS, "metric count changed");
    for (uint32_t i = 0; i < count; ++i) {
        properties[i].stype = ZET_STRUCTURE_TYPE_METRIC_PROPERTIES;
        TRY(zetMetricGetProperties(metrics[i], &properties[i]));
        printf("{\"kind\":\"property\",\"index\":%u,\"name\":", i);
        string(properties[i].name, sizeof(properties[i].name));
        printf(",\"units\":"); string(properties[i].resultUnits, sizeof(properties[i].resultUnits));
        printf(",\"result_type\":%u}\n", (unsigned)properties[i].resultType);
    }
    ze_context_desc_t cd = {0}; cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    TRY(zeContextCreate(driver, &cd, &context));
    uint32_t ordinals = 0, ordinal = UINT32_MAX;
    TRY(zeDeviceGetCommandQueueGroupProperties(device, &ordinals, NULL));
    CHECK(ordinals && ordinals <= 16, "queue group count");
    ze_command_queue_group_properties_t qp[16] = {{0}};
    for (uint32_t i = 0; i < ordinals; ++i) qp[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    capacity = ordinals;
    TRY(zeDeviceGetCommandQueueGroupProperties(device, &ordinals, qp));
    CHECK(ordinals == capacity, "queue group count changed");
    for (uint32_t i = 0; i < ordinals; ++i)
        if (qp[i].numQueues && (qp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) { ordinal = i; break; }
    CHECK(ordinal != UINT32_MAX, "compute queue absent");
    ze_command_queue_desc_t qd = {0}; qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.ordinal = ordinal; qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    TRY(zeCommandQueueCreate(context, device, &qd, &queue));
    ze_command_list_desc_t ld = {0}; ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC; ld.commandQueueGroupOrdinal = ordinal;
    TRY(zeCommandListCreate(context, device, &ld, &list));
    ze_host_mem_alloc_desc_t hd = {0}; hd.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    ze_device_mem_alloc_desc_t dd = {0}; dd.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    TRY(zeMemAllocHost(context, &hd, sizeof(uint32_t) * WORDS, 64, (void **)&host));
    TRY(zeMemAllocDevice(context, &dd, sizeof(uint32_t) * WORDS, 64, device, (void **)&device_data));
    for (uint32_t i = 0; i < WORDS; ++i) host[i] = SENTINEL;
    for (uint32_t i = 0; i < ELEMENTS; ++i) host[GUARD + i] = i % 1024;
    input = fopen(argv[2], "rb"); CHECK(input, "open module");
    CHECK(!fseek(input, 0, SEEK_END), "seek module");
    long extent = ftell(input); CHECK(extent > 0 && extent <= 1 << 20, "module size");
    rewind(input); spirv = malloc((size_t)extent); CHECK(spirv, "module host allocation");
    CHECK(fread(spirv, 1, (size_t)extent, input) == (size_t)extent && !ferror(input), "read module");
    int close_result = fclose(input); input = NULL;
    CHECK(!close_result, "close module");
    ze_module_desc_t md = {0}; md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    md.format = ZE_MODULE_FORMAT_IL_SPIRV; md.inputSize = (size_t)extent; md.pInputModule = spirv; md.pBuildFlags = "";
    TRY(zeModuleCreate(context, device, &md, &module, &log));
    if (log) { TRY(zeModuleBuildLogDestroy(log)); log = NULL; }
    ze_kernel_desc_t kd = {0}; kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC; kd.pKernelName = "strata_metric_integer_work";
    TRY(zeKernelCreate(module, &kd, &kernel));
    TRY(zeKernelSetArgumentValue(kernel, 0, sizeof(device_data), &device_data));
    uint32_t gx = 0, gy = 0, gz = 0;
    TRY(zeKernelSuggestGroupSize(kernel, ELEMENTS, 1, 1, &gx, &gy, &gz));
    CHECK(gx && gx <= 1024 && gy == 1 && gz == 1, "suggested group dimensions");
    TRY(zeKernelSetGroupSize(kernel, gx, gy, gz));
    ze_group_count_t launch = {(ELEMENTS + gx - 1) / gx, 1, 1};
    printf("{\"kind\":\"launch\",\"local_x\":%u,\"groups_x\":%u}\n", gx, launch.groupCountX);
    CHECK(gate("before-work"), "before-work acknowledgement");
    TRY(zeCommandListAppendMemoryCopy(list, device_data, host, sizeof(uint32_t) * WORDS, NULL, 0, NULL));
    CHECK(execute(queue, list), "upload list close");
    TRY(zeCommandListReset(list));
    if (query_on) {
        TRY(zetContextActivateMetricGroups(context, device, 1, &group)); active = 1;
        zet_metric_query_pool_desc_t pd = {0}; pd.stype = ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC;
        pd.type = ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE; pd.count = 1;
        TRY(zetMetricQueryPoolCreate(context, device, group, &pd, &pool));
        TRY(zetMetricQueryCreate(pool, 0, &query));
        TRY(zetCommandListAppendMetricQueryBegin(list, query));
    }
    if (work_on) TRY(zeCommandListAppendLaunchKernel(list, kernel, &launch, NULL, 0, NULL));
    if (query_on) TRY(zetCommandListAppendMetricQueryEnd(list, query, NULL, 0, NULL));
    CHECK(execute(queue, list), "work/query list close");
    if (query_on) {
        size_t bytes = 0;
        TRY(zetMetricQueryGetData(query, &bytes, NULL));
        CHECK(bytes && bytes <= 1 << 20, "raw metric size");
        size_t raw_capacity = bytes; raw = malloc(bytes); CHECK(raw, "raw metric allocation");
        TRY(zetMetricQueryGetData(query, &bytes, raw));
        CHECK(bytes && bytes <= raw_capacity, "actual raw metric size");
        count = 0;
        TRY(zetMetricGroupCalculateMetricValues(group, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES, bytes, raw, &count, NULL));
        CHECK(count && count <= METRICS * 16 && count % METRICS == 0, "calculated metric count");
        capacity = count; values = calloc(count, sizeof(*values)); CHECK(values, "calculated value allocation");
        TRY(zetMetricGroupCalculateMetricValues(group, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES, bytes, raw, &count, values));
        CHECK(count == capacity, "calculated metric count changed");
        printf("{\"kind\":\"collection\",\"raw_bytes\":%zu,\"values\":%u,\"sets\":%u}\n", bytes, count, count / METRICS);
        for (uint32_t i = 0; i < count; ++i) {
            CHECK(values[i].type == properties[i % METRICS].resultType, "calculated value type differs");
            printf("{\"kind\":\"value\",\"set\":%u,\"index\":%u,\"type\":%u,\"value\":", i / METRICS, i % METRICS, (unsigned)values[i].type);
            switch (values[i].type) {
                case ZET_VALUE_TYPE_UINT32: printf("%u", values[i].value.ui32); break;
                case ZET_VALUE_TYPE_UINT64: printf("%llu", (unsigned long long)values[i].value.ui64); break;
                case ZET_VALUE_TYPE_FLOAT32:
                    if (isfinite(values[i].value.fp32)) printf("%.17g", (double)values[i].value.fp32);
                    else printf("null");
                    break;
                case ZET_VALUE_TYPE_FLOAT64:
                    if (isfinite(values[i].value.fp64)) printf("%.17g", values[i].value.fp64);
                    else printf("null");
                    break;
                case ZET_VALUE_TYPE_BOOL8: printf("%s", values[i].value.b8 ? "true" : "false"); break;
                default: printf("null"); break;
            }
            printf(",\"raw_hex\":\"");
            for (size_t j = 0; j < sizeof(values[i].value); ++j) printf("%02x", ((unsigned char *)&values[i].value)[j]);
            puts("\"}");
        }
    }
    TRY(zeCommandListReset(list));
    TRY(zeCommandListAppendMemoryCopy(list, host, device_data, sizeof(uint32_t) * WORDS, NULL, 0, NULL));
    CHECK(execute(queue, list), "readback list close");
    for (uint32_t i = 0; i < WORDS; ++i) {
        uint32_t expected = SENTINEL;
        if (i >= GUARD && i < GUARD + ELEMENTS) expected = work_on ? ((i - GUARD) % 1024) * 3 + 7 : (i - GUARD) % 1024;
        CHECK(host[i] == expected, "integer payload/guard mismatch");
    }
    completed = 1;
cleanup:
    if (input) fclose(input);
    /* No work remains in flight here. Refuse a destruction cascade on error. */
#define DESTROY(expr) do { if (!API(expr)) { fflush(NULL); _Exit(13); } } while (0)
    if (list) DESTROY(zeCommandListDestroy(list));
    if (query) DESTROY(zetMetricQueryDestroy(query));
    if (pool) DESTROY(zetMetricQueryPoolDestroy(pool));
    if (active) DESTROY(zetContextActivateMetricGroups(context, device, 0, NULL));
    if (kernel) DESTROY(zeKernelDestroy(kernel));
    if (log) DESTROY(zeModuleBuildLogDestroy(log));
    if (module) DESTROY(zeModuleDestroy(module));
    if (device_data) DESTROY(zeMemFree(context, device_data));
    if (host) DESTROY(zeMemFree(context, host));
    if (queue) DESTROY(zeCommandQueueDestroy(queue));
    if (context) DESTROY(zeContextDestroy(context));
    free(spirv); free(raw); free(values);
    printf("{\"kind\":\"complete\",\"work_checked\":%s,\"query_requested\":%s,\"cleanup_complete\":true,\"status\":%d}\n",
           completed ? "true" : "false", query_on ? "true" : "false", status);
    if (!gate("before-exit")) return 11;
    return status;
}
