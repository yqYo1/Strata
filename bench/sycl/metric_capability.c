/* Enumerate this workstation's Level Zero metric properties, without queues,
 * contexts, group activation, queries or streamers. Repository LICENSE.
 * Main must pin source/headers/loader/providers and a supervised recipe first.
 * Run only after the frozen model comparison closes, with immutable pre-exec
 * ZET_ENABLE_METRICS=1. APIs: Level Zero Tools Programming Guide and the
 * installed ze_api.h/zet_api.h; no external implementation was copied.
 * https://oneapi-src.github.io/level-zero-spec/level-zero/latest/tools/PROG.html
 */
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { MAX_DRIVERS = 8, MAX_DEVICES = 16, MAX_GROUPS = 128,
       MAX_GROUP_METRICS = 1024, MAX_TOTAL_METRICS = 4096 };

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

static void before(const char *name) {
    fprintf(stderr, "metric capability: before %s\n", name);
    fflush(stderr);
}

static int result(const char *name, ze_result_t code) {
    printf("{\"kind\":\"api\",\"name\":"); string(name, strlen(name));
    printf(",\"result\":%u}\n", (unsigned)code);
    return code == ZE_RESULT_SUCCESS;
}

#define CALL(expr) do { \
    before(#expr); \
    if (!result(#expr, (expr))) return 10; \
} while (0)

static int refuse(const char *reason) {
    printf("{\"kind\":\"refusal\",\"reason\":"); string(reason, strlen(reason));
    puts("}");
    return 11;
}

static int gate(const char *stage) {
    printf("{\"kind\":\"gate\",\"stage\":\"%s\",\"pid\":%ld}\n", stage, (long)getpid());
    fflush(stdout);
    char answer[16];
    return fgets(answer, sizeof(answer), stdin) && strcmp(answer, "continue\n") == 0;
}

static int environment(const char *name, const char *expected) {
    const char *value = getenv(name);
    return value && strcmp(value, expected) == 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (!environment("STRATA_METRIC_CAPABILITY_GATES", "1") ||
        !environment("ZET_ENABLE_METRICS", "1") ||
        !environment("ZES_ENABLE_SYSMAN", "0") ||
        !environment("NEOReadDebugKeys", "1") ||
        !environment("EnableDirectSubmission", "0"))
        return refuse("required pre-exec environment missing");
    puts("{\"kind\":\"schema\",\"version\":1,\"scope\":\"enumeration_only\","
         "\"strings\":\"API bytes escaped individually\","
         "\"application_queues_created\":false,\"kernels_launched\":false}");
    if (!gate("before-runtime")) return refuse("before-runtime acknowledgement");
    CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driver_count = 0;
    CALL(zeDriverGet(&driver_count, NULL));
    if (!driver_count || driver_count > MAX_DRIVERS) return refuse("driver count bound");
    ze_driver_handle_t drivers[MAX_DRIVERS] = {0};
    uint32_t capacity = driver_count;
    CALL(zeDriverGet(&driver_count, drivers));
    if (driver_count != capacity) return refuse("driver enumeration changed");

    ze_device_handle_t selected = NULL;
    unsigned matches = 0;
    for (uint32_t d = 0; d < driver_count; ++d) {
        uint32_t device_count = 0;
        CALL(zeDeviceGet(drivers[d], &device_count, NULL));
        if (!device_count) continue;
        if (device_count > MAX_DEVICES) return refuse("device count bound");
        ze_device_handle_t devices[MAX_DEVICES] = {0};
        capacity = device_count;
        CALL(zeDeviceGet(drivers[d], &device_count, devices));
        if (device_count != capacity) return refuse("device enumeration changed");
        for (uint32_t i = 0; i < device_count; ++i) {
            ze_device_properties_t properties = {0};
            properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            CALL(zeDeviceGetProperties(devices[i], &properties));
            if (properties.type != ZE_DEVICE_TYPE_GPU || properties.vendorId != 0x8086 ||
                properties.deviceId != 0xe20c ||
                (properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)) continue;
            ze_pci_ext_properties_t pci = {0};
            pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
            CALL(zeDevicePciGetPropertiesExt(devices[i], &pci));
            if (pci.address.domain != 0 || pci.address.bus != 5 ||
                pci.address.device != 0 || pci.address.function != 0) continue;
            selected = devices[i]; ++matches;
            printf("{\"kind\":\"device\",\"driver_index\":%u,\"device_index\":%u,"
                   "\"pci\":\"0000:05:00.0\",\"vendor_id\":%u,\"device_id\":%u,\"name\":",
                   d, i, properties.vendorId, properties.deviceId);
            string(properties.name, sizeof(properties.name));
            printf(",\"uuid_hex\":\"");
            for (size_t u = 0; u < sizeof(properties.uuid.id); ++u)
                printf("%02x", (unsigned)properties.uuid.id[u]);
            puts("\"}");
        }
    }
    if (matches != 1 || !selected) return refuse("unique B570 PCI selection required");
    if (!gate("runtime-ready")) return refuse("runtime-ready acknowledgement");

    uint32_t group_count = 0, total_metrics = 0;
    CALL(zetMetricGroupGet(selected, &group_count, NULL));
    if (group_count > MAX_GROUPS) return refuse("metric group count bound");
    zet_metric_group_handle_t groups[MAX_GROUPS] = {0};
    if (group_count) {
        capacity = group_count;
        CALL(zetMetricGroupGet(selected, &group_count, groups));
        if (group_count != capacity) return refuse("group enumeration changed");
    }
    for (uint32_t g = 0; g < group_count; ++g) {
        zet_metric_group_properties_t properties = {0};
        properties.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
        CALL(zetMetricGroupGetProperties(groups[g], &properties));
        uint32_t count = 0;
        CALL(zetMetricGet(groups[g], &count, NULL));
        if (count != properties.metricCount || count > MAX_GROUP_METRICS ||
            count > MAX_TOTAL_METRICS - total_metrics) return refuse("metric count bound/mismatch");
        printf("{\"kind\":\"group\",\"group\":%u,\"name\":", g);
        string(properties.name, sizeof(properties.name));
        printf(",\"description\":"); string(properties.description, sizeof(properties.description));
        printf(",\"sampling_flags\":%u,\"domain\":%u,\"metric_count\":%u}\n",
               properties.samplingType, properties.domain, count);
        zet_metric_handle_t metrics[MAX_GROUP_METRICS] = {0};
        if (count) {
            capacity = count;
            CALL(zetMetricGet(groups[g], &count, metrics));
            if (count != capacity) return refuse("metric enumeration changed");
        }
        for (uint32_t m = 0; m < count; ++m) {
            zet_metric_properties_t metric = {0};
            metric.stype = ZET_STRUCTURE_TYPE_METRIC_PROPERTIES;
            CALL(zetMetricGetProperties(metrics[m], &metric));
            printf("{\"kind\":\"metric\",\"group\":%u,\"index\":%u,\"name\":", g, m);
            string(metric.name, sizeof(metric.name));
            printf(",\"description\":"); string(metric.description, sizeof(metric.description));
            printf(",\"component\":"); string(metric.component, sizeof(metric.component));
            printf(",\"units\":"); string(metric.resultUnits, sizeof(metric.resultUnits));
            printf(",\"tier\":%u,\"metric_type\":%u,\"result_type\":%u}\n",
                   metric.tierNumber, (unsigned)metric.metricType, (unsigned)metric.resultType);
        }
        total_metrics += count;
    }
    printf("{\"kind\":\"complete\",\"groups\":%u,\"metrics\":%u,"
           "\"collection_qualified\":false,\"hardware_utilization_measured\":false}\n",
           group_count, total_metrics);
    if (!gate("before-exit")) return refuse("before-exit acknowledgement");
    return 0;
}
