/* Main-authored, repository LICENSE. Installed Level Zero API declarations.
 * Metadata only: no context, queue, command list, allocation or submission.
 */
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fputs("queue inventory refused: " #x "\n", stderr); return 2; } } while (0)
#define API(x) do { fprintf(stderr, "queue inventory: before %s\n", #x); CHECK((x) == ZE_RESULT_SUCCESS); } while (0)

static int gate(const char *name) {
    char input[16];
    printf("{\"kind\":\"gate\",\"name\":\"%s\"}\n", name);
    return fgets(input, sizeof(input), stdin) && !strcmp(input, "continue\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    CHECK(gate("before-runtime"));
    API(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t nd = 0;
    API(zeDriverGet(&nd, NULL));
    CHECK(nd && nd <= 8);
    ze_driver_handle_t drivers[8] = {0};
    uint32_t original = nd;
    API(zeDriverGet(&nd, drivers));
    CHECK(nd == original);
    ze_device_handle_t chosen = NULL;
    ze_driver_handle_t driver = NULL;
    unsigned matches = 0;
    for (uint32_t d = 0; d < nd; ++d) {
        uint32_t count = 0;
        API(zeDeviceGet(drivers[d], &count, NULL));
        CHECK(count <= 16);
        if (!count) continue;
        ze_device_handle_t devices[16] = {0};
        original = count;
        API(zeDeviceGet(drivers[d], &count, devices));
        CHECK(count == original);
        for (uint32_t i = 0; i < count; ++i) {
            ze_device_properties_t p = {0}; p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            API(zeDeviceGetProperties(devices[i], &p));
            if (p.type != ZE_DEVICE_TYPE_GPU || p.vendorId != 0x8086 || p.deviceId != 0xe20c ||
                (p.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)) continue;
            ze_pci_ext_properties_t pci = {0}; pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
            API(zeDevicePciGetPropertiesExt(devices[i], &pci));
            if (pci.address.domain || pci.address.bus != 5 || pci.address.device || pci.address.function) continue;
            chosen = devices[i]; driver = drivers[d]; ++matches;
        }
    }
    CHECK(matches == 1 && chosen && driver);
    CHECK(gate("runtime-ready"));
    ze_api_version_t version = 0;
    ze_driver_properties_t dp = {0}; dp.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    API(zeDriverGetApiVersion(driver, &version));
    API(zeDriverGetProperties(driver, &dp));
    printf("{\"kind\":\"driver\",\"API_major\":%u,\"API_minor\":%u,\"driver_version\":%u}\n",
           ZE_MAJOR_VERSION(version), ZE_MINOR_VERSION(version), dp.driverVersion);
    uint32_t count = 0;
    API(zeDeviceGetCommandQueueGroupProperties(chosen, &count, NULL));
    CHECK(count && count <= 32);
    ze_command_queue_group_properties_t groups[32] = {{0}};
    for (uint32_t i = 0; i < count; ++i) groups[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    original = count;
    API(zeDeviceGetCommandQueueGroupProperties(chosen, &count, groups));
    CHECK(count == original);
    for (uint32_t i = 0; i < count; ++i) {
        CHECK(groups[i].numQueues);
        printf("{\"kind\":\"queue_group\",\"ordinal\":%u,\"flags\":%u,\"physical_engines\":%u,"
               "\"compute\":%s,\"copy\":%s,\"max_fill_pattern_bytes\":%zu}\n", i, groups[i].flags,
               groups[i].numQueues, (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) ? "true" : "false",
               (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) ? "true" : "false",
               groups[i].maxMemoryFillPatternSize);
    }
    CHECK(gate("before-exit"));
    puts("{\"kind\":\"done\",\"GPU_work_submitted\":false}");
    return 0;
}
