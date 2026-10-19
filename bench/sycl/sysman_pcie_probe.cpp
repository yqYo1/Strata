// Main-authored read-only Sysman capability probe; repository LICENSE applies.
// Uses installed Level Zero declarations. No queues, workloads or device changes.
#include <level_zero/zes_api.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <thread>
#include <vector>

template<class F>
ze_result_t observe(const char* name, F operation) {
    std::printf("{\"api\":\"%s\",\"stage\":\"enter\"}\n", name);
    std::fflush(stdout);
    const ze_result_t result = operation();
    std::printf("{\"api\":\"%s\",\"stage\":\"exit\",\"result\":\"0x%08x\"}\n",
                name, static_cast<unsigned>(result));
    std::fflush(stdout);
    return result;
}

void snapshot(int index, const zes_pci_stats_t& sample) {
    std::printf("{\"snapshot\":%d,\"timestamp_us\":%" PRIu64
                ",\"rx_counter\":%" PRIu64 ",\"tx_counter\":%" PRIu64
                ",\"packet_counter\":%" PRIu64 ",\"replay_counter\":%" PRIu64
                ",\"reported_gen\":%d,\"reported_width\":%d,\"reported_max_bandwidth\":%lld}\n",
                index, sample.timestamp, sample.rxCounter, sample.txCounter,
                sample.packetCounter, sample.replayCounter, sample.speed.gen,
                sample.speed.width, static_cast<long long>(sample.speed.maxBandwidth));
    std::fflush(stdout);
}

int main() {
    if (observe("zesInit", [] { return zesInit(0); }) != ZE_RESULT_SUCCESS) return 1;
    uint32_t count = 0;
    if (observe("zesDriverGet/count", [&] { return zesDriverGet(&count, nullptr); }) != ZE_RESULT_SUCCESS ||
        count == 0 || count > 16) return 1;
    std::vector<zes_driver_handle_t> drivers(count);
    if (observe("zesDriverGet/list", [&] { return zesDriverGet(&count, drivers.data()); }) != ZE_RESULT_SUCCESS ||
        count > drivers.size()) return 1;
    drivers.resize(count);
    for (auto driver : drivers) {
        count = 0;
        if (observe("zesDeviceGet/count", [&] { return zesDeviceGet(driver, &count, nullptr); }) != ZE_RESULT_SUCCESS ||
            count == 0 || count > 16) return 1;
        std::vector<zes_device_handle_t> devices(count);
        if (observe("zesDeviceGet/list", [&] { return zesDeviceGet(driver, &count, devices.data()); }) != ZE_RESULT_SUCCESS ||
            count > devices.size()) return 1;
        devices.resize(count);
        for (auto device : devices) {
            zes_pci_properties_t properties{};
            properties.stype = ZES_STRUCTURE_TYPE_PCI_PROPERTIES;
            if (observe("zesDevicePciGetProperties", [&] { return zesDevicePciGetProperties(device, &properties); }) != ZE_RESULT_SUCCESS)
                return 1;
            const auto& address = properties.address;
            std::printf("{\"pci\":\"%04x:%02x:%02x.%x\",\"bandwidth_counters\":%u,\"packet_counters\":%u,\"replay_counters\":%u}\n",
                        address.domain, address.bus, address.device, address.function,
                        properties.haveBandwidthCounters, properties.havePacketCounters, properties.haveReplayCounters);
            std::fflush(stdout);
            if (address.domain != 0 || address.bus != 5 || address.device != 0 || address.function != 0) continue;
            if (!properties.haveBandwidthCounters) return 3;
            zes_pci_stats_t before{}, after{};
            if (observe("zesDevicePciGetStats/before", [&] { return zesDevicePciGetStats(device, &before); }) != ZE_RESULT_SUCCESS)
                return 1;
            snapshot(0, before);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            if (observe("zesDevicePciGetStats/after", [&] { return zesDevicePciGetStats(device, &after); }) != ZE_RESULT_SUCCESS)
                return 1;
            snapshot(1, after);
            if (after.timestamp <= before.timestamp || after.rxCounter < before.rxCounter ||
                after.txCounter < before.txCounter ||
                (properties.havePacketCounters && after.packetCounter < before.packetCounter) ||
                (properties.haveReplayCounters && after.replayCounter < before.replayCounter)) return 2;
            std::puts("{\"raw_read_capability\":true,\"traffic_calibrated\":false,\"physical_byte_layer_qualified\":false}");
            return 0;
        }
    }
    std::puts("{\"target_found\":false}");
    return 3;
}
