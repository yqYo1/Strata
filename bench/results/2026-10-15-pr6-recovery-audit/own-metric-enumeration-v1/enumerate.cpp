// Independently authored for Strata; repository LICENSE applies.
// Source delivery only. Main must review and qualify every execution separately.
#include "ze_api.h"
#include "zet_api.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
// Encode bytes as JSON ASCII escapes; a provider string cannot inject a row.
std::string quoted(const char* p, size_t cap) {
    std::string s = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < cap && p[i]; ++i) {
        const auto c = static_cast<unsigned char>(p[i]);
        if (c == '"' || c == '\\') { s += '\\'; s += static_cast<char>(c); }
        else if (c < 32 || c >= 127) {
            s += "\\u00"; s += hex[c >> 4]; s += hex[c & 15];
        } else s += static_cast<char>(c);
    }
    return s + '"';
}
template<size_t N> std::string q(const char (&p)[N]) { return quoted(p, N); }
const char* result_name(ze_result_t r) {
    switch (r) {
#define RESULT(name) case name: return #name;
#include "result_names.inc"
#undef RESULT
    default: return "ZE_RESULT_UNRECOGNIZED";
    }
}
const char* metric_type(zet_metric_type_t t) {
    switch (t) {
#define TYPE(n) case ZET_METRIC_TYPE_##n: return "ZET_METRIC_TYPE_" #n;
    TYPE(DURATION) TYPE(EVENT) TYPE(EVENT_WITH_RANGE) TYPE(THROUGHPUT)
    TYPE(TIMESTAMP) TYPE(FLAG) TYPE(RATIO) TYPE(RAW) TYPE(IP)
    TYPE(EVENT_EXP_TIMESTAMP) TYPE(EVENT_EXP_START) TYPE(EVENT_EXP_END)
    TYPE(EVENT_EXP_MONOTONIC_WRAPS_VALUE) TYPE(EXP_EXPORT_DMA_BUF)
#undef TYPE
    default: return "ZET_METRIC_TYPE_UNRECOGNIZED";
    }
}
const char* value_type(zet_value_type_t t) {
    switch (t) {
#define TYPE(n) case ZET_VALUE_TYPE_##n: return "ZET_VALUE_TYPE_" #n;
    TYPE(UINT32) TYPE(UINT64) TYPE(FLOAT32) TYPE(FLOAT64) TYPE(BOOL8)
    TYPE(STRING) TYPE(UINT8) TYPE(UINT16)
#undef TYPE
    default: return "ZET_VALUE_TYPE_UNRECOGNIZED";
    }
}
struct Stop { const char* status; };
void bounded(bool yes) { if (!yes) throw Stop{"INCOMPLETE"}; }
// Log entry before calling the provider and result immediately afterward.
template<class F> void api(const char* name, F fn) {
    std::cout << "{\"kind\":\"entry\",\"api\":" << quoted(name, 128) << "}\n" << std::flush;
    const ze_result_t r = fn();
    std::cout << "{\"kind\":\"result\",\"api\":" << quoted(name, 128)
              << ",\"code\":" << static_cast<uint32_t>(r) << ",\"hex\":\"0x"
              << std::hex << static_cast<uint32_t>(r) << std::dec << "\",\"name\":\""
              << result_name(r) << "\"}\n" << std::flush;
    if (r != ZE_RESULT_SUCCESS) {
        switch (r) {
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
        case ZE_RESULT_ERROR_NOT_AVAILABLE:
        case ZE_RESULT_ERROR_UNINITIALIZED: throw Stop{"UNAVAILABLE"};
        default: throw Stop{"ERROR"};
        }
    }
}
template<class H, class F> std::vector<H> handles(const char* name, uint32_t cap, F fn) {
    uint32_t n = 0;
    api(name, [&] { return fn(&n, nullptr); });
    std::cout << "{\"kind\":\"count\",\"api\":" << quoted(name, 128) << ",\"count\":" << n << "}\n";
    bounded(n <= cap);
    if (!n) return {};
    std::vector<H> out(n); uint32_t written = n;
    api(name, [&] { return fn(&written, out.data()); });
    bounded(written == n); // No automatic retry or silent topology change.
    for (auto h : out) bounded(h != nullptr);
    return out;
}
uint32_t total_groups = 0, total_metrics = 0, selected = 0, selected_groups = 0;
void groups(ze_device_handle_t device, bool target, uint32_t driver, uint32_t root) {
    const auto gs = handles<zet_metric_group_handle_t>("zetMetricGroupGet", 256,
        [&](uint32_t* n, zet_metric_group_handle_t* p) { return zetMetricGroupGet(device, n, p); });
    bounded(gs.size() + total_groups <= 1024);
    for (size_t g = 0; g < gs.size(); ++g) {
        ++total_groups;
        zet_metric_group_properties_t gp{}; gp.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
        api("zetMetricGroupGetProperties", [&] { return zetMetricGroupGetProperties(gs[g], &gp); });
        std::cout << "{\"kind\":\"group\",\"driver\":" << driver << ",\"root\":" << root
                  << ",\"group\":" << g << ",\"target\":" << (target ? "true" : "false")
                  << ",\"name\":" << q(gp.name) << ",\"description\":" << q(gp.description)
                  << ",\"domain\":" << gp.domain << ",\"metric_count\":" << gp.metricCount
                  << ",\"sampling_flags\":" << gp.samplingType
                  << ",\"query_event_based\":" << bool(gp.samplingType & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED)
                  << ",\"time_based\":" << bool(gp.samplingType & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED)
                  << ",\"tracer_based\":" << bool(gp.samplingType & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EXP_TRACER_BASED)
                  << ",\"unknown_sampling_bits\":" << (gp.samplingType & ~uint32_t(7)) << "}\n";
        uint32_t count = 0;
        api("zetMetricGet", [&] { return zetMetricGet(gs[g], &count, nullptr); });
        bounded(count == gp.metricCount);
        const uint32_t requested = std::min(count, uint32_t(8));
        std::cout << "{\"kind\":\"metric_scope\",\"available\":" << count << ",\"requested\":"
                  << requested << ",\"complete_catalog\":" << (count <= 8 ? "true" : "false") << "}\n";
        if (!requested) continue;
        bounded(total_metrics + requested <= 4096);
        std::vector<zet_metric_handle_t> ms(requested); uint32_t got = requested;
        api("zetMetricGet", [&] { return zetMetricGet(gs[g], &got, ms.data()); });
        bounded(got == requested);
        for (uint32_t m = 0; m < got; ++m) {
            bounded(ms[m] != nullptr); ++total_metrics;
            zet_metric_properties_t mp{}; mp.stype = ZET_STRUCTURE_TYPE_METRIC_PROPERTIES;
            api("zetMetricGetProperties", [&] { return zetMetricGetProperties(ms[m], &mp); });
            std::cout << "{\"kind\":\"metric\",\"driver\":" << driver << ",\"root\":" << root
                      << ",\"group\":" << g << ",\"metric\":" << m << ",\"name\":" << q(mp.name)
                      << ",\"description\":" << q(mp.description) << ",\"component\":" << q(mp.component)
                      << ",\"units\":" << q(mp.resultUnits) << ",\"tier\":" << mp.tierNumber
                      << ",\"metric_type\":" << uint32_t(mp.metricType) << ",\"metric_type_name\":\"" << metric_type(mp.metricType)
                      << "\",\"result_type\":" << uint32_t(mp.resultType) << ",\"result_type_name\":\"" << value_type(mp.resultType) << "\"}\n";
        }
        if (target) ++selected_groups; // Nonempty group with sampled property calls completed.
    }
}
void enumerate() {
    const char* enabled = std::getenv("ZET_ENABLE_METRICS");
    if (!enabled || std::strcmp(enabled, "1")) throw Stop{"ENVIRONMENT_REJECTED"};
    api("zeInit", [] { return zeInit(0); }); // Raw driver enumeration; no selector assumptions.
    const auto ds = handles<ze_driver_handle_t>("zeDriverGet", 8,
        [](uint32_t* n, ze_driver_handle_t* p) { return zeDriverGet(n, p); });
    if (ds.empty()) throw Stop{"UNAVAILABLE"};
    for (uint32_t d = 0; d < ds.size(); ++d) {
        ze_driver_properties_t dp{}; dp.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
        api("zeDriverGetProperties", [&] { return zeDriverGetProperties(ds[d], &dp); });
        ze_api_version_t version{};
        api("zeDriverGetApiVersion", [&] { return zeDriverGetApiVersion(ds[d], &version); });
        std::cout << "{\"kind\":\"driver\",\"driver\":" << d << ",\"version\":" << dp.driverVersion
                  << ",\"api_version\":" << uint32_t(version) << ",\"uuid\":\"";
        for (auto b : dp.uuid.id) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
        std::cout << std::dec << "\"}\n";
        // Generic properties list is bounded; unlike handle arrays these are values.
        uint32_t ec = 0;
        api("zeDriverGetExtensionProperties", [&] { return zeDriverGetExtensionProperties(ds[d], &ec, nullptr); });
        bounded(ec <= 512); std::vector<ze_driver_extension_properties_t> ex(ec);
        uint32_t en = ec;
        if (ec) api("zeDriverGetExtensionProperties", [&] { return zeDriverGetExtensionProperties(ds[d], &en, ex.data()); });
        bounded(en == ec); bool pci_supported = false;
        for (const auto& e : ex) {
            if (std::strncmp(e.name, ZE_PCI_PROPERTIES_EXT_NAME, sizeof(e.name)) == 0 &&
                e.version >= ZE_PCI_PROPERTIES_EXT_VERSION_1_0) pci_supported = true;
        }
        auto pci = reinterpret_cast<decltype(&zeDevicePciGetPropertiesExt)>(dlsym(RTLD_DEFAULT, "zeDevicePciGetPropertiesExt"));
        std::cout << "{\"kind\":\"pci_interface\",\"advertised\":" << (pci_supported ? "true" : "false")
                  << ",\"loader_symbol\":" << (pci ? "true" : "false") << "}\n";
        const auto roots = handles<ze_device_handle_t>("zeDeviceGet", 32,
            [&](uint32_t* n, ze_device_handle_t* p) { return zeDeviceGet(ds[d], n, p); });
        for (uint32_t r = 0; r < roots.size(); ++r) {
            ze_device_properties_t p{}; p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            api("zeDeviceGetProperties", [&] { return zeDeviceGetProperties(roots[r], &p); });
            bool target = false;
            std::cout << "{\"kind\":\"root\",\"driver\":" << d << ",\"root\":" << r
                      << ",\"name\":" << q(p.name) << ",\"vendor_id\":" << p.vendorId
                      << ",\"device_id\":" << p.deviceId << ",\"type\":" << uint32_t(p.type)
                      << ",\"flags\":" << p.flags << ",\"uuid\":\"";
            for (auto b : p.uuid.id) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
            std::cout << std::dec << "\"}\n";
            if (pci_supported && pci) {
                ze_pci_ext_properties_t b{}; b.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
                api("zeDevicePciGetPropertiesExt", [&] { return pci(roots[r], &b); });
                std::cout << "{\"kind\":\"pci\",\"domain\":" << b.address.domain << ",\"bus\":" << b.address.bus
                          << ",\"device\":" << b.address.device << ",\"function\":" << b.address.function << "}\n";
                target = p.type == ZE_DEVICE_TYPE_GPU && p.vendorId == 0x8086 && p.deviceId == 0xe20c &&
                    !(p.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) && b.address.domain == 0 && b.address.bus == 5 &&
                    b.address.device == 0 && b.address.function == 0;
            }
            if (target) ++selected;
            groups(roots[r], target, d, r);
        }
    }
    if (selected != 1 || selected_groups == 0) throw Stop{"UNAVAILABLE"};
}
}
int main(int argc, char** argv) {
    // Main must request a hold, inspect owned PID/start/env/maps, then send exactly X.
    if (argc != 2 || std::strcmp(argv[1], "--main-owned-hold")) return 64;
    std::cout << std::boolalpha;
    const char* status = "ENUMERATION_PASS";
    try { enumerate(); }
    catch (const Stop& e) { status = e.status; }
    catch (const std::exception&) { status = "HOST_ERROR"; }
    std::cout << "{\"kind\":\"summary\",\"status\":\"" << status << "\",\"selected_roots\":" << selected
              << ",\"selected_nonempty_groups\":" << selected_groups << ",\"groups\":" << total_groups
              << ",\"sampled_metrics\":" << total_metrics << ",\"collection_qualified\":false}\n"
              << "{\"kind\":\"main_snapshot_hold\",\"milliseconds\":15000}\n" << std::flush;
    pollfd input{STDIN_FILENO, POLLIN, 0};
    char ack = 0;
    if (poll(&input, 1, 15000) != 1 || !(input.revents & POLLIN) || read(STDIN_FILENO, &ack, 1) != 1 || ack != 'X') return 65;
    return std::strcmp(status, "ENUMERATION_PASS") == 0 ? 0 : 2;
}
