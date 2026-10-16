// Independently authored Strata diagnostic; repository LICENSE applies.
// Source only. Main owns review, build, deadlines, process/fault gates and execution.
#include "ze_api.h"
#include "zet_api.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
constexpr char config[] = "\"AssertEnable\"=dword:00000000\n"
                          "\"DebugEnable\"=dword:0000001f\n"
                          "\"DebugLevel\"=dword:e7000007\n";
std::string quoted(const char* p, size_t cap) {
    std::string s = "\""; constexpr char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < cap && p[i]; ++i) {
        const auto c = static_cast<unsigned char>(p[i]);
        if (c == '"' || c == '\\') { s += '\\'; s += static_cast<char>(c); }
        else if (c < 32 || c >= 127) { s += "\\u00"; s += hex[c >> 4]; s += hex[c & 15]; }
        else s += static_cast<char>(c);
    }
    return s + '"';
}
template<size_t N> std::string q(const char (&p)[N]) { return quoted(p, N); }
template<size_t N> std::string uuid(const uint8_t (&p)[N]) {
    std::ostringstream s;
    for (auto b : p) s << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
    return s.str();
}
template<class T> uintptr_t ptr(T p) { return reinterpret_cast<uintptr_t>(p); }
template<class... T> std::string text(T&&... v) {
    std::ostringstream s; (s << ... << v); return s.str();
}
void row(const std::string& s) { std::cout << s << '\n' << std::flush; }
const char* result_name(ze_result_t r) {
    switch (r) {
#define RESULT(name) case name: return #name;
#include "result_names.inc"
#undef RESULT
    default: return "ZE_RESULT_UNRECOGNIZED";
    }
}
struct Stop { const char* reason; };
void need(bool ok, const char* why) { if (!ok) throw Stop{why}; }
uint32_t sequence = 0;
// Data captures contain the actual arguments both before and after each call.
// A pending entry without a result is UNKNOWN completion, never cleanup success.
template<class F, class D> ze_result_t api(const char* name, F fn, D data) {
    const uint32_t call = ++sequence;
    row(text("{\"kind\":\"entry\",\"call\":", call, ",\"api\":", quoted(name, 96), ",\"data\":", data(), "}"));
    const ze_result_t r = fn();
    std::ostringstream hex; hex << "0x" << std::hex << uint32_t(r);
    row(text("{\"kind\":\"result\",\"call\":", call, ",\"api\":", quoted(name, 96),
             ",\"code\":", uint32_t(r), ",\"hex\":", quoted(hex.str().c_str(), 16),
             ",\"name\":", quoted(result_name(r), 96), ",\"output_valid\":", r == ZE_RESULT_SUCCESS ? "true" : "false",
             ",\"data\":", data(), "}"));
    return r;
}
void success(ze_result_t r, const char* name) { need(r == ZE_RESULT_SUCCESS, name); }
template<class H, class F> std::vector<H> handles(const char* name, uint32_t cap, uintptr_t parent, F fn) {
    uint32_t n = 0; H* array = nullptr;
    auto data = [&] { return text("{\"parent\":", parent, ",\"count_ptr\":", ptr(&n),
                                  ",\"count\":", n, ",\"array\":", ptr(array), "}"); };
    success(api(name, [&] { return fn(&n, array); }, data), name);
    need(n <= cap, "enumeration_count_cap");
    if (!n) return {};
    const uint32_t expected = n;
    std::vector<H> out(n); array = out.data();
    success(api(name, [&] { return fn(&n, array); }, data), name);
    need(n == expected, "enumeration_count_changed");
    for (size_t i = 0; i < out.size(); ++i) {
        need(out[i] != nullptr, "enumeration_null_handle");
        for (size_t j = 0; j < i; ++j) need(out[i] != out[j], "enumeration_duplicate_handle");
    }
    return out;
}
void admission() {
    for (const auto& pair : {std::pair{"ZET_ENABLE_METRICS", "1"},
                            std::pair{"ZE_FLAT_DEVICE_HIERARCHY", "FLAT"},
                            std::pair{"ZE_ENABLE_VALIDATION_LAYER", "1"}}) {
        const char* value = std::getenv(pair.first);
        need(value && !std::strcmp(value, pair.second), "fixed_environment_mismatch");
    }
    // Exact 1.0.231 names. No generation/API overrides or injected collector.
    for (const char* name : {"MetricsLibraryApi", "MetricsLibraryGpu", "LD_PRELOAD", "LD_AUDIT"})
        need(std::getenv(name) == nullptr, "unexpected_override_or_injection");
    // Match the library's 260-byte path limit, and reject truncation before zeInit.
    char exe[260]{};
    const ssize_t count = readlink("/proc/self/exe", exe, sizeof(exe));
    need(count > 0 && count < 259, "executable_path_invalid_or_too_long");
    std::string path(exe, static_cast<size_t>(count));
    const auto slash = path.rfind('/'); need(slash != std::string::npos, "executable_parent_missing");
    path.resize(slash + 1); path += "iu.config";
    need(path.size() < 260, "config_path_too_long");
    const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    need(fd >= 0, "config_open_failed");
    struct stat st{}; char bytes[sizeof(config)]{};
    const bool regular = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size == sizeof(config) - 1;
    const ssize_t got = regular ? read(fd, bytes, sizeof(bytes)) : -1;
    const int closed = close(fd);
    need(regular && got == sizeof(config) - 1 && !std::memcmp(bytes, config, sizeof(config) - 1)
         && closed == 0, "config_missing_different_or_read_failure");
    row(text("{\"kind\":\"admission\",\"config\":", quoted(path.c_str(), 260),
             ",\"config_bytes\":", sizeof(config) - 1, ",\"exact_bytes\":true,\"fixed_environment\":true}"));
}
bool hold(const char* phase) {
    pollfd input{STDIN_FILENO, POLLIN, 0};
    // An early queued X cannot authorize a later phase. EOF is never an ack.
    const int early = poll(&input, 1, 0);
    if (early != 0) {
        row(text("{\"kind\":\"hold_result\",\"phase\":", quoted(phase, 32),
                 ",\"status\":\"ERROR\",\"reason\":\"early_input_or_closed_stdin\"}"));
        return false;
    }
    row(text("{\"kind\":\"hold\",\"phase\":", quoted(phase, 32), ",\"milliseconds\":20000,\"ack\":\"X\"}"));
    const int ready = poll(&input, 1, 20000);
    char ack[2]{}; ssize_t got = -1;
    if (ready == 1 && (input.revents & POLLIN)) got = read(STDIN_FILENO, ack, sizeof(ack));
    const bool ok = ready == 1 && (input.revents & POLLIN) && got == 1 && ack[0] == 'X';
    row(text("{\"kind\":\"hold_result\",\"phase\":", quoted(phase, 32), ",\"status\":\"", ok ? "ACK" : "ERROR",
             "\",\"poll_result\":", ready, ",\"revents\":", input.revents, ",\"read_bytes\":", got, "}"));
    return ok;
}
ze_driver_handle_t driver = nullptr;
ze_device_handle_t device = nullptr;
zet_metric_group_handle_t group = nullptr;
ze_context_handle_t context = nullptr;
zet_metric_query_pool_handle_t pool = nullptr;
uint32_t roots_seen = 0, gpu_roots = 0, matches = 0, group_matches = 0;
void enumerate() {
    success(api("zeInit", [] { return zeInit(0); }, [] { return "{\"flags\":0}"; }), "zeInit");
    const auto ds = handles<ze_driver_handle_t>("zeDriverGet", 8, 0,
        [](uint32_t* n, ze_driver_handle_t* p) { return zeDriverGet(n, p); });
    need(!ds.empty(), "no_drivers");
    for (uint32_t d = 0; d < ds.size(); ++d) {
        ze_driver_properties_t dp{}; dp.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
        success(api("zeDriverGetProperties", [&] { return zeDriverGetProperties(ds[d], &dp); },
            [&] { return text("{\"driver\":", ptr(ds[d]), ",\"properties\":", ptr(&dp),
                              ",\"stype\":", dp.stype, ",\"pNext\":", ptr(dp.pNext), ",\"version\":", dp.driverVersion, "}"); }), "zeDriverGetProperties");
        row(text("{\"kind\":\"driver\",\"index\":", d, ",\"handle\":", ptr(ds[d]), ",\"version\":", dp.driverVersion,
                 ",\"uuid\":\"", uuid(dp.uuid.id), "\"}"));
        uint32_t ec = 0;
        success(api("zeDriverGetExtensionProperties", [&] { return zeDriverGetExtensionProperties(ds[d], &ec, nullptr); },
            [&] { return text("{\"driver\":", ptr(ds[d]), ",\"count\":", ec, ",\"array\":0}"); }), "extension_count");
        need(ec <= 512, "extension_count_cap");
        std::vector<ze_driver_extension_properties_t> ex(ec); uint32_t en = ec;
        if (ec) success(api("zeDriverGetExtensionProperties", [&] { return zeDriverGetExtensionProperties(ds[d], &en, ex.data()); },
            [&] { return text("{\"driver\":", ptr(ds[d]), ",\"count\":", en, ",\"array\":", ptr(ex.data()), "}"); }), "extension_list");
        need(en == ec, "extension_count_changed"); bool advertised = false;
        for (const auto& e : ex)
            if (!std::strncmp(e.name, ZE_PCI_PROPERTIES_EXT_NAME, sizeof(e.name)) && e.version >= ZE_PCI_PROPERTIES_EXT_VERSION_1_0) advertised = true;
        auto pci = reinterpret_cast<decltype(&zeDevicePciGetPropertiesExt)>(dlsym(RTLD_DEFAULT, "zeDevicePciGetPropertiesExt"));
        const auto roots = handles<ze_device_handle_t>("zeDeviceGet", 32, ptr(ds[d]),
            [&](uint32_t* n, ze_device_handle_t* p) { return zeDeviceGet(ds[d], n, p); });
        for (uint32_t r = 0; r < roots.size(); ++r) {
            ++roots_seen;
            ze_device_properties_t p{}; p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            success(api("zeDeviceGetProperties", [&] { return zeDeviceGetProperties(roots[r], &p); },
                [&] { return text("{\"device\":", ptr(roots[r]), ",\"properties\":", ptr(&p),
                                  ",\"stype\":", p.stype, ",\"pNext\":", ptr(p.pNext), "}"); }), "device_properties");
            row(text("{\"kind\":\"root\",\"driver_index\":", d, ",\"root_index\":", r, ",\"device\":", ptr(roots[r]),
                     ",\"name\":", q(p.name), ",\"vendor_id\":", p.vendorId, ",\"device_id\":", p.deviceId,
                     ",\"type\":", uint32_t(p.type), ",\"flags\":", p.flags, ",\"uuid\":\"", uuid(p.uuid.id), "\"}"));
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;
            constexpr uint32_t device_flags = ZE_DEVICE_PROPERTY_FLAG_INTEGRATED | ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE
                | ZE_DEVICE_PROPERTY_FLAG_ECC | ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING;
            need(!(p.flags & ~device_flags), "unknown_device_flags");
            ++gpu_roots; need(advertised && pci, "gpu_pci_interface_missing");
            ze_pci_ext_properties_t b{}; b.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
            success(api("zeDevicePciGetPropertiesExt", [&] { return pci(roots[r], &b); },
                [&] { return text("{\"device\":", ptr(roots[r]), ",\"properties\":", ptr(&b), ",\"domain\":", b.address.domain,
                                  ",\"bus\":", b.address.bus, ",\"pci_device\":", b.address.device, ",\"function\":", b.address.function, "}"); }), "pci_properties");
            const bool target = p.vendorId == 0x8086 && p.deviceId == 0xe20c && !(p.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)
                && b.address.domain == 0 && b.address.bus == 5 && b.address.device == 0 && b.address.function == 0;
            if (target) { ++matches; driver = ds[d]; device = roots[r]; }
        }
    }
    need(gpu_roots == 1 && matches == 1, "sole_target_root_required");
    const auto gs = handles<zet_metric_group_handle_t>("zetMetricGroupGet", 256, ptr(device),
        [&](uint32_t* n, zet_metric_group_handle_t* p) { return zetMetricGroupGet(device, n, p); });
    constexpr uint32_t known = ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED | ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED
                            | ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EXP_TRACER_BASED;
    for (uint32_t g = 0; g < gs.size(); ++g) {
        zet_metric_group_properties_t gp{}; gp.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
        success(api("zetMetricGroupGetProperties", [&] { return zetMetricGroupGetProperties(gs[g], &gp); },
            [&] { return text("{\"group\":", ptr(gs[g]), ",\"properties\":", ptr(&gp), ",\"stype\":", gp.stype,
                              ",\"pNext\":", ptr(gp.pNext), ",\"name\":", q(gp.name), ",\"domain\":", gp.domain,
                              ",\"metric_count\":", gp.metricCount, ",\"sampling_flags\":", gp.samplingType,
                              ",\"unknown_sampling_bits\":", (gp.samplingType & ~known), "}"); }), "group_properties");
        need(!(gp.samplingType & ~known), "unknown_sampling_bits");
        if (!std::strncmp(gp.name, "ComputeBasic", sizeof(gp.name)) && gp.domain == 3 && gp.metricCount == 83
            && gp.samplingType == ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED) { ++group_matches; group = gs[g]; }
    }
    need(group_matches == 1, "unique_exact_group_required");
    row(text("{\"kind\":\"selection\",\"driver\":", ptr(driver), ",\"device\":", ptr(device), ",\"group\":", ptr(group),
             ",\"identity\":\"8086:e20c@0000:05:00.0\",\"non_subdevice\":true,\"group_name\":\"ComputeBasic\","
             "\"domain\":3,\"metric_count\":83,\"sampling_flags\":1,\"pool_count\":64}"));
}
const char *context_state = "not_attempted", *activation_state = "not_attempted", *pool_state = "not_attempted";
const char *pool_destroy = "not_attempted", *deactivation = "not_attempted", *context_destroy = "not_attempted";
bool context_owned = false, active = false, pool_owned = false, unknown = false;
template<class F, class D> ze_result_t tracked(const char*& state, const char* name, F fn, D data) {
    state = "pending"; const auto r = api(name, fn, data);
    state = r == ZE_RESULT_SUCCESS ? "success" : "error"; return r;
}
std::string state_data() {
    return text("{\"context\":", ptr(context), ",\"device\":", ptr(device), ",\"group\":", ptr(group), ",\"pool\":", ptr(pool), "}");
}
void probe() {
    ze_context_desc_t cd{}; cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    auto r = tracked(context_state, "zeContextCreate", [&] { return zeContextCreate(driver, &cd, &context); },
        [&] { return text("{\"driver\":", ptr(driver), ",\"desc\":", ptr(&cd), ",\"stype\":", cd.stype,
                          ",\"pNext\":", ptr(cd.pNext), ",\"flags\":", cd.flags, ",\"output\":", ptr(&context), ",\"context\":", ptr(context), "}"); });
    context_owned = r == ZE_RESULT_SUCCESS && context;
    unknown = (r == ZE_RESULT_SUCCESS) != (context != nullptr);
    success(r, "zeContextCreate"); need(context_owned, "null_context_on_success");
    r = tracked(activation_state, "zetContextActivateMetricGroups", [&] { return zetContextActivateMetricGroups(context, device, 1, &group); },
        [&] { return text("{\"context\":", ptr(context), ",\"device\":", ptr(device), ",\"group\":", ptr(group), ",\"count\":1,\"groups_ptr\":", ptr(&group), "}"); });
    active = r == ZE_RESULT_SUCCESS; success(r, "metric_group_activation");
    zet_metric_query_pool_desc_t pd{}; pd.stype = ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC;
    pd.type = ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE; pd.count = 64;
    r = tracked(pool_state, "zetMetricQueryPoolCreate", [&] { return zetMetricQueryPoolCreate(context, device, group, &pd, &pool); },
        [&] { return text("{\"context\":", ptr(context), ",\"device\":", ptr(device), ",\"group\":", ptr(group), ",\"desc\":", ptr(&pd),
                          ",\"stype\":", pd.stype, ",\"pNext\":", ptr(pd.pNext), ",\"type\":", uint32_t(pd.type),
                          ",\"count\":", pd.count, ",\"output\":", ptr(&pool), ",\"pool\":", ptr(pool), "}"); });
    pool_owned = r == ZE_RESULT_SUCCESS && pool;
    unknown |= (r == ZE_RESULT_SUCCESS) != (pool != nullptr);
    success(r, "zetMetricQueryPoolCreate"); need(pool_owned, "null_pool_on_success");
}
bool cleanup() {
    bool ok = true;
    if (pool_owned) ok &= tracked(pool_destroy, "zetMetricQueryPoolDestroy", [] { return zetMetricQueryPoolDestroy(pool); }, state_data) == ZE_RESULT_SUCCESS;
    if (active) ok &= tracked(deactivation, "zetContextActivateMetricGroups", [] { return zetContextActivateMetricGroups(context, device, 0, nullptr); },
        [] { return text("{\"context\":", ptr(context), ",\"device\":", ptr(device), ",\"count\":0,\"groups_ptr\":0}"); }) == ZE_RESULT_SUCCESS;
    if (context_owned) ok &= tracked(context_destroy, "zeContextDestroy", [] { return zeContextDestroy(context); }, state_data) == ZE_RESULT_SUCCESS;
    for (const char* state : {context_state, activation_state, pool_state, pool_destroy, deactivation, context_destroy})
        if (!std::strcmp(state, "pending")) unknown = true;
    row(text("{\"kind\":\"lifecycle\",\"context_create\":", quoted(context_state, 32), ",\"activate\":", quoted(activation_state, 32),
             ",\"pool_create\":", quoted(pool_state, 32), ",\"pool_destroy\":", quoted(pool_destroy, 32), ",\"deactivate\":", quoted(deactivation, 32),
             ",\"context_destroy\":", quoted(context_destroy, 32), ",\"unknown_completion_or_output\":", unknown ? "true" : "false", "}"));
    return ok && !unknown;
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2 || std::strcmp(argv[1], "--main-owned-hold")) return 64;
    const char* operation = "INCOMPLETE"; const char* reason = "not_started";
    bool pre_attempted = false, pre_ack = false, clean = false;
    try {
        admission(); pre_attempted = true; pre_ack = hold("pre_init"); need(pre_ack, "pre_init_hold_failed");
        admission(); enumerate(); probe(); operation = "COMPLETE"; reason = "pool_creation_succeeded";
    } catch (const Stop& e) { reason = e.reason; }
      catch (const std::exception&) { operation = "ERROR"; reason = "host_exception"; }
    try { clean = cleanup(); }
    catch (...) { clean = false; row("{\"kind\":\"cleanup\",\"status\":\"UNKNOWN\",\"reason\":\"host_exception\"}"); }
    row(text("{\"kind\":\"probe_summary\",\"operation_status\":", quoted(operation, 32), ",\"reason\":", quoted(reason, 96),
             ",\"cleanup_status\":\"", clean ? "COMPLETE" : "ERROR_OR_UNKNOWN", "\",\"roots_seen\":", roots_seen,
             ",\"gpu_roots\":", gpu_roots, ",\"target_matches\":", matches, ",\"group_matches\":", group_matches,
             ",\"collection_qualified\":false,\"numerical_result\":false,\"process_hold_status\":\"PENDING\"}"));
    const bool post_ack = hold("post_probe");
    const bool complete = pre_ack && post_ack && clean && !std::strcmp(operation, "COMPLETE");
    // A pool/operation/cleanup failure remains 125 even if the later hold fails.
    int code = complete ? 0 : 125;
    if ((pre_attempted && !pre_ack) || (!post_ack && clean && !std::strcmp(operation, "COMPLETE"))) code = 65;
    row(text("{\"kind\":\"process_summary\",\"operation_status\":", quoted(operation, 32), ",\"cleanup_complete\":", clean ? "true" : "false",
             ",\"pre_init_hold_status\":\"", !pre_attempted ? "NOT_ATTEMPTED" : pre_ack ? "ACK" : "ERROR",
             "\",\"post_probe_hold_status\":\"", post_ack ? "ACK" : "ERROR", "\",\"exit_code\":", code,
             ",\"diagnostic_complete\":", complete ? "true" : "false", ",\"collection_qualified\":false}"));
    return code;
}
