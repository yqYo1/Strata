// Independently authored project proposal under the repository LICENSE.
// SOURCE ONLY: Main owns integration, build, external supervision and GPU runs.
#include "strata/sycl_queue.hpp"
#include "strata/sycl_retirement.hpp"
#include "strata/prefill/draft_uploads.hpp"
#include "strata/core/validation_control.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unistd.h>

#ifdef DPCT_USM_LEVEL_NONE
#error "This probe requires the historical in-order USM queue contract"
#endif
static_assert(!strata::core::detail::concurrent_doorbell_admitted());
namespace {
constexpr size_t cells = 32;
constexpr size_t bytes = cells * sizeof(uint32_t);
static_assert(sizeof(uint32_t) == 4 && bytes == 128);
class ProviderVectorKernel;
struct AcceptedHostError {}; // Deliberate HOST boundary, never a provider fault.
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void phase(const char* name) {
    std::fprintf(stderr, "pr6_provider_probe phase=%s\n", name);
    std::fflush(stderr);
}
void main_gate(const char* name) {
    require(std::printf("PR6_PROVIDER_READY %s\n", name) > 0 &&
            std::fflush(stdout) == 0, "cannot publish Main gate");
    char byte = 0;
    ssize_t count;
    do { count = ::read(STDIN_FILENO, &byte, 1); } while (count < 0 && errno == EINTR);
    require(count == 1 && byte == 'X', "Main gate not released");
}
struct Options {
    std::string_view name;
    unsigned device = 0;
    bool owned = false, partial = false, async_hold = false;
};
Options parse(int argc, char** argv) {
    require(argc == 5 && std::string_view(argv[1]) == "--case" &&
            std::string_view(argv[3]) == "--device", "usage: --case NAME --device ORDINAL");
    Options o; o.name = argv[2];
    std::string_view value = argv[4];
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), o.device);
    require(error == std::errc{} && end == value.data() + value.size() && o.device <= INT32_MAX,
            "invalid nonnegative device ordinal");
    if (o.name == "baseline-default") {}
    else if (o.name == "baseline-owned") o.owned = true;
    else if (o.name == "partial-host-default") o.partial = true;
    else if (o.name == "partial-host-owned") o.owned = o.partial = true;
    else if (o.name == "async87-host-list") o.async_hold = true;
    else throw std::runtime_error("unknown case");
    return o;
}
struct Probe {
    // These host owners are never replaced/mutated until the queue retires.
    std::array<uint32_t, cells> input{}, output{};
    sycl::queue* borrowed = nullptr;
    sycl::queue* owned = nullptr;
    uint32_t* device = nullptr;
    unsigned uploads = 0, kernels = 0, readbacks = 0;
    unsigned pending_events = 0, complete_events = 0, pending_queues = 0;
    unsigned injected_host_errors = 0, unwind_drains = 0;
    bool default_survival = false, expired_owner_rejected = false, closed = false;
    Probe() = default;
    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;
    void init(const Options& o) {
        phase("runtime-init");
        // No environment mutation, filters, reset or device switches after this.
        dpct::select_device(o.device);
        borrowed = &dpct::get_in_order_queue();
        require(strata::q_of(nullptr) == borrowed && strata::q_of(borrowed) == borrowed,
                "borrowed default lacks actual checked registration");
        auto d = borrowed->get_device();
        const auto groups = d.get_info<sycl::info::device::sub_group_sizes>();
        require(d.is_gpu() && borrowed->get_backend() == sycl::backend::ext_oneapi_level_zero &&
                d.has(sycl::aspect::fp16) && d.has(sycl::aspect::usm_device_allocations) &&
                d.has(sycl::aspect::usm_host_allocations) &&
                std::find(groups.begin(), groups.end(), size_t(32)) != groups.end(),
                "requires Level Zero GPU, FP16, host/device USM and SG32");
        require(borrowed->has_property<sycl::property::queue::in_order>(), "default is not in order");
        std::fprintf(stderr, "pr6_provider_probe device=%s driver=%s\n",
                     d.get_info<sycl::info::device::name>().c_str(),
                     d.get_info<sycl::info::device::driver_version>().c_str());
        if (o.owned) {
            owned = dpct::get_current_device().create_queue(true);
            require(owned && owned != borrowed && strata::q_of(owned) == owned,
                    "owned DPCT queue lacks actual checked registration");
            require(owned->has_property<sycl::property::queue::in_order>() &&
                    owned->get_context() == borrowed->get_context() &&
                    owned->get_device() == borrowed->get_device(),
                    "owned/default queue order/context/device mismatch");
        }
        device = sycl::malloc_device<uint32_t>(cells, *borrowed);
        require(device != nullptr, "small device allocation returned null");
        phase("owners-ready");
    }
    void observe(sycl::event event, sycl::queue& queue) noexcept {
        // A quick kernel need not expose a pending sample. Never force a hang.
        try {
            const auto status = event.get_info<sycl::info::event::command_execution_status>();
            if (status == sycl::info::event_command_status::complete) ++complete_events;
            else if (status == sycl::info::event_command_status::submitted ||
                     status == sycl::info::event_command_status::running) ++pending_events;
            else strata::prefill::detail::retirement_fail_stop("unknown event status");
            if (!queue.ext_oneapi_empty()) ++pending_queues;
            queue.throw_asynchronous();
        } catch (...) { strata::prefill::detail::retirement_fail_stop("status query unavailable"); }
    }
    void finished(sycl::event event, sycl::queue& queue) {
        strata::retire_event(event);
        strata::retire_queue(queue); // Real provider barrier, not an empty success stand-in.
        try {
            require(event.get_info<sycl::info::event::command_execution_status>() ==
                    sycl::info::event_command_status::complete, "retired event not complete");
            require(queue.ext_oneapi_empty(), "retired queue not empty");
            queue.throw_asynchronous();
        } catch (...) { strata::prefill::detail::retirement_fail_stop("final completion unavailable"); }
    }
    void prepare(sycl::queue& queue, uint32_t seed) {
        strata::retire_queue(queue); // Closes the previous lease before host mutation.
        for (size_t i = 0; i < cells; ++i) { input[i] = seed + uint32_t(i); output[i] = 0xdeadbeefu; }
    }
    void baseline(sycl::queue& queue, uint32_t seed) {
        prepare(queue, seed);
        phase("upload");
        auto upload = queue.memcpy(device, input.data(), bytes); ++uploads;
        observe(upload, queue);
        auto* data = device;
        phase("integer-kernel");
        auto kernel = queue.submit([=](sycl::handler& cgh) {
            cgh.parallel_for<ProviderVectorKernel>(
                sycl::nd_range<1>(sycl::range<1>(cells), sycl::range<1>(cells)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    const size_t i = item.get_global_linear_id();
                    data[i] = data[i] * uint32_t(3) + uint32_t(7);
                });
        }); ++kernels;
        observe(kernel, queue);
        auto download = queue.memcpy(output.data(), device, bytes); ++readbacks;
        observe(download, queue);
        finished(download, queue);
        for (size_t i = 0; i < cells; ++i) {
            require(input[i] == seed + uint32_t(i), "retained upload source changed");
            require(output[i] == (seed + uint32_t(i)) * 3u + 7u, "integer kernel/readback mismatch");
        }
        phase("integer-result-verified");
    }
    void partial_host(sycl::queue& queue) {
        prepare(queue, 149u);
        bool saw_expected = false;
        phase("partial-host-upload");
        try {
            // Production guard precedes submission and outlives the borrowed
            // queue reference. Probe owns all buffers throughout the unwind.
            strata::prefill::detail::DraftUploadGuard lease([&] {
                strata::retire_queue(queue); ++unwind_drains; return true;
            }, [] { strata::prefill::detail::retirement_fail_stop("unwind retirement unavailable"); });
            lease.arm();
            // Discard the returned event intentionally: emulate a HOST failure
            // before event publication, AFTER the real copy call returns.
            (void)queue.memcpy(device, input.data(), bytes); ++uploads;
            ++injected_host_errors;
            throw AcceptedHostError{};
        } catch (const AcceptedHostError&) { saw_expected = true; }
        require(saw_expected && injected_host_errors == 1 && unwind_drains == 1,
                "expected host boundary/production unwind drain not observed");
        auto download = queue.memcpy(output.data(), device, bytes); ++readbacks;
        observe(download, queue);
        finished(download, queue);
        for (size_t i = 0; i < cells; ++i)
            require(input[i] == 149u + uint32_t(i) && output[i] == input[i],
                    "accepted copy/source lifetime mismatch after host exception");
        phase("accepted-copy-after-host-error-verified");
        // This is a real copy plus simulated host failure. No claim of an
        // actual provider partial-acceptance/publication failure follows.
    }
    void destroy_owned() {
        require(owned != nullptr && owned != borrowed, "no owned queue to destroy");
        strata::retire_queue(*owned);
        strata::retire_queue(*borrowed);
        auto* old = owned;
        // Cleanup exceptions are terminal with buffers/borrowed queue still live.
        try {
            dpct::get_current_device().destroy_queue(owned);
        } catch (...) { std::_Exit(88); }
        require(owned == nullptr, "DPCT did not clear destroyed owner");
        try { (void)strata::pr6_require_checked_queue(old); }
        catch (const std::invalid_argument&) { expired_owner_rejected = true; }
        require(expired_owner_rejected, "expired raw pointer was still registered");
        require(strata::q_of(nullptr) == borrowed && strata::q_of(borrowed) == borrowed,
                "borrowed default changed after owned queue destruction");
        phase("borrowed-default-survival");
        baseline(*borrowed, 73u); // Meaningful copy + kernel after owned destruction.
        default_survival = true;
    }
    void close() noexcept {
        if (closed) return;
        bool quiescent = false;
        try {
            strata::prefill::detail::RetirementDeadline deadline;
            if (owned) strata::retire_queue(*owned);
            if (borrowed) strata::retire_queue(*borrowed);
            quiescent = true;
            // No free before accepted work is retired, including submit throws.
            if (device) sycl::free(device, *borrowed);
            device = nullptr;
            if (owned) dpct::get_current_device().destroy_queue(owned);
            borrowed = nullptr; // Borrowed queue is NEVER destroyed here.
            closed = true;
        } catch (...) { std::_Exit(quiescent ? 88 : 86); }
    }
    ~Probe() { close(); }
};
} // namespace
int main(int argc, char** argv) {
    std::set_terminate([] { std::_Exit(88); });
    Options o;
    try { o = parse(argc, argv); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 64; }
    if (o.async_hold) {
        // Installed exception_list exposes iteration but only private PushBack.
        // No empty-list call, private API, ABI hack, host_task or device fault.
        std::fprintf(stderr, "HOLD async87: pinned SYCL exception_list has no public nonempty host constructor; handler delivery UNQUALIFIED\n");
        return 77;
    }
    const char* selector = std::getenv("ONEAPI_DEVICE_SELECTOR");
    if (!selector || std::string_view(selector) != "level_zero:gpu") {
        std::fprintf(stderr, "set ONEAPI_DEVICE_SELECTOR=level_zero:gpu in child environment before exec\n");
        return 64;
    }
    try {
        // Starts before ANY DPCT/runtime call. Main still owns the external
        // 90s direct / 120s total deadline, PID/start/maps and fault receipts.
        strata::prefill::detail::RetirementDeadline deadline;
        Probe p;
        main_gate("before-runtime");
        p.init(o);
        main_gate("before-operations");
        auto& queue = *(o.owned ? p.owned : p.borrowed);
        if (o.partial) p.partial_host(queue);
        p.baseline(queue, 11u);
        if (o.owned) p.destroy_owned();
        require(p.kernels == (o.owned ? 2u : 1u) &&
                p.uploads == p.kernels + (o.partial ? 1u : 0u) && p.readbacks == p.uploads &&
                p.injected_host_errors == (o.partial ? 1u : 0u), "incomplete operation receipt");
        p.close();
        phase("cleanup-complete");
        main_gate("after-cleanup");
        const int written = std::printf(
            "{\"status\":\"PROBE_CASE_COMPLETE\",\"case\":\"%.*s\",\"device_ordinal\":%u,"
            "\"cells\":32,\"usm_bytes\":128,\"uploads\":%u,\"kernels\":%u,\"readbacks\":%u,"
            "\"pending_event_observations\":%u,\"complete_event_observations\":%u,\"pending_queue_observations\":%u,"
            "\"injected_HOST_errors\":%u,\"unwind_drains\":%u,\"default_survival\":%s,"
            "\"expired_owner_rejected\":%s,\"cleanup_complete\":true,\"external_fault_receipt_required\":true}\n",
            int(o.name.size()), o.name.data(), o.device, p.uploads, p.kernels, p.readbacks,
            p.pending_events, p.complete_events, p.pending_queues, p.injected_host_errors, p.unwind_drains,
            p.default_survival ? "true" : "false", p.expired_owner_rejected ? "true" : "false");
        return written > 0 && std::fflush(stdout) == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PROBE_FAILED after owner retirement: %s\n", e.what()); return 1;
    } catch (...) { std::fprintf(stderr, "PROBE_FAILED unexpected host exception after retirement\n"); return 1; }
}
