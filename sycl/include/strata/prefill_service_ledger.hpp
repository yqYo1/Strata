#pragma once
// Diagnostic-only returned-event accounting. No queue commands or waits here.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace strata::prefill::diagnostic {
struct ServiceId {
    int64_t chunk = -1, layer = -1, expert = -1;
    int role = 0; // 0 GU, 1 Down, 2 expert copy, 3 activation, 4 device residual
    int source = 0; // 0 GEMM, 1 pinned blob, 2 host stager, 3 activation, 4 device
    int ring = -1;
    int64_t rows = 0, n = 0, k = 0, ldy = 0;
};
struct ServiceTotals {
    uint64_t calls = 0, bytes = 0, event_ns = 0, queue_delay_ns = 0;
    uint64_t rows = 0, expert_id_sum = 0, expert_id_xor = 0;
    uint64_t first_sequence = 0, last_sequence = 0;
    int ring_min = -1, ring_max = -1;
    double host_submit_ms = 0, stager_wait_ms = 0, ring_wait_ms = 0;
};
// Ops supplies admission, completion, async error delivery and the three timestamps.
// Each instance has exactly one producer; fold/report run only after its producer joins.
template<class Event, class Queue, class Ops> class ServiceLedger {
public:
    const bool on;
    const char* name;
    bool valid = true, admitted = false;
    const char* reason = "none";
    const char* fatal_reason = "none";
    size_t diagnostics_printed = 0;
    Queue* queue = nullptr;
    size_t attempts = 0, submitted = 0, retained = 0, folded = 0, dropped = 0;
    size_t query_attempts = 0, query_successes = 0, query_failures = 0, backward = 0;
    size_t event_cap, bucket_cap;
    uint64_t raw_submit = 0, raw_start = 0, raw_end = 0;
    uint64_t rejected_submit = 0, rejected_start = 0, rejected_end = 0;
    struct Record { Event event; ServiceId id; uint64_t sequence, bytes; double host_ms, stager_ms, ring_ms; };
    std::vector<Record> pending;
    std::optional<Event> latest;
    using Shape = std::tuple<int,int,int64_t,int64_t,int64_t,int64_t>;
    using Coverage = std::tuple<int64_t,int64_t,int>;
    std::map<Shape, ServiceTotals> shapes;
    std::map<Coverage, ServiceTotals> coverage;
    ServiceTotals total;
    using PairKey = std::tuple<int64_t,int64_t,int64_t>;
    std::map<PairKey,int64_t> expected_pairs;
    uint64_t expected_registered = 0, completed_pairs = 0;
    bool reported = false;
    // One existing chunk wait can follow 48*512*2=49152 expert products.
    // Keep that entire chunk bounded without adding a completion wait.
    explicit ServiceLedger(bool enabled, const char* label, size_t events = 65536, size_t buckets = 16384)
        : on(enabled), name(label), event_cap(events), bucket_cap(buckets) {}
    void invalidate(const char* why) { if (valid) reason = why; valid = false; }
    void admit(Queue* q) {
        if (!on) return;
        if (queue) { if (queue != q) invalidate("queue_changed"); return; }
        queue = q;
        try { admitted = q && Ops::profiling(*q); }
        catch (...) { invalidate("admission_query_failure"); return; }
        if (!admitted) invalidate("profiling_unavailable");
        // Admission precedes the issuer thread and timed submissions. Retain
        // vector capacity across folds instead of growing it in record().
        if (admitted) {
            try { pending.reserve(event_cap); }
            catch (...) { fatal("record_reservation_failure"); }
        }
    }
    void expect_gemm(const ServiceId& id) {
        if (!on) return;
        if (id.chunk < 0 || id.layer < 0 || id.expert < 0 || id.rows <= 0) { invalidate("expected_identity"); return; }
        if (expected_pairs.size() >= event_cap) { invalidate("expected_pair_cap_overflow"); return; }
        try {
            if (!expected_pairs.emplace(PairKey{id.chunk,id.layer,id.expert},id.rows).second)
                invalidate("duplicate_expected_pair");
            else add(expected_registered,1);
        } catch (...) { fatal("expected_pair_storage_failure"); }
    }
    void attempt() { if (on) ++attempts; }
    void record(const Event& event, Queue* q, ServiceId id, uint64_t bytes, double host_ms, double stager_ms = 0, double ring_ms = 0) {
        if (!on) return;
        ++submitted;
        latest = event; // Also guard completion of dropped records on error returns.
        admit(q);
        if (id.layer < 0 || id.role < 0 || id.role > 4 ||
            (id.role < 2 && (id.chunk < 0 || id.expert < 0 || id.rows <= 0 || id.n <= 0 || id.k <= 0 || id.ldy < id.n)) ||
            (id.role == 2 && (id.expert < 0 || id.source < 1 || id.source > 2)) ||
            !std::isfinite(host_ms) || host_ms < 0 || !std::isfinite(stager_ms) || stager_ms < 0 || !std::isfinite(ring_ms) || ring_ms < 0)
            invalidate("record_identity_or_host_time");
        if (!valid) { ++dropped; return; }
        if (pending.size() >= event_cap) { invalidate("event_cap_overflow"); ++dropped; return; }
        try { pending.push_back({event, id, (uint64_t)submitted, bytes, host_ms, stager_ms, ring_ms}); ++retained; }
        catch (...) { fatal("record_storage_failure"); }
    }
    bool add(uint64_t& destination, uint64_t amount) {
        if (amount > UINT64_MAX - destination) { invalidate("arithmetic_overflow"); return false; }
        destination += amount; return true;
    }
    void accumulate(ServiceTotals& t, const Record& r, uint64_t event_ns, uint64_t delay_ns) {
        if (!t.calls) t.first_sequence = r.sequence;
        t.last_sequence = r.sequence;
        add(t.calls, 1); add(t.bytes, r.bytes); add(t.event_ns, event_ns); add(t.queue_delay_ns, delay_ns);
        if (r.id.rows > 0) add(t.rows, (uint64_t)r.id.rows);
        if (r.id.expert >= 0) { add(t.expert_id_sum, (uint64_t)r.id.expert); t.expert_id_xor ^= (uint64_t)r.id.expert; }
        if (r.id.ring >= 0) { t.ring_min = t.ring_min < 0 ? r.id.ring : std::min(t.ring_min, r.id.ring); t.ring_max = std::max(t.ring_max,r.id.ring); }
        t.host_submit_ms += r.host_ms; t.stager_wait_ms += r.stager_ms; t.ring_wait_ms += r.ring_ms;
        if (!std::isfinite(t.host_submit_ms) || !std::isfinite(t.stager_wait_ms) || !std::isfinite(t.ring_wait_ms)) invalidate("host_sum_nonfinite");
    }
    // required=false retires only already complete events at an existing compute sync.
    // required=true follows the existing wait for this lane's queue.
    void fold(bool required = true) {
        if (!on || !queue) return;
        try { Ops::async_errors(*queue); } catch (...) { fatal("async_queue_failure"); }
        size_t keep = 0;
        // Exact GU/Down pair check while this bounded completed batch still has identities.
        std::map<std::tuple<int64_t,int64_t,int64_t>, std::pair<int,int64_t>> pairs;
        for (size_t i = 0; i < pending.size(); ++i) {
            bool complete = false;
            try { complete = Ops::complete(pending[i].event); }
            catch (...) { fatal("completion_unknown"); }
            if (!complete) {
                if (required) fatal("event_incomplete");
                if (keep != i) pending[keep] = std::move(pending[i]);
                ++keep; continue;
            }
            const auto& r = pending[i];
            ++folded;
            if (r.id.role < 2) {
                try {
                    auto& pair = pairs[{r.id.chunk,r.id.layer,r.id.expert}];
                    const int bit = 1 << r.id.role;
                    if ((pair.first & bit) || (pair.first && pair.second != r.id.rows)) invalidate("gemm_pair_identity");
                    pair.first |= bit; pair.second = r.id.rows;
                } catch (...) { fatal("pair_storage_failure"); }
            }
            uint64_t stamp[3] = {};
            bool queried = true;
            for (int field = 0; field < 3; ++field) {
                ++query_attempts;
                try { stamp[field] = Ops::timestamp(r.event, field); ++query_successes; }
                catch (const std::exception& error) {
                    ++query_failures; queried = false; invalidate("profiling_query_failure");
                    if (diagnostics_printed++ < 4)
                        std::fprintf(stderr,"strata service profiling query failure: %s sequence %llu field %d: %s\n",
                            name,(unsigned long long)r.sequence,field,error.what());
                } catch (...) {
                    ++query_failures; queried = false; invalidate("profiling_query_failure");
                    if (diagnostics_printed++ < 4) std::fprintf(stderr,"strata service profiling query failure: %s unknown exception\n",name);
                }
            }
            if (!queried) continue;
            raw_submit = stamp[0]; raw_start = stamp[1]; raw_end = stamp[2];
            if (raw_submit > raw_start || raw_start > raw_end) {
                if (!backward) { rejected_submit = raw_submit; rejected_start = raw_start; rejected_end = raw_end; }
                ++backward; invalidate("backward_event_timestamps");
                if (diagnostics_printed++ < 4) std::fprintf(stderr, "strata service event rejected: %s chunk %lld layer %lld expert %lld role %d submit %llu start %llu end %llu\n",
                    name, (long long)r.id.chunk, (long long)r.id.layer, (long long)r.id.expert, r.id.role,
                    (unsigned long long)raw_submit, (unsigned long long)raw_start, (unsigned long long)raw_end);
                continue;
            }
            if (!valid) continue; // Never aggregate partial invalid telemetry.
            const Shape shape{r.id.role,r.id.source,r.id.rows,r.id.n,r.id.k,r.id.ldy};
            const Coverage span{r.id.chunk,r.id.layer,r.id.role};
            if ((!shapes.count(shape) && shapes.size() >= bucket_cap) ||
                (!coverage.count(span) && coverage.size() >= bucket_cap)) {
                invalidate("aggregate_cap_overflow"); continue;
            }
            try {
                accumulate(total, r, raw_end - raw_start, raw_start - raw_submit);
                accumulate(shapes[shape], r, raw_end - raw_start, raw_start - raw_submit);
                accumulate(coverage[span], r, raw_end - raw_start, raw_start - raw_submit);
            } catch (...) { fatal("aggregate_storage_failure"); }
        }
        for (const auto& pair : pairs) {
            const auto expected = expected_pairs.find(pair.first);
            if (pair.second.first != 3) invalidate("gemm_pair_incomplete");
            else if (expected == expected_pairs.end() || expected->second != pair.second.second)
                invalidate("routed_pair_coverage");
            else { add(completed_pairs,1); expected_pairs.erase(expected); }
        }
        pending.resize(keep);
        if (required && latest) {
            try { if (!Ops::complete(*latest)) fatal("last_event_incomplete"); }
            catch (...) { fatal("last_completion_unknown"); }
        }
    }
    void receipt() const {
        std::fprintf(stderr, "strata prefill service validity: {\"lane\":\"%s\",\"status\":\"%s\",\"reason\":\"%s\",\"fatal_reason\":\"%s\","
            "\"admitted\":%s,\"attempts\":%zu,\"submitted\":%zu,\"retained\":%zu,\"folded\":%zu,\"pending\":%zu,\"dropped\":%zu,"
            "\"query_attempts\":%zu,\"query_successes\":%zu,\"query_failures\":%zu,\"backward\":%zu,"
            "\"raw_submit_ns\":%llu,\"raw_start_ns\":%llu,\"raw_end_ns\":%llu,\"rejected_submit_ns\":%llu,\"rejected_start_ns\":%llu,\"rejected_end_ns\":%llu,\"event_cap\":%zu,\"bucket_cap\":%zu,\"expected_registered\":%llu,\"completed_role_pairs\":%llu,\"expected_pending\":%zu}\n",
            name, valid ? "valid" : "invalid", reason, fatal_reason, admitted ? "true" : "false", attempts, submitted, retained,
            folded, pending.size(), dropped, query_attempts, query_successes, query_failures, backward,
            (unsigned long long)raw_submit, (unsigned long long)raw_start, (unsigned long long)raw_end, (unsigned long long)rejected_submit, (unsigned long long)rejected_start,
            (unsigned long long)rejected_end, event_cap, bucket_cap, (unsigned long long)expected_registered,
            (unsigned long long)completed_pairs, expected_pairs.size());
    }
    [[noreturn]] void fatal(const char* why) {
        fatal_reason = why; invalidate(why); receipt(); std::fflush(stderr); std::_Exit(EXIT_FAILURE);
    }
    static void totals_json(const ServiceTotals& t) {
        std::fprintf(stderr, "\"first_sequence\":%llu,\"last_sequence\":%llu,\"rows\":%llu,\"expert_id_sum\":%llu,\"expert_id_xor\":%llu,\"ring_min\":%d,\"ring_max\":%d,\"calls\":%llu,\"logical_bytes\":%llu,\"returned_event_ms\":%.6f,\"submit_to_start_ms\":%.6f,\"host_submit_ms\":%.6f,\"host_stager_wait_ms\":%.6f,\"host_ring_publication_wait_ms\":%.6f",
            (unsigned long long)t.first_sequence, (unsigned long long)t.last_sequence, (unsigned long long)t.rows, (unsigned long long)t.expert_id_sum, (unsigned long long)t.expert_id_xor,
            t.ring_min, t.ring_max, (unsigned long long)t.calls, (unsigned long long)t.bytes, t.event_ns/1e6, t.queue_delay_ns/1e6,
            t.host_submit_ms, t.stager_wait_ms, t.ring_wait_ms);
    }
    void report() {
        if (!on) return;
        if (!admitted || attempts != submitted || retained + dropped != submitted ||
            folded != retained || !pending.empty() || query_attempts != 3*folded ||
            query_successes != query_attempts || query_failures || backward || dropped)
            invalidate("final_reconciliation");
        if (expected_registered != completed_pairs || !expected_pairs.empty()) invalidate("routed_pair_reconciliation");
        if (valid && total.calls != submitted) invalidate("aggregate_reconciliation");
        receipt(); reported = true;
        if (!valid) return;
        std::fprintf(stderr, "strata prefill returned-event ledger: {\"lane\":\"%s\",\"scope\":\"returned routine or copy event; not exclusive kernel busy\",", name);
        totals_json(total);
        std::fprintf(stderr, ",\"shapes\":[");
        bool first = true;
        for (const auto& item : shapes) {
            const auto& key = item.first;
            std::fprintf(stderr, "%s{\"role\":%d,\"source\":%d,\"rows\":%lld,\"N\":%lld,\"K\":%lld,\"ldy\":%lld,",
                first ? "" : ",", std::get<0>(key), std::get<1>(key), (long long)std::get<2>(key),
                (long long)std::get<3>(key), (long long)std::get<4>(key), (long long)std::get<5>(key));
            first = false; totals_json(item.second); std::fprintf(stderr, "}");
        }
        std::fprintf(stderr, "],\"coverage\":["); first = true;
        for (const auto& item : coverage) {
            std::fprintf(stderr, "%s{\"chunk_position\":%lld,\"layer\":%lld,\"role\":%d,",
                first ? "" : ",", (long long)std::get<0>(item.first), (long long)std::get<1>(item.first), std::get<2>(item.first));
            first = false; totals_json(item.second); std::fprintf(stderr, "}");
        }
        std::fprintf(stderr, "]}\n");
    }
    ~ServiceLedger() {
        if (!on || !queue) return;
        if (std::uncaught_exceptions()) fatal("exception_unwinding");
        try {
            if (latest && !Ops::complete(*latest)) fatal("destruction_event_incomplete");
            Ops::async_errors(*queue);
        } catch (...) { fatal("destruction_completion_or_async_unknown"); }
        if (!reported && (attempts || submitted)) { invalidate("report_not_reached"); receipt(); }
    }
};
} // namespace strata::prefill::diagnostic
