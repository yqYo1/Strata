// Main-authored host bound/order fixture; repository LICENSE.
// Payload callbacks check metadata,not GPU copies or full-byte payloads.
#include "strata/core/adaptive_residency_audit.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
using Audit = strata::core::AdaptiveResidencyAudit;
using Swap = Audit::Swap;
using Map = std::vector<int32_t>;
constexpr std::size_t model_max_bytes = 2662400;
static_assert(Audit::max_expert_payload_bytes == (64u << 20));
static_assert(Audit::max_cycle_payload_bytes == (256u << 20));
static_assert(96 * model_max_bytes <= Audit::max_cycle_payload_bytes);
std::size_t cases = 0;

void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

struct Cycle {
    std::size_t count, bytes, victims = 0, incoming = 0, waits = 0;
    Map host, device, private_map;
    std::vector<Swap> changes;
    Audit audit;
    std::string err;

    Cycle(std::size_t n, std::size_t extent, std::size_t limit = 0)
        : count(n), bytes(extent), host(512, -1), device(512, -1), private_map(512, -1),
          audit({extent}, 512, n, limit ? limit : n) {
        for (std::size_t i = 0; i < n; ++i) {
            host[i] = device[i] = private_map[i] = int32_t(i);
            changes.push_back({256 + i, i, int32_t(i)});
        }
    }
    auto lookup() { return [this](auto row) { return private_map.at(row); }; }
    bool prepare() {
        return audit.prepare(host, device, changes, lookup(),
            [&](const Swap& s, auto extent, std::string&) {
                ++victims;
                return extent == bytes && s.outgoing == std::size_t(s.slot) &&
                       s.incoming == 256 + s.outgoing;
            }, err);
    }
    bool mask() {
        for (const auto& s : changes) host[s.outgoing] = device[s.outgoing] = private_map[s.outgoing] = -1;
        return audit.masked(host, device, lookup(), err);
    }
    bool complete(bool event_ok = true, std::size_t bad = std::size_t(-1)) {
        return audit.complete([&] { ++waits; return event_ok; },
            [&](const Swap& s, auto extent, std::string&) {
                require(waits == 1, "payload callback preceded the checked event");
                ++incoming;
                return extent == bytes && std::size_t(s.slot) != bad;
            }, err);
    }
    bool publish(int corrupt = -1) {
        for (const auto& s : changes) host[s.incoming] = device[s.incoming] = private_map[s.incoming] = s.slot;
        if (corrupt == 0) host[256] = -1;
        if (corrupt == 1) device[256] = -1;
        if (corrupt == 2) private_map[256] = -1;
        return audit.published(host, device, lookup(), err);
    }
};

void positive(std::size_t count, std::size_t bytes) {
    Cycle c(count, bytes);
    require(c.prepare() && c.victims == count && c.audit.generation() == 1,
            "bounded group preparation failed");
    require(c.mask() && c.complete() && c.incoming == count && c.publish() &&
            c.audit.idle() && c.audit.completed() == 1 && c.err.empty(),
            "bounded group did not publish exactly once");
    ++cases;
}

void over_cycle() {
    Cycle c(5, Audit::max_expert_payload_bytes);
    require(!c.prepare() && c.victims == 0 && c.incoming == 0 && c.waits == 0 &&
            c.audit.failed() && c.audit.completed() == 0 &&
            c.err == "adaptive cache audit: replacement payload budget",
            "aggregate overflow reached a payload callback");
    require(!c.prepare() && !c.publish(), "failed aggregate audit was reused");
    ++cases;

    Map map(1024, -1);
    std::vector<Swap> swaps;
    for (int32_t i = 0; i < 4; ++i) { map[i] = i; swaps.push_back({std::size_t(256 + i), std::size_t(i), i}); }
    map[512] = 4;
    swaps.push_back({768, 512, 4});
    Audit a({Audit::max_expert_payload_bytes, 1}, 512, 5, 5);
    std::string err;
    std::size_t callbacks = 0;
    require(!a.prepare(map, map, swaps, [&](auto row) { return map.at(row); },
        [&](const Swap&, auto, std::string&) { ++callbacks; return true; }, err) &&
        callbacks == 0 && a.failed() && a.completed() == 0 &&
        err == "adaptive cache audit: replacement payload budget", "cycle cap plus one byte passed");
    ++cases;
}

void refusals() {
    for (const auto bytes : {std::size_t(0), Audit::max_expert_payload_bytes + 1,
            Audit::max_cycle_payload_bytes, std::numeric_limits<std::size_t>::max()}) {
        bool refused = false;
        try { Audit a({bytes}, 2, 1, 1); }
        catch (const std::invalid_argument&) { refused = true; }
        require(refused, "invalid per-expert extent was accepted");
        ++cases;
    }
    Cycle limit(33, model_max_bytes, 32);
    require(!limit.prepare() && limit.victims == 0 && limit.audit.failed(), "swap limit was weakened");
    ++cases;

    Cycle event(32, model_max_bytes);
    require(event.prepare() && event.mask() && !event.complete(false) && event.waits == 1 &&
            event.incoming == 0 && event.audit.failed() && !event.publish() &&
            event.audit.completed() == 0, "failed event permitted incoming checks or publication");
    ++cases;

    Cycle payload(96, model_max_bytes);
    require(payload.prepare() && payload.mask() && !payload.complete(true, 95) && payload.incoming == 96 &&
            payload.audit.failed() && !payload.publish() && payload.audit.completed() == 0,
            "last incoming refusal permitted partial publication");
    ++cases;
    for (int view = 0; view < 3; ++view) {
        Cycle c(96, model_max_bytes);
        require(c.prepare() && c.mask() && c.complete() && !c.publish(view) &&
                c.audit.failed() && c.audit.completed() == 0, "partial map publication passed");
        ++cases;
    }
}
} // namespace

int main() {
    positive(32, model_max_bytes);
    positive(96, model_max_bytes);
    positive(4, Audit::max_expert_payload_bytes - 1);
    positive(4, Audit::max_expert_payload_bytes);
    over_cycle();
    refusals();
    require(cases == 16, "incomplete payload budget fixture");
    std::printf("adaptive_payload_budget cases=%zu status=PASS\n", cases);
}
