// Main-authored host fixture; repository LICENSE.
#include "strata/core/adaptive_residency_audit.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
using Audit = strata::core::AdaptiveResidencyAudit;
using Swap = Audit::Swap;
using Map = std::vector<int32_t>;
const Map before{-1, 0, 1, -1, -1, 2, 3, -1};
const Map masked{-1, -1, 1, -1, -1, -1, 3, -1};
const Map published{0, -1, 1, -1, 2, -1, 3, -1};
const std::vector<Swap> swaps{{0, 1, 0}, {4, 5, 2}};
std::size_t cases = 0;

void require(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

struct Fixture {
    Map host = before, device = before, private_map = before;
    std::vector<std::vector<uint8_t>> ram, slots;
    std::size_t victim_checks = 0, incoming_checks = 0, waits = 0;
    bool completed = false;

    Fixture() : ram(8), slots(4) {
        for (std::size_t row = 0; row < ram.size(); ++row) {
            ram[row].resize(row < 4 ? 16 : 24);
            for (std::size_t i = 0; i < ram[row].size(); ++i)
                ram[row][i] = uint8_t(31 * row + i);
            if (before[row] >= 0) slots[std::size_t(before[row])] = ram[row];
        }
    }

    int32_t lookup(std::size_t row) const { return private_map.at(row); }
    bool payload(const Swap& swap, std::size_t row, std::size_t bytes) const {
        return ram.at(row).size() == bytes && slots.at(std::size_t(swap.slot)).size() == bytes &&
               std::memcmp(ram[row].data(), slots[std::size_t(swap.slot)].data(), bytes) == 0;
    }
    bool begin(Audit& audit, std::string& err, const std::vector<Swap>& changes = swaps) {
        return audit.prepare(host, device, changes, [&](auto row) { return lookup(row); },
            [&](const Swap& swap, auto bytes, std::string&) {
                ++victim_checks;
                return payload(swap, swap.outgoing, bytes);
            }, err);
    }
    void mask() { host = device = private_map = masked; }
    bool check_mask(Audit& audit, std::string& err) {
        return audit.masked(host, device, [&](auto row) { return lookup(row); }, err);
    }
    bool copy_and_check(Audit& audit, std::string& err, bool wait_ok = true,
                        bool corrupt = false) {
        for (const auto& swap : swaps) slots[std::size_t(swap.slot)] = ram[swap.incoming];
        if (corrupt) slots[2].back() ^= 1;
        return audit.complete([&] { ++waits; completed = wait_ok; return wait_ok; },
            [&](const Swap& swap, auto bytes, std::string&) {
                require(completed, "payload was inspected before the checked completion");
                ++incoming_checks;
                return payload(swap, swap.incoming, bytes);
            }, err);
    }
    bool publish(Audit& audit, std::string& err) {
        return audit.published(host, device, [&](auto row) { return lookup(row); }, err);
    }
};

void success() {
    Fixture f;
    Audit a({16, 24}, 4, 4, 2);
    std::string err;
    require(a.idle() && f.begin(a, err), "valid preparation failed");
    require(f.victim_checks == 2 && a.generation() == 1 && !a.idle(), "lost prepared group");
    f.mask();
    require(f.check_mask(a, err) && f.copy_and_check(a, err), "valid mask/copy failed");
    f.host = f.device = f.private_map = published;
    require(f.publish(a, err) && a.idle() && !a.failed() && a.completed() == 1,
            "valid publication failed");
    // A later real group reuses one slot with different row identities.
    const std::vector<Swap> next{{1, 0, 0}};
    require(f.begin(a, err, next) && a.generation() == 2, "next generation did not renew");
    f.host[0] = f.device[0] = f.private_map[0] = -1;
    require(f.check_mask(a, err), "next generation mask failed");
    f.slots[0] = f.ram[1];
    require(a.complete([] { return true; }, [&](const Swap& swap, auto bytes, std::string&) {
        return f.payload(swap, swap.incoming, bytes);
    }, err), "next generation payload failed");
    f.host[1] = f.device[1] = f.private_map[1] = 0;
    require(f.publish(a, err) && a.completed() == 2 && err.empty(), "next generation did not close");
    ++cases;
}

void invalid_preparation() {
    const std::vector<std::vector<Swap>> invalid{
        {}, {{0, 1, 0}, {0, 2, 1}}, {{0, 1, 0}, {3, 1, 0}},
        {{0, 5, 2}}, {{8, 1, 0}}, {{0, 8, 0}}, {{0, 1, -1}},
        {{0, 1, 4}}, {{2, 1, 0}}, {{0, 3, 0}}, {{0, 1, 1}},
        {{0, 1, 0}, {4, 5, 2}, {7, 6, 3}}};
    for (const auto& changes : invalid) {
        Fixture f;
        Audit a({16, 24}, 4, 4, 2);
        std::string err;
        require(!f.begin(a, err, changes) && a.failed() && !err.empty(),
                "invalid preparation passed");
        require(f.victim_checks == 0 && f.waits == 0 && a.completed() == 0,
                "invalid geometry reached a payload or wait callback");
        ++cases;
    }
    for (int which = 0; which < 4; ++which) {
        Fixture f;
        Audit a({16, 24}, 4, 4, 2);
        std::string err;
        if (which == 0) f.host[3] = 1;           // aliased before map
        if (which == 1) f.device[1] = -1;        // partial publication
        if (which == 2) f.private_map[1] = -1;   // stale callback view
        if (which == 3) f.slots[0].back() ^= 1;  // outgoing payload mismatch
        require(!f.begin(a, err) && a.failed() && a.completed() == 0,
                "corrupt original snapshot/payload passed");
        ++cases;
    }
    for (int which = 0; which < 2; ++which) {
        Fixture f;
        Audit a({16, 24}, 4, 4, 2);
        std::string err;
        if (which == 0) f.host.pop_back();
        else f.device.pop_back();
        require(!f.begin(a, err) && a.failed() && f.victim_checks == 0,
                "truncated snapshot reached payload inspection");
        ++cases;
    }
}

void failed_transition(int which) {
    Fixture f;
    Audit a({16, 24}, 4, 4, 2);
    std::string err;
    require(f.begin(a, err), "failure fixture preparation");
    f.mask();
    if (which == 0) f.private_map[1] = 0;       // victim callback still hits
    if (which == 1) f.device[0] = 0;            // incoming published before copy
    const bool masked_ok = f.check_mask(a, err);
    if (which < 2) require(!masked_ok, "unsafe mask passed");
    else {
        require(masked_ok, "valid failure-fixture mask failed");
        if (which == 2) {
            require(!f.copy_and_check(a, err, false), "failed completion passed");
            require(f.incoming_checks == 0, "failed completion inspected incoming bytes");
        } else if (which == 3) {
            require(!f.copy_and_check(a, err, true, true), "corrupt incoming bytes passed");
        } else {
            require(f.copy_and_check(a, err), "valid failure-fixture copy failed");
            f.host = f.private_map = published;
            f.device = which == 4 ? masked : published;
            if (which == 5) f.private_map[0] = -1;
            if (which == 6) f.host[1] = 0;      // outgoing reappears
            require(!f.publish(a, err), "partial/corrupt publication passed");
        }
    }
    require(a.failed() && a.completed() == 0, "failure closed a successful cycle");
    require(!f.begin(a, err), "failed auditor was reused");
    require(!f.publish(a, err), "failed auditor authorized publication");
    ++cases;
}

void ordering_and_bounds() {
    for (int which = 0; which < 3; ++which) {
        Fixture f;
        Audit a({16, 24}, 4, 4, 2);
        std::string err;
        if (which == 0) {
            require(!a.complete([] { return true; }, [](const Swap&, auto, std::string&) {
                require(false, "unordered incoming callback ran"); return false;
            }, err), "completion without prepare/mask passed");
        } else {
            require(f.begin(a, err), "ordering preparation failed");
            if (which == 1) require(!f.publish(a, err), "publication before mask/copy passed");
            if (which == 2) require(!f.begin(a, err), "overlapping adaptive groups passed");
        }
        require(a.failed() && a.completed() == 0, "bad ordering did not fail closed");
        ++cases;
    }
    Fixture f;
    Audit a({40u << 20, 40u << 20}, 4, 4, 2);
    std::string err;
    require(!f.begin(a, err) && f.victim_checks == 0, "payload budget was checked after callbacks");
    ++cases;
    for (int which = 0; which < 8; ++which) {
        bool refused = false;
        try {
            const std::vector<std::size_t> bytes = which == 0 ? std::vector<std::size_t>{} :
                std::vector<std::size_t>{which == 6 ? 0 : which == 7 ?
                    Audit::max_cycle_payload_bytes + 1 : std::size_t(16)};
            Audit bad(bytes,
                      which == 1 ? 0 : which == 2 ? Audit::max_entries + 1 : 4,
                      which == 3 ? 0 : which == 5 ? Audit::max_entries + 1 : 4,
                      which == 4 ? 5 : 2);
        } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "invalid constructor geometry passed");
        ++cases;
    }
}

void exceptions_fail_closed() {
    for (int which = 0; which < 3; ++which) {
        Fixture f;
        Audit a({16, 24}, 4, 4, 2);
        std::string err;
        bool caught = false;
        try {
            if (which == 0) {
                (void)a.prepare(f.host, f.device, swaps, [&](auto row) { return f.lookup(row); },
                    [](const Swap&, auto, std::string&) -> bool {
                        throw std::runtime_error("victim check");
                    }, err);
            } else {
                require(f.begin(a, err), "exception preparation failed");
                f.mask();
                require(f.check_mask(a, err), "exception mask failed");
                if (which == 1) {
                    (void)a.complete([]() -> bool { throw std::runtime_error("wait"); },
                        [](const Swap&, auto, std::string&) {
                            require(false, "payload callback ran after wait threw"); return false;
                        }, err);
                } else {
                    require(f.copy_and_check(a, err), "exception copy failed");
                    f.host = f.device = f.private_map = published;
                    (void)a.published(f.host, f.device, [](std::size_t) -> int32_t {
                        throw std::runtime_error("private lookup");
                    }, err);
                }
            }
        } catch (const std::runtime_error&) { caught = true; }
        require(caught && a.failed() && a.completed() == 0, "exception reopened an audit cycle");
        require(!f.begin(a, err), "auditor reused after callback exception");
        ++cases;
    }
}
} // namespace

int main() {
    success();
    invalid_preparation();
    for (int i = 0; i < 7; ++i) failed_transition(i);
    ordering_and_bounds();
    exceptions_fail_closed();
    std::printf("adaptive_residency_audit cases=%zu status=PASS\n", cases);
}
