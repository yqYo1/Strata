#include "strata/core/verify_window_policy.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

using strata::core::VerifyInput;
using strata::core::VerifyWindowGraphCache;
using strata::core::verify_q6_columns;

static void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() try {
    // Expected selections come from the committed component decision, not a
    // second implementation of the dispatch predicate. Bit c means columns c.
    const int eligible[][3] = {
        {2560, 10240, 0x11e}, {6144, 2560, 0x11e}, {2560, 6144, 0x11e},
        {2560, 640, 0x11a}, {2560, 512, 0x100}, {2560, 12288, 0x11e},
        {2560, 248320, 0x11e},
    };
    int selected = 0;
    for (const auto& row : eligible) for (int c = 0; c <= 9; ++c) {
        const bool expected = (row[2] & (1 << c)) != 0;
        need(verify_q6_columns(true, VerifyInput::decode, true, 14, row[0], row[1], c) == expected,
             "dispatch differs from the repeated component decision");
        selected += expected;
        need(!verify_q6_columns(true, VerifyInput::prompt, true, 14, row[0], row[1], c),
             "decode candidate reached prompt reads");
        need(!verify_q6_columns(false, VerifyInput::decode, true, 14, row[0], row[1], c),
             "disabled candidate reached decode");
        need(!verify_q6_columns(true, VerifyInput::decode, false, 14, row[0], row[1], c),
             "exact-layout candidate reached another reduction layout");
        for (int type = -1; type < 40; ++type) if (type != 14)
            need(!verify_q6_columns(true, VerifyInput::decode, true, type, row[0], row[1], c),
                 "Q6 candidate changed another weight format");
    }
    need(selected == 30, "wrong number of qualified shapes");
    for (const auto& shape : {std::array<int, 2>{-2560, 10240}, {2560, 10239},
                              {2561, 10240}, {0, 0}, {2560, -1}})
        need(!verify_q6_columns(true, VerifyInput::decode, true, 14, shape[0], shape[1], 4),
             "unmeasured geometry reached candidate");

    // Actual cache: prompt4 -> decode4 -> prompt4 must recover the original
    // handles and route metadata. A T-only cache fails this request sequence.
    VerifyWindowGraphCache<int> cache;
    auto& prompt = cache.at(VerifyInput::prompt, 4);
    prompt.layer = {11}; prompt.resident = {12}; prompt.pcie = {13};
    prompt.q6_reference_calls = 41;
    prompt.ready = true;
    auto& decode = cache.at(VerifyInput::decode, 4);
    need(decode.layer.empty() && decode.resident.empty() && decode.pcie.empty(), "phase cache alias");
    decode.layer = {21}; decode.resident = {22}; decode.pcie = {23};
    decode.q6_columns_calls = 42;
    need(!decode.ready, "a partial graph cache became replayable");
    decode.ready = true;
    need(cache.at(VerifyInput::prompt, 4).layer == std::vector<int>{11} &&
         cache.at(VerifyInput::prompt, 4).q6_columns_calls == 0 &&
         cache.at(VerifyInput::prompt, 4).q6_reference_calls == 41 && prompt.ready,
         "decode changed an already captured prompt window");
    cache.at(VerifyInput::decode, 8).layer = {31};
    need(cache.at(VerifyInput::decode, 4).layer == std::vector<int>{21}, "token cache alias");
    int refused = 0;
    for (int t : {0, 9, -1}) try { cache.at(VerifyInput::decode, t); }
        catch (const std::out_of_range&) { ++refused; }
    try { cache.at(static_cast<VerifyInput>(255), 4); }
        catch (const std::out_of_range&) { ++refused; }
    need(refused == 4, "invalid graph cache key accepted");

    // An interrupted destroy keeps the remaining owners; a retry must not
    // destroy successful predecessors twice or omit the other phase/families.
    std::vector<int> destroyed;
    bool interrupted = false;
    try {
        cache.clear([&](int handle) {
            if (handle == 12) throw std::runtime_error("intentional host-only destroy refusal");
            destroyed.push_back(handle);
        });
    } catch (const std::runtime_error&) { interrupted = true; }
    need(interrupted && destroyed == std::vector<int>{11} && prompt.resident[0] == 12 &&
         decode.layer[0] == 21, "failed cleanup released remaining graph owners");
    cache.clear([&](int handle) { destroyed.push_back(handle); });
    need(destroyed == std::vector<int>({11, 12, 13, 21, 22, 23, 31}),
         "retirement missed or double-destroyed a phase graph");
    need(prompt.layer.empty() && prompt.resident.empty() && prompt.pcie.empty() &&
         decode.layer.empty() && decode.resident.empty() && decode.pcie.empty() &&
         prompt.q6_reference_calls == 0 && decode.q6_columns_calls == 0 && !prompt.ready && !decode.ready,
         "retirement left stale graphs or route metadata");
    cache.clear([&](int) { throw std::runtime_error("second retirement is not empty"); });
    cache.at(VerifyInput::prompt, 4).layer = {51};
    need(cache.at(VerifyInput::decode, 4).layer.empty(), "retired decode graph survived prompt recapture");
    cache.clear([](int) {});
    std::puts("verify-window-policy: passed selections=30 phase-roundtrip=1 invalid-keys=4 retirement=1");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "verify-window-policy: %s\n", error.what());
    return 1;
}
