// Main-authored ownership/admission checks; repository LICENSE applies.
// This is not independent arithmetic or performance qualification.
#include "cache.hpp"
#include <array>
#include <iostream>
#include <thread>

using compact_gu_cache::Cache;
using compact_gu_cache::Lease;
using strata::kernels::cpu::NativeFmt;
using strata::kernels::cpu::NativeGuPhase;

[[gnu::cold]] void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        need(compact_gu_cache::available(), "this component needs AVX2/FMA/F16C");
        NativeFmt format;
        format.gu_type = GGML_TYPE_IQ3_XXS;
        format.d_type = GGML_TYPE_IQ4_NL;
        format.gu_act = GGML_TYPE_Q8_K;
        format.n_embd = 2560;
        format.n_ff = 640;
        format.gu_row = 980;
        format.up_off = compact_gu_cache::raw_projection_bytes;
        format.down_off = 2 * format.up_off;
        format.bytes = format.down_off;
        format.act_bytes = 10 * sizeof(block_q8_K);

        std::vector<uint8_t> raw(2 * compact_gu_cache::raw_projection_bytes);
        for (size_t i = 0; i < raw.size(); i += sizeof(block_iq3_xxs)) {
            const uint16_t half_one = 0x3c00;
            std::memcpy(raw.data() + i, &half_one, sizeof half_one);
            for (size_t j = 2; j < sizeof(block_iq3_xxs); ++j)
                raw[i + j] = uint8_t(i / sizeof(block_iq3_xxs) + 37 * j);
        }
        std::array<block_q8_K, 10> activation{};
        for (auto& block : activation) {
            block.d = 0.0078125f;
            for (int i = 0; i < 256; ++i) block.qs[i] = int8_t(i % 31 - 15);
        }
        const auto act = std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(activation.data()), sizeof activation);
        std::array<float, 640> before{}, after{};
        Lease lease;
        {
            Cache cache(1, 2, 8u << 20);
            const size_t empty_bytes = cache.known_bytes();
            need(!cache.lookup({0, 0}, format, NativeGuPhase::Decode, 1), "unfrozen lookup");
            need(!cache.prepare({1, 0}, format, raw), "out-of-range preparation");
            need(!cache.prepare({0, 0}, format, std::span<const uint8_t>(raw).first(raw.size() - 1)),
                 "short raw source");
            need(cache.prepare({0, 0}, format, raw), "prepare valid IQ3_XXS GU");
            need(!cache.prepare({0, 0}, format, raw), "duplicate key must refuse replacement");
            need(cache.prepared_entries() == 1 &&
                cache.known_bytes() == empty_bytes + Cache::entry_known_bytes(), "known byte accounting");
            cache.freeze();
            need(!cache.prepare({0, 1}, format, raw), "insertion after publication");
            need(!cache.lookup({0, 0}, format, NativeGuPhase::Prefill, 1), "prefill must retain raw route");
            need(!cache.lookup({0, 0}, format, NativeGuPhase::Decode, 2), "NT2 must retain raw route");
            auto wrong = format;
            wrong.gu_type = GGML_TYPE_IQ2_S;
            need(!cache.lookup({0, 0}, wrong, NativeGuPhase::Decode, 1), "wrong GU format");
            wrong = format;
            wrong.d_type = 42;
            need(!cache.lookup({0, 0}, wrong, NativeGuPhase::Decode, 1), "different model format key");
            need(!cache.lookup({0, 1}, format, NativeGuPhase::Decode, 1), "unprepared key");
            lease = cache.lookup({0, 0}, format, NativeGuPhase::Decode, 1);
            need(bool(lease) && lease.guards_intact(), "published lease");
            need(lease.gu_rows(NativeGuPhase::Decode, 1, act, before, 0, 640), "first owned call");
            need(!lease.gu_rows(NativeGuPhase::Prefill, 1, act, after, 0, 640), "lease phase handoff");
            need(!lease.gu_rows(NativeGuPhase::Decode, 1, act, after, -1, 640), "negative row bounds");
            need(!lease.gu_rows(NativeGuPhase::Decode, 1, act, after, 0, 641), "row upper bound");
            for (float value : after) need(value == 0.f, "refused calls must not write output");
            std::array<bool, 6> ok{};
            std::array<std::jthread, 6> workers;
            for (size_t t = 0; t < workers.size(); ++t) {
                workers[t] = std::jthread([&, t] {
                    auto part = cache.lookup({0, 0}, format, NativeGuPhase::Decode, 1);
                    const int first = int(640 * t / workers.size());
                    const int last = int(640 * (t + 1) / workers.size());
                    ok[t] = part.gu_rows(NativeGuPhase::Decode, 1, act, after, first, last);
                });
            }
            for (auto& worker : workers) worker.join();
            for (bool part_ok : ok) need(part_ok, "concurrent frozen lookup/row partition");
            need(std::memcmp(before.data(), after.data(), sizeof before) == 0,
                 "disjoint row partitions changed the lifetime result");

            Cache bounded(1, 2, empty_bytes + Cache::entry_known_bytes());
            need(bounded.prepare({0, 0}, format, raw), "one-entry budget");
            need(!bounded.prepare({0, 1}, format, raw) && bounded.prepared_entries() == 1,
                 "byte budget must refuse second entry");
        }
        raw.clear();
        raw.shrink_to_fit();
        need(lease.gu_rows(NativeGuPhase::Decode, 1, act, after, 0, 640), "lease survives source/cache retirement");
        need(std::memcmp(before.data(), after.data(), sizeof before) == 0 && lease.guards_intact(),
             "owned view changed after retirement");
        for (float value : after) need(std::isfinite(value), "finite lifetime output");
        bool rejected = false;
        try { Cache zero(0, 1, 8u << 20); }
        catch (const std::invalid_argument&) { rejected = true; }
        need(rejected, "zero index geometry");
        rejected = false;
        try { Cache overflow(std::numeric_limits<size_t>::max(), 2, 8u << 20); }
        catch (const std::invalid_argument&) { rejected = true; }
        need(rejected, "index multiplication overflow");
        rejected = false;
        try { Cache underbudget(size_t(1) << 40, 1, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        need(rejected, "index budget refusal must precede a huge allocation");
        std::cout << "compact_cache_lifetime=PASS phase_guard=PASS budget=PASS "
                     "source_retirement=PASS numeric_oracle=UNRUN performance=UNRUN\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
