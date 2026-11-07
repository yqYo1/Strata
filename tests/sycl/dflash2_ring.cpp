// Independently authored stock-DFlash2 state checks; repository LICENSE.
// No target model, performance claim, graph or device-fault injection.
#include "strata/core/dflash2.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/dflash2_taps.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Drafter = strata::core::Dflash2Drafter;
using Tail = strata::prefill::Dflash2PrefillTaps;
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using strata::sycl_backend::Runtime;
constexpr int Width = 2560, Stride = 5 * Width, Heads = 8, HeadDim = 128;
void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
struct Drain {
    sycl::queue& q;
    ~Drain() noexcept {
        try { q.wait_and_throw(); }
        catch (...) { strata::sycl_backend::cleanup_fail_stop("DFlash2 ring fixture completion"); }
    }
};
void pass(const char* name) { std::cout << "dflash2 ring " << name << " PASS\n" << std::flush; }

void tail_transitions(const std::shared_ptr<Runtime>& rt) {
    Tail tail;
    std::string error;
    std::vector<float> input(3 * Stride);
    Allocation device(rt, input.size() * 4, MemoryKind::Device);
    auto& q = rt->compute();
    Drain drain{q};
    auto value = [](int64_t p, int slot, int i) { return float(p) + slot * 0.25f + (i % 17) * 0.03125f; };
    for (int64_t length : {2046, 2047, 2048}) {
        need(tail.begin(length, 0, error), "transition begin");
        for (int64_t start : {int64_t(0), length - 3}) {
            for (int r = 0; r < 3; ++r) for (int slot = 0; slot < 5; ++slot) for (int i = 0; i < Width; ++i)
                input[r * Stride + slot * Width + i] = value(start + r, slot, i);
            q.memcpy(device.data(), input.data(), input.size() * 4).wait_and_throw();
            need(tail.capture_window(device.as<float>(), 3, start, error), "transition window");
        }
        const int64_t first = length == 2048 ? 1 : 0;
        const std::vector<std::pair<int64_t, int64_t>> expected = {{first, 3 - first}, {length - 3, 2}};
        size_t run = 0;
        need(tail.consume([&](const float* data, int64_t rows, int64_t pos, std::string&) {
            need(run < expected.size() && expected[run++] == std::make_pair(pos, rows), "transition positions");
            std::vector<float> actual(size_t(rows) * Stride);
            Drain reading{q};
            q.memcpy(actual.data(), data, actual.size() * 4).wait_and_throw();
            for (int64_t r = 0; r < rows; ++r) for (int slot = 0; slot < 5; ++slot) for (int i = 0; i < Width; ++i)
                need(actual[size_t(r) * Stride + slot * Width + i] == value(pos + r, slot, i), "transition payload");
            return true;
        }, error), "transition consume");
        need(run == 2 && tail.captured_rows() == 5 - first, "transition complete count");
    }
    pass("tail_transitions_2046_2047_2048");
}

struct Snapshot { std::vector<float> kv; std::vector<int64_t> tags; int32_t epoch = -1; };

struct Harness {
    std::shared_ptr<Runtime> rt;
    std::string error;
    int rows;
    Allocation head, taps;
    std::vector<int32_t> vocab;
    bool fail_embed = false;
    int embed_calls = 0;
    Drafter df; // retire its borrowed head/taps before those allocations

    Harness(std::shared_ptr<Runtime> runtime, const std::string& pack, int n)
        : rt(std::move(runtime)), rows(n),
          head(rt, strata::kernels::native_mmvq_weight_bytes(8, Width, 256), MemoryKind::Device),
          taps(rt, size_t(n) * Stride * 4, MemoryKind::Device), vocab(256) {
        std::iota(vocab.begin(), vocab.end(), 100);
        rt->compute().memset(head.data(), 0, head.size()).wait_and_throw();
        set_rows(0, rows, 0);
        ok(df.load(pack, 0, error));
        ok(df.bind_raw([this](const int32_t* ids, int nrows, float* out, sycl::queue* q) {
            ++embed_calls;
            if (fail_embed) throw std::runtime_error("intentional host embedding refusal after context submission");
            q->parallel_for(sycl::range<1>(size_t(nrows) * Width), [=](sycl::id<1> at) {
                const size_t i = at[0];
                out[i] = float((ids[i / Width] % 13) - 6) * 0.03125f + float(int(i % 17) - 8) * 0.015625f;
            });
        }, head.data(), 8, int64_t(vocab.size()), vocab, error));
    }
    void ok(bool value) { if (!value) throw std::runtime_error(error); }
    void set_rows(int begin, int n, int seed) {
        need(begin >= 0 && n >= 0 && begin <= rows && n <= rows - begin, "fixture input extent");
        std::vector<float> input(size_t(n) * Stride);
        for (int r = 0; r < n; ++r) for (int i = 0; i < Stride; ++i)
            input[size_t(r) * Stride + i] = float((begin + r + seed) % 97 - 48) * 0.003f +
                float(i % 29 - 14) * 0.01f + float(i / Width) * 0.03f;
        auto& q = rt->compute();
        Drain drain{q};
        q.memcpy(taps.as<float>() + size_t(begin) * Stride, input.data(), input.size() * 4).wait_and_throw();
    }
    const float* at(int p) const { need(p >= 0 && p < rows, "fixture row pointer"); return taps.as<float>() + size_t(p) * Stride; }
    void context(int begin, int n) { ok(df.context(at(begin), n, begin, error)); }
    Snapshot read() {
        Snapshot out;
        Drain drain{*df.stream()}; // debug readback storage outlives exceptional completion
        ok(df.debug_ring(out.kv, out.tags, error));
        out.epoch = df.epoch();
        need(out.tags.size() == Drafter::kRing && out.kv.size() == size_t(5 * 2 * Heads * Drafter::kRing * HeadDim), "snapshot geometry");
        need(std::all_of(out.kv.begin(), out.kv.end(), [](float v) { return std::isfinite(v); }), "finite ring state");
        return out;
    }
    void forward(int p, int count = Drafter::kDrafts, bool probabilities = true) {
        std::array<int32_t, Drafter::kDrafts + 2> ids;
        std::array<float, Drafter::kDrafts + 2> probs;
        ids.fill(-1234567);
        probs.fill(-12345.0f);
        df.set_max_drafts(count);
        ok(df.draft(at(p), 1, p, 42, std::span(ids).subspan(1, Drafter::kDrafts),
            probabilities ? std::span(probs).subspan(1, Drafter::kDrafts) : std::span<float>{}, error));
        need(ids.front() == -1234567 && ids.back() == -1234567 &&
             probs.front() == -12345.0f && probs.back() == -12345.0f, "draft output guards");
        for (int i = 0; i < Drafter::kDrafts; ++i) {
            if (i < count) {
                need(std::find(vocab.begin(), vocab.end(), ids[i + 1]) != vocab.end(), "draft token in synthetic target vocabulary");
                if (probabilities) need(std::isfinite(probs[i + 1]) && probs[i + 1] >= 0 && probs[i + 1] <= 1,
                                        "finite draft probability");
            } else {
                need(ids[i + 1] == 0, "inactive draft token is zero");
                if (probabilities) need(probs[i + 1] == 0, "inactive draft probability is zero");
            }
            if (!probabilities) need(probs[i + 1] == -12345.0f, "omitted probabilities untouched");
        }
    }
};

int64_t tag(int32_t epoch, int64_t pos) { return (int64_t(epoch) << 32) | pos; }
const float* cell(const Snapshot& s, int layer, int kind, int head, int64_t p) {
    return s.kv.data() + (size_t(layer * 2 + kind) * Heads * Drafter::kRing +
        size_t(head) * Drafter::kRing + size_t(p % Drafter::kRing)) * HeadDim;
}
void compare(const Snapshot& want, const Snapshot& got, int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
        const size_t slot = size_t(p % Drafter::kRing);
        need(want.tags[slot] == tag(want.epoch, p) && got.tags[slot] == tag(got.epoch, p), "epoch/absolute ring tag");
        for (int l = 0; l < 5; ++l) for (int kind = 0; kind < 2; ++kind) for (int h = 0; h < Heads; ++h)
            if (std::memcmp(cell(want, l, kind, h, p), cell(got, l, kind, h, p), HeadDim * 4))
                throw std::runtime_error("ring replay differs at position " + std::to_string(p) + ",layer " + std::to_string(l));
    }
}
std::vector<int32_t> history(int n) {
    std::vector<int32_t> out(size_t(n), 0);
    for (int i = 0; i < n; ++i) out[size_t(i)] = (i * 37) % 100000;
    return out;
}

void small(const std::shared_ptr<Runtime>& rt, const std::string& pack) {
    tail_transitions(rt);
    {
        Harness h(rt, pack, 8);
        auto tokens = history(8);
        bool cold = true;
        h.ok(h.df.begin_request(tokens.data(), 8, 0, &cold));
        need(!cold, "fresh empty prefix");
        h.df.note_rows(tokens.data(), 8, 0);
        h.context(0, 8);
        auto snapshot = h.read();
        for (int p = 0; p < 8; ++p) need(snapshot.tags[size_t(p)] == tag(snapshot.epoch, p), "small committed tag");
        need(std::any_of(snapshot.kv.begin(), snapshot.kv.end(), [](float v) { return v != 0; }), "nontrivial stock projection");
        pass("stock_pack_context");
        h.forward(7);
        need(h.embed_calls == 1 && h.df.rounds == 1, "one completed block forward");
        pass("stock_block_forward");
        const auto before = h.read();
        for (int kind = 0; kind < 2; ++kind) {
            for (int length = kind; length < Drafter::kDrafts; ++length) {
                std::array<int32_t, Drafter::kDrafts> ids;
                std::array<float, Drafter::kDrafts> probs;
                ids.fill(-1234567); probs.fill(-12345.0f);
                const auto tokens_out = std::span(ids).first(kind ? Drafter::kDrafts : length);
                const auto probs_out = std::span(probs).first(kind ? length : Drafter::kDrafts);
                h.error.clear();
                need(!h.df.draft(h.at(7), 1, 7, 42, tokens_out, probs_out, h.error), "undersized draft output refused");
                need(h.error == "dflash2: draft outputs require seven entries", "output extent refusal mechanism");
                need(std::all_of(ids.begin(), ids.end(), [](int32_t v) { return v == -1234567; }) &&
                     std::all_of(probs.begin(), probs.end(), [](float v) { return v == -12345.0f; }), "refused output untouched");
                need(h.embed_calls == 1 && h.df.rounds == 1, "output refusal before block submission");
            }
        }
        const auto after = h.read();
        need(before.epoch == after.epoch && before.tags == after.tags && before.kv == after.kv,
             "output refusal preserves ring state and owner");
        pass("draft_output_span_admission");
        for (int count = 1; count <= Drafter::kDrafts; ++count) h.forward(7, count);
        h.forward(7, 3, false);
        need(h.embed_calls == 9 && h.df.rounds == 9, "narrow forwards and optional probabilities complete");
        pass("narrow_draft_output_guards");
        h.ok(h.df.context(nullptr, 0, 262144, h.error));
        need(!h.df.context(h.at(0), 1, 262144, h.error), "positive one-past context");
        h.ok(h.df.context(h.at(0), 1, 262143, h.error));
        int32_t proposals[7] = {};
        need(!h.df.draft(h.at(0), 1, 262143, 42, proposals, {}, h.error), "no anchor beyond capacity");
        snapshot = h.read();
        need(snapshot.tags[size_t(262143 % Drafter::kRing)] == tag(snapshot.epoch, 262143), "final physical drafter row");
        pass("physical_drafter_tail"); // does not fill the target's physical context
    }
    {
        Harness h(rt, pack, 8);
        h.fail_embed = true;
        int32_t proposals[7] = {};
        need(!h.df.draft(h.at(0), 1, 0, 42, proposals, {}, h.error) && h.embed_calls == 1, "controlled partial operation failure");
        need(h.error.find("intentional host embedding refusal") != std::string::npos, "expected injected mechanism");
        need(!h.df.context(h.at(0), 1, 0, h.error), "failed context owner cannot retry");
        need(!h.df.prepare(h.at(0), h.error), "failed owner cannot prepare");
        auto tokens = history(8); bool cold = false;
        need(!h.df.begin_request(tokens.data(), 8, 0, &cold) && cold, "failed owner cannot change lineage");
        bool refused = false;
        try { h.df.reset(); } catch (const std::runtime_error&) { refused = true; }
        need(refused, "failed owner cannot reset");
        need(!h.df.draft(h.at(0), 1, 0, 42, proposals, {}, h.error) && h.embed_calls == 1 && h.df.rounds == 0,
             "failed owner cannot submit another block");
        pass("failed_owner_containment");
    }
}

void full(const std::shared_ptr<Runtime>& rt, const std::string& pack) {
    constexpr int Count = Drafter::kRing + 16, Prefix = 256;
    Harness h(rt, pack, Count);
    auto original = history(Count);
    bool cold = true;
    h.ok(h.df.begin_request(original.data(), Count, 0, &cold));
    h.df.note_rows(original.data(), Count, 0);
    h.context(0, Prefix);
    const auto prefix = h.read();
    h.context(Prefix, Count - Prefix);
    const auto wrapped = h.read();
    for (int64_t p = Count - Drafter::kRing; p < Count; ++p)
        need(wrapped.tags[size_t(p % Drafter::kRing)] == tag(wrapped.epoch, p), "wrapped absolute position");
    pass("wrapped_context");

    h.ok(h.df.begin_request(original.data(), Prefix + 8, Prefix, &cold));
    need(!cold && h.df.last_rebuilt == 16 && h.df.last_missing == 0, "rewind rebuilds only overwritten prefix");
    compare(prefix, h.read(), 0, Prefix);
    pass("prefix_rebuild");

    auto branch = std::vector<int32_t>(original.begin(), original.begin() + Prefix + 8);
    for (int i = Prefix; i < Prefix + 8; ++i) branch[size_t(i)] += 100001;
    h.df.note_rows(branch.data() + Prefix, 8, Prefix);
    h.set_rows(Prefix, 8, 71);
    h.context(Prefix, 8);
    {
        const auto fork = h.read();
        bool differs = false;
        for (int l = 0; l < 5; ++l) for (int k = 0; k < 2; ++k)
            differs |= std::memcmp(cell(wrapped, l, k, 0, Prefix), cell(fork, l, k, 0, Prefix), HeadDim * 4) != 0;
        need(differs, "fork data must distinguish stale lineage");
    }
    h.ok(h.df.begin_request(original.data(), Count, Count, &cold));
    need(!cold && h.df.last_swapped && h.df.last_rebuilt == 2047 && h.df.last_missing == 0 &&
         h.df.ring_stats.stash_hits >= 1, "parked original lineage restored");
    const auto restored = h.read();
    compare(wrapped, restored, Count - 2047, Count);
    pass("parked_branch_restore");

    h.df.reset();
    h.set_rows(Prefix, 8, 0);
    h.ok(h.df.begin_request(original.data(), Count, 0, &cold));
    h.df.note_rows(original.data(), Count, 0);
    h.context(0, Count);
    compare(restored, h.read(), Count - 2047, Count);
    pass("fresh_replay_equivalence");

    h.df.reset();
    h.df.note_rows(original.data(), 130, 0);
    h.context(0, 64);
    h.context(65, 65);
    h.ok(h.df.begin_request(original.data(), 130, 130, &cold));
    need(cold && h.df.last_missing == 1 && h.df.last_rebuilt == 0, "one missing context row reported cold");
    const auto missing = h.read();
    need(missing.tags[64] != tag(missing.epoch, 64), "missing row retains no current epoch tag");
    for (int p = 0; p < 130; ++p) if (p != 64)
        need(missing.tags[size_t(p)] == tag(missing.epoch, p), "present row stays live");
    h.forward(129); // consume an actual tag-masked hole, not a forced fallback
    pass("missing_context_mask");
}
} // namespace

int main(int argc, char** argv) try {
    const bool supervised = argc == 4 && std::string(argv[3]) == "--supervised";
    need((argc == 3 || supervised) && (std::string(argv[1]) == "--small" || std::string(argv[1]) == "--full"),
         "usage: sycl_dflash2_ring_test --small|--full PACK [--supervised]");
    const bool is_small = std::string(argv[1]) == "--small";
    std::cout << "dflash2 ring: stock draft weights,synthetic target,no performance claim\n" << std::flush;
    auto runtime = strata::sycl_backend::runtime_for(0);
    if (is_small) small(runtime, argv[2]); else full(runtime, argv[2]);
    runtime->wait();
    std::cout << "dflash2 ring " << (is_small ? "small" : "full") << " cases=" << (is_small ? 7 : 5) << " PASS\nDF2_RING_WAIT\n" << std::flush;
    if (supervised) {
        std::string command;
        need(bool(std::getline(std::cin, command)) && command == "QUIT", "Main runtime-map retirement gate");
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "dflash2 ring FAILED: " << e.what() << '\n'; return 1;
}
