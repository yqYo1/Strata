// Main-authored actual-pool parity/lifetime gate; repository LICENSE applies.
#include "cache.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <array>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
namespace gu = cpu::compact_gu;
constexpr int H = 2560, FF = 640, E = 32;
constexpr float sentinel = 123.f;
[[gnu::cold, noreturn]] void fail(const char* message) { throw std::runtime_error(message); }
inline void need(bool value, const char* message) { if (!value) fail(message); }
uint64_t number(const char* value) {
    const std::string s(value);
    need(!s.empty() && s.size() <= 20 && std::all_of(s.begin(), s.end(),
        [](char c) { return c >= '0' && c <= '9'; }), "decimal required");
    return std::stoull(s);
}
struct File {
    int fd;
    explicit File(const char* path) : fd(open(path, O_RDONLY | O_CLOEXEC)) { need(fd >= 0, "open model"); }
    ~File() { close(fd); }
    void read(uint8_t* out, size_t bytes, uint64_t offset) {
        struct stat s{};
        need(!fstat(fd, &s) && s.st_size >= 0 && offset <= uint64_t(s.st_size) &&
             bytes <= uint64_t(s.st_size) - offset, "model span");
        while (bytes) {
            const auto n = pread(fd, out, bytes, offset);
            need(n > 0, "short model read"); out += n; bytes -= n; offset += n;
        }
    }
};
struct alignas(64) Output {
    std::array<float, H + 2> data{};
    Output() { data.fill(sentinel); }
    float* values() { return data.data() + 1; }
    void check() const {
        need(data.front() == sentinel && data.back() == sentinel, "Down output bounds");
        for (int i = 1; i <= H; ++i) need(std::isfinite(data[i]), "finite Down output");
    }
};
int main(int argc, char** argv) {
    try {
        need(argc == 7, "pool SHARD DOWN GATE UP DOWN_OFFSET BLOB_BYTES");
        need(gu::available(), "actual AVX2 cache target");
        cpu::NativeFmt f; std::string error;
        const auto down = number(argv[2]);
        need((down == 20 || down == 42) && cpu::native_fmt(18, int(down), H, FF, f, error) &&
             gu::supported(f) && f.bytes == number(argv[6]), "actual native format");
        const std::array<uint64_t, 3> offsets{number(argv[3]), number(argv[4]), number(argv[5])};
        const std::array<size_t, 3> sizes{f.up_off, f.up_off, f.bytes - f.down_off};
        File file(argv[1]);
        std::vector<uint8_t> raw(E * f.bytes);
        for (int e = 0; e < E; ++e) {
            auto* destination = raw.data() + size_t(e) * f.bytes;
            for (size_t role = 0; role < 3; ++role) {
                need(offsets[role] <= UINT64_MAX - E * sizes[role], "offset overflow");
                file.read(destination, sizes[role], offsets[role] + e * sizes[role]); destination += sizes[role];
            }
        }
        const gu::LayerView layer{0, E, f.bytes, raw.size(), raw.data(), f};
        auto cache = gu::prepare(1, E, 64u << 20, {&layer, 1}, error);
        need(cache && cache->prepared_entries() == E, "all-or-nothing real cache preparation");
        size_t budget = 0;
        for (const char* s : std::array<const char*, 4>{nullptr, "", "0", "016384"})
            need(gu::parse_mib_budget(s, budget), "valid MiB budget");
        for (const char* s : {"-1", "+1", " 1", "1 ", "nan", "18446744073709551615"})
            need(!gu::parse_mib_budget(s, budget) && !budget, "invalid MiB budget");
        need(!gu::prepare(1, E, 1, {&layer, 1}, error), "index-budget admission");
        std::array<gu::LayerView, 2> duplicate{layer, layer};
        need(!gu::prepare(2, E, 128u << 20, duplicate, error), "duplicate source layer");
        auto truncated = layer; --truncated.bytes;
        need(!gu::prepare(1, E, 64u << 20, {&truncated, 1}, error), "short immutable layer");
        std::array<std::vector<uint8_t>, 4> acts;
        uint32_t seed = 20261113;
        for (size_t t = 0; t < acts.size(); ++t) {
            std::array<float, H> input{};
            for (int i = 0; i < H; ++i) {
                seed = 1664525u * seed + 1013904223u;
                input[i] = t == 1 ? 0.f : (int(seed >> 8) - 8388608) * 0x1p-23f;
                if (t == 2) input[i] *= i % 37 == 0 ? 32.f : .002f;
            }
            acts[t].resize(f.act_bytes); cpu::native_quant_act(f, input.data(), acts[t].data());
        }
        std::array<gu::Lease, E> retained;
        for (int e = 0; e < E; ++e) retained[e] = cache->lookup({0, size_t(e)}, f, cpu::NativeGuPhase::Decode, 1);
        cache.reset(); // Leases own GU; the raw source remains live for Down.
        uint64_t cases = 0, raw_rows = 0, compact_rows = 0;
        for (int workers : {1, 5}) {
            cpu::ExpertPool pool(workers, true, workers == 5);
            for (auto phase : {cpu::NativeGuPhase::Decode, cpu::NativeGuPhase::Prefill})
                for (int n : {1, 6, 97, 193}) for (int policy : {0, 1, 2}) {
                    // NT1, NT2, or mixed; parity is always through the same
                    // complete production GU->quantize->Down entrypoint.
                    std::vector<cpu::ExpertJobMulti> jobs(n);
                    std::vector<gu::Lease> views(n);
                    std::vector<Output> reference(size_t(n) * 2), candidate(size_t(n) * 2);
                    uint64_t nt1_rows = 0, eligible_rows = 0;
                    for (int j = 0; j < n; ++j) {
                        auto& job = jobs[j]; job.blob = raw.data() + size_t(j % E) * f.bytes;
                        job.nt = policy == 0 ? 1 : policy == 1 ? 2 : 1 + (j & 1);
                        // Leave every third key uncached to test mixed fallback.
                        if (j % 3 != 0) views[j] = retained[j % E];
                        if (job.nt == 1) {
                            nt1_rows += FF;
                            if (phase == cpu::NativeGuPhase::Decode && views[j]) eligible_rows += FF;
                        }
                        for (int t = 0; t < job.nt; ++t) {
                            job.nact[t] = acts[size_t(j + t) % acts.size()].data();
                            job.out[t] = reference[size_t(j) * 2 + t].values();
                        }
                    }
                    const auto before = cpu::iq3_gu_audit_snapshot();
                    need(before.enabled && !before.overflow, "actual paired audit required");
                    pool.run_split_multi_native(f, jobs.data(), n, phase);
                    const auto middle = cpu::iq3_gu_audit_snapshot();
                    need(middle.nt1_rows - before.nt1_rows == nt1_rows &&
                        middle.pair_rows - before.pair_rows == (phase == cpu::NativeGuPhase::Decode ? nt1_rows : 0),
                        "actual baseline phase/NT dispatch");
                    for (int j = 0; j < n; ++j) for (int t = 0; t < jobs[j].nt; ++t)
                        jobs[j].out[t] = candidate[size_t(j) * 2 + t].values();
                    pool.run_split_multi_native(f, jobs.data(), n, phase, views);
                    const auto after = cpu::iq3_gu_audit_snapshot();
                    need(!after.overflow && after.nt1_rows - middle.nt1_rows == nt1_rows - eligible_rows &&
                        after.pair_rows - middle.pair_rows ==
                            (phase == cpu::NativeGuPhase::Decode ? nt1_rows - eligible_rows : 0),
                        "actual compact/fallback phase/NT dispatch");
                    for (int j = 0; j < n; ++j) for (int t = 0; t < jobs[j].nt; ++t) {
                        const auto idx = size_t(j) * 2 + t;
                        reference[idx].check(); candidate[idx].check();
                        need(std::memcmp(reference[idx].data.data(), candidate[idx].data.data(), sizeof(Output::data)) == 0,
                             "whole-pool GU/hidden-quant/Down bits");
                    }
                    ++cases; raw_rows += nt1_rows - eligible_rows; compact_rows += eligible_rows;
                }
            std::array<cpu::ExpertJobMulti, 2> invalid{}; std::array<gu::Lease, 1> short_views{};
            bool refused = false;
            try { pool.run_split_multi_native(f, invalid.data(), 2, cpu::NativeGuPhase::Decode, short_views); }
            catch (const std::invalid_argument&) { refused = true; }
            need(refused, "short lease span before publishing");
            pool.run_split_multi_native(f, nullptr, 0, cpu::NativeGuPhase::Decode);
        }
        for (const auto& lease : retained) need(lease.guards_intact(), "final encoded GU guard");
        std::cout << "{\"passed\":true,\"down\":" << down << ",\"whole_pool_cases\":" << cases
            << ",\"fallback_nt1_rows\":" << raw_rows << ",\"compact_rows\":" << compact_rows
            << ",\"performance_qualified\":false}\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
