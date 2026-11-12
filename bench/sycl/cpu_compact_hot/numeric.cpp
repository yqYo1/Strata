// Main-authored real-weight ownership-component check; repository LICENSE.
// Uses the actual production dispatcher/quantizer, not a copied raw oracle.
#include "cache.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <array>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
using compact_gu_cache::Cache;
constexpr int H = 2560, FF = 640, experts = 32;
constexpr float sentinel = 123.f;

[[gnu::cold]] void need(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
uint64_t number(const char* value) {
    const std::string text(value);
    need(!text.empty() && text.size() <= 20 && std::all_of(text.begin(), text.end(),
        [](char c) { return c >= '0' && c <= '9'; }), "unsigned decimal required");
    return std::stoull(text);
}
uint64_t hash(std::span<const uint8_t> bytes) {
    uint64_t value = 14695981039346656037ull;
    for (uint8_t byte : bytes) { value ^= byte; value *= 1099511628211ull; }
    return value;
}
struct Buffer {
    std::vector<uint8_t> storage;
    size_t bytes;
    explicit Buffer(size_t size) : storage(size + 128, 0xa7), bytes(size) {
        need(size && size <= (4u << 20), "buffer extent");
    }
    uint8_t* data() { return storage.data() + 64; }
    std::span<const uint8_t> view() const { return {storage.data() + 64, bytes}; }
    void guards() const {
        for (size_t i = 0; i < 64; ++i)
            need(storage[i] == 0xa7 && storage[64 + bytes + i] == 0xa7, "input canary");
    }
};
struct File {
    int fd;
    explicit File(const char* name) : fd(open(name, O_RDONLY | O_CLOEXEC)) {
        need(fd >= 0, "model open");
    }
    ~File() { close(fd); }
    void read(uint8_t* output, size_t bytes, uint64_t offset) const {
        struct stat stat{};
        need(!fstat(fd, &stat) && stat.st_size >= 0 && offset <= uint64_t(stat.st_size) &&
            bytes <= uint64_t(stat.st_size) - offset, "model slice extent");
        while (bytes) {
            const auto count = pread(fd, output, bytes, offset);
            need(count > 0, "model slice read");
            output += count; bytes -= count; offset += count;
        }
    }
};
using Output = std::array<float, FF + 2>;
std::span<float> values(Output& output) { return {output.data() + 1, FF}; }
void check_output(const Output& output, int first, int last) {
    need(output.front() == sentinel && output.back() == sentinel, "output canary");
    for (int row = 0; row < FF; ++row)
        need(row >= first && row < last ? std::isfinite(output[row + 1]) :
            output[row + 1] == sentinel, "output row bounds/finite");
}

int main(int argc, char** argv) {
    try {
        need(argc == 8, "usage: numeric SHARD DOWN GATE UP DOWN_OFFSET BLOB_BYTES SLOTS");
        const auto down_value = number(argv[2]);
        need(down_value == 20 || down_value == 42, "actual model Down format required");
        const int down = int(down_value);
        need(number(argv[7]) == 512 && compact_gu_cache::available() &&
             !cpu::cpu_avx512_ok(), "this AVX2 model/CPU contract required");
        cpu::NativeFmt format;
        std::string error;
        const bool valid = cpu::native_fmt(GGML_TYPE_IQ3_XXS, down, H, FF, format, error);
        need(valid, error.c_str());
        need(compact_gu_cache::supported(format) && number(argv[6]) == format.bytes,
             "actual supported native geometry required");
        const std::array<uint64_t, 3> offsets{number(argv[3]), number(argv[4]), number(argv[5])};
        const std::array<size_t, 3> extents{format.up_off, format.up_off, format.bytes - format.down_off};
        for (size_t i = 0; i < offsets.size(); ++i)
            need(offsets[i] <= std::numeric_limits<uint64_t>::max() - experts * extents[i],
                 "model offset overflow");
        File file(argv[1]);
        std::vector<Buffer> blobs;
        std::vector<uint64_t> original;
        blobs.reserve(experts); original.reserve(experts);
        Cache cache(1, experts, 64u << 20);
        for (int expert = 0; expert < experts; ++expert) {
            blobs.emplace_back(format.bytes);
            auto* destination = blobs.back().data();
            for (size_t i = 0; i < offsets.size(); ++i) {
                file.read(destination, extents[i], offsets[i] + expert * extents[i]);
                destination += extents[i];
            }
            original.push_back(hash(blobs.back().view()));
            need(cache.prepare({0, size_t(expert)}, format,
                 blobs.back().view().first(format.down_off)), "prepare real model GU");
        }
        cache.freeze();
        std::array<Buffer, 4> activations{Buffer(format.act_bytes), Buffer(format.act_bytes),
            Buffer(format.act_bytes), Buffer(format.act_bytes)};
        std::array<uint64_t, 4> activation_hashes{};
        uint32_t seed = 20261112;
        int q8_min = 127, q8_max = -128;
        for (size_t token = 0; token < activations.size(); ++token) {
            std::array<float, H> input{};
            for (int i = 0; i < H; ++i) {
                seed = 1664525u * seed + 1013904223u;
                input[i] = token == 1 ? 0.f : (int(seed >> 8) - 8388608) * 0x1p-23f;
                if (token == 2) input[i] *= i % 37 == 0 ? 32.f : 0.002f;
                if (token == 3 && i % 256 == 0) input[i] = -16.f;
            }
            cpu::native_quant_act(format, input.data(), activations[token].data());
            activation_hashes[token] = hash(activations[token].view());
            for (size_t b = 0; b < 10; ++b) {
                block_q8_K block;
                std::memcpy(&block, activations[token].data() + b * sizeof block, sizeof block);
                for (int8_t value : block.qs) { q8_min = std::min(q8_min, int(value)); q8_max = std::max(q8_max, int(value)); }
            }
        }
        auto before = cpu::iq3_gu_audit_snapshot();
        need(before.enabled && !before.overflow && before.calls == 0, "fresh actual dispatch audit required");
        uint64_t calls = 0, rows = 0, paired = 0, cases = 0;
        auto raw = [&](int expert, size_t token, Output& output, int first, int last,
                       cpu::NativeGuPhase phase = cpu::NativeGuPhase::Decode) {
            const void* act[]{activations[token].data()}; float* out[]{output.data() + 1};
            cpu::native_gu_rows(format, blobs[expert].data(), act, 1, out, first, last, phase);
        };
        for (int expert = 0; expert < experts; ++expert) for (size_t token = 0; token < 4; ++token)
            for (auto range : {std::pair{0, FF}, std::pair{1, FF - 1}, std::pair{128, 256}}) {
                Output reference, candidate; reference.fill(sentinel); candidate = reference;
                raw(expert, token, reference, range.first, range.second);
                auto lease = cache.lookup({0, size_t(expert)}, format, cpu::NativeGuPhase::Decode, 1);
                need(lease.gu_rows(cpu::NativeGuPhase::Decode, 1, activations[token].view(),
                    values(candidate), range.first, range.second), "real candidate call refused");
                need(std::memcmp(reference.data(), candidate.data(), sizeof reference) == 0,
                     "real-model production/compact GU bits");
                check_output(candidate, range.first, range.second);
                need(lease.guards_intact(), "encoded view canary");
                ++calls; ++cases; rows += range.second - range.first; paired += range.second - range.first;
            }
        Output reference, threaded_raw, threaded_compact;
        reference.fill(sentinel); threaded_raw = threaded_compact = reference;
        raw(0, 0, reference, 0, FF); ++calls; rows += FF; paired += FF;
        std::array<std::exception_ptr, 6> errors{};
        {
            std::array<std::jthread, 6> threads;
            for (size_t t = 0; t < threads.size(); ++t) threads[t] = std::jthread([&, t] {
                try {
                    const int first = int(FF * t / 6), last = int(FF * (t + 1) / 6);
                    raw(0, 0, threaded_raw, first, last);
                    auto lease = cache.lookup({0, 0}, format, cpu::NativeGuPhase::Decode, 1);
                    need(lease.gu_rows(cpu::NativeGuPhase::Decode, 1, activations[0].view(),
                        values(threaded_compact), first, last), "threaded compact call");
                } catch (...) { errors[t] = std::current_exception(); }
            });
        }
        for (auto error : errors) if (error) std::rethrow_exception(error);
        calls += 6; rows += FF; paired += FF;
        need(std::memcmp(reference.data(), threaded_raw.data(), sizeof reference) == 0 &&
             std::memcmp(reference.data(), threaded_compact.data(), sizeof reference) == 0,
             "disjoint production/compact row partition bits");
        Output prefill; prefill.fill(sentinel);
        raw(0, 0, prefill, 128, 256, cpu::NativeGuPhase::Prefill); ++calls; rows += 128;
        check_output(prefill, 128, 256);
        need(!cache.lookup({0, 0}, format, cpu::NativeGuPhase::Prefill, 1) &&
             !cache.lookup({0, 0}, format, cpu::NativeGuPhase::Decode, 2), "phase/NT guard");
        auto lease = cache.lookup({0, 0}, format, cpu::NativeGuPhase::Decode, 1);
        auto alias = std::span<float>(reinterpret_cast<float*>(activations[0].data()), FF);
        need(!lease.gu_rows(cpu::NativeGuPhase::Decode, 1, activations[0].view(), alias, 0, FF),
             "activation/output alias accepted");
        for (int e = 0; e < experts; ++e) {
            blobs[e].guards(); need(hash(blobs[e].view()) == original[e], "raw weight mutation");
            need(cache.lookup({0, size_t(e)}, format, cpu::NativeGuPhase::Decode, 1).guards_intact(),
                 "final encoded view canary");
        }
        for (size_t t = 0; t < 4; ++t) {
            activations[t].guards(); need(hash(activations[t].view()) == activation_hashes[t], "activation mutation");
        }
        const auto audit = cpu::iq3_gu_audit_snapshot();
        need(audit.enabled && !audit.overflow && audit.threads == 7 && audit.calls == calls &&
             audit.nt1_rows == rows && audit.pair_rows == paired, "actual raw/prefill dispatch differs");
        std::cout << "{\"component\":\"compact_cache_numeric\",\"passed\":true,\"down\":" << down
            << ",\"experts\":" << experts << ",\"serial_cases\":" << cases
            << ",\"threaded_rows\":640,\"raw_calls\":" << calls << ",\"raw_nt1_rows\":" << rows
            << ",\"raw_pair_rows\":" << paired << ",\"q8_min\":" << q8_min << ",\"q8_max\":" << q8_max
            << ",\"performance_qualified\":false}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
