// SPDX-FileCopyrightText: 2026 yayoi
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace strata::program::dense_prefill_nll {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "head result must be IEEE754 FP32");

inline bool recipe_digest(std::string_view s) {
    if (s.size() != 64) return false;
    for (char c : s) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

// Reduces the actual FP32 head result in CPU FP64. Subtract the target before
// adding log(sum): a common large offset cannot cancel the small loss. No
// fast-math: NaN/Inf input must fail, rather than disappear in the reduction.
bool target_loss(const float* values, size_t vocab, size_t target, double& out);

// IO is substitutable only for host fault fixtures; production uses Stdio.
struct Stdio {
    static std::FILE* open(const char* path) { return std::fopen(path, "wbx"); }
    static size_t write(const void* p, size_t n, std::FILE* f) { return std::fwrite(p, 1, n, f); }
    static int flush(std::FILE* f) { return std::fflush(f); }
    static int close(std::FILE* f) { return std::fclose(f); }
    static int seek(std::FILE* f, long pos) { return std::fseek(f, pos, SEEK_SET); }
};

// Raw v2: 64-byte LE header, then (uint32 target, IEEE754 binary64 NLL)
// records in implicit position order. First token + every target reconstruct
// all input IDs exactly. Completion is accepted ONLY with the published .meta.
// At 262144 input IDs: 3145780 raw bytes + <4096 metadata <4 MiB total.
// Caller owns the immutable token vector and must drain every GPU use of row()
// before append(), exception unwinding, or destruction of this host owner.
template<class IO = Stdio>
class BasicWriter {
public:
    static constexpr uint64_t raw_limit = 4ull << 20;
    static constexpr uint32_t max_input = 262144;
    static constexpr uint32_t max_vocab = 262144; // fixed <=1 MiB host row
    BasicWriter() = default;
    BasicWriter(const BasicWriter&) = delete;
    BasicWriter& operator=(const BasicWriter&) = delete;
    ~BasicWriter() { if (raw_) IO::close(raw_); if (meta_) IO::close(meta_); }

    bool begin(const std::string& path, const int64_t* input, size_t count, int64_t vocab,
               bool native, int head_type, std::string_view recipe_sha256, std::string& err) {
        if (begun_ || failed_ || path.empty() || !input || count < 2 || count > max_input ||
            vocab <= 0 || vocab > max_vocab || !recipe_digest(recipe_sha256))
            return fail(err, "invalid input dimensions/path/recipe digest");
        for (size_t i = 0; i < count; ++i)
            if (input[i] < 0 || input[i] >= vocab) return fail(err, "input ID outside vocabulary");
        final_meta_ = path + ".meta";
        partial_meta_ = final_meta_ + ".partial";
        std::error_code ec;
        if (std::filesystem::exists(final_meta_, ec) || ec)
            return fail(err, "completed metadata exists or cannot be checked");
        row_.resize(size_t(vocab));
        raw_ = IO::open(path.c_str());
        if (!raw_) return fail(err, "cannot exclusively create raw destination");
        meta_ = IO::open(partial_meta_.c_str());
        if (!meta_) return fail(err, "cannot exclusively create partial metadata");
        input_ = input; count_ = uint32_t(count); vocab_ = uint32_t(vocab);
        unsigned char header[64]{};
        std::memcpy(header, "STNLLV2\n", 8);
        put32(header + 8, 2); put32(header + 12, 64);
        put32(header + 16, count_); put32(header + 20, vocab_);
        put32(header + 24, 0); // written only after whole Prefill::run succeeds
        put32(header + 28, uint32_t(input[0]));
        put32(header + 32, uint32_t(head_type)); put32(header + 36, uint32_t(native));
        const std::string meta = "strata-dense-prefill-nll-v2\nrecipe_sha256=" + std::string(recipe_sha256) +
            "\ncoverage=initial-prefill-all-n_input-minus-one\nformat=little-endian-header64-target-u32-loss-f64\n"
            "projection=existing-single-row-lm-head\nperformance_cohort=excluded\n";
        if (!write(raw_, header, sizeof(header)) || !write(meta_, meta.data(), meta.size()))
            return fail(err, "header write failed");
        begun_ = true;
        return true;
    }

    bool chunk(int64_t position, int64_t count, std::string& err) {
        if (!begun_ || failed_ || finished_ || position != seen_ || written_ != seen_ || count <= 0 ||
            uint64_t(count) > uint64_t(count_ - 1 - seen_))
            return fail(err, "gap/repeat/overlap or unconsumed callback rows");
        seen_ += uint32_t(count);
        return true;
    }
    float* row() { return row_.data(); }
    size_t row_bytes() const { return row_.size() * sizeof(float); }
    uint32_t written() const { return written_; }
    bool append(int64_t position, int64_t target, std::string& err) {
        if (!begun_ || failed_ || finished_ || position != written_ || written_ >= seen_ ||
            target != input_[size_t(written_) + 1])
            return fail(err, "missing/repeated row or wrong next-token identity");
        double nll = 0.0;
        if (!target_loss(row_.data(), row_.size(), size_t(target), nll))
            return fail(err, "non-finite logit/loss or invalid target");
        unsigned char record[12];
        put32(record, uint32_t(target));
        uint64_t bits = 0;
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                      "NLL requires IEEE754 binary64");
        std::memcpy(&bits, &nll, sizeof(bits));
        for (unsigned b = 0; b < 8; ++b) record[4 + b] = static_cast<unsigned char>(bits >> (8 * b));
        if (!write(raw_, record, sizeof(record))) return fail(err, "record write failed");
        ++written_;
        return true;
    }

    // Called after the outer native-bank/head/MTP boundaries return successfully.
    // Failure, cancellation, incomplete final-chunk coverage, or a later callback failure
    // never publishes completion. This does not certify a later decode/RESTORE.
    bool finish(double callback_ms, std::string& err) {
        if (!begun_ || failed_ || finished_ || written_ != count_ - 1 || seen_ != written_ ||
            !std::isfinite(callback_ms) || callback_ms < 0)
            return fail(err, "incomplete coverage or invalid diagnostic cost");
        unsigned char count[4]; put32(count, written_);
        if (IO::seek(raw_, 24) || !write(raw_, count, sizeof(count)))
            return fail(err, "completion count write failed");
        if (!close_checked(raw_)) return fail(err, "raw flush/close failed");
        char footer[384];
        const int n = std::snprintf(footer, sizeof(footer),
            "n_input=%u\nvocab=%u\nrows=%u\nraw_bytes=%llu\ncallback_ms=%.17g\ncomplete=1\n",
            count_, vocab_, written_, static_cast<unsigned long long>(64ull + 12ull * written_), callback_ms);
        if (n <= 0 || size_t(n) >= sizeof(footer) || !write(meta_, footer, size_t(n)))
            return fail(err, "footer write failed");
        if (!close_checked(meta_)) return fail(err, "metadata flush/close failed");
        std::error_code ec;
        std::filesystem::create_hard_link(partial_meta_, final_meta_, ec);
        if (ec) return fail(err, "metadata no-clobber publication failed");
        finished_ = true;
        return true;
    }

private:
    static void put32(unsigned char* p, uint32_t v) {
        for (unsigned b = 0; b < 4; ++b) p[b] = static_cast<unsigned char>(v >> (8 * b));
    }
    static bool write(std::FILE* f, const void* p, size_t n) { return IO::write(p, n, f) == n; }
    static bool close_checked(std::FILE*& f) {
        const int flush = IO::flush(f);
        const int close = IO::close(f); f = nullptr;
        return !flush && !close;
    }
    bool fail(std::string& err, const char* message) {
        failed_ = true; err = std::string("dense prefill NLL: ") + message; return false;
    }
    std::FILE* raw_ = nullptr;
    std::FILE* meta_ = nullptr;
    std::string final_meta_, partial_meta_;
    const int64_t* input_ = nullptr;
    std::vector<float> row_;
    uint32_t count_ = 0, vocab_ = 0, seen_ = 0, written_ = 0;
    bool begun_ = false, failed_ = false, finished_ = false;
};
using Writer = BasicWriter<>;
} // namespace strata::program::dense_prefill_nll
