#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace strata::program::prefill_logits {

static_assert(sizeof(float) == 4, "prefill logit fixtures require FP32 storage");

// Diagnostic storage selection, never a change to the conditioning token IDs.
// At most 64 strictly rising decimal positions; no signs, spaces or empty fields.
inline bool parse_positions(std::string_view text, int64_t batched,
                            std::vector<int64_t>& out, std::string& err) {
    out.clear();
    if (batched <= 0 || text.empty() || text.size() > 64 * 21) {
        err = "prefill logits: need 1..64 positions and at least two input tokens";
        return false;
    }
    size_t start = 0;
    while (start < text.size()) {
        const size_t comma = text.find(',', start);
        const size_t end = comma == std::string_view::npos ? text.size() : comma;
        int64_t p = -1;
        const auto result = std::from_chars(text.data() + start, text.data() + end, p);
        if (out.size() == 64 || start == end || text[start] < '0' || text[start] > '9' ||
            result.ec != std::errc{} || result.ptr != text.data() + end || p < 0 || p >= batched ||
            (!out.empty() && p <= out.back()) || (comma != std::string_view::npos && end + 1 == text.size())) {
            out.clear();
            err = "prefill logits: positions must be 1..64 strictly rising decimals in [0,n_batched)";
            return false;
        }
        out.push_back(p);
        start = end + 1;
    }
    return true;
}

// One host row only. Caller must retire every GPU use of row() before any
// return, exception or file operation. The existing checked upload guard is
// used at the call site; this class deliberately has no GPU/runtime dependency.
class Writer {
public:
    Writer() = default;
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    ~Writer() {
        if (data_) std::fclose(data_);
        if (meta_) std::fclose(meta_);
    }

    bool begin(const std::string& path, const std::vector<int64_t>& positions,
               int64_t batched, int64_t vocab, bool native, int head_type, std::string& err) {
        constexpr uint64_t budget = 64ull << 20;
        constexpr uint64_t metadata_budget = 8192;
        if (started_ || failed_ || positions.empty() || positions.size() > 64 || vocab <= 0 ||
            vocab > std::numeric_limits<int32_t>::max() ||
            uint64_t(vocab) > (budget - metadata_budget - 8) / (4 * positions.size()))
            return fail(err, "invalid dimensions or output exceeds 64 MiB");
        int64_t previous = -1;
        for (int64_t p : positions) {
            if (p <= previous || p >= batched) return fail(err, "invalid positions at open");
            previous = p;
        }
        positions_ = positions;
        batched_ = batched;
        row_.resize(size_t(vocab));
        final_meta_ = path + ".meta";
        pending_meta_ = final_meta_ + ".partial";
        std::error_code ec;
        const bool exists = std::filesystem::exists(final_meta_, ec);
        if (ec || exists) return fail(err, "metadata destination already exists or cannot be checked");
        // Exclusive creation: existing raw fixtures/partial metadata are never overwritten.
        data_ = std::fopen(path.c_str(), "wbx");
        if (!data_) return fail(err, "cannot exclusively create logits file");
        meta_ = std::fopen(pending_meta_.c_str(), "wbx");
        if (!meta_) return fail(err, "cannot exclusively create partial metadata");
        const int32_t header[2] = {int32_t(vocab), 0}; // actual count is committed only after Prefill::run
        if (std::fwrite(header, sizeof(header), 1, data_) != 1 ||
            std::fprintf(meta_, "strata-prefill-logits-v1\nformat=native-endian-int32-vocab-count-float32-rows\n"
                               "vocab=%lld\ncount=%zu\nbatched=%lld\nhead=%s\nhead_ggml_type=%d\n"
                               "row\tposition\tnext_token_id\n",
                         (long long)vocab, positions.size(), (long long)batched,
                         native ? "native" : "pack-lm-head", native ? head_type : -1) < 0)
            return fail(err, "header write failed");
        started_ = true;
        return true;
    }

    bool chunk(int64_t position, int64_t count, std::string& err) {
        if (!started_ || failed_ || position != seen_ || count <= 0 || count > batched_ - seen_)
            return fail(err, "invalid, repeated or out-of-order callback range");
        seen_ += count;
        return true;
    }
    int64_t next_position() const { return written_ < positions_.size() ? positions_[written_] : -1; }
    float* row() { return row_.data(); }
    size_t row_bytes() const { return row_.size() * sizeof(float); }

    bool append(int64_t position, int64_t target, std::string& err) {
        if (!started_ || failed_ || next_position() < 0 || position != next_position() || position >= seen_ ||
            target < 0 || uint64_t(target) >= row_.size())
            return fail(err, "invalid output row or target token");
        for (float v : row_) if (!std::isfinite(v)) return fail(err, "non-finite logits");
        if (std::fwrite(row_.data(), sizeof(float), row_.size(), data_) != row_.size() ||
            std::fprintf(meta_, "%zu\t%lld\t%lld\n", written_, (long long)position, (long long)target) < 0)
            return fail(err, "row write failed");
        ++written_;
        return true;
    }

    // Only call after the complete Prefill::run (including its MTP callback) succeeds.
    // Consumers require PATH.meta plus exact binary dimensions/size. A partial
    // metadata file, even one with a footer, is never a completed fixture.
    bool finish(std::string& err) {
        if (!started_ || failed_ || seen_ != batched_ || written_ != positions_.size())
            return fail(err, "incomplete callback coverage or selected rows");
        const int32_t count = int32_t(written_);
        if (std::fseek(data_, sizeof(int32_t), SEEK_SET) != 0 ||
            std::fwrite(&count, sizeof(count), 1, data_) != 1)
            return fail(err, "final row count write failed");
        const int flush_data = std::fflush(data_);
        const int close_data = std::fclose(data_);
        data_ = nullptr;
        if (flush_data || close_data) return fail(err, "logits flush/close failed");
        const uint64_t bytes = 8 + uint64_t(written_) * row_bytes();
        if (std::fprintf(meta_, "complete\t%zu\t%llu\n", written_, (unsigned long long)bytes) < 0)
            return fail(err, "completion footer write failed");
        const int flush_meta = std::fflush(meta_);
        const int close_meta = std::fclose(meta_);
        meta_ = nullptr;
        if (flush_meta || close_meta) return fail(err, "metadata flush/close failed");
        // Atomic no-clobber publication after both files close successfully.
        // The small .partial path remains a hard link, without duplicate blocks.
        std::error_code ec;
        std::filesystem::create_hard_link(pending_meta_, final_meta_, ec);
        if (ec) return fail(err, "metadata publication failed (hard-link support required)");
        started_ = false;
        return true;
    }

private:
    bool fail(std::string& err, const char* reason) {
        failed_ = true;
        err = std::string("prefill logits: ") + reason;
        return false;
    }
    std::FILE* data_ = nullptr;
    std::FILE* meta_ = nullptr;
    std::string final_meta_, pending_meta_;
    std::vector<int64_t> positions_;
    std::vector<float> row_;
    size_t written_ = 0;
    int64_t seen_ = 0, batched_ = 0;
    bool started_ = false, failed_ = false;
};

} // namespace strata::program::prefill_logits
