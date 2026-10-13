// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "capture requires IEEE FP32");
// Host-only diagnostic sink. A completed file represents one row, never a
// teacher-forced sequence or a quality verdict. No device ownership lives here.
inline bool serve_first_row_opt_in(const char* value) {
    return value && std::strcmp(value, "1") == 0;
}
inline bool serve_first_row_bytes(int64_t vocab, uint64_t& bytes) {
    if (vocab <= 0 || vocab > std::numeric_limits<int32_t>::max()) return false;
    bytes = 8 + uint64_t(vocab) * sizeof(float);
    return bytes <= (64u << 20);
}
// Shared by production and CPU write-fault fixtures; always close, even after a
// short write or failed flush. A failure must never reach publication.
struct ServeFirstRowIO {
    using Write = size_t (*)(const void*, size_t, size_t, std::FILE*);
    using Flush = int (*)(std::FILE*);
    using Close = int (*)(std::FILE*);
    Write write = std::fwrite;
    Flush flush = std::fflush;
    Close close = std::fclose;
};
inline bool serve_first_row_write(std::FILE* file, const std::vector<float>& row,
                                  ServeFirstRowIO io = {}) {
    const int32_t header[2] = {int32_t(row.size()), 1};
    bool ok = io.write(header, sizeof(int32_t), 2, file) == 2;
    if (ok) ok = io.write(row.data(), sizeof(float), row.size(), file) == row.size();
    if (io.flush(file) != 0) ok = false;
    if (io.close(file) != 0) ok = false;
    return ok;
}
class ServeFirstRowCapture {
public:
    bool open(const char* directory, std::string& error) {
        if (!directory || !*directory) { error = "explicit STRATA_SERVE_FIRST_ROW_DIR required"; return false; }
        std::error_code ec;
        auto path = std::filesystem::canonical(directory, ec);
        if (ec || !std::filesystem::is_directory(path, ec) || ec) {
            error = "capture directory must already exist"; return false;
        }
        directory_ = std::move(path);
        return true;
    }
    bool begin(int64_t vocab, std::string& error) {
        uint64_t bytes = 0;
        if (active_ || requests_ >= 64 || !serve_first_row_bytes(vocab, bytes) || bytes > budget - bytes_) {
            error = "capture request/byte budget or active request"; return false;
        }
        ++requests_;
        bytes_ += bytes; // reserve attempts too; a failure does not restore quota
        row_bytes_ = bytes;
        vocab_ = vocab;
        active_ = true;
        return true;
    }
    bool publish(const std::vector<float>& row, int64_t prompt, std::string& error) {
        if (!active_ || prompt < 2 || row.size() != size_t(vocab_) ||
            !std::all_of(row.begin(), row.end(), [](float v) { return std::isfinite(v); })) {
            error = "capture incomplete/nonfinite/invalid row"; return false;
        }
        const auto final = directory_ / ("request-" + std::to_string(requests_) + ".bin");
        const auto partial = directory_ / ("request-" + std::to_string(requests_) + ".partial");
        // C11 exclusive creation refuses regular files and symlinks alike.
        std::FILE* file = std::fopen(partial.string().c_str(), "wbx");
        if (!file) { error = "exclusive capture create failed"; return false; }
        if (!serve_first_row_write(file, row)) { error = "capture write/flush/close failed; partial retained"; return false; }
        std::error_code ec;
        if (std::filesystem::file_size(partial, ec) != row_bytes_ || ec) {
            error = "capture completed size mismatch"; return false;
        }
        // Hard-link publication is atomic and refuses any existing final name.
        // Retain the partial link for Main's serialized hash/retention review.
        std::filesystem::create_hard_link(partial, final, ec);
        if (ec) { error = "exclusive capture publication failed; partial retained"; return false; }
        active_ = false;
        std::fprintf(stderr, "strata serve-first-row: COMPLETE request=%llu prompt_P=%lld predicted_position=%lld "
                             "source_row=0 actual_reuse=0 vocab=%lld bytes=%llu file=%s diagnostic_only=1\n",
                     (unsigned long long) requests_, (long long) prompt, (long long) (prompt - 1),
                     (long long) vocab_, (unsigned long long) row_bytes_, final.string().c_str());
        std::fflush(stderr);
        return true;
    }
    uint64_t request() const { return requests_; }
private:
    static constexpr uint64_t budget = 64u << 20;
    std::filesystem::path directory_;
    uint64_t requests_ = 0, bytes_ = 0, row_bytes_ = 0;
    int64_t vocab_ = 0;
    bool active_ = false;
};
} // namespace strata::core
