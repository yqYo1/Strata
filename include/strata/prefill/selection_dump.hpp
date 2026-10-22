#pragma once
// Main-authored checked diagnostic readback; repository LICENSE applies.
#include "strata/prefill/draft_uploads.hpp"
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

namespace strata::prefill::detail {

// copy(dst, src, bytes) and drain() return success. fatal() must terminate:
// uncertain DMA completion must never unwind through its host destination.
template<class Copy, class Drain, class Fatal>
bool selection_dump(const char* path, int64_t layer, int64_t first,
                    int64_t rows, int64_t cap, const int32_t* source,
                    Copy copy, Drain drain, Fatal fatal, std::string& error) {
    if (!path || !*path || !source || layer < 0 || first < 0 || rows <= 0 || cap <= 0 ||
        layer > INT32_MAX || first > INT32_MAX || rows > INT32_MAX || cap > INT32_MAX ||
        first > INT32_MAX - rows ||
        uint64_t(rows) > std::numeric_limits<size_t>::max() / sizeof(int32_t) / uint64_t(cap)) {
        error = "QSA selection dump: invalid geometry/path"; return false;
    }
    try {
        const size_t count = size_t(rows) * size_t(cap);
        std::vector<int32_t> host(count);
        auto stop = [&]() noexcept {
            try { fatal(); } catch (...) {}
            std::abort();
        };
        DraftUploadGuard pending(drain, stop); // declared after host
        pending.arm(); // covers partial submission and a throwing callback
        if (!copy(host.data(), source, count * sizeof(int32_t))) {
            error = "QSA selection dump: readback submission failed"; return false;
        }
        if (!drain()) stop();
        pending.retire();
        const int32_t header[] = {int32_t(layer), int32_t(first), int32_t(rows), int32_t(cap)};
        std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::fopen(path, "ab"), &std::fclose);
        if (!file) { error = "QSA selection dump: open failed"; return false; }
        if (std::fwrite(header, sizeof(int32_t), 4, file.get()) != 4 ||
            std::fwrite(host.data(), sizeof(int32_t), count, file.get()) != count) {
            error = "QSA selection dump: write failed"; return false;
        }
        if (std::fclose(file.release()) != 0) {
            error = "QSA selection dump: close failed"; return false;
        }
        return true;
    } catch (const std::exception& exc) {
        error = std::string("QSA selection dump: ") + exc.what(); return false;
    }
}
} // namespace strata::prefill::detail
