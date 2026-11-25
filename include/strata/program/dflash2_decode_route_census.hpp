#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace strata::program {

// The measured single-card DFlash2 verifier visits every layer once per window.
// All offered rows count, including drafts later rejected by the target.
class Dflash2DecodeRouteCensus {
public:
    static constexpr int layers = 48;
    static constexpr int experts = 512;
    static constexpr int fanout = 10;
    static constexpr int max_width = 4;
    static constexpr int max_windows = 256;

    void begin_window(int width) {
        if (width_ != 0 || width < 1 || width > max_width || windows_ == max_windows) {
            valid_ = false;
            return;
        }
        width_ = width;
        next_layer_ = 0;
    }

    void observe_layer(int64_t layer, const int32_t* ids, int64_t width, int64_t k) {
        if (!valid_ || width_ == 0 || width != width_ || k != fanout ||
            layer != next_layer_ || layer >= layers || ids == nullptr) {
            valid_ = false;
            return;
        }

        auto* counts = counts_.data() + static_cast<std::size_t>(layer) * experts;
        for (int i = 0; i < width_ * fanout; ++i) {
            const int32_t expert = ids[i];
            if (expert < 0 || expert >= experts) {
                valid_ = false;
                return;
            }
            ++counts[expert];
        }
        ++next_layer_;
    }

    void complete_window() {
        if (!valid_ || width_ == 0 || next_layer_ != layers) {
            valid_ = false;
            return;
        }
        ++windows_;
        offered_rows_ += width_;
        width_ = 0;
    }

    // One little-endian record: uint64 request, windows, offered rows,
    // then the layer-major uint32 expert counts. Write after the final verifier commit.
    bool write(std::FILE* file, uint64_t request) const {
        static_assert(std::endian::native == std::endian::little);
        if (!valid_ || width_ != 0 || file == nullptr) return false;
        const uint64_t header[] = {request, windows_, offered_rows_};
        return std::fwrite(header, sizeof(uint64_t), 3, file) == 3 &&
               std::fwrite(counts_.data(), sizeof(uint32_t), counts_.size(), file) == counts_.size() &&
               std::fflush(file) == 0;
    }

private:
    std::array<uint32_t, layers * experts> counts_{};
    uint64_t windows_ = 0;
    uint64_t offered_rows_ = 0;
    int width_ = 0;
    int next_layer_ = 0;
    bool valid_ = true;
};

} // namespace strata::program
