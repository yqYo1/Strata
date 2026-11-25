// GPU-local prompt tail for the optional DFlash2 drafter. Target arithmetic is
// untouched; only the requested layer means are retained, including BF16 input.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core { class Dflash2Drafter; }
namespace strata::prefill {

class Dflash2PrefillTaps {
public:
    Dflash2PrefillTaps() = default;
    ~Dflash2PrefillTaps();
    Dflash2PrefillTaps(const Dflash2PrefillTaps&) = delete;
    Dflash2PrefillTaps& operator=(const Dflash2PrefillTaps&) = delete;

    /// A target prompt contains n tokens; its last token is the first decode
    /// anchor, so prefill produces positions through n-2. Keep at most2047
    /// positions, never a full-context activation capture. Fresh/rewound
    /// requests clear presence metadata; cached-prefix rows are not invented.
    bool begin(int64_t n, int device, std::string& err);
    bool capture(int64_t layer, int64_t pos0, int64_t rows, const float* R,
                 const uint16_t* R_bf16, void* stream, std::string& err);
    // The caller and prefill can identify this observer through std::function.
    struct LayerTap {
        Dflash2PrefillTaps& owner;

        bool operator()(int64_t layer, int64_t pos0, int64_t rows, const float* R,
                        const uint16_t* R_bf16, void* stream, std::string& err) const {
            return owner.capture(layer, pos0, rows, R, R_bf16, stream, err);
        }
    };
    static constexpr bool captures_layer(int64_t layer) noexcept {
        return layer_slot(layer) != kLayers;
    }
    /// A completed prompt verifier window already contains all five means.
    /// Retire its copy before the caller may reuse the verifier's tap arena.
    bool capture_window(const float* taps, int64_t rows, int64_t pos0, std::string& err);
    /// Drain producers and visit contiguous complete rows, in position order.
    /// Each callback borrows GPU FP32 means [rows][5*2560]; it must complete
    /// its reads before returning. No host tensor copy is performed here.
    /// Partial/cached-prefix holes are omitted, never filled with stale data.
    using Consumer = std::function<bool(const float*, int64_t, int64_t, std::string&)>;
    bool consume(const Consumer& consumer, std::string& err);
    /// Checked drain before another queue reads the completed rows. Missing
    /// cached-prefix rows remain absent; the target still verifies every draft.
    bool ingest(core::Dflash2Drafter& drafter, std::string& err);
    void release();
    int64_t captured_rows() const { return captured_; }
    static constexpr int64_t kWidth = 2560, kStreams = 4, kLayers = 5, kTail = 2047;
    static constexpr int layers[kLayers] = {5, 15, 25, 35, 45};
    static constexpr uint64_t bytes() { return kTail * kLayers * kWidth * sizeof(float); }

private:
    static constexpr int64_t layer_slot(int64_t layer) noexcept {
        for (int64_t slot = 0; slot < kLayers; ++slot) {
            if (layers[slot] == layer) return slot;
        }
        return kLayers;
    }

    float* data_ = nullptr;
    int device_ = -1;
    int64_t first_ = 0, end_ = 0, captured_ = 0;
    std::vector<uint8_t> present_;
};

} // namespace strata::prefill
