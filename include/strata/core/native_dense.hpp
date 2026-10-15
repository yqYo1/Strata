#pragma once

#include <cstdint>
#include <map>
#include "strata/core/weights.hpp"
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, bool capture_layer_bank = false,
              uint64_t layer_bank_ram_limit = (3ULL << 30));
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return weights_.size(); }

    // Explicit startup opt-in only: capture_layer_bank copies ORIGINAL GGUF bytes
    // during load. No late D2H snapshot or source-file reread is performed.
    enum class Residency { Full, Bank };
    struct ReaderRetirement {
        // Caller asserts ALL target-NativeDense-reading decode/token/MTP/verifier graphs are destroyed,
        // all streams/callbacks drained and no normal-table reader may resume
        // until restore_full succeeds and graphs have been recaptured.
        bool all_readers_retired = false;
    };
    struct PackedSource {
        std::string name, shard;
        uint64_t gguf_offset = 0, data_start = 0, file_bytes = 0;
        int layer = -1;
        WeightRef original;
        std::vector<uint8_t> packed; // owned immutable original bytes
    };
    bool prepare_layer_bank(std::string& err); // Full-only allocation; no publications/free
    void cancel_prepared_bank(); // Full-only checked release, no submitted use
    bool enter_layer_bank(ReaderRetirement retirement, std::string& err);
    // Synchronous publication: a view remains valid until the NEXT activate or
    // restore. Caller must retire every prior view reader first, on this same
    // ordered stream. No other stream, retained callback or graph may use it.
    bool activate_layer(int layer, void* stream, ReaderRetirement retirement, std::string& err);
    bool restore_full(void* stream, ReaderRetirement retirement, std::string& err);
    const std::map<std::string, WeightRef>& layer_view() const { return layer_view_; }
    const std::vector<PackedSource>& packed_sources() const { return sources_; }
    Residency residency() const { return residency_; }
    uint64_t bank_bytes() const { return bank_bytes_; }
    uint64_t snapshot_bytes() const { return sources_.empty() ? 0 : bytes_; }
    // Requires exclusive owner access. Not thread-safe. Destructor in Bank
    // state fail-stops: cancellation must explicitly drain and restore first.

private:
    std::vector<PackedSource> sources_;
    WeightTable* table_ = nullptr; // must outlive NativeDense
    Residency residency_ = Residency::Full;
    uint64_t bank_bytes_ = 0;
    void* bank_ = nullptr;
    std::map<std::string, WeightRef> layer_view_;
    bool validate_table(bool withdrawn, std::string& err) const;
    std::vector<void*> weights_;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
};
} // namespace strata::core
