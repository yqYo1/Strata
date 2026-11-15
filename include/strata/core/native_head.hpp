#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Opt-in native output projection. Uploads GGUF blocks once at startup and owns its
/// activation scratch. The normal canonical head remains available for numerical A/B comparisons.
/// Plan v0.3 P6: any type the native MMVQ takes (Q5_K in the Q2_0 / IQ3_XXS files, IQ4_XS in IQ2_XS).
/// `load` takes the model's shards (strata::gguf_split_paths): the architecture is checked on the first, the
/// tensor is read from the shard that holds it.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    bool load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err);
    // Narrow SYCL prompt boundary. Caller first preflights all consumers, then
    // retires their graphs/streams before detach. Snapshot budget: 1 GiB canonical
    // weights, no files; scratch and tensor metadata remain owned and unchanged.
    // Mode2 keeps the packed representation through this same byte-exact lease.
    // Opt-in host backing retains the immutable initial upload instead of
    // reading weights back from the GPU. Graph/queue retirement is unchanged.
    bool prepare_prefill_weights(std::string& err);
    void cancel_prefill_weights() noexcept;
    void detach_prefill_weights() noexcept;
    void restore_prefill_weights() noexcept;
    void resume_prefill_weights() noexcept;
    bool head_detached() const { return head_detached_; }
    bool run(const float* mixed, float* logits, void* stream, std::string& err) const;
    uint64_t weight_bytes() const { return bytes_; }
    bool has_host_backing() const { return head_host_backed_; }
    uint64_t host_backing_bytes() const { return uint64_t(head_backing_.size()) * sizeof(uint16_t); }
    bool loaded() const { return weights_ != nullptr; }
    /// Plan v0.3 P6: the GGUF blocks, for the verify window's multi-column head.
    // Row-major consumers fail before launch when the owner is packed.
    const void* weights() const;
    bool primary_packed() const { return primary_packed_; }
    const void* packed_weights() const { return primary_packed_ ? weights_ : nullptr; }
    int type() const { return type_; }
    // Mode1 uses canonical rows for C1/MTP/DFlash2 and DPAS for decode C4.
    // Mode2 is a separate sole-packed-owner experiment;selection is immutable.
    bool decode_dpas_enabled() const { return decode_dpas_; }
    bool decode_dpas_ready() const {
        return !head_detached_ && weights_ && decode_dpas_;
    }
    // Four exact Q8_1 columns only; caller selects the decode phase explicitly.
    // May throw on an invalid owner/shape/span/queue. No internal fallback.
    void run_decode_c4(const void* q8, float* logits, void* stream) const;
    // Explicit sole-packed-owner entry. Input/output contain exactly1..4columns.
    void run_packed_columns(const void* q8, float* logits, int columns, void* stream) const;
    /// Canonical bytes per vocabulary row;packed storage is not row-addressable.
    size_t row_bytes() const { return n_out_ > 0 ? (size_t) (bytes_ / (uint64_t) n_out_) : 0; }

private:
    bool head_prepared_=false,head_detached_=false,head_release_used_=false;
    bool decode_dpas_=false; // immutable after load; reads the canonical owner
    bool primary_packed_=false; // immutable;one packed allocation,no canonical GPU copy
    bool head_host_backed_=false; // immutable startup policy,default off
    std::vector<uint16_t> head_backing_; // exact initial canonical/packed bytes;never device-written
    std::vector<uint8_t> head_snapshot_; // exact selected storage bytes,request-scoped RAM
    void* weights_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    int n_in_ = 0, n_out_ = 0;
    int type_ = -1;
};

/// Plan v0.3 P6: `token_embd.weight` in its GGUF form (the IQ model files), in mapped pinned host memory: a row
/// is read over PCIe per token, so the table costs no VRAM.  `embed_row`, the verify window and the MTP drafter
/// use it instead of the canonical table when it is set (`set_native_embed`).
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;
    bool load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err);
    /// Rows for device token ids.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
    /// One row for a host token id.
    void gather_one(int64_t token, float* out, void* stream) const;
    uint64_t bytes() const { return bytes_; }
    int type() const { return type_; }

private:
    void* host_ = nullptr;
    const void* dev_ = nullptr;
    uint64_t bytes_ = 0;
    size_t row_ = 0;
    int64_t n_embd_ = 0, n_vocab_ = 0;
    int type_ = -1;
};
void set_native_embed(const NativeEmbed* e);
const NativeEmbed* native_embed();

}  // namespace strata::core
