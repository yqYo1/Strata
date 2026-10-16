#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace strata::prefill { class Prefill; }

namespace strata::core {

/// Opt-in native output projection. Uploads unchanged GGUF blocks once at startup and owns its
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
    bool run(const float* mixed, float* logits, void* stream, std::string& err) const;
    uint64_t weight_bytes() const { return bytes_; }
    bool loaded() const { return weights_ != nullptr; }
    /// Plan v0.3 P6: the GGUF blocks, for the verify window's multi-column head.
    const void* weights() const { return weights_; }
    int type() const { return type_; }
    int64_t vocabulary() const { return loaded() ? n_out_ : 0; }
    /// Bytes of one vocabulary row.
    size_t row_bytes() const { return n_out_ > 0 ? (size_t) (bytes_ / (uint64_t) n_out_) : 0; }

private:
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
    /// Direct device entrypoint: downloads/checks IDs before submitting a gather.
    /// Graph callers below use a private path after their own host admission.
    /// Caller owns an immutable ID range through gather completion; its producers
    /// and this call must use the same in-order queue.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
    /// One row for a host token id.
    void gather_one(int64_t token, float* out, void* stream) const;
    uint64_t bytes() const { return bytes_; }
    int type() const { return type_; }
    int64_t vocabulary() const { return dev_ ? n_vocab_ : 0; }

private:
    friend class MtpDrafter;
    friend class Verifier;
    friend class strata::prefill::Prefill;
    // Only these callers may record this: they admit every host input before
    // submitting a graph, and require sampled/subset IDs to fit this vocabulary.
    void gather_dev_admitted(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
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
