#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::prefill::detail {

// Parse once in Prefill::run, before that request submits any work. Runtime
// environment is immutable; no per-layer/per-chunk getenv or environment edits.
inline bool layer_major_kv_reuse_requested(const char* value) {
    if (!value || std::strcmp(value,"0")==0) return false;
    if (std::strcmp(value,"1")==0) return true;
    throw std::invalid_argument("STRATA_PREFILL_LAYER_MAJOR_KV_REUSE must be 0 or 1");
}

// Checked address arithmetic only; never reads the pointed-to allocation.
// The active full-context KV stage and identity table must survive draft_kv's
// reuse of the Prefill operator scratch across final-layer chunk callbacks.
inline bool layer_major_kv_disjoint(const void* a, uint64_t a_bytes,
                                    const void* b, uint64_t b_bytes) {
    const auto ap=reinterpret_cast<uintptr_t>(a), bp=reinterpret_cast<uintptr_t>(b);
    if (!a || !b || !a_bytes || !b_bytes || a_bytes>UINTPTR_MAX-ap || b_bytes>UINTPTR_MAX-bp)
        throw std::invalid_argument("invalid layer-major KV scratch/store address span");
    return ap+a_bytes<=bp || bp+b_bytes<=ap;
}

class LayerMajorKvPrefixReuse;
inline void layer_major_kv_require_idle(const LayerMajorKvPrefixReuse* active) {
    if (active) throw std::logic_error("prefill KV reuse: reentry/reset/init/relayout requires an idle request");
}

// Host scheduling state only, borrowed by Impl for one checked layer-major
// request. It is NOT a session validity marker or a new KV owner/commit protocol.
// Decisions describe same-in-order-queue submissions, not completed GPU writes.
// Failed/cancelled requests discard this object; every new call seeds afresh.
class LayerMajorKvPrefixReuse {
public:
    struct Decision { bool seed; int64_t seed_blocks; };

    LayerMajorKvPrefixReuse(int64_t rows,int64_t pos0,int64_t chunk,int64_t max_cells)
        : rows_(rows),pos0_(pos0),chunk_(chunk) {
        if (max_cells<=0 || max_cells>262144 || pos0<0 || pos0>max_cells ||
            rows<=0 || rows>max_cells-pos0 || chunk<=0 || chunk>8192)
            throw std::invalid_argument("invalid layer-major KV prefix-reuse geometry");
    }

    void begin_layer(int64_t layer,int64_t qsa_index) {
        if (active_layer_!=-1 || qsa_index!=completed_layers_ || qsa_index<0 ||
            qsa_index>=12 || layer!=4*qsa_index+3)
            throw std::logic_error("prefill KV reuse: invalid QSA layer order or reentry");
        active_layer_=layer; next_=pos0_;
    }

    Decision chunk(int64_t layer,int64_t qsa_index,int64_t position,int64_t count) {
        const int64_t end=pos0_+rows_; // Constructor checked against max_cells.
        if (active_layer_!=layer || qsa_index!=completed_layers_ || position!=next_ ||
            position<pos0_ || position>=end || count<=0 || count!=std::min(chunk_,end-position))
            throw std::logic_error("prefill KV reuse: nonmonotonic or invalid chunk");
        const bool seed=position==pos0_;
        // Match the existing copy's ceil(p0/4), including a partial first page.
        const int64_t blocks=position/4+(position%4!=0);
        next_+=count;
        if (seed) { ++seeds_; seed_blocks_+=uint64_t(blocks); }
        else { ++reused_chunks_; avoided_prefix_blocks_+=uint64_t(blocks); }
        return {seed,seed?blocks:0};
    }

    // Called only after run_impl's pass, outer checked drain and chunk cursor
    // all succeed. The stage may then be reused by a different QSA layer.
    void finish_layer(int64_t layer) {
        if (active_layer_!=layer || next_!=pos0_+rows_)
            throw std::logic_error("prefill KV reuse: incomplete QSA layer");
        active_layer_=-1; ++completed_layers_;
    }

    bool pass_matches(int64_t lb,int64_t le) const {
        return active_layer_==-1 ? (lb>=0 && lb<48 && le==lb+1 && lb%4!=3)
                                 : lb==active_layer_ && le==lb+1;
    }
    bool complete() const { return active_layer_==-1 && completed_layers_==12; }
    uint64_t seeds() const { return seeds_; }
    uint64_t seed_blocks() const { return seed_blocks_; }
    uint64_t reused_chunks() const { return reused_chunks_; }
    uint64_t avoided_prefix_blocks() const { return avoided_prefix_blocks_; }

private:
    int64_t rows_,pos0_,chunk_;
    int64_t active_layer_=-1,completed_layers_=0,next_=0;
    // Bounds: <=12 * ceil(262144/1) chunks, <=65536 blocks per decision.
    // These counters fit uint64_t even at the physical planning limit.
    uint64_t seeds_=0,seed_blocks_=0,reused_chunks_=0,avoided_prefix_blocks_=0;
};

} // namespace strata::prefill::detail
