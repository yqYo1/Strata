#!/usr/bin/env python3
"""Derive a proposal in memory; never write product source or a full candidate."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "src/prefill/prefill.cpp"
ORIGINAL = "484764fccef4cf58909bfe27fb01934da19dbffaca7f7be2565424d36ab86578"
raw = SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest() != ORIGINAL:
    raise SystemExit("Source differs from the authorized basis; stop and rebase explicitly.")
before = raw.decode()
after = before

def replace(old, new):
    global after
    if after.count(old) != 1:
        raise SystemExit("Generation anchor missing or ambiguous: " + old[:100])
    after = after.replace(old, new, 1)

replace("#include <memory>\n", "#include <memory>\n#include <limits>\n")
replace("uint64_t gdn_set_bytes(size_t T) {", r'''// Allocation-only HC opt-in. Fused normalization carries grs/xn16 (and its low
// part) across halves, so only the down/SiLU/up temporaries share phase scratch.
struct HcScratchPlan { bool requested, effective; const char* reason; };
HcScratchPlan hc_scratch_plan(const core::ModelGeometry& g, const core::SessionState& ss, size_t T) {
    const char* value = std::getenv("STRATA_PREFILL_HC_SCRATCH_COMPACT");
    const bool requested = value && std::strcmp(value, "1") == 0;
    if (!requested) return {false, false, value ? "value-not-exact-1" : "off"};
#if !defined(STRATA_ENABLE_SYCL)
    return {true, false, "sycl-only"};
#else
    if (gr_unfused()) return {true, false, "unfused-GR"};
    if (bf16x2_mode() < 0 || bf16x2_mode() > 2) return {true, false, "BF16X2-mode"};
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || ss.k != K ||
        g.n_layers != 48 || g.qsa_interval != 4 || g.n_expert != NE || g.n_ff != 640 ||
        g.ssm_state_size != 128 || g.ssm_k_heads != 16 || g.ssm_v_heads != HV ||
        g.ssm_d_conv != 4 || g.ssm_conv_channels != C || g.ssm_value_dim != ZV ||
        g.n_head != 24 || g.n_head_kv != 2 || g.head_dim != 256 ||
        g.idx_q_heads != 4 || g.idx_key_dim != 128) return {true, false, "geometry"};
    // Guard every new product and the padded HC aggregate before multiplying.
    // Keep size_t global ranges; no INT32 grid restriction is introduced here.
    constexpr uint64_t per_token_max = 4 * D + 8 * LR + 4 * N; // HC low plus shared bo
    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<size_t>::max(),
                                             std::numeric_limits<uint64_t>::max());
    if (T == 0 || T > (limit - 5 * 511) / per_token_max ||
        T > (uint64_t) std::numeric_limits<int64_t>::max() / D)
        return {true, false, "HC-byte-range"};
    return {true, true, "fused-fixed-geometry"};
#endif
}
struct HcScratch {
    float *lo, *gated;
    uint16_t *lo16, *lo16_lo;
};
HcScratch take_hc_scratch(Alloc& a, size_t T, bool& ok) {
    float* lo = a.take<float>(T * LR, ok);
    uint16_t* lo16 = a.take<uint16_t>(T * LR, ok);
    float* gated = a.take<float>(T * D, ok);
    uint16_t* low = bf16x2_hc() ? a.take<uint16_t>(T * LR, ok) : nullptr;
    return {lo, gated, lo16, low};
}
uint64_t hc_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    take_hc_scratch(a, T, ok);
    return a.used;
}
uint64_t hc_outer_removed_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * N, ok); // bo shares the embedding take
    return hc_set_bytes(T) + a.used;
}

uint64_t gdn_set_bytes(size_t T) {''')

replace('''    bool ok = true;
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok);''', '''    bool ok = true;
    const HcScratchPlan hc = hc_scratch_plan(g, ss, T);
    const uint64_t carve_start = o.used;
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok);''')
replace('''    m.xn16 = o.take<uint16_t>(T * D, ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
    m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);''', '''    m.xn16 = o.take<uint16_t>(T * D, ok);
    if (!hc.effective) {
        m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
        m.gated = o.take<float>(T * D, ok);
    }
    m.inj = o.take<float>(T * HC, ok);''')
replace('''    m.mixed_h = o.take<uint16_t>(T * N, ok); m.bo = o.take<float>(T * N, ok);
    if (bf16x2_hc()) { m.xn16_lo = o.take<uint16_t>(T * D, ok); m.lo16_lo = o.take<uint16_t>(T * LR, ok); }''', '''    m.mixed_h = o.take<uint16_t>(T * N, ok);
    // emb's last read is the initial gr_broadcast into R on the same queue.
    // PLE uses ple_emb, callbacks use R, and mixed remains a distinct take.
    m.bo = hc.effective ? m.emb : o.take<float>(T * N, ok);
    if (bf16x2_hc()) {
        m.xn16_lo = o.take<uint16_t>(T * D, ok);
        if (!hc.effective) m.lo16_lo = o.take<uint16_t>(T * LR, ok);
    }''')
replace('''    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {''', '''    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    uint64_t phase_original = 0, phase_hc = 0;
    {''')
replace('''        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert, fz)});''', '''        phase_original = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert, fz)});
        phase_hc = hc.effective ? hc_set_bytes(T) : 0;
        const uint64_t region = std::max(phase_original, phase_hc);''')
replace('''        m.region_bytes = region;
        Alloc a;''', '''        m.region_bytes = region;
        if (hc.effective && base != nullptr) {
            Alloc h;
            h.base = base; h.cap = region; h.owned = &m.owned;
            const HcScratch scratch = take_hc_scratch(h, T, ok);
            m.lo = scratch.lo; m.lo16 = scratch.lo16;
            m.gated = scratch.gated; m.lo16_lo = scratch.lo16_lo;
            if (h.used != phase_hc) ok = false;
        }
        Alloc a;''')
replace('''    m.T = (int64_t) T;
    if (mmq_plan().fallback''', '''    m.T = (int64_t) T;
    if (std::getenv("STRATA_PREFILL_HC_SCRATCH_COMPACT") != nullptr) {
        std::fprintf(stderr, "strata prefill: HC scratch compact requested=%d effective=%d reason=%s, "
                             "chunk=%zu, HC phase=%llu, original phase=%llu, shared phase=%llu, "
                             "outer removed=%llu, phase growth=%llu, chunk carved=%llu, accounted=%llu, ok=%d bytes\\n",
                     hc.requested, hc.effective, hc.reason, T, (unsigned long long) phase_hc,
                     (unsigned long long) phase_original, (unsigned long long) m.region_bytes,
                     (unsigned long long) (hc.effective ? hc_outer_removed_bytes(T) : 0),
                     (unsigned long long) (m.region_bytes - phase_original),
                     (unsigned long long) (o.used - carve_start), (unsigned long long) o.used, ok);
        std::fflush(stderr);
    }
    if (mmq_plan().fallback''')
replace('''    f(T * N); f(T * D); f(T * D); o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
    if (bf16x2_hc()) { o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok); }''', '''    const HcScratchPlan hc = hc_scratch_plan(g, ss, T);
    if (hc.effective) {
        f(T * N); f(T * D);
        f(T * D); // retain the existing estimator's conservative unfused-xn allowance
        f(T * HC); // grs is live across fused halves (absent in the old estimator)
        o.take<uint16_t>(T * D, ok);
        f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
        if (bf16x2_hc()) o.take<uint16_t>(T * D, ok);
    } else {
        // Default and declined requests preserve the original count sequence.
        f(T * N); f(T * D); f(T * D); o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
        f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
        if (bf16x2_hc()) { o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok); }
    }''')
replace('''    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert, fused_layout(T, true))}), ok);''', '''    const uint64_t phase_original = std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                              moe_set_bytes(T, g.n_expert, fused_layout(T, true))});
    o.take<uint8_t>((size_t) std::max(phase_original, hc.effective ? hc_set_bytes(T) : uint64_t{0}), ok);''')
patch = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                     fromfile="a/src/prefill/prefill.cpp",
                                     tofile="b/src/prefill/prefill.cpp"))
patch_raw = patch.encode()
(HERE / "fused-hc-scratch.patch").write_bytes(patch_raw)
manifest = {
    "status": "proposal-only; unbuilt; untested; default-off",
    "source": "src/prefill/prefill.cpp",
    "original_sha256": ORIGINAL,
    "derived_sha256": hashlib.sha256(after.encode()).hexdigest(),
    "patch_sha256": hashlib.sha256(patch_raw).hexdigest(),
    "generation_recipe": "python3 bench/results/2026-10-14-primary-compact-moe-recovery/fused-hc-proposal/generate.py",
    "full_candidate_retained": False,
    "consumer": "Main's serialized post-modelpair source review and validation",
    "next_review": "before any application; then after four-arm capacity/correctness qualification",
}
def span(n, width):
    return (n * width + 511) & ~255

# Source-derived arithmetic only, not an allocation or executed validation.
byte_plan = []
for t in (1, 17, 4096, 8192):
    for mode in (0, 1, 2):
        hc_bytes = span(t * 320, 4) + span(t * 320, 2) + span(t * 10240, 4)
        if mode == 1:
            hc_bytes += span(t * 320, 2)
        bo_bytes = span(t * 2560, 4)
        grs_bytes = span(t * 4, 4)
        byte_plan.append({
            "capacity_T": t, "BF16X2_mode": mode,
            "HC_phase_bytes": hc_bytes, "bo_take_saved_bytes": bo_bytes,
            "outer_removed_bytes": hc_bytes + bo_bytes,
            "estimator_grs_added_bytes": grs_bytes,
            "carve_reduction_if_phase_does_not_grow": hc_bytes + bo_bytes,
            "bytes_needed_reduction_if_phase_does_not_grow": hc_bytes + bo_bytes - grs_bytes,
        })
(HERE / "byte-plan.json").write_text(json.dumps({
    "status": "source-derived arithmetic only; not measured or tested",
    "span_formula": "A(n,width) = align256(n*width + 256)",
    "phase_formula": "max(HC_phase,GDN_phase,QSA_phase,MoE_phase)",
    "carve_delta_formula": "outer_removed - (new_phase - original_phase)",
    "estimator_delta_formula": "carve_delta - A(T*HC,4); existing xn allowance retained",
    "plans": byte_plan,
}, indent=2) + "\n")
(HERE / "identity.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
