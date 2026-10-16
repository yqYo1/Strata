from pathlib import Path
import difflib, hashlib, json
p=Path('src/prefill/prefill.cpp'); s=p.read_text(); d=s
helper='''// Narrow HC/base-only candidate. Keep all GDN/QSA/MoE allocations and arithmetic.
// The unfused HC route never carries xn16 across the attention/MoE phase.
const char* hc_compact_reason(const core::ModelGeometry& g, const core::SessionState& ss) {
    static const bool requested = [] {
        const char* v = std::getenv("STRATA_PREFILL_HC_COMPACT");
        return v && std::strcmp(v, "1") == 0;
    }();
    if (!requested) return "not-requested";
#if !defined(STRATA_ENABLE_SYCL)
    return "requires-SYCL-backend";
#endif
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K)
        return "unsupported-geometry";
    if (std::getenv("STRATA_DBG_NAN") != nullptr) return "DBG_NAN";
    if (!gr_unfused()) return "requires-GR_UNFUSED";
    if (bf16x2()) return "BF16X2";
    const auto& mp = mmq_plan();
    if (mp.any || !mp.fallback) return "requires-FP16-fallback";
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        if (lay.fmt.empty()) return "unknown-native-geometry";
        for (const auto& f : lay.fmt)
            if (f.n_embd != N || f.n_ff != 640) return "unsupported-native-geometry";
    }
    return "enabled";
}
bool hc_compact(const core::ModelGeometry& g, const core::SessionState& ss) {
    return std::strcmp(hc_compact_reason(g, ss), "enabled") == 0;
}
template <typename V>
void hc_take(Alloc& a, Prefill::Impl* m, V* Prefill::Impl::*field, size_t n, bool& ok) {
    V* value = a.take<V>(n, ok);
    if (m) m->*field = value;
}
// Shared count/carve sequence. All spans remain distinct inside each live phase.
void take_hc_phase(Alloc& a, Prefill::Impl* m, size_t T, bool& ok) {
    hc_take(a, m, &Prefill::Impl::xn, T * D, ok);
    hc_take(a, m, &Prefill::Impl::xn16, T * D, ok);
    hc_take(a, m, &Prefill::Impl::lo, T * LR, ok);
    hc_take(a, m, &Prefill::Impl::lo16, T * LR, ok);
    hc_take(a, m, &Prefill::Impl::gated, T * D, ok);
    if (m) m->xn16_lo = m->lo16_lo = nullptr;
}
uint64_t hc_phase_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    take_hc_phase(a, nullptr, T, ok);
    return a.used;
}
void take_hc_base(Alloc& a, Prefill::Impl* m, size_t T, bool& ok) {
    float* image = a.take<float>(T * N, ok);
    if (m) m->emb = m->mixed = m->bo = image;
    hc_take(a, m, &Prefill::Impl::R, T * D, ok);
    hc_take(a, m, &Prefill::Impl::grs, T * HC, ok);
    hc_take(a, m, &Prefill::Impl::inj, T * HC, ok);
    hc_take(a, m, &Prefill::Impl::mixed_bf, T * N, ok);
    hc_take(a, m, &Prefill::Impl::mixed_h, T * N, ok);
    hc_take(a, m, &Prefill::Impl::steps_dev, T * strata::kernels::kStepCount, ok);
    if (m) m->mixed_bf_lo = nullptr;
}

'''
needle='// #136 P3: a layout whose chunks run the fused experts'
i=d.index(needle); d=d[:i]+helper+d[i:]
d=d.replace('    bool borrowed = false;\n    cudaStream_t cs', '    bool borrowed = false;\n    bool compact_hc_base = false;\n    cudaStream_t cs',1)
start=d.index('    m.emb = o.take<float>(T * N, ok); m.R ='); end=d.index('    strata::kernels::QsaShapes s =',start)
old=d[start:end]
d=d[:start]+'''    m.compact_hc_base = hc_compact(g, ss);
    // Conservative product bound before the candidate's shared shape multiplications.
    if (m.compact_hc_base && T > SIZE_MAX / (256u * D)) return false;
    if (m.compact_hc_base) {
        take_hc_base(o, &m, T, ok);
    } else {
'''+''.join('    '+l+'\n' for l in old.splitlines())+'''    }
'''+d[end:]
d=d.replace('std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap', 'std::max({m.compact_hc_base ? hc_phase_bytes(T) : 0ull, gdn_set_bytes(T), qsa_set_bytes(T, m.cap',1)
needle='        Alloc a;\n        a.base = base; a.cap = region; a.owned = &m.owned;'
d=d.replace(needle,'''        if (m.compact_hc_base) {
            Alloc h; h.base = base; h.cap = region; h.owned = &m.owned;
            take_hc_phase(h, &m, T, ok);
        }
        if (std::getenv("STRATA_PREFILL_HC_COMPACT") != nullptr)
            std::fprintf(stderr, "strata prefill HC compact: effective=%d reason=%s T=%zu HC=%llu phase-region=%llu arena-through-region=%llu QSA-q-attn=distinct\\n",
                         (int) m.compact_hc_base, hc_compact_reason(g, ss), T,
                         (unsigned long long) (m.compact_hc_base ? hc_phase_bytes(T) : 0),
                         (unsigned long long) region, (unsigned long long) o.used);
'''+needle,1)
start=d.index('    f(T * N); f(T * D); f(T * D);'); end=d.index('    strata::kernels::QsaShapes s =',start); old=d[start:end]
d=d[:start]+'''    const bool compact_hc_base = hc_compact(g, ss);
    if (compact_hc_base && (chunk <= 0 || T > SIZE_MAX / (256u * D))) return UINT64_MAX;
    if (compact_hc_base) {
        take_hc_base(o, nullptr, T, ok);
    } else {
'''+''.join('    '+l+'\n' for l in old.splitlines())+'''    }
'''+d[end:]
d=d.replace('std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap', 'std::max({compact_hc_base ? hc_phase_bytes(T) : 0ull, gdn_set_bytes(T), qsa_set_bytes(T, cap',1)
# uint64_t is not necessarily unsigned long long: avoid initializer_list deduction mismatch.
d=d.replace(' : 0ull, gdn_set_bytes', ' : uint64_t(0), gdn_set_bytes')
out=Path('bench/results/2026-10-14-pre-Xe-model-compare/compact-hc-proposal')
patch=''.join(difflib.unified_diff(s.splitlines(True),d.splitlines(True),fromfile='a/src/prefill/prefill.cpp',tofile='b/src/prefill/prefill.cpp'))
(out/'compact-hc.patch').write_text(patch)
oldpath=Path('.tools/pre-xe-1eb-source-v1/sycl/src/prefill/prefill.cpp')
receipt={'status':'proposal-not-applied-not-built-not-tested','base_head_supplied':'4808d87e','origin_revision':'1eb89482a4afd20277ae0405780ed4f8eb98eb20','source_path':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'derived_source_sha256':hashlib.sha256(d.encode()).hexdigest(),'origin_source_sha256':hashlib.sha256(oldpath.read_bytes()).hexdigest(),'patch_sha256':hashlib.sha256(patch.encode()).hexdigest()}
(out/'hashes.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
