#!/usr/bin/env python3
"""Read live source; emit a localized patch only under this artifact. Never apply it.
Each replacement must occur exactly once. Pin whole read sources and old/new hunk
text; unrelated phase-audit edits are retained on regeneration. Main owns review,
patch application, build and execution. No Git invocation, no model/GPU access.
"""
from pathlib import Path
import difflib, hashlib, json, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
assert (ROOT / 'src/prefill/prefill.cpp').is_file(), ROOT
pins = {}
changes = []
def change(path, replacements):
    raw = (ROOT / path).read_bytes()
    original = raw.decode('utf-8')
    modified = original
    records = []
    for old, new in replacements:
        count = modified.count(old)
        if count != 1:
            raise SystemExit(f'anchor mismatch ({count}): {path}: {old[:110]!r}; re-read source and repair local anchor')
        records.append({'old_sha256': hashlib.sha256(old.encode()).hexdigest(),
                        'new_sha256': hashlib.sha256(new.encode()).hexdigest(), 'old': old})
        modified = modified.replace(old, new, 1)
    pins[path] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'anchors': records}
    changes.append(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                        fromfile='a/'+path, tofile='b/'+path, n=3)))
change('src/prefill/prefill.cpp', [
('#include "strata/prefill/gemm.hpp"', '#include "strata/prefill/gemm.hpp"\n#if defined(STRATA_ENABLE_SYCL)\n#include "strata/prefill/sycl_qsa_score_gemm.hpp"\n#endif'),
('    float* sel_scores = nullptr;          // [sel_batch, max_blocks]', '    float* sel_scores = nullptr;          // [sel_batch, max_blocks]\n#if defined(STRATA_ENABLE_SYCL)\n    QsaScoreGemm qsa_score_gemm; // View only; QSA arena owns every byte.\n#endif'),
('    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);\n    return a.used;',
'''    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
#if defined(STRATA_ENABLE_SYCL)
    if (qsa_score_mode() != QsaScoreMode::off) {
        if (s.idx_dim != 128 || s.idx_n_head != 4 || s.idx_block != 4)
            throw std::invalid_argument("QSA score GEMM requires 4 heads, 128 dimensions and blocks of 4");
        a.take<uint8_t>(qsa_score_scratch_bytes(max_blocks, sel_batch), ok);
    }
#endif
    return a.used;'''),
('    (void) prefill_fallback_attn_layout();\n    Impl& m = *impl_;',
'''    (void) prefill_fallback_attn_layout();
#if defined(STRATA_ENABLE_SYCL)
    try {
        if (qsa_score_mode() != QsaScoreMode::off &&
            (!stream || ss.max_cells < 1 || ss.max_cells > 262144 ||
             g.idx_q_heads != 4 || g.idx_key_dim != 128)) {
            err = "QSA score GEMM: requires non-null in-order SYCL queue and logical context <=262144";
            return false;
        }
    } catch (const std::exception& e) { err = e.what(); return false; }
#endif
    Impl& m = *impl_;'''),
('        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);',
'''        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
#if defined(STRATA_ENABLE_SYCL)
        if (qsa_score_mode() != QsaScoreMode::off) {
            const size_t score_bytes = qsa_score_scratch_bytes(m.max_blocks, m.sel_batch);
            void* score_arena = b.take<uint8_t>(score_bytes, ok);
            if (ok) m.qsa_score_gemm.bind(score_arena, score_bytes, m.max_blocks, m.sel_batch);
        }
#endif'''),
('    // the same allocation sequence as `init`, counted\n    const size_t T = (size_t) chunk;',
'''    // the same allocation sequence as `init`, counted
#if defined(STRATA_ENABLE_SYCL)
    if (qsa_score_mode() != QsaScoreMode::off &&
        (ss.max_cells < 1 || ss.max_cells > 262144 || g.idx_q_heads != 4 || g.idx_key_dim != 128))
        throw std::invalid_argument("QSA score GEMM: unsupported sizing geometry/logical context");
#endif
    const size_t T = (size_t) chunk;'''),
('                    pt.mark(kPfQsaSel, cs);\n                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {',
'''                    pt.mark(kPfQsaSel, cs);
#if defined(STRATA_ENABLE_SYCL)
                    const bool score_gemm = qsa_score_mode() != QsaScoreMode::off;
                    if (score_gemm) {
                        const int64_t last_n = m.steps_host[(T - 1) * strata::kernels::kStepCount + strata::kernels::kStepNKv];
                        m.qsa_score_gemm.prepare(st.idx_pooled, st.idx_dead, last_n, s, m.cs);
                    }
#endif
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {'''),
('                        if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,',
'''#if defined(STRATA_ENABLE_SYCL)
                        if (score_gemm) {
                            m.qsa_score_gemm.scores(m.q_idx + t0 * 512, steps0, nb, active, m.sel_scores, m.cs);
                        } else
#endif
                        if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,'''),
('                    // STRATA_SEL_OVERLAP (debug, D-1\'s question): how much do neighbouring queries\' selections share?',
'''#if defined(STRATA_ENABLE_SYCL)
                    if (score_gemm) m.qsa_score_gemm.invalidate();
#endif
                    // STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share?'''),
])
change('cmake/sycl_backend.cmake', [
('src/prefill/sycl/moe_mmq.cpp src/prefill/sycl/moe_fused.cpp)',
 'src/prefill/sycl/moe_mmq.cpp src/prefill/sycl/moe_fused.cpp src/prefill/sycl/qsa_score_gemm.cpp)'),
('# Standalone synthetic hot-operand row/column oneMKL descriptor diagnostic.',
'''# Manual-only score precision component oracle/benchmark; never added to CTest.
add_executable(sycl_qsa_score_gemm_component EXCLUDE_FROM_ALL tools/sycl/qsa_score_gemm_component.cpp)
target_link_libraries(sycl_qsa_score_gemm_component PRIVATE strata_prefill)
target_compile_options(sycl_qsa_score_gemm_component PRIVATE -fno-fast-math -ffp-contract=off)
# Standalone synthetic hot-operand row/column oneMKL descriptor diagnostic.'''),
])
for path in sorted((HERE / 'new').rglob('*')):
    if path.is_file():
        rel = path.relative_to(HERE/'new').as_posix()
        if (ROOT/rel).exists():
            raise SystemExit(f'new-file collision: {rel}')
        new = path.read_text()
        changes.append(''.join(difflib.unified_diff([],new.splitlines(True),fromfile='/dev/null',tofile='b/'+rel,n=3)))
for rel in ['src/prefill/layer_major.inc','src/prefill/sycl/gemm.cpp','include/strata/prefill/gemm.hpp',
            'src/kernels/sycl/qsa_select.cpp','include/strata/kernels/qsa_select.hpp',
            'include/strata/sycl/launch.hpp','include/strata/sycl/runtime.hpp',
            'src/sycl/runtime.cpp','src/prefill/layer_major.inc','CMakeLists.txt',
            'docs/ARTIFACT_RETENTION.md','AGENTS.md']:
    p = ROOT/rel
    raw=p.read_bytes()
    pins[rel]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'modified_by_candidate':False}
for name in ['research-index.json','research-prefill-qsa-xmx-scores-round589-v1.txt',
             'research-own-prefill-gemm-down-gap-round563-v1.txt',
             'research-current-qsa-matrix-work-round564-v1.txt',
             'research-qsa-headtile-slm-round568-v1.txt','research-qsa-order-divergence-round581-v1.txt',
             'research-gemm-memory-divergence-round583-v1.txt']:
    p=HERE.parent.parent/name
    raw=p.read_bytes()
    pins[str(p.relative_to(ROOT))]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'modified_by_candidate':False}
artifact_files={}
for p in sorted(HERE.parent.rglob('*')):
    if p.is_file() and p.name not in ('pins.json','candidate.patch'):
        raw=p.read_bytes()
        artifact_files[p.relative_to(HERE.parent).as_posix()]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
result=''.join(changes)
(HERE/'candidate.patch').write_text(result)
(HERE.parent/'pins.json').write_text(json.dumps({'status':'proposal-unbuilt-unexecuted',
    'generated_from_live_source':True,'patch_sha256':hashlib.sha256(result.encode()).hexdigest(),
    'files':pins,'artifact_files':artifact_files},indent=2)+'\n')
print('Wrote artifact patch and pins only; no patch application, build, test, Git, GPU, service or cleanup.')
