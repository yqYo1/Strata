#!/usr/bin/env python3
"""Generate an unapplied diff in this directory; never write product source."""
from pathlib import Path
import hashlib, json, difflib
P = Path(__file__).resolve().parent
ROOT = P.parents[3]
path = ROOT / 'src/prefill/prefill.cpp'
a = path.read_text()
expected = '484764fccef4cf58909bfe27fb01934da19dbffaca7f7be2565424d36ab86578'
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'source pin changed'
b = a
def replace(old, new):
    global b
    assert b.count(old) == 1, old[:120]
    b = b.replace(old, new)
replace('size_t fp16_moe_rows(size_t T)', '''bool fp16_gather_batch_requested() {
    static const bool requested = [] {
        const char* v = std::getenv("STRATA_PREFILL_FP16_GATHER_BATCH");
        return v && std::strcmp(v, "1") == 0;
    }();
    return requested;
}
size_t fp16_moe_rows(size_t T)''')
replace('    const int64_t LB = stage_lb_, LE = stage_le_;', '''    const int64_t LB = stage_lb_, LE = stage_le_;
    // Opt-in diagnostics: at most four aggregate lines per run; no device reads.
    const char* fp16_batch_diag_env = std::getenv("STRATA_PREFILL_FP16_GATHER_BATCH_DIAG");
    const bool fp16_batch_diag = fp16_batch_diag_env && std::strcmp(fp16_batch_diag_env, "1") == 0;
    size_t fp16_batch_diag_lines = 0;''')
replace('                        n_order = order.size();', '''                        n_order = order.size();
                        const bool fp16_batch_requested = fp16_gather_batch_requested();
                        bool fp16_batch = false;
                        const char* fp16_batch_decline = "unset";
#if defined(STRATA_ENABLE_SYCL)
                        if (fp16_batch_requested) {
                            fp16_batch_decline = "compact-or-fallback";
                            if (m.compact_moe_fp16 && !use_mmq && !mmq_plan().any && mmq_plan().fallback) {
                                fp16_batch_decline = "peer-or-pipeline";
                                if (!m.pp && !next_ && LB == 0 && LE == m.g->n_layers && on_peer.empty()) {
                                    fp16_batch_decline = "layout";
                                    // Validate the entire immutable consumer order before choosing batching.
                                    // Any unsupported layout retains the existing per-expert compact gather.
                                    int64_t end = 0;
                                    int32_t previous = -1;
                                    bool valid = T > 0 && T <= m.T && !order.empty();
                                    for (int32_t id : order) {
                                        if (!valid || id <= previous || id < 0 || id >= m.g->n_expert) { valid = false; break; }
                                        const int64_t n = m.cnt[(size_t) id], off = m.off[(size_t) id];
                                        if (n <= 0 || n > T || off != end || off < 0 || off > T * K - n) { valid = false; break; }
                                        end = off + n;
                                        previous = id;
                                    }
                                    fp16_batch = valid && end == T * K;
                                    if (fp16_batch) fp16_batch_decline = "none";
                                }
                            }
                        }
#else
                        if (fp16_batch_requested) fp16_batch_decline = "non-sycl";
#endif
                        size_t fp16_batch_end_j = 0, fp16_gather_calls = 0;
                        int64_t fp16_batch_start = 0, fp16_batch_rows = 0, fp16_gather_rows = 0;''')
replace('''                            if (m.compact_moe_fp16) {
                                if (T <= 0''', '''                            int64_t input_row = m.compact_moe_fp16 ? 0 : o0;
                            if (m.compact_moe_fp16) {
                                if (T <= 0''')
replace('''                                pt.mark(kPfGather, cs);
                                gather_rows16(m.mixed_h, m.src_dev + o0, m.Xs, ne, N, m.cs);''', '''                                if (fp16_batch) {
                                    if (j >= fp16_batch_end_j) {
                                        fp16_batch_start = o0;
                                        fp16_batch_rows = 0;
                                        fp16_batch_end_j = j;
                                        // Capacity is the allocated scratch m.T, including on a short tail.
                                        while (fp16_batch_end_j < order.size()) {
                                            const int32_t id = order[fp16_batch_end_j];
                                            const int64_t n = m.cnt[(size_t) id], off = m.off[(size_t) id];
                                            if (n <= 0 || n > T || off != fp16_batch_start + fp16_batch_rows ||
                                                off < 0 || off > T * K - n) {
                                                err = "prefill: FP16 batch metadata changed"; return false;
                                            }
                                            if (n > m.T - fp16_batch_rows) break;
                                            fp16_batch_rows += n;
                                            ++fp16_batch_end_j;
                                        }
                                        if (fp16_batch_end_j <= j || fp16_batch_rows <= 0 || fp16_batch_rows > m.T) {
                                            err = "prefill: FP16 batch exceeds scratch"; return false;
                                        }
                                        pt.mark(kPfGather, cs);
                                        gather_rows16(m.mixed_h, m.src_dev + fp16_batch_start, m.Xs, fp16_batch_rows, N, m.cs);
                                        ++fp16_gather_calls;
                                        fp16_gather_rows += fp16_batch_rows;
                                    }
                                    input_row = o0 - fp16_batch_start;
                                    if (input_row < 0 || input_row > fp16_batch_rows - ne) {
                                        err = "prefill: FP16 batch input exceeds gathered rows"; return false;
                                    }
                                } else {
                                    pt.mark(kPfGather, cs);
                                    gather_rows16(m.mixed_h, m.src_dev + o0, m.Xs, ne, N, m.cs);
                                    ++fp16_gather_calls;
                                    fp16_gather_rows += ne;
                                }''')
replace('m.gemm.f16(m.Xs + scratch_row * N, m.dq_gu[q]', 'm.gemm.f16(m.Xs + input_row * N, m.dq_gu[q]')
replace('''                            release_to(m.g->n_expert);
                        }
                    }
                    pt.mark(kPfCombine''', '''                            release_to(m.g->n_expert);
                        }
                        if (fp16_batch_diag && fp16_batch_diag_lines < 4) {
                            ++fp16_batch_diag_lines;
                            std::fprintf(stderr, "strata prefill: FP16 activation gather batch requested=%d effective=%d decline=%s "
                                                 "layer=%lld actual_rows=%lld allocated_rows=%lld experts=%zu "
                                                 "compact_gathers=%zu gathered_rows=%lld\\n",
                                         fp16_batch_requested, fp16_batch, fp16_batch_decline,
                                         (long long) l, (long long) T, (long long) m.T, order.size(),
                                         fp16_gather_calls, (long long) fp16_gather_rows);
                        }
                    }
                    pt.mark(kPfCombine''')
patch = ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), fromfile='a/src/prefill/prefill.cpp', tofile='b/src/prefill/prefill.cpp'))
(P / 'proposal.patch').write_text(patch)
identity = dict(status='unapplied-unbuilt-untested-unmeasured-unadopted', base_head='9a6385f6', source_sha256=expected,
                candidate_in_memory_sha256=hashlib.sha256(b.encode()).hexdigest(), patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), native_pin_prefix='6361e706',
                sole_upstream='Niko1221/Strata', origin='current fork implementation; own pre-Xe 1eb historical full gather is context, not Xe donor')
(P / 'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
print(json.dumps(identity, indent=2))
