"""Prepare a private layer-major dequant-output memo; no device submission."""
from pathlib import Path
import datetime
import hashlib
import json

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = base/'expanded-fp16-cache-prepared'
out.mkdir(mode=0o700)
source = root/'sycl/src/prefill/prefill.cpp'
original = source.read_text()
assert hashlib.sha256(original.encode()).hexdigest() == 'bdc0dea41753d345f5b5c329d807d3f9013280e2488e39df1d615a3b84506b92'
candidate = original
patches = []

def replace(old, new):
    global candidate
    assert candidate.count(old) == 1, old[:90]
    candidate = candidate.replace(old, new)
    patches.append({'old': old, 'new': new})

replace('#include "strata/host_wait.hpp"', '#include "strata/host_wait.hpp"\n#include "expanded_fp16_layer_cache.hpp"')
replace('    PfTimer* phase_context = nullptr;', '    PfTimer* phase_context = nullptr;\n    ExpandedFp16LayerCache* fp16_cache = nullptr;')
replace('    PfTimer phases;\n    // Restore decode\'s cache', '''    PfTimer phases;
    ExpandedFp16LayerCache expanded_cache;
    const int expanded_request = ExpandedFp16LayerCache::requested(g.n_expert);
    if (expanded_request && n > first_len && layout.native) {
        size_t free_bytes = 0, total_bytes = 0;
        dpct::get_current_device().get_memory_info(free_bytes, total_bytes);
        expanded_cache.initialize(*m.cs, g.n_expert, expanded_request, free_bytes);
    }
    if (expanded_request)
        std::fprintf(stderr, "strata prefill FP16 cache allocation: requested %d, capacity %d, bytes %llu\\n",
                     expanded_request, expanded_cache.capacity(), (unsigned long long) expanded_cache.bytes());
    // Restore decode's cache''')
replace('        decltype(on_chunk) chunk; decltype(on_stage_chunk) stage;\n        ~Restore()', '        decltype(on_chunk) chunk; decltype(on_stage_chunk) stage;\n        ExpandedFp16LayerCache* fp16_cache;\n        ~Restore()')
replace('            m.cache = cache; m.host_res = residency; m.transfer_context = timer;', '            m.cache = cache; m.host_res = residency; m.transfer_context = timer;\n            m.fp16_cache = fp16_cache;')
replace('              m.residual_reused_tokens, m.residual_inplace, m.phase_context, on_chunk, on_stage_chunk};', '              m.residual_reused_tokens, m.residual_inplace, m.phase_context, on_chunk, on_stage_chunk, m.fp16_cache};')
replace('    m.cache = &layer_cache; m.host_res = residency.data(); m.transfer_context = &transfers;', '    m.cache = &layer_cache; m.host_res = residency.data(); m.transfer_context = &transfers;\n    m.fp16_cache = expanded_cache.enabled() ? &expanded_cache : nullptr;')
replace('    trace_memory("prefill completed with temporary layer cache");\n    layer_cache.close();', '''    trace_memory("prefill completed with temporary layer cache");
    if (expanded_request) expanded_cache.report();
    m.fp16_cache = restore.fp16_cache;
    expanded_cache.reset();
    layer_cache.close();''')
old = '''                            const int q = (int) (j % DQ);
                            if (lay.native) {
                                // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                                const auto& f = lay.fmt[(size_t) l];
                                strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                                   m.dq_gu[q], m.cs);
                                strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs);
                            } else {
                                blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                            }'''
new = '''                            const int q = (int) (j % DQ);
                            ExpandedFp16LayerCache::Entry expanded;
                            if (m.fp16_cache && lay.native) {
                                const auto& f = lay.fmt[(size_t) l];
                                if (f.n_ff == 640 && f.n_embd == 2560)
                                    expanded = m.fp16_cache->acquire(l, e);
                            }
                            uint16_t* const dq_gu = expanded.gu ? expanded.gu : m.dq_gu[q];
                            uint16_t* const dq_d = expanded.down ? expanded.down : m.dq_d[q];
                            if (!expanded.hit) {
                                if (lay.native) {
                                    // Same formulas, launch geometry and FP16 outputs as the uncached path.
                                    const auto& f = lay.fmt[(size_t) l];
                                    strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                                       dq_gu, m.cs);
                                    strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, dq_d, m.cs);
                                } else {
                                    blob_dequant_f16(blob_dev, dq_gu, dq_d, m.cs);
                                }
                                if (expanded.slot >= 0) m.fp16_cache->publish(expanded);
                            }'''
replace(old, new)
replace('m.gemm.f16(m.Xs + scratch_row * N, m.dq_gu[q], m.GU + scratch_row * 1280, ne, 1280, N);', 'm.gemm.f16(m.Xs + scratch_row * N, dq_gu, m.GU + scratch_row * 1280, ne, 1280, N);')
replace('m.gemm.f16(m.Hh + scratch_row * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);', 'm.gemm.f16(m.Hh + scratch_row * 640, dq_d, m.Dm + o0 * N, ne, N, 640);')
(out/'prefill.candidate.cpp').write_text(candidate)
(out/'expanded_fp16_layer_cache.hpp').write_bytes((base/'expanded_fp16_layer_cache.hpp').read_bytes())
record = {'scope': 'CPU-only preparation of opt-in unchanged-FP16 expert reuse within a layer-major request; GPU parity, allocation/lifetime diagnostics, timings and 256K fallback untested',
          'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
          'original_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
          'candidate_source_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
          'header_sha256': hashlib.sha256((base/'expanded_fp16_layer_cache.hpp').read_bytes()).hexdigest(),
          'patches': patches, 'default_experts': 0, 'reserve_bytes': 256*1024*1024,
          'bytes_per_expert': (1280*2560+2560*640)*2,
          'preserved': ['IQ dequant bodies and wrappers', 'FP16 values', 'GEMM dimensions/row counts/activation operands',
                        'phase waits', 'layer loading and residual transfers', 'main/MTP release/restore callbacks'],
          'required': ['same in-order queue for allocation/producers/GEMMs and wait before free',
                       'reset expert tags before reuse for a new layer', 'publish only after both producers enqueue',
                       'free optional allocation before main/MTP restore', 'normal-MTP repeated heads and full capacity gates']}
(out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k != 'patches'}, indent=2))
