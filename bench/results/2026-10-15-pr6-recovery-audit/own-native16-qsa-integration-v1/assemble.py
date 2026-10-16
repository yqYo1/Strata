# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Niko1221 and the Strata contributors
"""Package assembly only: hash/read source, emit unified patch inside this package.
Never modifies product files, invokes Git/builds/tests or loads a GPU library.
"""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDIT = HERE.parent
OLD = AUDIT / 'own-native16-attention-v1'
BASE = {
    'src/prefill/prefill.cpp': 'eb513f3a8579254f34f81f137a0c790be94b763ac3ac79e4fdb2114f220dd297',
    'src/prefill/layer_major.inc': '2097f08fe4a4d152d8f66000164c0adc0dad4a64d24958400c14c301abc4682c',
    'cmake/sycl_backend.cmake': '70ebb97dac6e5d8347022305178fabcfb3d8c09d56fc149990324bbb06f8af0b',
    'include/strata/kernels/qsa_prompt_attn.hpp': 'ad662c1dbb166e1c31b933ced070fa65250ffda20b03f86a7f3b38add8138b75',
    'src/kernels/sycl/prompt_attention.cpp': 'c88bd802d17ec5f9e1df05eee5581c9f50be86e87b1a454783b5c75aca55be3d',
}
def digest(data): return hashlib.sha256(data).hexdigest()
base = {}
for name, pin in BASE.items():
    value = (ROOT / name).read_bytes()
    if digest(value) != pin:
        raise RuntimeError(f'Base moved: {name}; re-review rather than replacing it')
    base[name] = value.decode()
new = dict(base)
def replace(path, before, after):
    if new[path].count(before) != 1:
        raise RuntimeError(f'Nonunique patch anchor: {path}: {before[:80]}')
    new[path] = new[path].replace(before, after)

# Rebase only the actual reviewed appended R787 BF16 addition. Its FP32 source
# is still byte-identical to the current prompt file, checked here, not assumed.
prompt = 'src/kernels/sycl/prompt_attention.cpp'
old_prompt = (OLD / 'files' / prompt).read_text()
if not old_prompt.startswith(base[prompt]): raise RuntimeError('R787 prefix changed')
if digest(old_prompt.encode()) != '743e5781346939e952f5196cd1f70d8e7de10700df3cd5ce38bcec37648f1a53':
    raise RuntimeError('R787 source changed')
addition = old_prompt[len(base[prompt]):]
start = addition.index('void prompt_bf16_registered(')
end = addition.index('void launch_prompt_bf16(', start)
addition = addition[:start] + '''void prompt_bf16_registered(sycl::queue &queue, Span span) {
  // Full range, live Strata device allocation, and exact queue context/device.
  // The caller still supplies true suballocation capacities from the owners.
  validate_device_span(span.pointer, span.bytes, queue);
}
''' + addition[end:]
api = 'bool qsa_prompt_attn_bf16_batch('
if addition.count(api) != 1: raise RuntimeError('R787 API changed')
addition = addition.replace(api, 'bool qsa_prompt_attn_bf16_validate(', 1)
addition = addition.replace('!p.k_q || !p.v_q || !p.k_scale || !p.v_scale || p.k_q4 || p.v_q4)',
    '!p.k_q || !p.v_q || !p.k_scale || !p.v_scale || p.k_pool || p.v_pool || p.k_q4 || p.v_q4)')
before = '''  launch_prompt_bf16(q, p, ids, steps, cap, s, out, nq, bounds, stream);
  return true;
}
} // namespace strata::kernels
'''
after = '''  return true;
}

bool qsa_prompt_attn_bf16_batch(const uint16_t *q, const QsaAttnPools &p,
                                const int32_t *ids, const int32_t *steps,
                                int64_t cap, const QsaShapes &s, uint16_t *out,
                                int64_t nq, const QsaPromptBf16Bounds &bounds,
                                void *stream) {
  if (!qsa_prompt_attn_bf16_validate(q, p, ids, steps, cap, s, out, nq, bounds, stream))
    return false;
  if (nq != 0) launch_prompt_bf16(q, p, ids, steps, cap, s, out, nq, bounds, stream);
  return true;
}
} // namespace strata::kernels
'''
if addition.count(before) != 1: raise RuntimeError('R787 validation tail changed')
addition = addition.replace(before, after)
new[prompt] += addition
# Preserve the exact reviewed numerical/collective core.
old_core = old_prompt[old_prompt.index('void launch_prompt_bf16('):old_prompt.index('\n} // namespace\n\nbool qsa_prompt_attn_bf16_batch(')]
assert old_core in new[prompt] and new[prompt].startswith(base[prompt])

header = 'include/strata/kernels/qsa_prompt_attn.hpp'
old_header = (OLD / 'files' / header).read_text()
if digest(old_header.encode()) != '94f47f825e5fd3dac6c61f4d37973b039ccad1cb2411b09a0db6044d3ce5a4e4':
    raise RuntimeError('R787 header changed')
piece = old_header[old_header.index('#if defined(STRATA_ENABLE_SYCL)'):old_header.rindex('\n}  // namespace strata::kernels')]
piece = piece.replace('Invalid/null/unregistered/overlapping spans throw before submission.',
    'Invalid/null/foreign/host/unregistered/overlapping spans throw before submission.\n/// Full spans must fit live Strata device allocations in the queue context/device.')
declaration = piece[piece.index('bool qsa_prompt_attn_bf16_batch('):piece.index('\n#endif')]
piece = piece.replace('\n#endif', '\n/// Same complete validation/capability checks; never submits or waits.\n' +
    declaration.replace('qsa_prompt_attn_bf16_batch', 'qsa_prompt_attn_bf16_validate') + '\n#endif')
replace(header, '\n}  // namespace strata::kernels\n', '\n' + piece + '\n}  // namespace strata::kernels\n')

pf = 'src/prefill/prefill.cpp'
replace(pf, '#include "strata/prefill/native16.hpp"',
    '#include "strata/prefill/native16.hpp"\n#include "strata/prefill/qsa_native16.hpp"\n#include "strata/kernels/mrope.hpp"')
replace(pf, '    bool native16_active = false;', '''    bool native16_active = false;
    // Disjoint typed subviews of existing QSA reservations; no new allocation.
    uint16_t *qsa_Kraw16 = nullptr, *qsa_Krope16 = nullptr, *qsa_V16 = nullptr,
             *qsa_Qfull16 = nullptr, *qsa_q16 = nullptr, *qsa_attn16 = nullptr;''')
replace(pf, '    (void)native16::requested(); // validate the immutable startup setting',
    '    (void)native16::requested(); // validate the immutable startup setting\n    (void)native16::qsa::requested();')
anchor = '        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);'
replace(pf, anchor, anchor + '''
#if defined(STRATA_ENABLE_SYCL)
        // K uses both halves of its original float reservation. Raw and
        // normalized/rotated BF16 views must coexist during the fused kernel.
        m.qsa_Kraw16 = reinterpret_cast<uint16_t*>(m.Kc);
        m.qsa_Krope16 = m.qsa_Kraw16 ? m.qsa_Kraw16 + T * 512 : nullptr;
        m.qsa_V16 = reinterpret_cast<uint16_t*>(m.Vc);
        m.qsa_Qfull16 = reinterpret_cast<uint16_t*>(m.Qf);
        m.qsa_q16 = reinterpret_cast<uint16_t*>(m.q);
        m.qsa_attn16 = reinterpret_cast<uint16_t*>(m.attn);
#endif''')
anchor = '                    const core::QsaState& st = ss.qsa_states[qsa_index];'
replace(pf, anchor, anchor + '''
#if defined(STRATA_ENABLE_SYCL)
                    const bool typed_qsa = m.native16_active && native16::qsa::requested();
#endif''')
before = '''                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;'''
replace(pf, before, '''#if defined(STRATA_ENABLE_SYCL)
                    if (typed_qsa) {
                        if (!native_proj16(m.gemm, wk, m.mixed_h, m.qsa_Kraw16, T, v.name("attn_k.weight"), err) ||
                            !native_proj16(m.gemm, wv, m.mixed_h, m.qsa_V16, T, v.name("attn_v.weight"), err) ||
                            !native_proj16(m.gemm, wq, m.mixed_h, m.qsa_Qfull16, T, v.name("attn_q.weight"), err)) return false;
                    } else {
#endif
''' + before + '''
#if defined(STRATA_ENABLE_SYCL)
                    }
#endif''')
before = '''                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, strata::kernels::rope_scaling(), m.cs);'''
replace(pf, before, '''#if defined(STRATA_ENABLE_SYCL)
                    if (typed_qsa) {
                        native16::qsa::norm_rope_split(m.qsa_Kraw16, static_cast<const float*>(wkn->data),
                            m.qsa_Krope16, T, 2, 256, 512, 512, p0, st.max_cells,
                            st.cos_tab, st.sin_tab, EPS, m.cs);
                    } else {
#endif
''' + before + '''
#if defined(STRATA_ENABLE_SYCL)
                    }
#endif''')
anchor = '                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)'
replace(pf, anchor, '''#if defined(STRATA_ENABLE_SYCL)
                    if (typed_qsa) {
                        native16::qsa::append_q8(m.qsa_Krope16, m.qsa_V16, T, 512, p0,
                            {st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table},
                            s, {st.n_pages, st.n_slots}, m.cs);
                    } else {
#endif
''' + anchor)
before = '''                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, strata::kernels::rope_scaling(), m.cs);
                    if (st.kv_rot) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);'''
replace(pf, before, '''#if defined(STRATA_ENABLE_SYCL)
                    }
                    if (typed_qsa) {
                        native16::qsa::norm_rope_split(m.qsa_Qfull16, static_cast<const float*>(wqn->data),
                            m.qsa_q16, T, 24, 512, 12288, 6144, p0, st.max_cells,
                            st.cos_tab, st.sin_tab, EPS, m.cs);
                    } else {
#endif
''' + before + '''
#if defined(STRATA_ENABLE_SYCL)
                    }
#endif''')
before = '''                    const bool matrix_used = !old_attn && strata::kernels::qsa_prompt_attn_batch(
                        m.q, pools, m.sel_ids, m.steps_dev, m.cap, s, m.attn, T, m.cs);'''
replace(pf, before, '''                    bool matrix_used = false;
#if defined(STRATA_ENABLE_SYCL)
                    if (typed_qsa) {
                        matrix_used = strata::kernels::qsa_prompt_attn_bf16_batch(
                            m.qsa_q16, pools, m.sel_ids, m.steps_dev, m.cap, s, m.qsa_attn16, T,
                            {st.n_pages, st.n_slots}, m.cs);
                        if (!matrix_used) {
                            err = "prefill QSA16: unsupported typed attention; no fallback submitted"; return false;
                        }
                    } else
#endif
                        matrix_used = !old_attn && strata::kernels::qsa_prompt_attn_batch(
                            m.q, pools, m.sel_ids, m.steps_dev, m.cap, s, m.attn, T, m.cs);''')
anchor = '                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);'
replace(pf, anchor, '''#if defined(STRATA_ENABLE_SYCL)
                    if (typed_qsa)
                        native16::qsa::gate(m.qsa_attn16, m.qsa_Qfull16, m.attn_h,
                            T, 6144, 12288, 6144, m.cs);
                    else
#endif
''' + anchor)

lm = 'src/prefill/layer_major.inc'
anchor = '    const bool native16_on = native16::requested();'
replace(lm, anchor, anchor + '''
    const bool qsa16_on = native16::qsa::requested();
    if (qsa16_on && !native16_on) {
        err = "prefill QSA16: requires STRATA_SYCL_PREFILL_NATIVE16=bf16"; return false;
    }''')
anchor = '''    // This route never submits a Stager job. Do not enter over a live previous'''
replace(lm, anchor, '''    // QSA16 v1 deliberately supports a resident text INT8 cache only. Reject
    // every unsupported layer/device/domain BEFORE this request submits work.
    if (qsa16_on) {
        native16::qsa::validate_device(m.cs);
        // The legacy OLD switch is presence-based, including a value of 0.
        if (std::getenv("STRATA_PROMPT_ATTN_OLD")) {
            err = "prefill QSA16: leave STRATA_PROMPT_ATTN_OLD unset"; return false;
        }
        const char* mapping = std::getenv("STRATA_SYCL_PROMPT_K_CELL_MAJOR");
        if (mapping && std::strcmp(mapping, "0")) {
            err = "prefill QSA16: requires prompt Mapping0"; return false;
        }
        const auto tab = strata::kernels::rope_table_for(strata::kernels::rope_scaling());
        if (strata::kernels::mrope_table() || !tab.cos) {
            err = "prefill QSA16: requires text positions and STRATA_ROPE_TABLE=1"; return false;
        }
        const auto shapes = strata::kernels::qsa_real_shapes();
        for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
            const auto& q = ss.qsa_states[i];
            if (q.kv_mode != 0 || !q.kv_int8 || q.kv_rot || q.kv_q4 || q.kv_hybrid || q.host.present() ||
                q.n_slots != q.n_pages || tab.cos != q.cos_tab || tab.sin != q.sin_tab ||
                tab.max_pos != q.max_cells) {
                err = "prefill QSA16: requires resident unrotated INT8 K/V and the owning session RoPE tables";
                return false;
            }
            sycl_backend::validate_device_span(q.cos_tab, size_t(q.max_cells) * 32 * sizeof(float),
                                               sycl_backend::queue_for(m.cs));
            sycl_backend::validate_device_span(q.sin_tab, size_t(q.max_cells) * 32 * sizeof(float),
                                               sycl_backend::queue_for(m.cs));
            // Owner n_slots sizes physical pools, n_pages sizes the table.
            // max_cells is NOT used to guess allocated physical pages.
            if (!strata::kernels::qsa_prompt_attn_bf16_validate(m.qsa_q16, core::qsa_attn_pools(q),
                    m.sel_ids, m.steps_dev, m.cap, shapes, m.qsa_attn16, m.T,
                    {q.n_pages, q.n_slots}, m.cs)) {
                err = "prefill QSA16: typed attention capability/geometry unsupported"; return false;
            }
        }
    }
''' + anchor)

cm = 'cmake/sycl_backend.cmake'
replace(cm, 'src/prefill/sycl/native16.cpp src/prefill/sycl/postops.cpp',
    'src/prefill/sycl/native16.cpp src/prefill/sycl/qsa_native16.cpp src/prefill/sycl/postops.cpp')
anchor = '  add_executable(sycl_prompt_attention_test tests/sycl/decode_attention.cpp)'
replace(cm, anchor, '''  # Manual only: Main owns serialized qualification after current GEMM cohorts.
  add_executable(sycl_native16_qsa_test EXCLUDE_FROM_ALL tests/sycl/qsa_native16.cpp)
  target_link_libraries(sycl_native16_qsa_test PRIVATE strata_prefill)
  target_compile_options(sycl_native16_qsa_test PRIVATE -fno-fast-math -ffp-contract=off)
  add_executable(sycl_prompt_native16_test EXCLUDE_FROM_ALL tests/sycl/prompt_native16.cpp)
  target_link_libraries(sycl_prompt_native16_test PRIVATE strata_kernels)
  target_compile_options(sycl_prompt_native16_test PRIVATE -fno-fast-math -ffp-contract=off)
''' + anchor)

# Preserve every R787 attention oracle/case/tolerance; harden only lifetime.
fixture = (OLD / 'files/tests/sycl/prompt_native16.cpp').read_text()
if digest(fixture.encode()) != 'd3791231919fe137cd4e3349d0d97ea1c3df44842d3bd2b6eb78cd9f45f396ab':
    raise RuntimeError('R787 fixture moved')
fixture = fixture.replace('#include "strata/sycl/runtime.hpp"',
    '#include "strata/sycl/runtime.hpp"\n#include "strata/sycl/checked_cleanup.hpp"')
fixture = fixture.replace('template <class T> struct Buffer {', '''void checked_drain() noexcept {
  sycl_backend::checked_cleanup([] { runtime->wait(); }, [] {},
      "R787 BF16 fixture drain failed");
}
struct Drain { ~Drain() noexcept { checked_drain(); } };
template <class T> struct Buffer {''')
fixture = fixture.replace('    runtime->wait(runtime->compute().memcpy(data(), values.data(), allocation.size()));',
    '''    try { runtime->wait(runtime->compute().memcpy(data(), values.data(), allocation.size())); }
    catch (...) { checked_drain(); throw; }''')
fixture = fixture.replace('    runtime->wait(runtime->compute().memcpy(values.data(), data(), allocation.size()));',
    '''    try { runtime->wait(runtime->compute().memcpy(values.data(), data(), allocation.size())); }
    catch (...) { checked_drain(); throw; }''')
before = '''  const uint16_t guard = bf16(1234.f);
  const std::vector<uint16_t> sentinel(query.size()+32,guard);
  out.put(sentinel); single.put(sentinel);
  struct Drain {
    ~Drain() { try { runtime->wait(); } catch (...) { std::abort(); } }
  } drain; // Before allocation owners, including exception paths.
'''
if fixture.count(before) != 1: raise RuntimeError('R787 fixture lifetime anchor changed')
fixture = fixture.replace(before, '  out.put(sentinel); single.put(sentinel);\n')
anchor = '  Buffer<uint16_t> dks(ks.size()), dvs(vs.size()), q(query.size()),'
fixture = fixture.replace(anchor, '''  const uint16_t guard = bf16(1234.f);
  const std::vector<uint16_t> sentinel(query.size()+32,guard);
''' + anchor)
fixture = fixture.replace('  dks.put(ks); dvs.put(vs); dk8.put(k8); dv8.put(v8);',
    '  Drain drain; // After ALL host/device owners, before the first upload.\n  dks.put(ks); dvs.put(vs); dk8.put(k8); dv8.put(v8);')
fixture_path = HERE / 'files/tests/sycl/prompt_native16.cpp'
fixture_path.write_text(fixture)
for path in sorted((HERE / 'files').rglob('*')):
    if path.is_file(): new[str(path.relative_to(HERE / 'files'))] = path.read_text()

patch = []
records = []
for name in sorted(new):
    before, after = base.get(name, ''), new[name]
    patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/' + name if name in base else '/dev/null', tofile='b/' + name))
    records.append({'path': name, 'base_sha256': digest(before.encode()) if name in base else None,
        'proposed_sha256': digest(after.encode()), 'proposed_bytes': len(after.encode()),
        'license': 'MIT', 'state': 'PROPOSAL_ONLY_UNRUN'})
(HERE / 'proposal.patch').write_text(''.join(patch))
(HERE / 'source-manifest.json').write_text(json.dumps({
    'base_commit_hint': '9c1a2020; supplied by Main, source snapshot authoritative',
    'base_identity': 'Fully hashed live source snapshot; no Git commands executed',
    'files': records, 'R787_core_unchanged_sha256': digest(old_core.encode()),
    'original_prompt_source_is_byte_identical_prefix': True,
    'patch_sha256': digest((HERE / 'proposal.patch').read_bytes()),
    'execution': dict.fromkeys(['build', 'compile', 'link', 'tests', 'GPU', 'numerical',
        'physical262144', 'RESTORE', 'performance', 'model_quality'], 'UNRUN')
}, indent=2) + '\n')
print(json.dumps({'patch_sha256': digest((HERE / 'proposal.patch').read_bytes()),
    'patch_bytes': (HERE / 'proposal.patch').stat().st_size,
    'changed_paths': [x['path'] for x in records]}))
