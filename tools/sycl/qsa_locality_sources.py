"""Main-authored counter-free logical-gather sensitivity diagnostic.

Own MIT6e40 source and archived cfabc1ee2 event-export scaffolding. No kernel
arithmetic, local layout, global address formula or production route is changed.
"""
import hashlib
from pathlib import Path
import struct
import subprocess

BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
PREFIX_SHA256 = '4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'
FIXTURE_SHA256 = 'a0410c7005f64a89d02becd42fcfd8155c92b68cd7b4d175543c172070a6338c'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('nonunique event-export overlay: ' + before[:80])
    return text.replace(before, after, 1)


def generated(project):
    raw = (project / PREFIX).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREFIX_SHA256:
        raise ValueError('exact selected6e40 prefix required')
    source = raw.decode()
    body = source[:source.index('bool qsa_bf16_prefix_batch(')]
    wrapper = source[source.index('bool ' + EXPORT + '('):
                     source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    edits = [
        ('namespace strata::kernels {', 'namespace strata::kernels::locality_control {'),
        ('QsaPrefixStateProbe probe = {}) {',
         'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr) {'),
        ('  sycl_backend::finish(stream, event);\n  return true;',
         '  if (submitted) *submitted = event;\n  sycl_backend::finish(stream, event);\n  return true;')]
    value = body
    for before, after in edits:
        value = once(value, before, after)
    restored = value
    for before, after in reversed(edits):
        restored = once(restored, after, before)
    if restored != body:
        raise ValueError('undeclared kernel edit')
    arm = once(wrapper, 'void *stream) {', 'void *stream, sycl::event *submitted) {')
    arm = once(arm, 'output, queries, bounds, live_cells, scratch, stream);',
               'output, queries, bounds, live_cells, scratch, stream, {}, submitted);')
    header = ('// Main-authored declarations; repository LICENSE.\n#pragma once\n'
              '#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <sycl/sycl.hpp>\n'
              'namespace strata::kernels::locality_control {\n' +
              arm[:arm.index(' {')] + ';\n}\n')
    return {'locality_control.cpp': (value + arm + '}\n').encode(),
            'qsa-locality.hpp': header.encode()}


def count_proof(project, fixture):
    """Recover deterministic caller counts, then validate initialized actual IDs.

    These counts are source-derived, not a device steps-buffer capture. Pin the
    original producer/caller/helper and retain that distinction in every result.
    """
    import qsa_selection_capture as capture
    paths = ('src/prefill/prefill.cpp', 'src/kernels/sycl/kv.cpp',
             'include/strata/kernels/qsa.hpp', PREFIX)
    source = {p: subprocess.check_output(['git', '-C', str(project), 'show', BASE + ':' + p])
              for p in paths}
    prefill, kv, header = (source[p].decode() for p in paths[:3])
    clauses = (
        'm.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);',
        'qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);',
        'cudaMemcpyAsync(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount * 4,',
        'const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};',
        'm.qsa_q16, pools, m.sel_ids, m.steps_dev, m.cap, s, m.qsa_attn16, T,')
    if not all(c in prefill for c in clauses):
        raise ValueError('caller/capture/attention source contract differs')
    if ('step[kStepWidth] = int32_t(qsa_selection_width(pos + 1, s));' not in kv or
            'const int64_t w = s.idx_top_k + s.idx_block - 1;' not in header or
            'return n_kv < w ? n_kv : w;' not in header):
        raise ValueError('step-width source contract differs')
    if hashlib.sha256(source[PREFIX]).hexdigest() != PREFIX_SHA256:
        raise ValueError('selected prefix differs')
    result = capture.inspect(fixture, [(0, 32768, 6491, 2051)], (0, 32768, 6491, 2051))
    if result['sha256'] != FIXTURE_SHA256:
        raise ValueError('actual selected fixture differs')
    counts = [min(32768 + q + 1, 2051) for q in range(6491)]
    raw = struct.pack('<6491i', *counts)
    return dict(native_source_commit=BASE,
        source_sha256={p: hashlib.sha256(v).hexdigest() for p, v in source.items()},
        count_origin='Source-derived caller qsa_step_fill(p0+q);not a captured device steps vector.',
        header=[0, 32768, 6491, 2051], queries=6491, minimum=2051, maximum=2051,
        counts_int32le_sha256=hashlib.sha256(raw).hexdigest(),
        selected_entries=result['selected']['valid_entries'],
        ignored_entries=result['selected']['unused_cap_entries'],
        fixture_sha256=result['sha256'],
        reasoning='Caller cap=qsa_selection_width(kTopkMaxCells,s)=2051<kTopkMaxCells '
          'fixes idx_top_k+idx_block-1=2051. Every position is32768..39258,so all '
          '6491widths=min(position+1,2051)=2051. The same steps_dev buffer reaches '
          'ordered-exp attention;the dump records p0,T,cap and those selected IDs. '
          'Every2051entry row is initialized,strictly increasing and causal.',
        GPU_executed=False)


def host_controls(project):
    output = generated(project)
    body = output['locality_control.cpp'].decode()
    if any(token in body for token in ('KPitch', 'VPitch', 'PPitch')):
        raise ValueError('local-layout candidate leaked into diagnostic')
    if body.count('if (submitted) *submitted = event;') != 1:
        raise ValueError('attention event export absent or duplicated')
    # Counterfactual IDs preserve count, distinctness, causality and four final
    # cells. They are not claims about the model selector's block-score ranking.
    for q in (0, 1, 3, 14, 15, 16, 511, 4095, 6490):
        pos = 32768 + q
        for bank in (0, q % 16):
            values = list(range(bank * 2047, (bank + 1) * 2047)) + list(range(pos - 3, pos + 1))
            if len(values) != 2051 or len(set(values)) != 2051 or values != sorted(values) or values[-1] != pos:
                raise ValueError('invalid finite locality bank')
    return dict(passed=True, reverse_overlay_exact=True, bank_boundary_cases=18,
                GPU_executed=False, compiler_executed=False)
