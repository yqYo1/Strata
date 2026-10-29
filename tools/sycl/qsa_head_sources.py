"""Independent M4/WG96 prefill-only component from our selected M8 kernel.

Repository LICENSE applies. These are exact, reversible geometry edits to our
own source, not a copied external kernel. No engine or decode caller is changed.
"""
import hashlib
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parents[2]
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
BASE_SHA = '4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
# The exact count rejects a changed parent instead of applying a fuzzy patch.
EDITS = [
    ('Cells = 32, Workgroup = 128;', 'Cells = 32, Workgroup = 96;', 1),
    ('16 * Dim', '12 * Dim', 8),
    ('16 * Cells', '12 * Cells', 3),
    ('16 * ScoreStride', '12 * ScoreStride', 2),
    ('3 * 16', '3 * 12', 1),
    ('16 * 4', '12 * 4', 1),
    ('== 28160', '== 25280', 1),
    ('== 28224', '== 25328', 1),
    ('== 20036', '== 19188', 1),
    ('fits(c.msize, c.max_msize, 8)', 'fits(c.msize, c.max_msize, 4)', 1),
    ('probe.head >= 16', 'probe.head >= Heads', 1),
    ('maximum(16, handler)', 'maximum(12, handler)', 1),
    ('normalizer(16, handler), alpha(16, handler)', 'normalizer(12, handler), alpha(12, handler)', 1),
    ('bad_head(16, handler)', 'bad_head(12, handler)', 1),
    ('head < 16; head += 8', 'head < Heads; head += 6', 1),
    ('if (tid < 16)', 'if (tid < Heads)', 2),
    ('accumulator, 8, 16', 'accumulator, 4, 16', 1),
    ('use::a, 8, 16', 'use::a, 4, 16', 1),
    ('Acc accumulated[4]', 'Acc accumulated[8]', 1),
    ('for (int j = 0; j < 4; ++j) mx::joint_matrix_fill',
     'for (int j = 0; j < 8; ++j) mx::joint_matrix_fill', 1),
    ('if (subgroup < 4)', 'if (subgroup < 6)', 2),
    ('(subgroup / 2) * 8', '(subgroup / 2) * 4', 2),
    ('h < 16; h += 8', 'h < Heads; h += 6', 1),
    ('(subgroup / 4) * 8, dim_base = (subgroup % 4) * 64',
     '(subgroup / 2) * 4, dim_base = (subgroup % 2) * 128', 2),
    ('        for (int j = 0; j < 4; ++j) {\n          Left a; Right b;',
     '        for (int j = 0; j < 8; ++j) {\n          Left a; Right b;', 1),
    ('      for (int j = 0; j < 4; ++j)\n        mi::joint_matrix_apply(sg, accumulated[j]',
     '      for (int j = 0; j < 8; ++j)\n        mi::joint_matrix_apply(sg, accumulated[j]', 1),
]
SIGNATURE = '''const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream'''


def change(text, before, after, count=1):
    if text.count(before) != count:
        raise RuntimeError('unexpected geometry/source count: ' + before)
    return text.replace(before, after)


def generated():
    raw = subprocess.check_output(['git', '-C', str(P), 'show', BASE + ':' + PREFIX])
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise RuntimeError('selected original source differs')
    body, separator, _ = raw.decode().partition('bool qsa_bf16_prefix_batch(')
    if not separator: raise RuntimeError('missing selected wrapper boundary')
    # Capture actual submitted events, never a final marker as a service proxy.
    base = body
    event_edits = [
        ('QsaPrefixStateProbe probe = {}) {',
         'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr, sycl::event *packed_event = nullptr) {', 1),
        ('  sycl_backend::finish(stream, prepared);',
         '  if (packed_event) *packed_event = prepared;\n  sycl_backend::finish(stream, prepared);', 1),
        ('  sycl_backend::finish(stream, event);',
         '  if (submitted) *submitted = event;\n  sycl_backend::finish(stream, event);', 1),
    ]
    for a,b,n in event_edits: base = change(base,a,b,n)
    candidate = base
    for a,b,n in EDITS: candidate = change(candidate,a,b,n)
    recovered = candidate
    for a,b,n in reversed(EDITS): recovered = change(recovered,b,a,n)
    if recovered != base: raise RuntimeError('undeclared geometry edit')
    for a,b,n in reversed(event_edits): recovered = change(recovered,b,a,n)
    if recovered != body: raise RuntimeError('undeclared diagnostic edit')
    outputs = {}
    for name,text in [('original',base),('candidate',candidate)]:
        namespace = 'qsa_head_' + name
        text = change(text, 'namespace strata::kernels {', 'namespace strata::kernels::' + namespace + ' {')
        for function,capture in [('batch',False),('capture',True)]:
            probe = ', QsaPrefixStateProbe probe' if capture else ''
            text += '\nbool ' + function + '(' + SIGNATURE + probe + ', sycl::event *submitted, sycl::event *packed_event) {\n'
            text += '  return prefix_batch_impl<33, false, false, true, true, true, true, ' + ('true' if capture else 'false') + '>(\n'
            text += '      query, pools, ids, steps, cap, shape, output, queries, bounds, live_cells, scratch, stream, '
            text += ('probe' if capture else '{}') + ', submitted, packed_event);\n}\n'
        text += '} // namespace strata::kernels::' + namespace + '\n'
        outputs[name + '.cpp'] = text.encode()
    header = '// Authored for Strata; repository LICENSE applies.\n#pragma once\n#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <sycl/sycl.hpp>\n'
    for name in ['original','candidate']:
        header += 'namespace strata::kernels::qsa_head_' + name + ' {\n'
        for function,capture in [('batch',False),('capture',True)]:
            header += 'bool ' + function + '(' + SIGNATURE + (', QsaPrefixStateProbe probe' if capture else '') + ', sycl::event *submitted = nullptr, sycl::event *packed_event = nullptr);\n'
        header += '}\n'
    outputs['qsa-head.hpp'] = header.encode()
    return outputs


def proof():
    rows = generated()
    return dict(original_source_commit=BASE, original_source_sha256=BASE_SHA,
        generated_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in rows.items()},
        geometry='M4,N16,K16,WG96,SG16;QK3headblocks*2cellblocks;PV3headblocks*2dim128blocks*8fragments;12real heads only.',
        softmax='h=subgroup;h<12;h+=6 gives exactly one writer per head,independent of QK head-block ownership.',
        declared_local_bytes=19188, reversible_geometry_edits=len(EDITS), reversible_event_edits=3,
        preservation='Same Q scale/BF16 rounding,d ascending16,K/V4loop bounds,FP32 cell sum c0..31,PV c0,16,output conversion;all whole-group barriers unchanged and uniform. No padded score/probability/state rows are consumed. M4matrix arithmetic/registers/speed still require native observation.',
        production_or_decode_changed=False)


if __name__ == '__main__':
    import json
    print(json.dumps(proof(),indent=2))
