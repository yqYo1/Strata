"""Exact-WG128 metadata prototype; authored under repository LICENSE.

Both arms retain selected6e40 M8 geometry and12real-head FP32 arithmetic.
Only the attention lambda receives its actual fixed work-group size.
"""
import hashlib
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parents[2]
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
BASE_SHA = '4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
EDITS = [
    ('        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {',
     '        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] [[sycl::reqd_work_group_size(128)]] {', 1),
]
SIGNATURE = """const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream"""


def change(text, before, after, count=1):
    if text.count(before) != count:
        raise RuntimeError('unexpected source count: ' + before)
    return text.replace(before, after)


def generated():
    raw = subprocess.check_output(['git', '-C', str(P), 'show', BASE + ':' + PREFIX])
    if hashlib.sha256(raw).hexdigest() != BASE_SHA or (P/PREFIX).read_bytes() != raw:
        raise RuntimeError('selected original source differs')
    body, separator, _ = raw.decode().partition('bool qsa_bf16_prefix_batch(')
    if not separator:
        raise RuntimeError('missing selected wrapper boundary')
    common = [
        ('QsaPrefixStateProbe probe = {}) {',
         'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr, sycl::event *packed_event = nullptr) {', 1),
        ('  sycl_backend::finish(stream, prepared);',
         '  if (packed_event) *packed_event = prepared;\n  sycl_backend::finish(stream, prepared);', 1),
        ('  sycl_backend::finish(stream, event);',
         '  if (submitted) *submitted = event;\n  sycl_backend::finish(stream, event);', 1),
        ('probe.head >= 16)', 'probe.head >= Heads)', 1),
    ]
    base = body
    for a,b,count in common:
        base = change(base,a,b,count)
    candidate = base
    for a,b,count in EDITS:
        candidate = change(candidate,a,b,count)
    recovered = candidate
    for a,b,count in reversed(EDITS):
        recovered = change(recovered,b,a,count)
    if recovered != base:
        raise RuntimeError('undeclared work-group annotation edit')
    for a,b,count in reversed(common):
        recovered = change(recovered,b,a,count)
    if recovered != body:
        raise RuntimeError('undeclared common diagnostic edit')
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
        generated_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in rows.items()},
        geometry='Same M8,N16,K16,actualWG128,SG16,16padded/12real matrix rows;bothsize_t IDs. Candidate alone declares exactWG128.',
        reversible_annotation_edits=1, reversible_real_math_edits=0,
        reversible_common_diagnostic_edits=4, declared_local_bytes=20036,
        arithmetic_contract='Every original padded/real-head softmax,QK/PV,ascendingFP32sum,Qscaling,BF16PV,FP32m/alpha/norm,outputdivision/cast and all barriers are textually unchanged.',
        workgroup_contract='Actual selected attention launch is alwaysnd_range1D(local128);sameeight SIMD16subgroups and twoheads each. Attribute may alter downstream native compilation even when optimizedLLVMbodies match. Compiled candidate requiredX128/Y1/Z1 is mandatory beforeenqueue;normal/state actualtraced launches must match.',
        production_or_decode_changed=False)


if __name__ == '__main__':
    import json
    print(json.dumps(proof(),indent=2))
