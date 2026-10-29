"""Own selected-QSA exact-workgroup CPU discriminator;repository LICENSE.

Only an exactWG128annotation changes. No production or decode caller changes.
"""
import hashlib
from pathlib import Path
import subprocess

P=Path(__file__).resolve().parents[2]
BASE='6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX='src/kernels/sycl/qsa_bf16_prefix.cpp'
BASE_SHA='4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
EDITS=[
    ('[[sycl::reqd_sub_group_size(16)]] {',
     '[[sycl::reqd_sub_group_size(16)]] [[sycl::reqd_work_group_size(128)]] {',1),
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
    rows=generated()
    return dict(original_source_commit=BASE,original_source_sha256=BASE_SHA,
        generated_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in rows.items()},
        geometry='Same M8,N16,K16,WG128,SG16,16padded/12real heads;only exactWG128metadata added.',
        reversible_geometry_edits=0,reversible_attribute_edits=len(EDITS),reversible_event_edits=3,
        declared_local_bytes=20036,production_or_decode_changed=False,
        preservation='All math/storage/control-flow/barriers unchanged. Rejected identity-rescale source recovers at032a3e15f;current candidate changes one exact-workgroup annotation only.')


if __name__=='__main__':
    import json
    print(json.dumps(proof(),indent=2))
