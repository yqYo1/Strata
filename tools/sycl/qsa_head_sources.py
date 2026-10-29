"""Own bounded-ID selected-QSA comparison;repository LICENSE.

The int-compiled private image preserves the size_t fallback before enqueue.
Math, array offsets and production/decode callers are unchanged.
"""
import hashlib
from pathlib import Path
import subprocess

P=Path(__file__).resolve().parents[2]
BASE='6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX='src/kernels/sycl/qsa_bf16_prefix.cpp'
BASE_SHA='4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
EDITS=[]

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
    anchor='  const size_t active_rows = checked_count(checked_count(active_pages, 2), shape.page_size);\n'
    values='query, pools, ids, steps, cap, shape, output, queries, bounds, live_cells, scratch, stream'
    fallback=anchor+'''  // Checked counts precede the range promise; never multiply the range first.
  // This flag is private to this TU. Broad public calls retain the size_t image.
  if (!id_range_eligible(active_rows, queries)) {
    if constexpr (CaptureState)
      return strata::kernels::qsa_head_original::capture('''+values+''', probe, submitted, packed_event);
    else
      return strata::kernels::qsa_head_original::batch('''+values+''', submitted, packed_event);
  }
'''
    candidate=change(candidate,anchor,fallback)
    if change(candidate,fallback,anchor)!=base:raise RuntimeError('undeclared bounded-ID source edit')
    outputs = {}
    for name,text in [('original',base),('candidate',candidate)]:
        namespace = 'qsa_head_' + name
        text = change(text, 'namespace strata::kernels {', 'namespace strata::kernels::' + namespace + ' {')
        if name=='candidate':text='#include "qsa-head.hpp"\n'+text
        for function,capture in [('batch',False),('capture',True)]:
            probe = ', QsaPrefixStateProbe probe' if capture else ''
            text += '\nbool ' + function + '(' + SIGNATURE + probe + ', sycl::event *submitted, sycl::event *packed_event) {\n'
            text += '  return prefix_batch_impl<33, false, false, true, true, true, true, ' + ('true' if capture else 'false') + '>(\n'
            text += '      query, pools, ids, steps, cap, shape, output, queries, bounds, live_cells, scratch, stream, '
            text += ('probe' if capture else '{}') + ', submitted, packed_event);\n}\n'
        text += '} // namespace strata::kernels::' + namespace + '\n'
        outputs[name + '.cpp'] = text.encode()
    header = '// Authored for Strata; repository LICENSE applies.\n#pragma once\n#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <sycl/sycl.hpp>\n#include <limits>\n'
    for name in ['original','candidate']:
        header += 'namespace strata::kernels::qsa_head_' + name + ' {\n'
        if name=='candidate':
            header+='''static_assert(std::numeric_limits<int>::max()==2147483647);
inline bool id_range_eligible(size_t active_rows, int64_t queries) noexcept {
  return active_rows<=size_t(std::numeric_limits<int>::max()/64) && queries>=0 &&
      uint64_t(queries)<=uint64_t(std::numeric_limits<int>::max()/(2*128));
}
'''
        for function,capture in [('batch',False),('capture',True)]:
            header += 'bool ' + function + '(' + SIGNATURE + (', QsaPrefixStateProbe probe' if capture else '') + ', sycl::event *submitted = nullptr, sycl::event *packed_event = nullptr);\n'
        header += '}\n'
    outputs['qsa-head.hpp'] = header.encode()
    return outputs


def proof():
    rows=generated()
    return dict(original_source_commit=BASE,original_source_sha256=BASE_SHA,
        generated_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in rows.items()},
        geometry='Same M8,N16,K16,WG128,SG16,16padded/12real heads;candidate TU IDrange=int only.',
        reversible_geometry_edits=0,reversible_math_edits=0,reversible_event_edits=3,
        declared_local_bytes=20036,production_or_decode_changed=False,
        fallback='After checked active_rows,before range multiplication/any enqueue,require active_rows<=INT_MAX/64 and queries<=INT_MAX/256;otherwise original size_t batch/capture. Signed rows,addresses and tensor offsets unchanged.',
        preservation='All device math/storage/control-flow/barriers unchanged. Exact-WG attribute rejected at9ee50d68d;identity-rescale rejected at032a3e15f.')


if __name__=='__main__':
    import json
    print(json.dumps(proof(),indent=2))
