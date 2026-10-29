"""Own only-normalizer parallel sum and matched lossless probes;repository LICENSE.

Both arms retain size_t IDs, original geometry/storage and FP32 online state.
No production/decode caller changes. Trace mode is separate from speed mode.
"""
import hashlib
from pathlib import Path
import subprocess

P=Path(__file__).resolve().parents[2]
BASE='6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX='src/kernels/sycl/qsa_bf16_prefix.cpp'
BASE_SHA='4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
SUM_FOLD='''              // Every lane executes every exchange with a uniform source ID.
              // Lane0 alone adds c=0..31 in the original ascending FP32 order.
              // Keeping p0/p1 in FP32 precedes the BF16 probability store.
              for (int c = 0; c < Cells; ++c) {
                const float p = sycl::select_from_group(sg, c < 16 ? p0 : p1, c % 16);
                if (!lane) sum += p;
              }'''
EDITS=[(SUM_FOLD,'''              // All16lanes contribute one materialized FP32pair exactly once.
              // The group tree is unspecified;no balanced-tree assumption.
              sum = sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());''',1)]

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
    if hashlib.sha256(raw).hexdigest() != BASE_SHA or (P/PREFIX).read_bytes()!=raw:
        raise RuntimeError('selected original source differs')
    body, separator, _ = raw.decode().partition('bool qsa_bf16_prefix_batch(')
    if not separator: raise RuntimeError('missing selected wrapper boundary')
    # Capture actual submitted events, never a final marker as a service proxy.
    base = body
    event_edits = [
        ('QsaPrefixStateProbe probe = {}) {',
         'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr, sycl::event *packed_event = nullptr, uint32_t *details = nullptr) {', 1),
        ('  sycl_backend::finish(stream, prepared);',
         '  if (packed_event) *packed_event = prepared;\n  sycl_backend::finish(stream, prepared);', 1),
        ('  sycl_backend::finish(stream, event);',
         '  if (submitted) *submitted = event;\n  sycl_backend::finish(stream, event);', 1),
    ]
    trace_edits=[
        ('    validate_device_span(probe.states, state_bytes, queue);',
         '''    validate_device_span(probe.states, state_bytes, queue);
    if (details) {
      const size_t tiles = (uint64_t(cap) + Cells - 1) / Cells;
      const size_t words = checked_count(tiles, 64);
      if (words > std::numeric_limits<size_t>::max() - 512)
        throw std::invalid_argument("QSA normalizer trace size");
      const size_t bytes = checked_count(words + 512, 4);
      validate_spans({{details, bytes}},
          {{probe.states, state_bytes}, pk, pv, bad, {output, query_bytes}, {query, query_bytes},
           {ids, checked_count(checked_count(queries, cap), 4)},
           {steps, checked_count(checked_count(queries, kStepCount), 4)},
           {pools.page_table, checked_count(bounds.logical_pages, 4)},
           {pools.k_q, checked_count(rows_count, Dim)}, {pools.v_q, checked_count(rows_count, Dim)},
           {pools.k_scale, checked_count(checked_count(rows_count, 4), 2)},
           {pools.v_scale, checked_count(checked_count(rows_count, 4), 2)}}, 1);
      validate_spans({}, {{details, bytes}}, 4);
      validate_device_span(details, bytes, queue);
    }''',1),
        ('            probability[h * Cells + c1] = BFloat(p1);',
         '''            probability[h * Cells + c1] = BFloat(p1);
            if constexpr (CaptureState) {
              if (details && batch == probe.query && kv_head == probe.kv_head && h == probe.head) {
                const size_t at = size_t(first / Cells) * 64;
                details[at + c0] = sycl::bit_cast<uint32_t>(p0);
                details[at + c1] = sycl::bit_cast<uint32_t>(p1);
                details[at + 32 + c0] = uint32_t(sycl::bit_cast<uint16_t>(probability[h * Cells + c0]));
                details[at + 32 + c1] = uint32_t(sycl::bit_cast<uint16_t>(probability[h * Cells + c1]));
              }
            }''',1),
        ('          if (h >= Heads) return;',
         '''          if constexpr (CaptureState) {
            if (details && batch == probe.query && kv_head == probe.kv_head && h == probe.head) {
              const size_t at = size_t((uint64_t(cap) + Cells - 1) / Cells) * 64;
              details[at + dim_base + j * 16 + c] = sycl::bit_cast<uint32_t>(value);
            }
          }
          if (h >= Heads) return;''',1),
        ('''          else if (normalizer[h] > 0.f)
            result = sycl::bit_cast<uint16_t>(BFloat(value / normalizer[h]));''',
         '''          else if (normalizer[h] > 0.f) {
            const float normalized = value / normalizer[h];
            if constexpr (CaptureState) {
              if (details && batch == probe.query && kv_head == probe.kv_head && h == probe.head) {
                const size_t at = size_t((uint64_t(cap) + Cells - 1) / Cells) * 64 + 256;
                details[at + dim_base + j * 16 + c] = sycl::bit_cast<uint32_t>(normalized);
              }
            }
            result = sycl::bit_cast<uint16_t>(BFloat(normalized));
          }''',1),
    ]
    for a,b,n in [*event_edits,*trace_edits]: base = change(base,a,b,n)
    candidate = base
    for a,b,n in EDITS: candidate = change(candidate,a,b,n)
    recovered = candidate
    for a,b,n in reversed(EDITS): recovered = change(recovered,b,a,n)
    if recovered != base: raise RuntimeError('undeclared normalizer edit')
    for a,b,n in reversed([*event_edits,*trace_edits]): recovered = change(recovered,b,a,n)
    if recovered != body: raise RuntimeError('undeclared diagnostic edit')
    outputs = {}
    for name,text in [('original',base),('candidate',candidate)]:
        namespace = 'qsa_head_' + name
        text = change(text, 'namespace strata::kernels {', 'namespace strata::kernels::' + namespace + ' {')
        text='#include <limits>\n'+text
        for function,capture in [('batch',False),('capture',True)]:
            probe = ', QsaPrefixStateProbe probe' if capture else ''
            detail = ', uint32_t *details' if capture else ''
            text += '\nbool ' + function + '(' + SIGNATURE + probe + ', sycl::event *submitted, sycl::event *packed_event' + detail + ') {\n'
            text += '  return prefix_batch_impl<33, false, false, true, true, true, true, ' + ('true' if capture else 'false') + '>(\n'
            text += '      query, pools, ids, steps, cap, shape, output, queries, bounds, live_cells, scratch, stream, '
            text += ('probe' if capture else '{}') + ', submitted, packed_event, ' + ('details' if capture else 'nullptr') + ');\n}\n'
        text += '} // namespace strata::kernels::' + namespace + '\n'
        outputs[name + '.cpp'] = text.encode()
    header = '// Authored for Strata; repository LICENSE applies.\n#pragma once\n#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <sycl/sycl.hpp>\n#include <limits>\n'
    for name in ['original','candidate']:
        header += 'namespace strata::kernels::qsa_head_' + name + ' {\n'
        for function,capture in [('batch',False),('capture',True)]:
            header += 'bool ' + function + '(' + SIGNATURE + (', QsaPrefixStateProbe probe' if capture else '') + ', sycl::event *submitted = nullptr, sycl::event *packed_event = nullptr' + (', uint32_t *details = nullptr' if capture else '') + ');\n'
        header += '}\n'
    outputs['qsa-head.hpp'] = header.encode()
    return outputs


def proof():
    rows=generated()
    return dict(original_source_commit=BASE,original_source_sha256=BASE_SHA,
        generated_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in rows.items()},
        geometry='Same M8,N16,K16,WG128,SG16,16padded/12real heads;bothsize_t IDs.',
        reversible_geometry_edits=0,reversible_math_edits=1,reversible_common_diagnostic_edits=7,
        declared_local_bytes=20036,production_or_decode_changed=False,
        trace_layout='Capture only:ceil(cap/32)*64uint32words(per tile32FP32p bits+32BF16PVbits),then256FP32A bits+256pre-BF16normalized bits. Allwrites validated disjoint;normal speed exports have no capture stores.',
        preservation='Only normalizer fold changes;same common p/alpha/max/PV/order/geometry/barriers. Fullinstrumented/uninstrumented correspondence,groupmapping,denormal/mathmode and interval/state/task/performance qualification pending. ClosedID-int source recovers at1a2583fb0.')


if __name__=='__main__':
    import json
    print(json.dumps(proof(),indent=2))
