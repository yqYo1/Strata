"""Generate isolated V tile variants from the owned, measured prefix source."""

import hashlib
from pathlib import Path

PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
PREFIX_SHA = 'eeebba1dd68db3c4c3ffa88d5156475408ac27fac37f594d9277d5df6c18306d'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('unique original QSA source anchor required: ' + old[:72])
    return text.replace(old, new, 1)


def project(raw, early):
    if hashlib.sha256(raw).hexdigest() != PREFIX_SHA:
        raise ValueError('requires the exact profiled owned QSA prefix source')
    source = raw.decode()
    start = source[:source.index('bool qsa_bf16_prefix_batch(')]
    wrapper = source[source.index('bool ' + EXPORT + '('):
                     source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    namespace = 'separate_early_v' if early else 'separate_late_v'
    text = replace_once(start, 'namespace strata::kernels {', f'namespace strata::kernels::{namespace} {{')
    text = replace_once(text, 'static_assert(storage_contract<Cells * Dim>());',
        'static_assert(storage_contract<2 * Cells * Dim>());')
    text = replace_once(text, 'LocalPayload<ScoreStride, CacheQueries>)) return false;',
        'LocalPayload<ScoreStride, CacheQueries> + Cells * Dim * sizeof(BFloat))) return false;')
    text = replace_once(text, 'sycl::local_accessor<MatrixTile<Cells * Dim>, 1>',
        'sycl::local_accessor<MatrixTile<2 * Cells * Dim>, 1>')
    text = replace_once(text, '      if constexpr (CacheQueries) qtile = kv;',
        '''      // K and V are disjoint spans of the same aligned BF16 array.
      const auto value_tile = kv + Cells * Dim;
      if constexpr (CacheQueries) qtile = kv;''')

    # Change V writes and PV reads together. All lane/cell/dimension mappings
    # and the existing zero-fill and barrier contracts survive both variants.
    begin = text.index('        if constexpr (VectorV4) {')
    end = text.index('        item.barrier', begin)
    gather = text[begin:end].replace('kv[', 'value_tile[')
    text = text[:begin] + gather + text[end:]
    text = replace_once(text,
        'mx::joint_matrix_load(sg, b, kv + ((c / 16) * 16',
        'mx::joint_matrix_load(sg, b, value_tile + ((c / 16) * 16')
    text = replace_once(text,
        'mx::joint_matrix_load(sg, b, kv + c * Dim + dim_base',
        'mx::joint_matrix_load(sg, b, value_tile + c * Dim + dim_base')
    if early:
        # Reuse the post-K barrier for both operands. The existing
        # post-softmax barrier still publishes P, and the final tile barrier
        # prevents the next iteration from overwriting V while PV reads it.
        anchor = '        if (tid < Cells) local_live |= rows[tid] >= 0;'
        text = replace_once(text, gather, '')
        text = replace_once(text, anchor, gather + anchor)

    declaration = wrapper.split(' {', 1)[0]
    state = replace_once(declaration, 'bool ' + EXPORT + '(',
        'bool qsa_bf16_prefix_v_tile_state_batch(')
    state = replace_once(state, 'void *stream)', 'const QsaPrefixStateProbe &probe, void *stream)')
    body = '''
  return prefix_batch_impl<Cells + 1, false, false, true, true, true, true, true>(
      query, pools, ids, steps, cap, shape, output, queries, bounds, live_cells, scratch, stream, probe);
}
'''
    return (text + wrapper + state + ' {' + body + '}\n').encode()


def generated(path: Path):
    raw = path.read_bytes()
    return {'separate_late_v.cpp': project(raw, False),
            'separate_early_v.cpp': project(raw, True)}
