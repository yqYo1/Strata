"""Specialize selected-row lookup for the long-prefill owner's immutable identity page table."""

import hashlib


def candidate(raw):
    if hashlib.sha256(raw).hexdigest() != 'eeebba1dd68db3c4c3ffa88d5156475408ac27fac37f594d9277d5df6c18306d':
        raise ValueError('expected the measured ordered-exp kernel')
    source = raw.decode()
    body = source[:source.index('bool qsa_bf16_prefix_batch(')]
    namespace = 'strata::kernels::prefill_stage_identity'
    body = body.replace('namespace strata::kernels {', f'namespace {namespace} {{', 1)
    lookup = '''            const int page = logical >= 0 && logical < bounds.logical_pages
                ? pools.page_table[logical] : -1;
            if (cell < live_cells && page >= 0 && page < bounds.physical_pages)'''
    # Prefix packing and the meaning of every caller-owned scratch byte stay identical.
    # Equal page counts plus the checked live extent bound every nonnegative live cell.
    if body.count(lookup) != 1:
        raise ValueError('expected exactly one selected-cell page lookup')
    body = body.replace(lookup, '            if (cell >= 0 && cell < live_cells)')
    start = source.index('bool qsa_bf16_prefix_ordered_exp_batch(')
    end = source.index('bool qsa_bf16_prefix_state_probe_batch(', start)
    entry = source[start:end].replace('bool qsa_bf16_prefix_ordered_exp_batch(',
                                      'static bool prefill_owned_identity_batch(', 1)
    selector = '''
// The owner supplies its independently constructed identity table, or null.
// An ordinary resident pool may have just as many pages and still be remapped.
decltype(&strata::kernels::qsa_bf16_prefix_ordered_exp_batch) select(
    const QsaAttnPools &pools, const QsaPromptBf16Bounds &bounds,
    const int32_t *owned_identity_table) {
  if (owned_identity_table && pools.page_table == owned_identity_table &&
      bounds.logical_pages == bounds.physical_pages)
    return prefill_owned_identity_batch;
  return strata::kernels::qsa_bf16_prefix_ordered_exp_batch;
}
'''
    return (body + entry + selector + f'}}  // namespace {namespace}\n').encode()
