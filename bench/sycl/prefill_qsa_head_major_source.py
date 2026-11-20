"""Isolate KV-head-major workgroup assignment in the measured ordered-exp QSA kernel."""

import hashlib


def candidate(raw):
    if hashlib.sha256(raw).hexdigest() != 'eeebba1dd68db3c4c3ffa88d5156475408ac27fac37f594d9277d5df6c18306d':
        raise ValueError('expected the measured ordered-exp kernel')
    source = raw.decode()
    implementation = source[:source.index('bool qsa_bf16_prefix_batch(')]
    old = '''      const int64_t batch = int64_t(item.get_group_linear_id()) / 2;
      const int kv_head = int(item.get_group_linear_id() % 2);'''
    new = '''      const int64_t group = int64_t(item.get_group_linear_id());
      const int kv_head = int(group >= queries);
      const int64_t batch = kv_head ? group - queries : group;'''
    if implementation.count(old) != 1:
        raise ValueError('ordered-exp attention group assignment changed')
    implementation = implementation.replace(old, new)
    namespace = 'strata::kernels::prefill_head_major'
    implementation = implementation.replace('namespace strata::kernels {', f'namespace {namespace} {{', 1)
    start = source.index('bool qsa_bf16_prefix_ordered_exp_batch(')
    end = source.index('bool qsa_bf16_prefix_state_probe_batch(', start)
    return (implementation + source[start:end] + f'}}  // namespace {namespace}\n').encode()
