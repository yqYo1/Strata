"""Build the isolated PV fragment-reuse arm from the measured Strata prefix."""

import hashlib
from pathlib import Path
import sys


SOURCE_SHA256 = 'eeebba1dd68db3c4c3ffa88d5156475408ac27fac37f594d9277d5df6c18306d'
NAMESPACE = 'strata::kernels::prefill_probability_reuse'


def candidate(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('expected the profiled Strata QSA prefix source')
    source = raw.decode('utf-8')
    prefix = source[:source.index('bool qsa_bf16_prefix_batch(')]
    prefix = prefix.replace('namespace strata::kernels {', f'namespace {NAMESPACE} {{', 1)
    start = prefix.index('        for (int j = 0; j < 4; ++j) {\n          Left a; Right b;')
    end = prefix.index('        item.barrier', start)
    # Only the two ordered-exp specializations below instantiate this loop.
    # The existing barriers publish P/V and retire both operands around it.
    pv = '''        static_assert(OrderedExp && !TilePacked && Cells == 32);
        Left fragments[Cells / 16];
        for (int c = 0; c < Cells; c += 16)
          mx::joint_matrix_load(sg, fragments[c / 16], probability + head * Cells + c, Cells);
        for (int j = 0; j < 4; ++j) {
          Right b;
          mi::joint_matrix_apply(sg, accumulated[j], [=](float &v, size_t r, size_t) {
            v *= alpha[head + r];
          });
          for (int c = 0; c < Cells; c += 16) {
            mx::joint_matrix_load(sg, b, kv + c * Dim + dim_base + j * 16, Dim);
            mx::joint_matrix_mad(sg, accumulated[j], fragments[c / 16], b, accumulated[j]);
          }
        }
'''
    start_export = source.index('bool qsa_bf16_prefix_ordered_exp_batch(')
    end_export = source.index('// FP32 subgroup max/sum', start_export)
    exports = source[start_export:end_export]
    serial = exports.index('  return prefix_batch_impl<Cells + 1, false, false, true, true, true, false, true>')
    end_serial = exports.index(';', serial) + 1
    exports = exports[:serial] + '  return false;  // The comparison only exercises ordered-exp.' + exports[end_serial:]
    return (prefix[:start] + pv + prefix[end:] + exports.rstrip() + '\n}\n').encode('utf-8')


if __name__ == '__main__':
    source_path, output_path = map(Path, sys.argv[1:])
    output_path.write_bytes(candidate(source_path.read_bytes()))
