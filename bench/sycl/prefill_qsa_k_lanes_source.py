"""Generate a single K-load lane mapping trial from the profiled prefill kernel."""

import hashlib
from pathlib import Path
import sys

SOURCE_SHA256 = 'eeebba1dd68db3c4c3ffa88d5156475408ac27fac37f594d9277d5df6c18306d'


def candidate(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('expected the measured ordered-exp prefix source')
    source = raw.decode()
    prefix = source[:source.index('bool qsa_bf16_prefix_batch(')]
    prefix = prefix.replace('namespace strata::kernels {', 'namespace strata::kernels::prefill_k_lanes4 {', 1)
    old = 'const int cell = i % Cells, d = (i / Cells) * 4;'
    if prefix.count(old) != 1:
        raise ValueError('expected exactly one vector K load loop')
    prefix = prefix.replace(old, '''// Four adjacent SIMD lanes read 32 contiguous K bytes for one cell.
            const int cell = (i / 4) % Cells;
            const int d = ((i / (4 * Cells)) * 4 + i % 4) * 4;''')
    prefix = prefix.replace('// Retain parallel rows and contiguous cell stores within a\n'
                            '            // subgroup. Each of the 8192 local destinations has one writer.',
                            '// Preserve the transposed SLM layout used by the matrix instruction.')
    start = source.index('bool qsa_bf16_prefix_ordered_exp_batch(')
    end = source.index('bool qsa_bf16_prefix_state_probe_batch(', start)
    return (prefix + source[start:end] + '}  // namespace strata::kernels::prefill_k_lanes4\n').encode()


if __name__ == '__main__':
    src, out = map(Path, sys.argv[1:])
    out.write_bytes(candidate(src.read_bytes()))
