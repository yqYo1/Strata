"""Generate the isolated four-cell BF16 K layout from the measured SYCL source.

The repository LICENSE applies. The production kernel stays in the source
projection and supplies the control export. This variant is linked only into
the component comparison.
"""

import argparse
import hashlib
from pathlib import Path
import sys


KERNEL = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
SOURCE_SHA256 = 'eeebba1dd68db3c4c3ffa88d5156475408ac27fac37f594d9277d5df6c18306d'
EXPORT = 'bool qsa_bf16_prefix_ordered_exp_batch('
KEY_OFFSET = '''// Each page/head has four logical rows. Preserve a contiguous four-element
// vector per cell and put the neighbouring cells' vectors next to it.
constexpr size_t packed_key_offset(size_t row, size_t feature) {
  constexpr size_t CellsPerPage = 4, VectorWidth = 4;
  const size_t page_head = row / CellsPerPage;
  const size_t cell = row % CellsPerPage;
  const size_t vector = feature / VectorWidth;
  return (page_head * Dim + vector * VectorWidth) * CellsPerPage
         + cell * VectorWidth + feature % VectorWidth;
}

'''


def replace_once(source, original, replacement):
    if source.count(original) != 1:
        raise ValueError('Unexpected QSA source: ' + original)
    return source.replace(original, replacement, 1)


def generate(original):
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError('The component requires the measured V2 ordered-exp source')
    source = original.decode()
    body = source[:source.index('bool qsa_bf16_prefix_batch(')]
    export = source[source.index(EXPORT):source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    changes = (
        ('namespace strata::kernels {', 'namespace strata::kernels::prefill_key_page {'),
        ('bool matrix_available(', KEY_OFFSET + 'bool matrix_available('),
        ('  static_assert(Dim % 4 == 0);', '  static_assert(Dim % 4 == 0 && VectorK4);\n'
                                         '  if (shape.page_size != 4) return false;'),
        ('packed.k[row * Dim + d] =', 'packed.k[packed_key_offset(row, d)] ='),
        ('packed.k + row * Dim + d));', 'packed.k + packed_key_offset(size_t(row), size_t(d))));'),
    )
    candidate = body
    for before, after in changes:
        candidate = replace_once(candidate, before, after)
    recovered = candidate
    for before, after in reversed(changes):
        recovered = replace_once(recovered, after, before)
    if recovered != body:
        raise ValueError('The isolated producer and consumer changed another source byte')
    return (candidate + export + '} // namespace strata::kernels::prefill_key_page\n').encode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measured_source', type=Path, help=f'Path to the exact measured {KERNEL}')
    args = parser.parse_args()
    sys.stdout.buffer.write(generate(args.measured_source.read_bytes()))
