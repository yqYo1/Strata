"""Own MIT source-only QSA direct-K row-invariant discriminator.

Uses only this fork's independently authored selected prefix. No production
dispatch, arithmetic, matrix layout, shared-memory extent or barrier changes.
"""
import hashlib

PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
PREFIX_SHA256 = '4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'

ORIGINAL = '''          for (int i = tid; i < Cells * (Dim / 4); i += Workgroup) {
            const int cell = i % Cells, d = (i / Cells) * 4;
            const int64_t row = rows[cell];
            sycl::vec<uint16_t, 4> words{uint16_t(0)};
            if (row >= 0) {
              // Typed vec::load needs only alignof(uint16_t), not an
              // eight-byte object or a type-punned uint64_t pointer.
              words.load(0, sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(packed.k + row * Dim + d));
            }
            // Retain parallel rows and contiguous cell stores within a
            // subgroup. Each of the 8192 local destinations has one writer.
            for (int j = 0; j < 4; ++j)
              kv[(d + j) * Cells + cell] = sycl::bit_cast<BFloat>(uint16_t(words[j]));
          }'''

REPLACEMENT = '''          static_assert(Cells == 32 && Dim == 256 && Workgroup == 128);
          // Workgroup is a multiple of Cells: one lane keeps the same row
          // throughout this K tile. No lane writes rows before the next tile.
          const uint32_t cell = uint32_t(tid) % uint32_t(Cells);
          const uint32_t first_dim = (uint32_t(tid) / uint32_t(Cells)) * 4;
          constexpr uint32_t dim_step = (Workgroup / Cells) * 4;
          const int64_t row = rows[cell];
          if (row >= 0) {
            // Form the pointer only for a valid row. Preserve the typed,
            // align2 vector load and all four original BF16 stores.
            const auto *row_k = packed.k + row * Dim;
            for (uint32_t d = first_dim; d < uint32_t(Dim); d += dim_step) {
              sycl::vec<uint16_t, 4> words;
              words.load(0, sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(row_k + d));
              for (int j = 0; j < 4; ++j)
                kv[(d + j) * Cells + cell] = sycl::bit_cast<BFloat>(uint16_t(words[j]));
            }
          } else {
            // No source pointer or read for an invalid row. Every local
            // destination is still written before the unchanged WG barrier.
            for (uint32_t d = first_dim; d < uint32_t(Dim); d += dim_step)
              for (int j = 0; j < 4; ++j)
                kv[(d + j) * Cells + cell] = sycl::bit_cast<BFloat>(uint16_t(0));
          }'''


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('nonunique direct-K overlay')
    return text.replace(before, after, 1)


def generated(project):
    raw = (project / PREFIX).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREFIX_SHA256:
        raise ValueError('require exact selected 6e40 prefix')
    source = raw.decode()
    body = source[:source.index('bool qsa_bf16_prefix_batch(')]
    wrapper = source[source.index('bool ' + EXPORT + '('):
                     source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    result = {}
    for name, changed in [('direct_k_control', False), ('direct_k_invariant', True)]:
        namespace = 'namespace strata::kernels::' + name + ' {'
        value = once(body, 'namespace strata::kernels {', namespace)
        if changed:
            value = once(value, ORIGINAL, REPLACEMENT)
        recovered = once(value, REPLACEMENT, ORIGINAL) if changed else value
        recovered = once(recovered, namespace, 'namespace strata::kernels {')
        if recovered != body:
            raise ValueError('undeclared source delta')
        result[name + '.cpp'] = (value + wrapper + '}\n').encode()
    return result


def mapping_proof():
    """Independent old/new address and single-writer comparison, not GPU proof."""
    masks = [0, 0xffffffff, 0xaaaaaaaa, 0x55555555]
    masks += [1 << i for i in range(32)]
    masks += [0xffffffff ^ (1 << i) for i in range(32)]
    row_values = [0, 1, 524256]
    for mask in masks:
        for base_row in row_values:
            rows = [-1 if mask >> c & 1 else base_row + c for c in range(32)]
            before, after, reads_before, reads_after = {}, {}, [], []
            for tid in range(128):
                for i in range(tid, 2048, 128):
                    cell, d = i % 32, (i // 32) * 4
                    row = rows[cell]
                    for j in range(4):
                        at = (d + j) * 32 + cell
                        if at in before:
                            raise ValueError('old duplicate writer')
                        source = (row, d + j) if row >= 0 else None
                        before[at] = (tid, source)
                        if source is not None:
                            reads_before.append(source)
                cell, row = tid % 32, rows[tid % 32]
                for d in range((tid // 32) * 4, 256, 16):
                    if not 0 <= d <= 252:
                        raise ValueError('vector crosses row end')
                    for j in range(4):
                        at = (d + j) * 32 + cell
                        if at in after:
                            raise ValueError('new duplicate writer')
                        source = (row, d + j) if row >= 0 else None
                        after[at] = (tid, source)
                        if source is not None:
                            reads_after.append(source)
            if before != after or len(before) != 8192 or reads_before != reads_after:
                raise ValueError('direct-K source/destination/owner/order differs')
    return dict(cases=len(masks) * len(row_values), masks=len(masks),
                destinations_per_case=8192, same_writer_and_source_per_destination=True,
                unchanged_per_lane_read_order=True, invalid_rows_have_no_source_reads=True,
                source_row_reads_per_lane_old=16, source_row_reads_per_lane_new=1,
                limit='Finite Python mapping proof;not C++/SYCL legality,numerics,native lowering or speed.')
