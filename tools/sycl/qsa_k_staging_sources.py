"""Own MIT QSA component: isolate four-cell/sixteen-dimension K staging.

Derived only from this fork's independently authored 6e40 prefix and its own
retired private-component scaffolding (39044cc68). No production dispatch edit.
The host census proves a finite integer permutation, not device execution,
compiler lowering, BF16 numerical correctness or a speed improvement.
"""
import hashlib

BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
PREFIX_SHA256 = '4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('nonunique K-staging overlay: '+before[:100])
    return text.replace(before, after, 1)


ORIGINAL_LOAD = '''          for (int i = tid; i < Cells * (Dim / 4); i += Workgroup) {
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

SUBGROUP_LOAD = '''          // Each SG16 loads four cells by sixteen dimensions. All lanes
          // execute every exchange, including lanes with invalid source rows.
          for (int tile = subgroup; tile < (Cells / 4) * (Dim / 16);
               tile += Workgroup / 16) {
            const int cell_base = (tile % (Cells / 4)) * 4;
            const int dim_base = (tile / (Cells / 4)) * 16;
            const int64_t row = rows[cell_base + lane / 4];
            sycl::vec<uint16_t, 4> words{uint16_t(0)};
            if (row >= 0)
              words.load(0, sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(
                      packed.k + row * Dim + dim_base + 4 * (lane % 4)));
            // Swap lane bit2/register bit0,then lane bit3/register bit1.
            // Each sender packs only the two values its peer needs. Both
            // exchanges are uniform;all casts/shifts are unsigned and exact.
            const bool odd = (lane & 4) != 0;
            const uint32_t pair0 =
                uint32_t(odd ? uint16_t(words[0]) : uint16_t(words[1])) |
                (uint32_t(odd ? uint16_t(words[2]) : uint16_t(words[3])) << 16);
            const uint32_t peer0 = sycl::permute_group_by_xor(sg, pair0, uint32_t(4));
            words[0] = odd ? uint16_t(peer0) : uint16_t(words[0]);
            words[1] = odd ? uint16_t(words[1]) : uint16_t(peer0);
            words[2] = odd ? uint16_t(peer0 >> 16) : uint16_t(words[2]);
            words[3] = odd ? uint16_t(words[3]) : uint16_t(peer0 >> 16);
            const bool upper = (lane & 8) != 0;
            const uint32_t pair1 =
                uint32_t(upper ? uint16_t(words[0]) : uint16_t(words[2])) |
                (uint32_t(upper ? uint16_t(words[1]) : uint16_t(words[3])) << 16);
            const uint32_t peer1 = sycl::permute_group_by_xor(sg, pair1, uint32_t(8));
            words[0] = upper ? uint16_t(peer1) : uint16_t(words[0]);
            words[1] = upper ? uint16_t(peer1 >> 16) : uint16_t(words[1]);
            words[2] = upper ? uint16_t(words[2]) : uint16_t(peer1);
            words[3] = upper ? uint16_t(words[3]) : uint16_t(peer1 >> 16);
            const int d = dim_base + 4 * (lane % 4) + lane / 4;
            // Keep BFloat object lifetime/type and the original matrix stride.
            // Only the staging schedule changes; no new WG barrier is added.
            for (int j = 0; j < 4; ++j)
              kv[d * Cells + cell_base + j] = sycl::bit_cast<BFloat>(uint16_t(words[j]));
          }'''


def generated(project):
    raw = (project/PREFIX).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREFIX_SHA256:
        raise ValueError('generate only from exact immutable6e40prefix')
    source = raw.decode()
    body = source[:source.index('bool qsa_bf16_prefix_batch(')]
    wrapper = source[source.index('bool '+EXPORT+'('):
                     source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    header = ('// Authored for Strata; repository LICENSE (MIT) applies.\n'
              '#pragma once\n#include "strata/kernels/qsa_bf16_prefix.hpp"\n'
              '#include <sycl/sycl.hpp>\n')
    outputs = {}
    for name, transpose in (('kstaging_control', False), ('kstaging_candidate', True)):
        namespace = 'namespace strata::kernels::'+name+' {'
        edits = [
            ('namespace strata::kernels {', namespace),
            ('QsaPrefixStateProbe probe = {}) {',
             'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr) {'),
            ('static_assert(Dim % 4 == 0);',
             'static_assert(Dim == 256 && Cells == 32 && Workgroup == 128);\n'
             '  static_assert(OrderedExp && CacheQueries && VectorK4 && VectorV4 && !TilePacked && !CaptureState);'),
            ('  sycl_backend::finish(stream, event);\n  return true;',
             '  if (submitted) *submitted = event;\n'
             '  sycl_backend::finish(stream, event);\n  return true;')]
        if transpose:
            edits.append((ORIGINAL_LOAD, SUBGROUP_LOAD))
        value = body
        for before, after in edits:
            value = once(value, before, after)
        restored = value
        for before, after in reversed(edits):
            restored = once(restored, after, before)
        if restored != body:
            raise ValueError('undeclared native source edit')
        wrapper_edits = [
            ('void *stream) {', 'void *stream, sycl::event *submitted) {'),
            ('output, queries, bounds, live_cells, scratch, stream);',
             'output, queries, bounds, live_cells, scratch, stream, {}, submitted);')]
        arm = wrapper
        for before, after in wrapper_edits:
            arm = once(arm, before, after)
        restored = arm
        for before, after in reversed(wrapper_edits):
            restored = once(restored, after, before)
        if restored != wrapper:
            raise ValueError('undeclared wrapper edit')
        outputs[name+'.cpp'] = (value+arm+'}\n').encode()
        header += namespace+'\n'+arm[:arm.index(' {')]+';\n}\n'
    outputs['qsa-k-staging.hpp'] = header.encode()
    return outputs


def staged(mask=0, defect=None, seed=None):
    """Independent dense transpose versus two packed exchanges per SG tile.

    Unique symbolic tokens prove the full permutation. Numeric seeds check
    unsigned half packing/extraction,including zero,all-one and sign-bit words.
    """
    def value(cell, dimension):
        if mask >> cell & 1:
            return None if seed is None else 0
        if seed is None:
            return (cell,dimension)
        if seed in ('zero','ones','sign'):
            return {'zero':0,'ones':0xffff,'sign':0x8000}[seed]
        return ((cell*8192+dimension*40503)^seed)&0xffff

    def pack(pair):
        if seed is None:
            return tuple(pair)
        if not all(0 <= word <= 0xffff for word in pair):
            raise ValueError('packing outside uint16 domain')
        return pair[0] | (pair[1] << 16)

    def unpack(pair):
        words=pair if seed is None else (pair & 0xffff,(pair >> 16) & 0xffff)
        return words[::-1] if defect == 'reversed_halves' else words

    output = {}; loaded = set(); exchanges = 0
    for subgroup in range(8):
        for tile in range(subgroup, 128, 8):
            cell_base, dim_base = tile % 8 * 4, tile // 8 * 16
            words = []
            for lane in range(16):
                cell = cell_base + lane // 4
                dimensions = [dim_base+4*(lane % 4)+j for j in range(4)]
                for d in dimensions:
                    if not 0 <= cell < 32 or not 0 <= d < 256 or (cell,d) in loaded:
                        raise ValueError('duplicate or out-of-bounds source')
                    loaded.add((cell,d))
                words.append([value(cell,d) for d in dimensions])
            for bit in range(2):
                if defect == 'skip_second_stage' and bit == 1:
                    continue
                indices=[((0,2) if lane & 4 else (1,3)) if bit==0 else
                         ((0,1) if lane & 8 else (2,3)) for lane in range(16)]
                packed=[pack([words[lane][j] for j in indices[lane]]) for lane in range(16)]
                next_words = [row[:] for row in words]
                for lane in range(16):
                    peer_lane = lane ^ (8 if defect=='wrong_mask' and bit==0 else 1 << (bit+2))
                    if not 0 <= peer_lane < 16:
                        raise ValueError('out-of-range subgroup peer')
                    for j,word in zip(indices[lane],unpack(packed[peer_lane])):
                        next_words[lane][j]=word
                    exchanges += 1
                words = next_words
            for lane in range(16):
                d = dim_base + 4*(lane % 4) + lane // 4
                for j in range(4):
                    cell = cell_base+j; address = d*32+cell
                    if not 0 <= address < 8192 or address in output:
                        raise ValueError('duplicate or out-of-bounds destination')
                    output[address] = words[lane][j]
    expected = {d*32+cell: value(cell,d)
                for d in range(256) for cell in range(32)}
    if len(loaded) != 8192 or output != expected:
        raise ValueError('staging differs from independent dense transpose')
    return exchanges


def host_controls(project):
    generated_sources = generated(project)
    masks = [0, (1 << 32)-1, 0x55555555, 0xaaaaaaaa]+[1 << i for i in range(32)]
    counts = [staged(mask) for mask in masks]
    seeds=('zero','ones','sign',0x5a17,0xc001,0x13579bdf)
    numeric=[staged(mask,seed=seed) for mask in masks for seed in seeds]
    if set(counts+numeric) != {4096}:
        raise ValueError('nonuniform symbolic collective count')
    rejected = 0
    for defect in ('skip_second_stage','wrong_mask','reversed_halves'):
        try:
            staged(defect=defect)
        except ValueError:
            rejected += 1
        else:
            raise ValueError('broken transpose escaped independent oracle')
    return dict(passed=True, mask_cases=len(masks), numeric_cases=len(numeric),values_per_case=8192,
                unique_source_and_destination=True, missing_rows_zero=True,
                symbolic_lane_collective_calls_per_WG=4096,
                subgroup_collective_instances_per_WG=256,
                count_scope='One WG per32selected-cell iteration over all256Kdimensions;source-derived,not executed instruction/traffic counts.',
                rejected_broken_transposes=rejected,
                generated_sha256={k:hashlib.sha256(v).hexdigest()
                                  for k,v in generated_sources.items()},
                GPU_executed=False, compiler_executed=False,
                limits='Finite symbolic staging proof only;not compiler/device correctness, numerical, hardware traffic or speed qualification.')


if __name__ == '__main__':
    import json
    from pathlib import Path
    print(json.dumps(host_controls(Path(__file__).resolve().parents[2])))
