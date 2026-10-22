#!/usr/bin/env python3
"""Offline admission model for the existing SYCL expert-cache layouts.

This does not allocate GPU memory or predict a hit rate. Native admission,
loan/refill correctness and phase-specific model comparisons remain required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

LAYERS, EXPERTS, PREFIX = 48, 512, 128
MAX_BYTES = (1 << 63) - 256


def require(value, message):
    if not value:
        raise ValueError(message)


def aligned(size):
    require(type(size) is int and 0 < size <= MAX_BYTES, 'invalid slot size')
    return (size + 255) // 256 * 256


def read_profile(path):
    raw = Path(path).read_bytes()
    count = LAYERS * EXPERTS
    require(len(raw) == 24 + count * 8 and raw[:4] == b'STRP', 'incomplete profile')
    require(struct.unpack_from('<5I', raw, 4) == (1, LAYERS, EXPERTS, count, count),
            'unsupported profile header')
    pairs = list(struct.iter_unpack('<HH', raw[24:24 + count * 4]))
    require(len(set(pairs)) == count and
            all(l < LAYERS and e < EXPERTS for l, e in pairs), 'invalid profile pairs')
    reverse = struct.unpack_from('<' + 'i' * count, raw, 24 + count * 4)
    require(all(reverse[l * EXPERTS + e] == i for i, (l, e) in enumerate(pairs)),
            'profile reverse map differs')
    return pairs


def read_manifest(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        require(len(fields) in (8, 9), 'unsupported native manifest row')
        values = [int(v) for v in fields[:8]]
        layer, gu, down, offset, size, gate, up, down_offset = values
        require(layer == len(rows) and gu > 0 and down > 0 and
                min(offset, gate, up, down_offset) >= 0, 'invalid manifest geometry')
        aligned(size)
        rows.append(dict(layer=layer, gu=gu, down=down, blob_bytes=size))
    require(len(rows) == LAYERS, 'complete48-layer manifest required')
    return rows


def per_layer(pairs, slots):
    """Return slot order, not profile order: the last layer owns the remainder."""
    require(type(slots) is int and PREFIX < slots <= LAYERS * EXPERTS, 'invalid cache slots')
    quotas = [slots // LAYERS] * (LAYERS - 1) + [slots // LAYERS + slots % LAYERS]
    require(max(quotas) <= EXPERTS, 'per-layer quota exceeds expert count')
    selected = [[] for _ in range(LAYERS)]
    for layer, expert in pairs:
        if len(selected[layer]) < quotas[layer]:
            selected[layer].append((layer, expert))
    require([len(row) for row in selected] == quotas, 'per-layer fill incomplete')
    return [pair for row in selected for pair in row]


def ranked(pairs, sizes, budget):
    """Native prefix admission stops at the first non-fitting pair, without skipping."""
    used, selected = 0, []
    for layer, expert in pairs:
        size = aligned(sizes[layer])
        if used + size > budget:
            break
        selected.append((layer, expert))
        used += size
    return selected


def membership(selected):
    masks = [0] * LAYERS
    for layer, expert in selected:
        require(not masks[layer] & (1 << expert), 'duplicate admitted expert')
        masks[layer] |= 1 << expert
    return masks


def reorder_cached_prefix(pairs, sizes, slots):
    """Stable size order inside the admitted set; never change layer-local rank."""
    require(len(sizes) == LAYERS and type(slots) is int and PREFIX < slots <= len(pairs),
            'invalid protected-prefix geometry')
    return sorted(pairs[:slots], key=lambda pair: aligned(sizes[pair[0]])) + pairs[slots:]


def encode_profile(pairs):
    count = LAYERS * EXPERTS
    require(len(pairs) == len(set(pairs)) == count and
            all(0 <= l < LAYERS and 0 <= e < EXPERTS for l, e in pairs), 'invalid complete profile')
    reverse = [0] * count
    for rank, (layer, expert) in enumerate(pairs):
        reverse[layer * EXPERTS + expert] = rank
    return (b'STRP' + struct.pack('<5I', 1, LAYERS, EXPERTS, count, count) +
            b''.join(struct.pack('<HH', *pair) for pair in pairs) + struct.pack('<'+'i'*count, *reverse))


def summary(selected, sizes):
    require(len(selected) > PREFIX, 'no positive loan after128-slot prefix')
    masks = membership(selected)
    extents = [aligned(sizes[layer]) for layer, _ in selected]
    total, prefix = sum(extents), sum(extents[:PREFIX])
    require(total <= (1 << 63)-1, 'cache extent exceeds native signed byte count')
    return dict(slots=len(selected), layer_slots=[m.bit_count() for m in masks],
        payload_bytes=sum(sizes[layer] for layer, _ in selected), allocated_bytes=total,
        prefix128_bytes=prefix, maximum_loan_bytes=total-prefix,
        membership_sha256=hashlib.sha256(b''.join(m.to_bytes(64, 'little') for m in masks)).hexdigest(),
        slot_map_sha256=hashlib.sha256(b''.join(struct.pack('<HH', *p) for p in selected)).hexdigest())


def plan(profile, manifest, slots=1256, protect_prefix=False):
    pairs, rows = read_profile(profile), read_manifest(manifest)
    sizes = [row['blob_bytes'] for row in rows]
    base_pairs = per_layer(pairs, slots)
    base = summary(base_pairs, sizes)
    # --expert-cache is denominated in maximum *unaligned* blob bytes in
    # generate.cpp. Reusing1256 in global mode would exceed the actual baseline.
    knob = base['allocated_bytes'] // max(sizes)
    require(0 < knob < (1 << 31), 'global cache argument exceeds native range')
    cap = knob * max(sizes)
    candidate_pairs = ranked(pairs, sizes, cap)
    original_candidate = candidate_pairs
    if protect_prefix:
        pairs = reorder_cached_prefix(pairs, sizes, len(candidate_pairs))
        candidate_pairs = ranked(pairs, sizes, cap)
        require(membership(candidate_pairs) == membership(original_candidate) and
                len(candidate_pairs) == len(original_candidate) and per_layer(pairs, slots) == base_pairs,
                'prefix ordering changed cache membership or per-layer slot map')
    candidate = summary(candidate_pairs, sizes)
    require(candidate['allocated_bytes'] <= cap <= base['allocated_bytes'], 'budget exceeds baseline')
    next_size = (aligned(sizes[pairs[len(candidate_pairs)][0]])
                 if len(candidate_pairs) < len(pairs) else None)
    require(next_size is None or cap-candidate['allocated_bytes'] < next_size, 'prefix stop differs')
    a, b = set(base_pairs), set(candidate_pairs)
    result = dict(schema='strata-cache-placement-offline-v1',
        inputs={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (profile, manifest)},
        baseline=dict(base, expert_cache_argument=slots, per_layer=True),
        candidate=dict(candidate, expert_cache_argument=knob, per_layer=False, nominal_cap_bytes=cap),
        common_experts=len(a & b), added_experts=len(b-a), removed_experts=len(a-b),
        unused_vs_baseline_bytes=base['allocated_bytes']-candidate['allocated_bytes'],
        required_free_before_cache_bytes=cap+(700 << 20),
        limits='Offline upper admission with700MiB reserve; native free-memory shrink is unmeasured. '
               'Ranks are not frequencies. No predicted hit rate, physical traffic, speed or adoption. '
               'Different prefix128 bytes can change prefill loan capacity. '
               'Actual membership, payload refill, prefill and decode must be checked separately.')
    if protect_prefix:
        result['profile_generation'] = dict(algorithm='stable ascending aligned blob bytes inside original admitted prefix',
            cached_prefix_slots=len(original_candidate), sha256=hashlib.sha256(encode_profile(pairs)).hexdigest(),
            original_prefix128_bytes=summary(original_candidate, sizes)['prefix128_bytes'],
            per_layer_slot_map_unchanged=True, global_membership_unchanged=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--slots', type=int, default=1256)
    parser.add_argument('--protect-prefix', action='store_true')
    args = parser.parse_args()
    print(json.dumps(plan(args.profile, args.manifest, args.slots, args.protect_prefix), indent=2))


if __name__ == '__main__':
    main()
