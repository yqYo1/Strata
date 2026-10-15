#!/usr/bin/env python3
"""Offline arithmetic on closed CPU-route counts and a static profile.

No model/kernel/test/benchmark is run. Historical profile content was not pinned
by the CPU-census producer, so all reconstructed-residency bounds are conditional.
"""
import base64
import hashlib
import json
from pathlib import Path
import re
import struct

A = Path(__file__).resolve().parent
P = A.parents[2]
INPUT_SHA = '17ddf01d0ec4d90c6e0567fc1d0e2eb3959fccb68da893f4e404de5d89b43cb1'
PROFILE_SHA = '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, cap):
    with path.open('rb') as stream:
        data = stream.read(cap + 1)
    require(len(data) <= cap, 'bounded offline input exceeded')
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    data = read(A / 'cpu-gu-reuse-long-v1.json', 1 << 20)
    profile = read(P / 'data/expert-profile.bin', 256 << 10)
    require(sha(data) == INPUT_SHA and sha(profile) == PROFILE_SHA, 'offline input identity changed')
    source = json.loads(data)
    require(source['passed'] and source['complete'] and not source['active'] and not source['fresh_faults'],
            'closed original census required')
    argv = source['argv']
    require(argv[argv.index('--adapt-swaps') + 1] == '0' and argv[argv.index('--pcie-frac') + 1] == '0' and
            argv[argv.index('--expert-cache') + 1] == '128' and '--expert-cache-per-layer' in argv,
            'historical static cache configuration changed')
    require(profile[:4] == b'STRP' and struct.unpack_from('<5I', profile, 4) == (1, 48, 512, 24576, 24576) and
            len(profile) == 196632, 'current shipped profile geometry')
    pairs = [struct.unpack_from('<HH', profile, 24 + 4 * i) for i in range(24576)]
    require(len(set(pairs)) == 24576 and all(l < 48 and e < 512 for l, e in pairs), 'ranked pair coverage')
    request = source['requests'][0]
    observation = request['recurrence']
    packed = base64.b64decode(observation['expert_counts_base64'], validate=True)
    require(len(packed) <= 1 << 20 and sha(packed) == observation['encoded_counts_sha256'],
            'bounded exact encoded census')
    values, value, shift = [], 0, 0
    for byte in packed:
        require(shift < 64, 'counter encoding overflow')
        value |= (byte & 127) << shift
        if byte & 128:
            shift += 7
        else:
            values.append(value)
            value, shift = 0, 0
    require(not shift and len(values) == 98304, 'complete 48*512*4 census')
    counts = {(l, e): sum((t + 1) * values[(l * 512 + e) * 4 + t] for t in range(4))
              for l in range(48) for e in range(512)}
    sizes, cpu_per_layer, owners = {}, [0] * 48, {}
    for line in request['ledger']['lines']:
        fields = dict(re.findall(r'(\w+)=(\d+)', line))
        if line.startswith('format l='):
            require(int(fields['l']) not in sizes, 'duplicate format layer')
            sizes[int(fields['l'])] = int(fields['bytes'])
        if line.startswith('dispatch l='):
            cpu_per_layer[int(fields['l'])] += int(fields['cpu_rows'])
        if line.startswith('owner o='):
            require(int(fields['o']) not in owners, 'duplicate owner')
            owners[int(fields['o'])] = int(fields['entries'])
    coverage = request['ledger']['coverage']
    routed_rows, total = int(coverage['R']), int(coverage['route_entries'])
    require(len(sizes) == 48 and total == routed_rows * 48 * 10 and
            sum(counts.values()) == int(coverage['cpu_rows']) == observation['header']['tokens'],
            'census/ledger conservation')
    require(sum(values) == int(coverage['cpu_jobs']) == observation['header']['jobs'] and
            sum(n > 0 for n in counts.values()) == observation['header']['first_touches_unique'],
            'job and unique-ID conservation')
    require(all(sum(counts[l, e] for e in range(512)) == cpu_per_layer[l] for l in range(48)),
            'per-layer census/dispatch conservation')
    gpu_per_layer = [routed_rows * 10 - n for n in cpu_per_layer]
    require(owners == {0: sum(counts.values()), 1: sum(gpu_per_layer), 2: 0, 3: 0} and
            all(n >= 0 for n in gpu_per_layer) and sum(gpu_per_layer) == total - sum(counts.values()),
            'complement route coverage')

    def quota(slots, balanced):
        q, r = divmod(slots, 48)
        return [q + (int(l < r) if balanced else (r if l == 47 else 0)) for l in range(48)]

    def partition(order, slots, balanced=False):
        limits, used, selected = quota(slots, balanced), [0] * 48, []
        for l, e in order:
            if used[l] < limits[l]:
                selected.append((l, e))
                used[l] += 1
        require(len(selected) == slots, 'incomplete partition')
        return set(selected)

    baseline = partition(pairs, 128)
    require(sum(counts[pair] for pair in baseline) == 0, 'reconstructed residents overlap observed CPU work')
    results = []

    def record(name, selected, **settings):
        known = sum(counts[pair] for pair in selected)
        retained_lo = retained_hi = 0
        for l in range(48):
            old = {e for layer, e in baseline if layer == l}
            retained = old & {e for layer, e in selected if layer == l}
            if retained == old:
                retained_lo += gpu_per_layer[l]
            if retained:
                retained_hi += gpu_per_layer[l]
        low, high = known + retained_lo, known + retained_hi
        results.append(dict(name=name, **settings, actual_slots=len(selected),
            arena_bytes=sum((sizes[l] + 255) // 256 * 256 for l, e in selected),
            retained_baseline_slots=len(selected & baseline), known_old_CPU_rows=known,
            conditional_hit_rows_lower=low, conditional_hit_rows_upper=high,
            conditional_hit_fraction_lower=low / total, conditional_hit_fraction_upper=high / total,
            conditional_net_hit_rows_lower=low - sum(gpu_per_layer),
            conditional_net_hit_rows_upper=high - sum(gpu_per_layer),
            selected_pairs_sha256=sha(json.dumps(sorted(selected)).encode())))

    for slots in (128, 192, 256):
        for balanced in (False, True):
            record('shipped-profile-partition', partition(pairs, slots, balanced),
                   requested_slots=slots, balanced=balanced, implementation='proposal' if balanced else 'current')
    for requested in (105, 128, 192, 256):
        cap, used, selected = requested * max(sizes.values()), 0, set()
        for pair in pairs:
            amount = (sizes[pair[0]] + 255) // 256 * 256
            if used + amount > cap:
                break
            used += amount
            selected.add(pair)
        record('shipped-profile-global-budget', selected, requested_slots=requested,
               uniform_budget_bytes=cap, implementation='current; live free-VRAM cap not evaluated')
    cpu_rank = sorted((pair for pair in pairs if counts[pair]), key=lambda pair: (-counts[pair], *pair))
    record('CPU-census-hindsight-global-128', set(cpu_rank[:128]), implementation='offline selected set only')
    record('CPU-census-hindsight-legacy-128', partition(cpu_rank, 128),
           implementation='offline ranking under current quota; no profile file written')
    result = dict(schema='offline-profile-cache-route-bounds-v1', analysis_complete=True,
        GPU_executed=False, new_inference_executed=False, performance_qualified=False, quality_qualified=False,
        adopted=False, recipe_sha256=sha(read(Path(__file__), 64 << 10)),
        original_census_sha256=sha(data), current_profile_sha256=sha(profile),
        original_GPU_inference_status=dict(passed=source['passed'], complete=source['complete'],
                                          fresh_faults=source['fresh_faults']),
        baseline_CPU_rows=sum(counts.values()), baseline_GPU_rows=sum(gpu_per_layer), route_entries=total,
        baseline_GPU_rows_per_layer=gpu_per_layer, conditional_reconstructed_resident_cpu_overlap=0,
        historical_profile_content_identity_proven=False,
        source_condition='Historical argv identifies this profile path but its producer did not pin the profile hash. '
                         'Bounds are conditional on the current profile being that startup profile. '
                         'Zero CPU overlap is a consistency check, not historical content provenance.',
        scope='Fixed historical route coverage only. Missing original GPU ID counts are bounded per layer. '
              'No new routing, live allocation, physical traffic, numerical quality, or wall-speed prediction. '
              'Hindsight ranking is trained on this same request and requires held-out inference before adoption.',
        rows=results)
    with (A / 'profile-cache-coverage-bounds-v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for row in results:
        print(row['name'], row.get('requested_slots'), row.get('balanced'), row['actual_slots'],
              row['arena_bytes'], row['known_old_CPU_rows'],
              row['conditional_hit_rows_lower'], row['conditional_hit_rows_upper'])


if __name__ == '__main__':
    main()
