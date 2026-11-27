"""Rank bounded expert swaps by early routes and score their later CPU work."""

from collections import Counter
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
import dflash2_decode_residency_replay as replay
import adaptive_cpu_leaf_evidence as leaves

sys.path.insert(0, str(replay.ROOT / 'tools'))
from make_profile import read_profile


LEAF_RECORD = '6bbd3e3e1cab1415a4bba7df675c729bf4213d59'
LEAF_KEY = 'dflash2_decode_cpu_leaves_v1'
ROWS_ANCHOR = (96, 197_913_600, 147_795, 303_143_424_000)  # 68f8c61b6, 256-window metric arm
LEAF_LOG_SHA = (
    '75775f6408f3f8714d4a7edfdac3f5235b654f41f6b6b36278f30672ed02fc05',
    'c964f2d066c9e867751ccc780dfb500b6ba570cd564f6e45a52bdbc1a32359dc',
)
PROFILE = 'data/expert-profile.bin'
EARLY, SLOTS, PROTECTED, SWAPS, COPY_LIMIT = 256, 1256, 132, 96, 197_913_600
METRICS = ('unique_jobs', 'routed_rows', 'leaf_cost')
CHECKPOINTS = (64, 256, 1024)
# Lower rounded sustained host-USM copy reference from prefill_copy_engine_v6.
COPY_REFERENCE_BYTES_PER_SECOND = 6_291_000_000


def historical(revision, key):
    return json.loads(replay.git('show', revision + ':' + replay.EVIDENCE))[key]


def calibration(capture):
    record = historical(LEAF_RECORD, LEAF_KEY)
    replay.require(record['stage'] == 'closed' and record['ownership_closed'] and not record['active']
                   and record['decision']['leaf_observer_cost_qualified']
                   and record['decision']['matched_observed_format_work']
                   and all(record[k] == capture[k] for k in ('binary', 'native_source_commit', 'driver', 'boot')),
                   'leaf clocks must come from an observed request on the same retained build')
    cpu_total, jobs_total = Counter(), Counter()
    selected = [p for p in record['processes'] if p['arm'] == 'leaves']
    for process, checksum in zip(selected, LEAF_LOG_SHA, strict=True):
        path = Path(record['plans'][process['index']]['root']) / 'engine.stderr'
        replay.require(path.stat().st_size <= 128 << 10, 'leaf evidence exceeds its envelope')
        data = path.read_bytes()
        replay.require(replay.sha(data) == checksum and process['ownership_closed'] and process['passed']
                       and process['native_exit_code'] == 0 and not process['fresh_faults'], 'leaf evidence changed')
        final = leaves.extract(data.decode(), process['evidence']['samples'], 4)[-1]
        expected = process['routes']['leaves']
        fields = ('gu', 'down', 'nt', 'jobs')
        replay.require(final['cpu_ns'] == expected['concurrent_cpu_ns'] and final['thread_count'] == 6
                       and [{k: group[k] for k in fields} for group in final['format_nt']] == expected['format_nt_jobs'],
                       'raw leaf bins no longer match the native census')
        for group in final['format_nt']:
            key = tuple(group[k] for k in fields[:3])
            cpu_total[key] += sum(op['cpu_ns'] for op in group['operations'].values())
            jobs_total[key] += group['jobs']
    return {k: max(1, round(cpu_total[k] / jobs)) for k, jobs in jobs_total.items()}


def slot_order(base):
    layers = [[] for _ in base]
    for layer, expert in read_profile(replay.ROOT / PROFILE):
        layers[layer].append(expert)
    replay.require(all(sum(1 << e for e in ordered[:mask.bit_count()]) == mask
                       for ordered, mask in zip(layers, base, strict=True)), 'stock slot order differs from capture')
    return layers


def choose(metric, counts, layers, base, formats, leaf_ns):
    proposals, offset = [], 0
    for layer, (histogram, experts, mask, (gu, down, size)) in enumerate(zip(counts, layers, base, formats, strict=True)):
        width_scores = (1, 1, 1, 1) if metric == 'unique_jobs' else (1, 2, 3, 4)
        if metric == 'leaf_cost':
            width_scores = tuple(leaf_ns[gu, down, nt] for nt in range(1, 5))
        totals = Counter()
        for (expert, nt), jobs in histogram.items():
            totals[expert] += jobs * width_scores[nt - 1]
        cap = mask.bit_count()
        locked = min(cap, max(0, PROTECTED - offset))
        offset += cap
        incoming = sorted(experts[cap:], key=totals.__getitem__, reverse=True)
        victims = sorted(range(locked, cap), key=lambda slot: totals[experts[slot]])
        for slot, expert in zip(victims, incoming):
            gain = totals[expert] - totals[experts[slot]]
            if gain < 2 * width_scores[0]:
                break
            proposals.append((-gain, layer, slot, expert, size))
    replay.require(offset == SLOTS, 'stock slot capacity changed')
    changes = [{} for _ in layers]
    count, moved = 0, 0
    for _, layer, slot, expert, size in sorted(proposals):
        if moved + size > COPY_LIMIT:
            continue
        changes[layer][layers[layer][slot]] = -1
        changes[layer][expert] = 1
        count += 1
        moved += size
        if count == SWAPS:
            break
    return changes, count, moved


def route_prefix_signature(early, windows):
    """Canonical all-expert row census for the bounded decode observer."""
    replay.require(len(early) == replay.LAYERS and 0 < EARLY < len(windows),
                   'complete captured decode prefix required')
    rows_per_layer = 10 * sum(width for _, width in windows[:EARLY])
    pack = struct.Struct(f'<{replay.EXPERTS}I')
    counts = []
    for histogram in early:
        routed = Counter()
        for (expert, width), jobs in histogram.items():
            routed[expert] += width * jobs
        replay.require(sum(routed.values()) == rows_per_layer,
                       'each layer must count all verifier rows in the prefix')
        counts.append(pack.pack(*(routed[e] for e in range(replay.EXPERTS))))
    payload = b''.join(counts)
    return dict(windows=EARLY, layout='layer-major uint32 little-endian',
                layers=replay.LAYERS, experts=replay.EXPERTS,
                entries_per_layer=rows_per_layer, total_entries=rows_per_layer * replay.LAYERS,
                bytes=len(payload), sha256=replay.sha(payload))


def replay_cost():
    revision, raw, windows, base, formats, observed = replay.inputs()
    for name in ('bench/dflash2_decode_adapt_cost_replay.py', PROFILE, 'tools/make_profile.py',
                 'tools/sycl/adaptive_cpu_leaf_evidence.py', 'tools/sycl/cpu_service_evidence.py',
                 'tools/sycl/decode_footer.py'):
        replay.require((replay.ROOT / name).read_bytes() == replay.git('show', revision + ':' + name),
                       'commit the exact adaptive cost replay inputs')
    capture = historical(replay.CAPTURE, replay.KEY)
    replay.require(replay.sha((replay.ROOT / PROFILE).read_bytes()) == capture['routing_metadata']['profile_sha256'],
                   'active profile does not match original captured bytes')
    output = capture['processes'][0]['evidence']['samples'][-1]['generated_tokens']
    leaf_ns = calibration(capture)
    layer_costs = [{nt: leaf_ns[g, d, nt] for nt in range(1, 5)} for g, d, _ in formats]
    layers = slot_order(base)
    source = replay.window_counts(raw, [w for _, w in windows])
    early = [Counter() for _ in layers]
    for _ in range(EARLY):
        for counts, jobs in zip(early, next(source), strict=True):
            counts.update(jobs.items())
    prefix = route_prefix_signature(early, windows)
    variants = [choose(metric, early, layers, base, formats, leaf_ns) for metric in METRICS]
    cumulative = [dict(cpu_jobs_removed=0, logical_cpu_bytes_removed=0, modeled_leaf_cpu_ns_removed=0)
                  for _ in METRICS]
    crossing, never_below = [None] * len(METRICS), [None] * len(METRICS)
    stock = [sum(c for (e, _), c in hist.items() if not mask >> e & 1) for hist, mask in zip(early, base, strict=True)]
    full_stock_jobs, suffix_jobs, suffix_cpu_ns, points = sum(stock), 0, 0, []
    start_output = windows[EARLY][0] - windows[0][0]
    for i, frame in enumerate(source, EARLY + 1):
        for layer, jobs in enumerate(frame):
            cost, size, mask = layer_costs[layer], formats[layer][2], base[layer]
            suffix_jobs += sum(not (mask >> expert & 1) for expert in jobs)
            suffix_cpu_ns += sum(cost[nt] for expert, nt in jobs.items() if not mask >> expert & 1)
            for j, (changes, _, _) in enumerate(variants):
                hit = changes[layer]
                delta = sum(hit.get(expert, 0) for expert in jobs)
                sums = cumulative[j]
                sums['cpu_jobs_removed'] += delta
                sums['logical_cpu_bytes_removed'] += delta * size
                sums['modeled_leaf_cpu_ns_removed'] += sum(hit.get(e, 0) * cost[nt] for e, nt in jobs.items())
        emitted = (windows[i][0] - windows[0][0] if i < len(windows) else output) - start_output
        for j, (_, _, moved) in enumerate(variants):
            elapsed = cumulative[j]['modeled_leaf_cpu_ns_removed']
            if moved and elapsed * COPY_REFERENCE_BYTES_PER_SECOND >= 6 * 2 * moved * 10**9:
                if crossing[j] is None:
                    crossing[j] = emitted
                if never_below[j] is None:
                    never_below[j] = emitted
            else:
                never_below[j] = None
        if i - EARLY in CHECKPOINTS or i == len(windows):
            points.append(dict(suffix_windows=i - EARLY, suffix_outputs=emitted, stock_jobs=suffix_jobs,
                               projected={metric: dict(entry) for metric, entry in zip(METRICS, cumulative, strict=True)}))
    replay.require(full_stock_jobs + suffix_jobs == observed['cpu_jobs'], 'rolling stock jobs differ from native capture')
    _, swaps, copied = variants[METRICS.index('routed_rows')]
    rows = cumulative[METRICS.index('routed_rows')]
    replay.require((swaps, copied, rows['cpu_jobs_removed'], rows['logical_cpu_bytes_removed']) == ROWS_ANCHOR,
                   'the routed-row comparison failed to reproduce the original result')
    return dict(source_commit=revision, route_sha256=observed['routing_sha256'], capture_commit=replay.CAPTURE,
                calibration_commit=LEAF_RECORD, early_windows=EARLY, early_outputs=start_output, stock_suffix_leaf_cpu_ns=suffix_cpu_ns,
                route_prefix=prefix, observations=points,
                methods={name: dict(swaps=count, requested_each_direction_bytes=moved,
                    first_copy_only_crossover_outputs=crossing[j], final_crossing_outputs=never_below[j])
                    for j, (name, (_, count, moved)) in enumerate(zip(METRICS, variants, strict=True))},
                copy_reference_bytes_per_second=COPY_REFERENCE_BYTES_PER_SECOND,
                scope='Same prose route and caps; CPU leaf means are from code. Copy reference charges H2D refill and '
                      'H2D restore. The six-way leaf-time and copy-only crossover prices neither GPU work nor publications. '
                      'No physical bus traffic, actual latency, second held-out workload or adoption.')


if __name__ == '__main__':
    print(json.dumps(replay_cost(), indent=2))
