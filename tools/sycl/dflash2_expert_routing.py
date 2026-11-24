"""Read existing DFlash2 routing evidence and score fixed decode placement."""

from collections import Counter
from pathlib import Path
import struct

import measurement_guards as guards


ROUTE_ROW = struct.Struct('<ii10i40s')
ZERO_WEIGHTS = bytes(40)


def verifier_routing_usage(path, widths, census, formats, membership, *, max_bytes):
    """Recover the last completed GEN's decoded expert visits from a routing trace.

    The caller binds these decode-only widths and file size to the last request
    and reads the file after owned process closure. Earlier prefill and warm-up
    rows precede the suffix. Only the batch callback's zero-weight rows qualify.
    """
    path = Path(path)
    guards.require(len(formats) == len(membership) == 48
                   and len(widths) == census['windows'] > 0 and all(1 <= w <= 4 for w in widths)
                   and sum(widths) == census['rows'], 'invalid verifier routing geometry')
    suffix_bytes = census['rows'] * 48 * ROUTE_ROW.size
    layer_histograms = [dict() for _ in formats]
    with path.open('rb') as capture:
        size = capture.seek(0, 2)
        guards.require(suffix_bytes <= size <= max_bytes and size % ROUTE_ROW.size == 0,
                       'routing trace is partial or outside the declared capture budget')
        capture.seek(size - suffix_bytes)
        for width in widths:
            for layer, histogram in enumerate(layer_histograms):
                window = Counter()
                for _ in range(width):
                    row = capture.read(ROUTE_ROW.size)
                    guards.require(len(row) == ROUTE_ROW.size, 'incomplete verifier routing row')
                    observed_layer, fanout, *ids, weights = ROUTE_ROW.unpack(row)
                    guards.require((observed_layer, fanout) == (layer, 10) and weights == ZERO_WEIGHTS
                                   and len(set(ids)) == 10 and all(0 <= e < 512 for e in ids),
                                   'wrong routing layer, callback or expert ID')
                    window.update(ids)
                for expert, rows in window.items():
                    counts = histogram.setdefault(expert, [0] * 4)
                    counts[rows - 1] += 1
        guards.require(capture.tell() == size and capture.read(1) == b'', 'unexpected routing suffix')

    usage = []
    reconstructed = {fmt: [0] * 8 for fmt in formats}
    for layer, histogram in enumerate(layer_histograms):
        jobs = reconstructed[formats[layer]]
        for expert, counts in sorted(histogram.items()):
            usage.append((layer, expert, counts))
            if not (membership[layer] >> expert) & 1:
                for i, count in enumerate(counts):
                    jobs[i] += count
    expected = {fmt: [0] * 8 for fmt in formats}
    for group in census['format_bins']:
        jobs = expected[(group['gu'], group['down'], group['blob_bytes'])]
        for i, count in enumerate(group['jobs_NT']):
            jobs[i] += count
    totals = projected_cpu_expert_work(usage, membership, formats)
    guards.require(reconstructed == expected and all(value == census[key] for key, value in totals.items()),
                   'routing suffix does not conserve the independent CPU census')
    return usage


def rank_by_verifier_jobs(reference_pairs, training_samples, *, cache_slots, protected_prefix):
    """Rank by per-window visits and keep mapped slots below the loan boundary.

    Layer interleaving and equal-score order follow the original profile.
    """
    guards.require(training_samples and len(reference_pairs) == len(set(reference_pairs)) == 48 * 512
                   and all(0 <= layer < 48 and 0 <= expert < 512 for layer, expert in reference_pairs),
                   'rank needs a complete expert profile and training requests')
    quota, remainder = divmod(cache_slots, 48)
    guards.require(0 <= protected_prefix <= cache_slots <= 48 * 512
                   and cache_slots > 0 and quota + remainder <= 512, 'invalid protected cache prefix')
    visits = [[0] * 512 for _ in range(48)]
    for sample in training_samples:
        guards.require(sample, 'empty per-expert training request')
        for layer, expert, counts in sample:
            visits[layer][expert] += sum(counts)
    by_layer = [[] for _ in range(48)]
    for layer, expert in reference_pairs:
        by_layer[layer].append(expert)
    ranks = []
    for layer, experts in enumerate(by_layer):
        capacity = quota + (remainder if layer == 47 else 0)
        locked = min(capacity, max(0, protected_prefix - layer * quota))
        tail = sorted(experts[locked:], key=visits[layer].__getitem__, reverse=True)
        ranks.append(iter(experts[:locked] + tail))
    return [(layer, next(ranks[layer])) for layer, _ in reference_pairs]


def projected_cpu_expert_work(usage, membership, formats):
    """Forecast verified held-out CPU work with fixed routes and no PCIe tier."""
    jobs, entries, bytes_visited = 0, 0, 0
    for layer, expert, counts in usage:
        if (membership[layer] >> expert) & 1:
            continue
        visits = sum(counts)
        jobs += visits
        entries += sum(width * count for width, count in enumerate(counts, 1))
        bytes_visited += visits * formats[layer][2]
    return dict(cpu_jobs=jobs, cpu_entries=entries, logical_job_blob_bytes=bytes_visited)
