"""Bound full-expert cache gains for a fixed, source-pinned DFlash2 route."""

from collections import Counter
import fcntl
from fractions import Fraction
import json
from math import ceil
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_layer_quota_replay as quota

replay = quota.replay
require = replay.require
THIS = str(Path(__file__).resolve().relative_to(replay.ROOT))
SCOPE_WINDOWS = 256
EXTRA_CAPS = (0, 256 << 20, 512 << 20, 768 << 20)
METRICS = ('cpu_jobs', 'logical_job_blob_bytes')


def count_scopes(trace, widths):
    half = len(widths) // 2
    train = [Counter() for _ in range(replay.LAYERS)]
    holdout = [[Counter() for _ in range(replay.LAYERS)]
               for _ in range((len(widths) - half + SCOPE_WINDOWS - 1) // SCOPE_WINDOWS)]
    for i, frame in enumerate(replay.window_counts(trace, widths)):
        scope = train if i < half else holdout[(i - half) // SCOPE_WINDOWS]
        for counts, jobs in zip(scope, frame, strict=True):
            counts.update(jobs.items())
    return train, holdout


def merge_counts(scopes):
    result = [Counter() for _ in range(replay.LAYERS)]
    for scope in scopes:
        for total, counts in zip(result, scope, strict=True):
            total.update(counts)
    return result


def minimum_remaining(holdout, sizes, protected, caps, use_bytes):
    """Relax equal-size slots to a byte budget with fractions of experts allowed."""
    fixed = set(protected)
    remaining = Counter()
    for layer, counts in enumerate(holdout):
        for (expert, _), jobs in counts.items():
            if (layer, expert) not in fixed:
                remaining[layer, expert] += jobs
    costs = [(sizes[layer], jobs * sizes[layer] if use_bytes else jobs)
             for (layer, _), jobs in remaining.items()]
    costs.sort(key=lambda item: Fraction(item[1], item[0]), reverse=True)
    total = sum(value for _, value in costs)
    protected_bytes = sum(sizes[layer] for layer, _ in protected)
    bounds = []
    for cap in caps:
        room = cap - protected_bytes
        require(room >= 0, 'the protected experts exceed the relaxed budget')
        saved = Fraction(0)
        for size, value in costs:
            if not room:
                break
            taken = min(room, size)
            saved += Fraction(taken * value, size)
            room -= taken
        bounds.append(ceil(total - saved))
    return bounds


def study():
    revision, trace, windows, membership, formats, observed = replay.inputs()
    for name in (THIS, quota.THIS, 'tools/make_profile.py'):
        require((replay.ROOT / name).read_bytes() == replay.git('show', revision + ':' + name),
                'commit exact static capacity replay inputs: ' + name)
    order, stock, sizes, slot_sizes = quota.stock_geometry(replay.ROOT / quota.PROFILE, membership, formats)
    train, epochs = count_scopes(trace, [width for _, width in windows])
    held = merge_counts(epochs)
    original = replay.score(merge_counts((train, held)), membership, formats)
    require(all(value == (observed[key][:4] if key == 'jobs_NT' else observed[key])
                for key, value in original.items()), 'capacity replay does not conserve observed stock work')

    def fixed_geometry(counts):
        ranked = quota.ranked_slots(stock, order, counts, lambda pair: sizes[pair[0]])
        require(ranked[:quota.PROTECTED] == stock[:quota.PROTECTED]
                and [sizes[layer] for layer, _ in ranked] == slot_sizes, 'actual cache geometry changed')
        return quota.masks(ranked)

    stock_held = replay.score(held, membership, formats)
    learned = replay.score(held, fixed_geometry(train), formats)
    ideal_static = replay.score(held, fixed_geometry(held), formats)
    ideal_scopes = [replay.score(scope, fixed_geometry(scope), formats) for scope in epochs]
    ideal_periodic = {key: sum(score[key] for score in ideal_scopes) for key in METRICS}
    caps = [quota.EXPECTED_BYTES + extra for extra in EXTRA_CAPS]
    byte_relaxed = dict(zip(METRICS, (minimum_remaining(held, sizes, stock[:quota.PROTECTED], caps, use_bytes)
                        for use_bytes in (False, True)), strict=True))
    require(all(byte_relaxed[key][0] <= ideal_static[key] <= learned[key] and
                ideal_periodic[key] <= ideal_static[key] <= stock_held[key] and
                all(a >= b for a, b in zip(byte_relaxed[key], byte_relaxed[key][1:]))
                for key in METRICS), 'inconsistent fixed-geometry and byte-only bounds')

    held_windows = len(windows) - len(windows) // 2
    return dict(source_commit=revision, capture_commit=replay.CAPTURE, routing_sha256=observed['routing_sha256'],
                training_windows=len(windows) - held_windows, holdout_windows=held_windows,
                oracle_window_period=SCOPE_WINDOWS, oracle_periods=len(epochs),
                fixed_bytes=quota.EXPECTED_BYTES, protected_slots=quota.PROTECTED, effective_slots=len(stock),
                heldout={label: {key: values[key] for key in METRICS} for label, values in (
                    ('stock', stock_held), ('training_rank', learned), ('oracle_static', ideal_static),
                    ('oracle_periodic', ideal_periodic))},
                byte_relaxed_static=[dict(cap_bytes=cap, **{key: values[i] for key, values in byte_relaxed.items()})
                                     for i, cap in enumerate(caps)],
                scope='One unchanged prose route. Oracle profiles know the evaluation routes in advance. '
                      'The periodic bound can change any unprotected slot every period at zero cost. '
                      'The byte-only static bound allows fractional experts and relaxes slot sizes/count. '
                      'These bounds concern CPU job counts and logical weight visits on this capture. '
                      'GPU service, transfers, actual DRAM use and decode rate are unmeasured.')


if __name__ == '__main__':
    require(sys.executable == quota.PYTHON and len(sys.argv) == 1, 'use the pinned Python without arguments')
    with (quota.STATE / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(study(), indent=2))
