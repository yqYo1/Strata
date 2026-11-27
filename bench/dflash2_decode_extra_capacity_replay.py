"""Estimate CPU expert work saved by adding decode-only cache capacity."""

from bisect import bisect_right
from collections import Counter
from fractions import Fraction
import fcntl
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_layer_quota_replay as quota

replay = quota.replay
require = replay.require
THIS = str(Path(__file__).resolve().relative_to(replay.ROOT))
CAPS = (0, 256 << 20, 512 << 20, 768 << 20)


def additions(order, stock, sizes, train):
    residents = set(stock)
    jobs = Counter()
    for layer, counts in enumerate(train):
        for (expert, _), count in counts.items():
            jobs[layer, expert] += count
    misses = (pair for pair in order if pair not in residents and jobs[pair] > 0)
    return sorted(misses, key=lambda pair: (-Fraction(jobs[pair], sizes[pair[0]]), pair))


def study():
    revision, trace, windows, membership, formats, observed = replay.inputs()
    for name in (THIS, quota.THIS, 'tools/make_profile.py'):
        require((replay.ROOT / name).read_bytes() == replay.git('show', revision + ':' + name),
                'commit the exact capacity replay input: ' + name)
    order, stock, sizes, _ = quota.stock_geometry(replay.ROOT / quota.PROFILE, membership, formats)
    train, holdout = replay.count_routes(trace, [width for _, width in windows])
    total = replay.score([a + b for a, b in zip(train, holdout, strict=True)], membership, formats)
    require(all(value == (observed[key][:4] if key == 'jobs_NT' else observed[key])
                for key, value in total.items()), 'capacity control does not conserve the captured work')

    ranked = additions(order, stock, sizes, train)
    cumulative = list(itertools.accumulate((sizes[layer] for layer, _ in ranked), initial=0))
    base = replay.score(holdout, membership, formats)
    cases = []
    for cap in CAPS:
        count = bisect_right(cumulative, cap) - 1
        slots = [*stock, *ranked[:count]]
        resident = quota.masks(slots)
        require(len(set(slots)) == len(slots) and all((a & b) == b for a, b in zip(resident, membership, strict=True))
                and cumulative[count] <= cap
                and (count == len(ranked) or cumulative[count + 1] > cap), 'capacity or existing placement changed')
        remaining = replay.score(holdout, resident, formats)
        saved = {key: base[key] - remaining[key] for key in ('cpu_jobs', 'cpu_entries', 'logical_job_blob_bytes')}
        require(min(saved.values()) >= 0, 'increasing residency added CPU work')
        cases.append(dict(cap_bytes=cap, added_slots=count, extra_bytes=cumulative[count],
                          extra_order_sha256=quota.rank_sha256(ranked[:count]),
                          extra_slots_by_layer=[(a ^ b).bit_count() for a, b in zip(resident, membership, strict=True)],
                          holdout=remaining, saved=saved))
    first = len(windows) // 2
    return dict(source_commit=revision, capture_commit=replay.CAPTURE, routing_sha256=observed['routing_sha256'],
                training_windows=first, holdout_windows=len(windows) - first, fixed_slots=len(stock),
                fixed_bytes=quota.EXPECTED_BYTES, fixed_order_sha256=quota.rank_sha256(stock), cases=cases,
                sort='Highest training CPU job count per aligned extra byte, tie by layer and expert ID.',
                scope='Fixed-work replay on the held-out half of one observed prose decode. Preserve every '
                      'existing slot, including protected and lent positions. Each budget admits a prefix '
                      'of the same extra ranking. The extra bytes are optional capacity hypotheses, '
                      'not measured available memory. No engine allocation, transfer or timing.')


if __name__ == '__main__':
    require(sys.executable == quota.PYTHON and len(sys.argv) == 1, 'use the pinned Python without arguments')
    with (quota.STATE / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(study(), indent=2))
