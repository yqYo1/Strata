"""Score decode residency across layers while preserving every slot size."""

from collections import Counter, defaultdict, deque
import fcntl
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
import dflash2_decode_residency_replay as replay

sys.path.insert(0, str(replay.ROOT / 'tools'))
from make_profile import read_profile

STATE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
THIS = str(Path(__file__).resolve().relative_to(replay.ROOT))
PROFILE = replay.PROFILE
PROFILE_SHA256 = '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'
PROTECTED = 132
EXPECTED_SLOTS = 1256
EXPECTED_BYTES = 2_575_206_400
EXPECTED_TAIL = 2_321_664_000
EXPECTED_FIRST_LENT = 1_971_200
require = replay.require


def ranked_slots(stock, order, train, bucket):
    """Keep the protected prefix and slot sequence, then fill each budget."""
    fixed = set(stock[:PROTECTED])
    previous = set(stock)
    tail = stock[PROTECTED:]
    capacity = Counter(map(bucket, tail))
    jobs = Counter()
    for layer, counts in enumerate(train):
        for (expert, _), count in counts.items():
            jobs[layer, expert] += count

    candidates = defaultdict(list)
    for pair in order:
        if pair not in fixed and bucket(pair) in capacity:
            candidates[bucket(pair)].append(pair)
    chosen = set(fixed)
    for group, count in capacity.items():
        ranked = sorted(candidates[group], key=lambda pair: (-jobs[pair], pair not in previous, pair))
        require(len(ranked) >= count, 'insufficient candidates for a protected slot budget')
        chosen.update(ranked[:count])
    remaining = defaultdict(deque)
    for pair in sorted(chosen - previous):
        remaining[bucket(pair)].append(pair)
    slots = list(stock[:PROTECTED])
    slots.extend(pair if pair in chosen else remaining[bucket(pair)].popleft() for pair in tail)
    require(set(slots) == chosen and len(slots) == len(chosen)
            and not any(remaining.values()), 'slot allocation is not a complete permutation of the selected residents')
    return slots


def global_profile(slots, order, sizes):
    """The global loader stops at the first pair that exceeds its byte cap."""
    resident = set(slots)
    tail = [pair for pair in order if pair not in resident]
    largest = max(sizes)
    stop = next(pair for pair in tail if sizes[pair[0]] == largest)
    ranked = [*slots, stop, *(pair for pair in tail if pair != stop)]
    total = sum(sizes[layer] for layer, _ in slots)
    requested = (total + largest - 1) // largest
    require(total <= requested * largest < total + sizes[stop[0]]
            and len(set(ranked)) == len(ranked) == len(order), 'global cache does not stop at the exact admitted extent')
    return ranked, requested


def masks(slots):
    out = [0] * replay.LAYERS
    for layer, expert in slots:
        out[layer] |= 1 << expert
    return out


def stock_geometry(profile, stock_masks, formats):
    require(replay.sha(profile.read_bytes()) == PROFILE_SHA256, 'stock profile ordering changed')
    order = read_profile(profile)
    stock = sorted((pair for pair in order if stock_masks[pair[0]] >> pair[1] & 1), key=lambda pair: pair[0])
    sizes = [replay.aligned(fmt[2]) for fmt in formats]
    slot_sizes = [sizes[layer] for layer, _ in stock]
    require(all(size == fmt[2] for size, fmt in zip(sizes, formats, strict=True))
            and masks(stock) == stock_masks and len(stock) == EXPECTED_SLOTS
            and sum(slot_sizes) == EXPECTED_BYTES and sum(slot_sizes[PROTECTED:]) == EXPECTED_TAIL
            and slot_sizes[PROTECTED] == EXPECTED_FIRST_LENT, 'the original cache or prefill loan geometry changed')
    return order, stock, sizes, slot_sizes


def rank_sha256(ranking):
    return replay.sha(b''.join(struct.pack('<HH', *pair) for pair in ranking))


def analyze():
    revision, route, windows, stock_masks, formats, observed = replay.inputs()
    for name in (THIS, PROFILE, 'tools/make_profile.py', 'src/program/generate.cpp', 'src/core/expert_cache.cpp',
                 'include/strata/core/expert_cache.hpp', 'include/strata/core/expert_slot_sizes.hpp'):
        require((replay.ROOT / name).read_bytes() == replay.git('show', revision + ':' + name),
                'commit exact residency replay and loader inputs: ' + name)
    order, stock, sizes, stock_sizes = stock_geometry(replay.ROOT / PROFILE, stock_masks, formats)

    train, holdout = replay.count_routes(route, [width for _, width in windows])
    original = replay.score([a + b for a, b in zip(train, holdout, strict=True)], stock_masks, formats)
    require(all(value == (observed[key][:4] if key == 'jobs_NT' else observed[key])
                for key, value in original.items()), 'stock replay does not conserve the native route')

    selections = dict(stock=stock,
                      within_layer=ranked_slots(stock, order, train, lambda pair: pair[0]),
                      across_equal_size_layers=ranked_slots(stock, order, train, lambda pair: sizes[pair[0]]))
    cases = {}
    for label, slots in selections.items():
        require(slots[:PROTECTED] == stock[:PROTECTED] and [sizes[layer] for layer, _ in slots] == stock_sizes,
                'protected experts or physical slot sequence changed')
        ranking, requested = global_profile(slots, order, sizes)
        selected = masks(slots)
        cases[label] = dict(ranking_sha256=rank_sha256(ranking),
                            requested_expert_cache=requested, effective_slots=len(slots),
                            layer_slots=[layer.bit_count() for layer in selected],
                            changed_members=len(set(stock) - set(slots)),
                            train=replay.score(train, selected, formats),
                            holdout=replay.score(holdout, selected, formats))
    center = len(windows) // 2
    cross = cases['across_equal_size_layers']['holdout']
    advance = all(cross[k] < cases[case]['holdout'][k] for case in ('stock', 'within_layer')
                  for k in ('cpu_jobs', 'logical_job_blob_bytes'))
    return dict(source_commit=revision, capture_commit=replay.CAPTURE, routing_sha256=observed['routing_sha256'],
                training_windows=center, holdout_windows=len(windows) - center,
                protected_slots=PROTECTED, cache_bytes=EXPECTED_BYTES, prefill_loan_bytes=EXPECTED_TAIL,
                cases=cases, advance_to_admission=advance,
                scope='Observed single-workload route replay. Global stock preserves the original '
                'slot identities and widths. The two rankings change only membership after slot 132. '
                'Future model checks require actual placement, the same loan/restore and independent workloads.')


if __name__ == '__main__':
    require(sys.executable == PYTHON and len(sys.argv) == 1, 'use the pinned Python without arguments')
    with (STATE / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(analyze(), indent=2))
