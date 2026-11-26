"""Screen fixed-budget DFlash2 expert maps against code and prose routes."""

from collections import Counter
import fcntl
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_layer_quota_replay as quota
import dflash2_decode_residency_replay as replay


THIS = str(Path(__file__).resolve().relative_to(replay.ROOT))
CODE_RUN = 'dflash2_long_routing_train_v1'
CODE_REPLAY = 'dflash2_long_routing_train_replay_v1'
METRICS = ('cpu_jobs', 'logical_job_blob_bytes')
TRAINING = ('code_train', 'prose_train')
require = replay.require


def verify_stock(counts, masks, formats, observed):
    scored = replay.score(counts, masks, formats)
    require(all(value == (observed[key][:4] if key == 'jobs_NT' else observed[key])
                for key, value in scored.items()), 'stock counters disagree with the capture')


def code_counts(archive, masks, formats):
    original, restored, prose = (archive[k] for k in (CODE_RUN, CODE_REPLAY, replay.KEY))
    plan, routes = restored['plan'], restored['routes']
    process, = original['processes']
    require(original['stage'] == 'failed' and original['ownership_closed']
            and restored['stage'] == 'closed' and restored['passed'] and not restored['native_executed']
            and process['ownership_closed'] and process['boot_unchanged'] and not process['fresh_faults']
            and process['error'] == plan['original_error'] and process['native_identity'] == plan['original_native_identity']
            and plan['native_exit_code'] == 0 and plan['original_source_commit'] == original['source_commit']
            and all(original[field] == prose[field] for field in ('binary', 'native_source_commit',
                    'model_assets_sha256', 'routing_metadata')), 'code replay provenance differs')
    path = Path(original['plans'][0]['root']) / 'verifier-usage.json'
    require(path == Path(routes['usage_file']) and path.stat().st_size <= 4 << 20, 'code usage path or extent differs')
    data = path.read_bytes()
    require(replay.sha(data) == routes['usage_sha256'], 'code usage hash differs')
    usage = json.loads(data)
    identity = original['routing_metadata']
    require((usage['native_source_commit'], usage['profile_sha256'], usage['format_sha256']) ==
            (original['native_source_commit'], identity['profile_sha256'], identity['manifest_sha256'])
            and usage['request_sha256'] == original['routing_workload']['request_sha256']
            and replay.sha(b''.join(mask.to_bytes(64, 'little') for mask in masks)) == routes['membership_sha256'],
            'code usage was counted with different expert inputs')
    layers = [Counter() for _ in masks]
    observed_pairs = set()
    for layer, expert, bins in usage['usage']:
        pair = (layer, expert)
        require(0 <= layer < replay.LAYERS and 0 <= expert < replay.EXPERTS and pair not in observed_pairs
                and len(bins) == 4 and all(type(count) is int and count >= 0 for count in bins) and any(bins),
                'invalid or duplicate layer/expert counts')
        observed_pairs.add(pair)
        layers[layer].update({(expert, nt): count for nt, count in enumerate(bins, 1) if count})
    require(len(observed_pairs) == routes['distinct_layer_expert_pairs']
            and all(sum(nt * count for (_, nt), count in layer.items()) == 10 * routes['target_rows']
                    for layer in layers), 'incomplete code usage')
    verify_stock(layers, masks, formats, routes)
    return layers, routes['usage_sha256'], routes['windows']


def blend(code, prose, code_quarters, stock):
    # Normalize each workload's counts by its stock misses. Cross-multiply to
    # retain exact integer ordering and the original tie rule in ranked_slots.
    code_scale = code_quarters * stock['prose_train']['cpu_jobs']
    prose_scale = (4 - code_quarters) * stock['code_train']['cpu_jobs']
    for a, b in zip(code, prose, strict=True):
        mixed = Counter({key: count * code_scale for key, count in a.items()})
        mixed.update({key: count * prose_scale for key, count in b.items()})
        yield mixed


def analyze():
    revision, data, windows, masks, formats, observed = replay.inputs()
    for name in (THIS, quota.THIS, 'tools/make_profile.py'):
        require((replay.ROOT / name).read_bytes() == replay.git('show', revision + ':' + name),
                'commit the exact blended rank replay inputs: ' + name)
    archive = json.loads(replay.git('show', replay.CAPTURE + ':' + replay.EVIDENCE))
    code, code_hash, code_windows = code_counts(archive, masks, formats)
    prose, holdout = replay.count_routes(data, [width for _, width in windows])
    verify_stock([a + b for a, b in zip(prose, holdout, strict=True)], masks, formats, observed)
    order, stock_slots, sizes, widths = quota.stock_geometry(replay.ROOT / quota.PROFILE, masks, formats)
    workloads = dict(code_train=code, prose_train=prose, prose_suffix=holdout)

    def measure(selected):
        return {name: {key: value for key, value in replay.score(counts, selected, formats).items()
                       if key in METRICS} for name, counts in workloads.items()}

    stock = measure(masks)
    candidates = []
    for weight in range(5):
        counts = blend(code, prose, weight, stock)
        slots = quota.ranked_slots(stock_slots, order, counts, lambda pair: pair[0])
        selected = quota.masks(slots)
        require(slots[:quota.PROTECTED] == stock_slots[:quota.PROTECTED]
                and [sizes[layer] for layer, _ in slots] == widths
                and [mask.bit_count() for mask in selected] == [mask.bit_count() for mask in masks],
                'layer quota, protected members or slot widths changed')
        ranking, requested = quota.global_profile(slots, order, sizes)
        candidates.append(dict(code_weight_quarters=weight, ranking_sha256=quota.rank_sha256(ranking),
                               requested_expert_cache=requested, work=measure(selected)))

    def worst_train_ratio(weight):
        work = candidates[weight]['work']
        return max(Fraction(work[name][key], stock[name][key]) for name in TRAINING for key in METRICS)

    selected = min(range(1, 4), key=lambda w: (worst_train_ratio(w), abs(w - 2), w))
    advance = all(candidates[selected]['work'][name][key] < stock[name][key]
                  for name in workloads for key in METRICS)
    return dict(source_commit=revision, capture_commit=replay.CAPTURE, code_usage_sha256=code_hash,
                prose_trace_sha256=observed['routing_sha256'], code_train_windows=code_windows,
                prose_train_windows=len(windows) // 2, prose_suffix_windows=(len(windows) + 1) // 2,
                slots=len(stock_slots), protected_slots=quota.PROTECTED, cache_bytes=sum(widths),
                stock=stock, candidates=candidates, selected_code_weight_quarters=selected,
                advance_to_independent_capture=advance)


if __name__ == '__main__':
    require(sys.executable == quota.PYTHON and len(sys.argv) == 1, 'use the pinned Python without arguments')
    with (quota.STATE / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(analyze(), indent=2))
