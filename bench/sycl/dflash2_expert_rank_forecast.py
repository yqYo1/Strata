"""Freeze a DFlash2 cache ranking from training and score untouched held-out work."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_long_routing as capture
import dflash2_long_routing_replay as training_replay

bank, basis, require = capture.bank, capture.basis, capture.require
census, routing = capture.census, capture.routing
sys.path.insert(0, str(bank.P / 'tools'))
import make_profile

KEY = capture.FORECAST
ROOT = bank.B / KEY.replace('_', '-')
PROFILE = ROOT / 'train-rank.bin'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
SOURCES = (THIS, capture.THIS, training_replay.THIS, 'tools/make_profile.py', 'tools/sycl/cpu_work_census.py',
           'tools/sycl/dflash2_expert_routing.py')
CACHE_SLOTS, PREFIX = 1256, 132
CAPACITY = [26] * 47 + [34]


def verify_sources(revision):
    pins = {}
    for name in SOURCES:
        path = bank.P / name
        require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                'uncommitted forecast input: ' + name)
        pins[name] = bank.sha(path)
    return pins


def observed(role):
    ctx = capture.experiment(role)
    row = basis.validation.committed(bank.RESULT, (ctx.KEY,))
    require(row['ownership_closed'] and not row['active'] and row['plan']['case'] == role,
            f'require the committed closed {role} capture')
    process, = row['processes']
    require(process['ownership_closed'] and not process['fresh_faults'], 'native owner or fault differs')
    if role == 'train':
        replay = basis.validation.committed(bank.RESULT, (training_replay.KEY,))
        require(row['stage'] == 'failed' and process['error'] == training_replay.ORIGINAL_ERROR
                and replay['stage'] == 'closed' and replay['passed'] and replay['native_executed'] is False
                and replay['plan']['original_record_sha256'] == bank.signature(row)
                and replay['plan']['original_native_identity'] == process['native_identity']
                and replay['plan']['native_exit_code'] == 0, 'require the separate exact training replay')
        routes = replay['routes']
    else:
        require(row['stage'] == 'closed' and row['passed'] and row['complete']
                and process['passed'] and process['native_exit_code'] == 0, 'held-out result differs')
        routes = process['routes']
    path = Path(routes['usage_file'])
    require(path == ctx.ROOT / '0' / capture.USAGE_FILE and not path.is_symlink()
            and path.is_file() and path.stat().st_size <= 4 << 20
            and bank.sha(path) == routes['usage_sha256'], 'retained verifier usage differs')
    usage = bank.previous.read(path)
    sample = process['evidence']['samples'][-1]
    meta = row['routing_metadata']
    require(all(bank.sha(meta[k]) == meta[k + '_sha256'] for k in ('profile', 'manifest'))
            and usage['request_sha256'] == sample['request_sha256'] == row['routing_workload']['request_sha256']
            and usage['format_sha256'] == meta['manifest_sha256']
            and usage['profile_sha256'] == meta['profile_sha256']
            and usage['native_source_commit'] == row['native_source_commit'],
            'usage, request or profile provenance differs')
    pairs = usage['usage']
    membership = census.expected_membership(meta['profile'], CACHE_SLOTS)
    formats = census.manifest_formats(meta['manifest'])
    expected = routing.projected_cpu_expert_work(pairs, membership, formats)
    require(len(pairs) == routes['distinct_layer_expert_pairs']
            and all(routes[k] == v for k, v in expected.items()),
            'retained routing observations differ from the native census')
    summary = dict(key=ctx.KEY, request_sha256=sample['request_sha256'],
                   candidate_freeze=row.get('rank_candidate_freeze'),
                   usage_sha256=routes['usage_sha256'],
                   generated_tokens=sample['generated_tokens'],
                   read_tokens=sample['read_tokens'], accepted=sample['drafts_accepted'],
                   offered=sample['drafts_offered'], jobs=expected)
    geometry = dict(meta, binary=row['binary']['sha256'], native_source_commit=row['native_source_commit'],
                    boot=row['boot'], driver=row['driver'])
    return pairs, formats, geometry, summary


def by_layer(pairs):
    result = [[] for _ in CAPACITY]
    for layer, expert in pairs:
        result[layer].append(expert)
    return result


def selected_delta(stock_pairs, trained_pairs):
    before, after = by_layer(stock_pairs), by_layer(trained_pairs)
    return {str(layer): top[:cap] for layer, (old, top, cap) in enumerate(zip(before, after, CAPACITY, strict=True))
            if old[:cap] != top[:cap]}


def apply_delta(stock_pairs, delta):
    """Reconstruct each admitted prefix, preserving inactive expert order."""
    stock = by_layer(stock_pairs)
    require(set(delta) <= {str(layer) for layer in range(len(CAPACITY))}, 'profile delta layer out of range')
    rebuilt, position = [], 0
    for layer, (experts, cap) in enumerate(zip(stock, CAPACITY, strict=True)):
        selected = delta.get(str(layer), experts[:cap])
        protected = max(0, min(cap, PREFIX - position))
        require(len(selected) == len(set(selected)) == cap and set(selected) <= set(experts)
                and selected[:protected] == experts[:protected], 'invalid or changed protected expert slots')
        inactive = [expert for expert in experts if expert not in selected]
        rebuilt.append(iter([*selected, *inactive]))
        position += cap
    require(position == CACHE_SLOTS, 'cache capacity changed')
    return [(layer, next(rebuilt[layer])) for layer, _ in stock_pairs]


def membership(pairs):
    return [sum(1 << expert for expert in order[:cap])
            for order, cap in zip(by_layer(pairs), CAPACITY, strict=True)]


def save(row):
    data = bank.previous.read(bank.RESULT)
    data[KEY] = row
    bank.stage.write(bank.RESULT, data)


def freeze(revision):
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
            'original rank selection already exists')
    require(not capture.experiment('holdout').ROOT.exists(), 'freeze the candidate before preparing the held-out run')
    usage, formats, geometry, train = observed('train')
    stock = make_profile.read_profile(geometry['profile'])
    trained = routing.rank_by_verifier_jobs(stock, [usage], cache_slots=CACHE_SLOTS, protected_prefix=PREFIX)
    delta = selected_delta(stock, trained)
    ranked = apply_delta(stock, delta)
    before, after = membership(stock), membership(ranked)
    require(before == census.expected_membership(geometry['profile'], CACHE_SLOTS)
            and after == membership(trained), 'generated rank or stock membership differs')
    estimated = routing.projected_cpu_expert_work(usage, after, formats)
    require(estimated['cpu_jobs'] <= train['jobs']['cpu_jobs']
            and estimated['logical_job_blob_bytes'] <= train['jobs']['logical_job_blob_bytes'],
            'training rank violates its fixed-capacity resource objective')
    sources = verify_sources(revision)
    ROOT.mkdir(mode=0o700)
    make_profile.write_profile(PROFILE, ranked)
    require(census.expected_membership(PROFILE, CACHE_SLOTS) == after, 'serialized admitted membership differs')
    save(dict(stage='frozen', source_commit=revision, sources=sources, geometry=geometry, train=train,
              train_forecast=estimated, profile=dict(path=str(PROFILE), sha256=bank.sha(PROFILE),
                  capacity=CACHE_SLOTS, protected_prefix=PREFIX, profile_delta=delta),
              rule='Select from training only. For the untouched held-out request require fewer CPU jobs '
                   'and no increase in logical blob bytes before a live rank comparison. Measure actual '
                   'prefill and decode independently. Compute a held-out oracle bound without selecting from it.'))


def score():
    record = basis.validation.committed(bank.RESULT, (KEY,))
    require(record['stage'] == 'frozen' and record['sources'] == verify_sources(record['source_commit']),
            'require the original committed pre-holdout rank')
    path = record['profile']
    require(path['path'] == str(PROFILE) and bank.sha(PROFILE) == path['sha256']
            and (path['capacity'], path['protected_prefix']) == (CACHE_SLOTS, PREFIX),
            'frozen profile artifact changed')
    usage, formats, geometry, heldout = observed('holdout')
    require(geometry == record['geometry']
            and heldout['request_sha256'] != record['train']['request_sha256']
            and heldout['candidate_freeze'] == bank.signature(record),
            'held-out body, recording geometry or frozen rank differs from the plan')
    stock = make_profile.read_profile(geometry['profile'])
    ranked = apply_delta(stock, path['profile_delta'])
    changed = census.expected_membership(PROFILE, CACHE_SLOTS)
    require(make_profile.read_profile(PROFILE) == ranked and changed == membership(ranked),
            'projected and actual candidate file differ')
    candidate = routing.projected_cpu_expert_work(usage, changed, formats)
    original = heldout['jobs']
    oracle = routing.rank_by_verifier_jobs(stock, [usage], cache_slots=CACHE_SLOTS, protected_prefix=PREFIX)
    bound = routing.projected_cpu_expert_work(usage, membership(oracle), formats)
    require(bound['cpu_jobs'] <= min(candidate['cpu_jobs'], original['cpu_jobs'])
            and bound['logical_job_blob_bytes'] <= min(candidate['logical_job_blob_bytes'],
                                                       original['logical_job_blob_bytes']),
            'held-out oracle violates the fixed quota bound')
    eligible = candidate['cpu_jobs'] < original['cpu_jobs'] and (
        candidate['logical_job_blob_bytes'] <= original['logical_job_blob_bytes'])
    record.update(stage='scored', holdout=heldout, holdout_forecast=candidate,
                  holdout_fixed_work_oracle=bound, eligible_for_live_comparison=eligible)
    save(record)
    print({'stock': original, 'candidate': candidate, 'fixed_work_oracle': bound, 'eligible': eligible}, flush=True)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('freeze', 'score'),
            'use the pinned Python with freeze or score')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'freeze':
            freeze(basis.validation.git('rev-parse', 'HEAD').decode().strip())
        else:
            score()


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
