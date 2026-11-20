"""Verify the two prefill selectors on the actual 64K and 117K bench.sh bodies."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_kv_gather_restore as state
import prefill_split_117k as split

bank, model, require = state.bank, state.model, state.require
basis = bank.basis
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'prefill_kv_gather_dispatch_v1'
ROOT = bank.B / 'prefill-kv-gather-dispatch-v1'
HISTORY = {
    65537: ('73d9394a9cb482a2dd091405ce42ca7343d708e3', 'prefill_resident_input_timing_v1'),
    117489: ('9c03b44d07e1bd9e4088d5d8f93a7b30b9172615', split.KEY),
}
ORDER = tuple((read, arm) for read in HISTORY for arm in state.ARMS)
RUNS = 2  # First-use and subsequent dispatch, both outside clean timing.


def signature(route, sample):
    return dict(request=sample['request_sha256'], offered=sample['drafts_offered'],
                accepted=sample['drafts_accepted'], generated=sample['generated_tokens'],
                GU=route['actual_GU_dispatch'], window=route['fast_window'], store=route['attention_store'])


def parent_work(native, speed):
    work = {}
    for read, (revision, key) in HISTORY.items():
        row = bank.previous.historic(revision, key)
        control = row['processes'][0]
        require(row['complete'] and row['passed'] and row['ownership_closed'] and not row['active']
                and row['native_source_commit'] == bank.INPUT_SOURCE
                and row['binary'] == native['projection']['parent_binary']
                and row['configuration'] == speed['configuration']
                and row['model_assets_sha256'] == bank.signature(speed['model_assets'])
                and control['passed'] and control['complete'] and control['ownership_closed']
                and control['native_exit_code'] == 0 and not control['fresh_faults']
                and len(control['samples']) == (3 if read == 65537 else 4)
                and [s['request_sha256'] for s in control['samples']] == row['request_hashes'][str(read)],
                'recover the measured parent binary, configuration and original workloads')
        work[str(read)] = []
        for route, sample in zip(control['routes'][:RUNS], control['samples'][:RUNS], strict=True):
            require(route['attention_input']['borrowed'] == route['attention_store']['direct'] == 0
                    and sample['actual_read_tokens'] == read and sample['actual_reused_tokens'] == 0
                    and sample['generated_tokens'] == 24, 'expected a fresh copy-control request')
            work[str(read)].append(signature(route, sample))
    return work


def admission():
    native, speed, build_hash, device_hash = state.inputs()
    gate = basis.validation.committed(bank.RESULT, (state.KEY,))
    require(gate['stage'] == 'closed' and gate['passed'] and gate['complete'] and gate['ownership_closed']
            and not gate['active'] and gate['binary'] == native['binaries']['strata']
            and gate['native_source_commit'] == native['native_source_commit']
            and gate['build_sha256'] == build_hash and gate['device_sha256'] == device_hash
            and list(gate['outcomes']) == list(state.ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in gate['outcomes'].values()),
            'complete all same-build state, dispatch and RESTORE comparisons first')
    prior = bank.previous.read(basis.reference.model.ROOT / 'record.json')
    require(prior['models'] == speed['model_assets'] and prior['probe'] == speed['native_validation_probe'],
            'official benchmark must use the measured assets and validation provider')
    prior['observer_admission'] = dict(
        fast_configuration_diagnostic=speed['observer_admission']['fast_configuration_diagnostic'],
        state_sha256=bank.signature(gate), ordinary_control=model.ordinary_control(),
        parent_work=parent_work(native, speed))
    manifest = speed['runtime_manifest']
    require(bank.sha(manifest['path']) == manifest['sha256'], 'runtime provider manifest changed')
    return native, prior, bank.previous.read(Path(manifest['path'])), None


def check_work(outcome, row, plan):
    read, arm = ORDER[plan['index']]
    expected = row['observer_admission']['parent_work'][str(read)]
    require(not outcome['clean_speed'] and [signature(r, s) for r, s in zip(
        outcome['routes'], outcome['samples'], strict=True)] == expected,
        'actual request, GU/Down, expert transfer, attention store or draft work differs from parent')
    if arm != 'base':
        base = row['processes'][ORDER.index((read, 'base'))]
        for route, old in zip(outcome['routes'], base['routes'], strict=True):
            kv, original = route['kv'], old['kv']
            require(route['gather'] == old['gather']
                    and kv['mirror_D2H_calls'] == original['mirror_D2H_calls']
                    and kv['mirror_D2H_bytes'] == original['mirror_D2H_bytes']
                    and kv['reloaded_pages'] + kv['reused_pages'] == original['reloaded_pages'],
                    'changed eligible gather rows, host publication or prefix extent')


def experiment():
    ctx = basis.core.source.device.load_module('kv_gather_official_dispatch', basis.core.__file__)
    historical = basis.reference.context()
    ctx.THIS, ctx.KEY, ctx.ROOT = THIS, KEY, ROOT
    ctx.READS, ctx.ORDER = [read for read, _ in ORDER], [(read, 1) for read, _ in ORDER]
    ctx.RUNS, ctx.CLEAN_FAST = RUNS, True
    ctx.basis, ctx.check_sample, ctx.after_execution = admission, historical.check_sample, check_work
    old_pins, old_save, old_current = ctx.pins, ctx.save, ctx.current

    def pins(revision):
        values = old_pins(revision)
        require({THIS, split.THIS, split.bank.THIS, *state.SCRIPTS} <= values.keys(),
                'missing benchmark controller dependency')
        return values

    def environment(row, plan):
        reuse, vec8 = state.ARMS[ORDER[plan['index']][1]]
        env = model.ordinary_environment(row, plan['root'], row['observer_admission']['ordinary_control'])
        env[model.INPUT] = env[model.OUTPUT] = '0'
        env[state.KV], env[state.GATHER] = str(int(reuse)), str(int(vec8))
        env['STRATA_TRACE'] = '1'
        return env

    def route(text, plan):
        read, arm = ORDER[plan['index']]
        rows = read - 1
        reuse, vec8 = state.ARMS[arm]
        value = historical.request_route(text, plan)
        store = model.restore.require_store(text, dict(direct=False), plan)
        require(store['gpu_rows'] == rows and store['host_rows'] == 0 and store['window_mode'] == 10
                and 'strata MoE prefix expert upload layer:' not in text, 'wrong resident window or expert queue')
        value['attention_store'] = store
        value['attention_input'] = model.input_receipt(text, rows, store, False)
        if read == split.READ:
            value['admission'] = split.check_split_budget(text)
        value['gather'] = state.gather_counts(text, vec8)
        value['kv'] = state.kv.check_request(text, 0, rows, model.restore.capacity.CHUNK, reuse=reuse)
        return value

    def save(row):
        if row['stage'] == 'prepared':
            for plan, (_, arm) in zip(row['plans'], ORDER, strict=True):
                plan.update(trace_progress=True, arm=arm)
            row['plan'] = dict(order=ORDER, requests=RUNS,
                scope='Traced same-build execution of the first two original bodies at each length. No speed claim.',
                gates='Match parent work, per-layer mirror/reload/reuse, actual vec8 rows and state-gated binary. '
                      'Stop at the first workload, ownership, numerical, dispatch or driver failure.')
            row['whole_bytes'] = len(ORDER) * row['budgets']['total_cap'] + (8 << 20)
        old_save(row)

    def current(row):
        require(row['plan']['order'] == [list(v) for v in ORDER] and row['plan']['requests'] == RUNS
                and [(p['index'], p['read'], p['arm'], p['trace_progress']) for p in row['plans']] ==
                    [(i, read, arm, True) for i, (read, arm) in enumerate(ORDER)],
                'changed dispatch order, length, arm or observation policy')
        return old_current(row)

    def decision(processes):
        require(len(processes) == len(ORDER) and all(p['index'] == i and p['passed'] and p['complete']
                and p['ownership_closed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                and not p['clean_speed'] and len(p['samples']) == RUNS for i, p in enumerate(processes)),
                'eight completed traced processes required')
        return dict(dispatch_verified=True, requests=RUNS * len(ORDER), speed_qualified=False)

    ctx.pins, ctx.environment, ctx.request_route = pins, environment, route
    ctx.save, ctx.current, ctx.decision = save, current, decision
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON
            and (action in ('prepare', 'host') or action in map(str, range(len(ORDER)))), 'invalid dispatch stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
