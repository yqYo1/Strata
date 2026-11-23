"""Measure shared Gate placement and reuse independently on long SYCL prefill."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_compare as upload

bank, basis, require = upload.bank, upload.basis, upload.require
previous = upload.previous
READ, SMALL_READ = upload.READ, 16386
QSA_PREFIX_PAGE_SIZE = 4
MODES = ('original', 'reload', 'reuse')
BLOCKS = (MODES, MODES[1:] + MODES[:1], MODES[2:] + MODES[:2])
CONTRASTS = (('reload', 'original'), ('reuse', 'reload'), ('reuse', 'original'))
SELECTOR = 'STRATA_SYCL_PREFILL_SHARED_GATE_WEIGHTS'
KEYS = {'admit': 'prefill_shared_gate_admission_v2', 'time': 'prefill_shared_gate_speed_v1'}
REFERENCE = 'be8caf6138b633d23df45cf0ee9ca10480589bcb'
THIS = str(Path(__file__).resolve().relative_to(bank.P))


def measured_basis(small, runs):
    control = upload.experiment('store-time')
    original = bank.previous.historic(REFERENCE, control.KEY)
    recorded = bank.previous.read(control.ROOT / 'record.json')
    native = bank.previous.read(upload.build.GATE_ROOT / 'record.json')
    upload.build.context('shared-gate').check(native)
    require(upload.gather.closed(original) and original == basis.compact(recorded)
            and native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['projection']['gate_parent_binary'] == original['binary']
            and native['driver'] == original['driver'], 'complete the exact shared Gate build and host controls')
    prior = bank.previous.read(basis.reference.model.ROOT / 'record.json')
    require(prior['probe'] == original['native_validation_probe'] and prior['boot'] == original['boot']
            and prior['driver'] == original['driver']
            and bank.signature(prior['models']) == original['model_assets_sha256']
            and prior['runtime_manifest']['sha256'] == original['runtime_manifest']['sha256']
            and bank.sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
            and bank.previous.read(Path(prior['runtime_manifest']['path'])) == prior['runtime_files'],
            'original model assets, validation, boot or runtime differs')
    proof = dict(reference=REFERENCE, reference_configuration=original['configuration'],
                 ordinary_control=original['observer_admission']['ordinary_control'],
                 fast_configuration_diagnostic=original['observer_admission']['fast_configuration_diagnostic'],
                 build_sha256=bank.signature(bank.builder.compact(native)))
    if not small:
        measured = next(p for p in original['processes'] if p['direct_output'])
        work = [previous.split.work(route, sample) for route, sample in
                zip(measured['routes'], measured['samples'], strict=True)]
        require(len(work) == runs and [w['request'] for w in work] == original['request_hashes'][str(READ)]
                and measured['passed'] and measured['ownership_closed'] and not measured['fresh_faults'],
                'require the exact four measured direct-output requests')
        gate = basis.validation.committed(bank.RESULT, (KEYS['admit'],))
        require(upload.gather.closed(gate) and gate['binary'] == native['binaries']['strata']
                and gate['native_source_commit'] == native['native_source_commit']
                and gate['observer_admission'] == proof and gate['configuration'] == original['configuration']
                and gate['decision'] == dict(admitted_modes=list(MODES), response_matched=True),
                'complete the checked same-build ragged-tail admission before timing')
        proof.update(expected_work=work, admission_sha256=bank.signature(gate))
    return native, dict(prior, observer_admission=proof), prior['runtime_files'], None


def request_route(text, plan):
    read, mode = plan['read'], plan['gate_mode']
    if plan['trace_progress']:
        value = previous.workspace.request_route(text, dict(plan, bank=384, window=10, relocate=0, upload='jit'),
                                                 prefix_page_size=plan['qsa_prefix_page_size'])
        store = previous.model.restore.require_store(text, dict(direct=True), plan)
        previous.model.input_receipt(text, read - 1, store, False, trace_layers=True)
        kv = previous.speed.state.kv.check_request(text, 0, read - 1, 8192, reuse=True)
        route = {name: value[name] for name in ('layer_cache_bytes', 'uploads', 'actual_GU_dispatch',
                                               'requested_bank_H2D_bytes')}
        route.update(attention_store=store, kv={k: kv[k] for k in ('reloaded_pages', 'reused_pages')},
                     gather=previous.speed.state.gather_counts(text, False))
    else:
        route = upload.request_route(text, dict(plan, direct_output=True, mode='batched'))
    windows = 48
    chunks = windows * ((read - 2) // 8192 + 1)
    marker = previous.workspace.records(text, 'strata MoE prefill shared Gate: ')
    if mode == 'original':
        require(not marker, 'original Gate path unexpectedly used the alternate buffer')
    else:
        expansions = chunks if mode == 'reload' else windows
        require(marker == [dict(windows=str(windows), chunks=str(chunks), expansions=str(expansions),
                                final_drain='checked')], 'actual Gate expansions or retirement differ')
    route['shared_gate'] = dict(mode=mode, alternate_buffer=marker)
    return route


def check_execution(outcome, row, plan):
    work = [previous.split.work(route, sample) for route, sample in
            zip(outcome['routes'], outcome['samples'], strict=True)]
    outcome['gate_mode'] = plan['gate_mode']
    if plan['trace_progress']:
        result, = (Path(plan['root']) / 'bench').glob('*/results.json')
        requests = bank.previous.read(result)['results']['prefill_only']['requests']
        require([s['request_sha256'] for s in requests] == [s['request_sha256'] for s in outcome['samples']],
                'ragged admission response does not match its timed request')
        outcome.update(response_sha256=[[s['reasoning_sha256'], s['output_sha256']] for s in requests],
                       exact_work=work)
        if row['processes']:
            original = row['processes'][0]
            require(outcome['response_sha256'] == original['response_sha256'] and work == original['exact_work'],
                    'shared Gate changed the ragged request work or response')
    else:
        require(outcome['clean_speed'] and work == row['observer_admission']['expected_work'],
                'clean split-prefill work differs from the exact historical bodies')


def decision(processes, order, small):
    require(tuple(p['gate_mode'] for p in processes) == order
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(order)
            and all(p['complete'] and p['passed'] and p['ownership_closed'] and not p['fresh_faults']
                    and p['native_exit_code'] == 0 and p['clean_speed'] == (not small) for p in processes),
            'complete every independent process in the frozen order')
    if small:
        return dict(admitted_modes=list(order), response_matched=True)
    rates = {mode: [] for mode in MODES}
    ratios = {a + '/' + b: [] for a, b in CONTRASTS}
    for i in range(0, len(order), len(MODES)):
        clocks = {p['gate_mode']: previous.timing(p) for p in processes[i:i + len(MODES)]}
        for mode, t in clocks.items():
            rates[mode].append(1000 * READ / t['warm_prompt_ms'])
        for a, b in CONTRASTS:
            ratios[a + '/' + b].append({k: clocks[a][k] / clocks[b][k] for k in clocks[a]})
    gains = {}
    for contrast, samples in ratios.items():
        times = [p['warm_prompt_ms'] for p in samples]
        gains[contrast] = sum(t < 1 for t in times) >= 2 and statistics.median(times) < 1
    return dict(warm_process_rates=rates, paired_time_ratios=ratios, repeated_warm_gain=gains,
                repeated_1000={mode: min(p) >= 1000 for mode, p in rates.items()}, adopted=False)


def experiment(phase):
    small = phase == 'admit'
    order = MODES if small else tuple(mode for block in BLOCKS for mode in block)
    read, runs = (SMALL_READ, 2) if small else (READ, 4)
    ctx = basis.core.source.device.load_module('shared_gate_' + phase, basis.core.__file__)
    same_route = upload.experiment('store-time')
    ctx.THIS, ctx.KEY, ctx.ROOT = THIS, KEYS[phase], bank.B / KEYS[phase].replace('_', '-')
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (read,) * len(order), [(read, 1)] * len(order), runs, True
    ctx.basis = lambda: measured_basis(small, runs)
    ctx.pins = lambda revision: same_route.pins(revision) | bank.stage.source.pin(bank.P, [THIS])['source_sha256']
    ctx.request_route, ctx.after_execution, ctx.check_sample = request_route, check_execution, same_route.check_sample
    ctx.decision = lambda processes: decision(processes, order, small)
    base_save, base_current = ctx.save, ctx.current
    plan = dict(order=list(order), read=read, requests=runs, trace_progress=small,
        scope='Ragged 8192+8192+1 workspace and same-build response admission.' if small else
            'Compare alternate-buffer placement, reuse at that address and their combined effect on the measured split path.',
        acceptance='Require exact Gate expansion counts, original work/counters/responses and checked retirement.' if small else
            'Match all four historical input bodies, actual work and Gate expansion counts. Use three balanced blocks. '
            'Two faster paired warm prompt times and a median ratio below one flag a gain; keep first-use and warmed '
            'TTFT separate. Adoption, longer decode and physical-context qualification remain open.',
        stop='Stop on the first identity, actual-work, route, validation, ownership, budget or runtime failure. '
             'Keep every eligible sample without replacement, speed-based stopping or reclassifying an earlier outcome.')
    if small:
        # The retained candidate's qsa_real_shapes() uses four-cell pages.
        plan.update(qsa_prefix_page_size=QSA_PREFIX_PAGE_SIZE, previous_failure='df349d38d')

    def environment(row, arm):
        env = same_route.environment(row, dict(arm, index=1))
        env[SELECTOR] = order[arm['index']]
        if small:
            env.update(STRATA_TRACE='1', STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384')
        return env

    def save(row):
        if row['stage'] == 'prepared':
            row.update(plan=plan, whole_bytes=len(order) * row['budgets']['total_cap'] + (8 << 20))
            if not small:
                row['request_hashes'][str(read)] = [w['request'] for w in row['observer_admission']['expected_work']]
            for arm, mode in zip(row['plans'], order, strict=True):
                arm.update(gate_mode=mode, trace_progress=small)
                if small:
                    arm['qsa_prefix_page_size'] = plan['qsa_prefix_page_size']
                    arm['command'].append('--prefill-short-diagnostic')
        base_save(row)

    def current(row):
        require(row['plan'] == plan and row['configuration'] == row['observer_admission']['reference_configuration']
                and [(p['index'], p['read'], p['gate_mode'], p['trace_progress']) for p in row['plans']]
                == [(i, read, m, small) for i, m in enumerate(order)]
                and all(p.get('qsa_prefix_page_size') == plan.get('qsa_prefix_page_size') for p in row['plans']),
                'frozen Gate comparison differs')
        return base_current(row)

    ctx.environment, ctx.current, ctx.save = environment, current, save
    return ctx


if __name__ == '__main__':
    phase, action = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and phase in KEYS, 'select admit or time with the benchmark interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment(phase)
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            require(action in map(str, range(len(ctx.ORDER))), 'select a planned process')
            basis.comparison.run(int(action), ctx)
