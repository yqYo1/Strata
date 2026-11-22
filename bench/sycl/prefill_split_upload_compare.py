"""Admit and compare exact split-prefill upload modes on one built engine."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_split_bank_compare as split
import prefill_split_upload_build as build
from prefill_prefix_expert_measure import timing
import prefill_prefix_expert_profile as host_profile

bank, basis, model, require = split.bank, split.basis, split.model, split.require
speed, workspace = split.speed, split.workspace
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READ, SLOTS = 117489, 128
REFERENCE = '7c88fb0123a76b35e2dc91e597a19439bd2ae5ca'
CLOSED_TIMING = '982211f5c91b4251ff89b8ad93f1f369a46fbad0'
STORE_FIXTURE_EVIDENCE = '90c572ca98efb28c1578fa23b6cb8cd13e103efe'
STORE_FIXTURE_KEY = 'prefill_split_direct_store_component_v1'
MODES = ('demand', 'batched', 'coalesced')
BLOCKS = (MODES, MODES[1:] + MODES[:1], MODES[2:] + MODES[:2])
CONTRASTS = (('batched', 'demand'), ('coalesced', 'batched'), ('coalesced', 'demand'))
UPLOAD = 'STRATA_SYCL_PREFILL_SPLIT_EXPERT_UPLOAD'
KEYS = {'admit': 'prefill_split_upload_admission_v1', 'time': 'prefill_split_upload_speed_v1',
        'profile': 'prefill_split_upload_cpu_wait_v1', 'store-admit': 'prefill_split_direct_store_admission_v1'}
MAX_OBSERVER_RATIO = 1.03


def admission(traced, runs):
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    fixture = native['binaries']['sycl_expert_bank_test']
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['host_control']['ownership_closed']
            and fixture['sha256'] == 'dd69762b225b612ba88ae1cf3542b0e9e0dcefcb6ab4f4b9db1c165c270b1326',
            'the checked engine and exact device-tested expert fixture are required')
    parent = bank.previous.historic(REFERENCE, 'decode_kv_boundary_preservation_117489_v1')
    require(parent['stage'] == 'closed' and parent['passed'] and parent['complete']
            and parent['ownership_closed'] and not parent['active']
            and parent['observer_admission']['native_builds']['boundary']['binary'] == native['projection']['parent_binary']
            and parent['observer_admission']['prefill_arm'] == 'reuse',
            'recover the measured boundary workload and scalar decode reference')
    process = parent['processes'][1]
    original = [split.work(dict(r, requested_bank_H2D_bytes=r['fast_window']['requested_bank_H2D_bytes']), s)
                for r, s in zip(process['routes'], process['samples'], strict=True)]
    require(process['kernel'] == 'boundary' and len(original) == 4
            and [x['request'] for x in original] == parent['request_hashes'][str(READ)],
            'require the exact four existing 117K request bodies')
    prior = bank.previous.read(basis.reference.model.ROOT / 'record.json')
    require(prior['probe'] == parent['native_validation_probe'] and prior['boot'] == parent['boot']
            and prior['driver'] == parent['driver'] == native['driver']
            and bank.signature(prior['models']) == parent['model_assets_sha256']
            and prior['runtime_manifest']['sha256'] == parent['runtime_manifest']['sha256']
            and bank.sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
            and bank.previous.read(Path(prior['runtime_manifest']['path'])) == prior['runtime_files'],
            'keep the original assets, providers and workstation state')
    old = parent['observer_admission']
    prior['observer_admission'] = dict(fast_configuration_diagnostic=old['fast_configuration_diagnostic'],
        ordinary_control=old['ordinary_control'], reference_configuration=parent['configuration'],
        expected_work=original[:runs], build_sha256=bank.signature(build.builder.compact(native)),
        traced=traced, reference=REFERENCE)
    if not traced:
        gate = basis.validation.committed(bank.RESULT, (KEYS['admit'],))
        require(gate['stage'] == 'closed' and gate['passed'] and gate['complete']
                and gate['ownership_closed'] and not gate['active']
                and gate['binary'] == native['binaries']['strata']
                and gate['native_source_commit'] == native['native_source_commit']
                and gate['observer_admission']['build_sha256'] == prior['observer_admission']['build_sha256']
                and gate['decision']['matched_decode_loan'] and len(gate['processes']) == len(MODES)
                and [p['mode'] for p in gate['processes']] == list(MODES),
                'finish the same-engine model-route and decode loan admission first')
        prior['observer_admission']['traced_admission_sha256'] = bank.signature(gate)
    return native, prior, prior['runtime_files'], None


def request_route(text, plan):
    mode = plan['mode']
    route = split.request_route(text, dict(plan, bank=SLOTS))
    records = workspace.records(text, 'strata MoE split expert upload: ')
    experts = route['uploads']
    if mode == 'demand':
        require(not records, 'original demand path emitted a batched upload')
        copies = experts
    else:
        record, = records
        copies = int(record['copy_commands'])
        require(record['windows'] == '48' and int(record['experts']) == experts
                and record['final_drain'] == 'checked', 'actual batched upload or retirement differs')
        runs = min(experts, 48 * 513 - experts)
        upper = runs + (experts - runs) // 16
        require(copies == experts if mode == 'batched' else 0 < copies <= upper < experts,
                'actual copy commands differ from the selected transport')
    route['split_upload'] = dict(mode=mode, experts=experts, copy_commands=copies)
    return route


def check_work(outcome, row, plan):
    require([split.work(r, s) for r, s in zip(outcome['routes'], outcome['samples'], strict=True)]
            == row['observer_admission']['expected_work']
            and outcome['clean_speed'] == (not plan['trace_progress']),
            'exact model work, prefill route or timing scope differs')
    outcome['mode'] = plan['mode']
    if plan['trace_progress'] and plan['index']:
        demand = row['processes'][0]
        require([r['loaned_slots'] for r in outcome['routes']] == [r['loaned_slots'] for r in demand['routes']],
                'the same engine restored a different decode expert loan')


def profile_basis(runs):
    native, prior, providers, extra = admission(False, runs)
    gate = basis.validation.committed(bank.RESULT, (KEYS['time'],))
    require(gate == bank.previous.historic(CLOSED_TIMING, KEYS['time'])
            and gate['stage'] == 'closed' and gate['passed'] and gate['complete']
            and gate['ownership_closed'] and not gate['active']
            and gate['binary'] == native['binaries']['strata']
            and gate['native_source_commit'] == native['native_source_commit']
            and gate['observer_admission']['build_sha256'] == prior['observer_admission']['build_sha256']
            and gate['decision']['repeated_warm_gain']['batched/demand']
            and not gate['decision']['repeated_warm_gain']['coalesced/batched'],
            'recover the finished same-build batch placement comparison')
    prior['observer_admission']['timing_evidence'] = CLOSED_TIMING
    prior['observer_admission']['timing_sha256'] = bank.signature(gate)
    return native, prior, providers, extra


def store_basis(runs):
    native, prior, providers, extra = profile_basis(runs)
    fixture = basis.validation.committed(bank.RESULT, (STORE_FIXTURE_KEY,))
    plan = fixture['plan']
    require(fixture == bank.previous.historic(STORE_FIXTURE_EVIDENCE, STORE_FIXTURE_KEY)
            and fixture['stage'] == 'closed' and fixture['passed'] and fixture['complete']
            and fixture['ownership_closed'] and not fixture['active']
            and plan['engine'] == native['binaries']['strata']
            and plan['fixture'] == native['binaries']['sycl_prefill_attention_bank_test']
            and plan['native_build_sha256'] == prior['observer_admission']['build_sha256']
            and plan['native_source_commit'] == native['native_source_commit']
            and plan['boot'] == prior['boot'] and plan['driver'] == native['driver']
            and fixture['result']['cases'] == 4 and not fixture['result']['fresh_faults'],
            'exact same-build BF16, bounds and cancellation fixture required')
    admission_row = basis.validation.committed(bank.RESULT, (KEYS['admit'],))
    copy = admission_row['processes'][MODES.index('batched')]
    observer = prior['observer_admission']
    original = [split.work(r, s) for r, s in zip(copy['routes'], copy['samples'], strict=True)]
    require(copy['mode'] == 'batched' and copy['passed'] and copy['ownership_closed']
            and copy['native_exit_code'] == 0 and not copy['fresh_faults']
            and original == observer['expected_work']
            and all(item['store']['pieces'] == 720 and item['store']['direct'] == 0
                    and item['store']['copy_bytes'] == 115495403520 for item in original),
            'same-build batched copy admission and exact store geometry required')
    observer.update(traced=True, direct_store_component_sha256=bank.signature(fixture),
                    expected_loaned_slots=[r['loaned_slots'] for r in copy['routes']],
                    copied_D2D_bytes=[item['store']['copy_bytes'] for item in original],
                    expected_work=[dict(item, store=dict(item['store'], direct=item['store']['pieces'], copy_bytes=0))
                                   for item in original])
    return native, prior, providers, extra


def check_direct_store(outcome, row, plan):
    check_work(outcome, row, plan)
    require([r['loaned_slots'] for r in outcome['routes']] == row['observer_admission']['expected_loaned_slots'],
            'direct store restored a different decode expert loan')


def profiled_route(text, plan):
    route = request_route(text, plan)
    if plan['cpu_wait_scopes']:
        route['cpu_partition'] = host_profile.cpu_partition(text)
    else:
        require(all(scope not in text for scope in host_profile.SCOPES), 'CPU scopes appeared in a clean request')
    return route


def check_profile_work(outcome, row, plan):
    check_work(outcome, row, plan)
    outcome.update(role=plan['role'], clean_speed=not plan['cpu_wait_scopes'])


def profile_decision(ctx, processes):
    require([p['role'] for p in processes] == list(host_profile.ROLES)
            and [p['clean_speed'] for p in processes] == [True, False, True]
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 3
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                    and not p['fresh_faults'] and p['native_exit_code'] == 0 for p in processes),
            'complete independent CPU-accounting bracket required')
    result = ctx.profile_decision(processes)
    eligible = all(1 / MAX_OBSERVER_RATIO <= value <= MAX_OBSERVER_RATIO
                   for point in result['observer_ratios'] for field in ('prompt', 'client_TTFT')
                   for value in point[field])
    result.update(maximum_observer_ratio=MAX_OBSERVER_RATIO,
                  observer_cost_qualified=eligible, normal_route_attribution_qualified=eligible,
                  limits='Calling-thread accounting of host regions and instrumented waits only. '
                         'Upload submission can contain provider blocking. Physical bus traffic and '
                         'GPU saturation remain unmeasured.')
    return result


def decision(processes, traced, order):
    identities = {(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}
    require(tuple(p['mode'] for p in processes) == order and len(identities) == len(order)
            and all(p['complete'] and p['passed'] and p['ownership_closed'] and not p['fresh_faults']
                    and p['native_exit_code'] == 0 and p['clean_speed'] == (not traced) for p in processes),
            'all planned independent processes must close successfully')
    if traced:
        return dict(matched_decode_loan=True, admitted_modes=list(order), clean_speed_qualified=False)
    rates = {arm: [] for arm in MODES}
    ratios = {a + '/' + b: [] for a, b in CONTRASTS}
    for i in range(0, len(order), len(MODES)):
        values = {p['mode']: timing(p) for p in processes[i:i + len(MODES)]}
        for arm, clocks in values.items():
            rates[arm].append(1000 * READ / clocks['warm_prompt_ms'])
        for a, b in CONTRASTS:
            ratios[a + '/' + b].append({k: values[a][k] / values[b][k] for k in values[a]})
    gains = {}
    for name, pairs in ratios.items():
        times = [p['warm_prompt_ms'] for p in pairs]
        gains[name] = sum(t < 1 for t in times) >= 2 and statistics.median(times) < 1
    return dict(warm_process_rates=rates, paired_time_ratios=ratios, repeated_warm_gain=gains,
                repeated_1000={arm: min(values) >= 1000 for arm, values in rates.items()},
                model_qualified=False, adopted=False)


def experiment(action):
    direct_store = action == 'store-admit'
    traced = action == 'admit' or direct_store
    profiled = action == 'profile'
    if direct_store:
        order = ('batched',)
    elif profiled:
        order = ('batched',) * len(host_profile.ROLES)
    else:
        order = MODES if traced else tuple(arm for block in BLOCKS for arm in block)
    runs = 2 if traced or profiled else 4
    ctx = basis.core.source.device.load_module('split_upload_' + action, basis.core.__file__)
    ctx.THIS, ctx.KEY, ctx.ROOT = THIS, KEYS[action], bank.B / (KEYS[action].replace('_', '-'))
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (READ,) * len(order), [(READ, 1)] * len(order), runs, True
    ctx.basis = lambda: admission(traced, runs)
    ctx.check_sample = basis.reference.context().check_sample
    ctx.request_route, ctx.after_execution = request_route, check_work
    ctx.decision = lambda processes: decision(processes, traced, order)
    old_pins, old_save, old_current = ctx.pins, ctx.save, ctx.current
    protocol = dict(order=list(order), requests=runs, trace=traced, bank=SLOTS,
                    acceptance='Match actual work and checked decode loan. For each contrast, use at least two '
                               'faster warm prompt times and a paired median below one. Keep both TTFT scopes.')
    if profiled:
        ctx.basis = lambda: profile_basis(runs)
        ctx.request_route, ctx.after_execution = profiled_route, check_profile_work
        ctx.decision = lambda processes: profile_decision(ctx, processes)
        protocol.update(roles=list(host_profile.ROLES), maximum_observer_ratio=MAX_OBSERVER_RATIO,
                        acceptance='Match actual work and close all owners. Compare first and second requests '
                                   'to both clean controls. Use a two-sided 3% observer screen for attribution.')

        def validate_model_text(text, plan):
            count = runs if plan['cpu_wait_scopes'] else 0
            require(all(sum(line.startswith(scope) for line in text.splitlines()) == count
                        for scope in host_profile.SCOPES), 'extra or missing CPU accounting scope')

        ctx.validate_model_text = validate_model_text
    if direct_store:
        ctx.basis = lambda: store_basis(runs)
        ctx.after_execution = check_direct_store
        protocol.update(direct_output=True, acceptance=(
            'Match the same-build traced batched copy work and decode loan for both inputs. '
            'Require 720 direct pieces and zero output copies per input.'))

    def pins(revision):
        values = old_pins(revision)
        paths = [split.THIS, build.THIS, speed.THIS, speed.capacity.THIS, speed.dispatch.THIS,
                 speed.dispatch.split.THIS, *speed.state.SCRIPTS]
        if profiled:
            paths.append(host_profile.THIS)
        values.update(bank.stage.source.pin(bank.P, paths)['source_sha256'])
        return values

    def environment(row, plan):
        env = model.ordinary_environment(row, plan['root'], row['observer_admission']['ordinary_control'])
        env.update({UPLOAD: order[plan['index']], split.SLOT_OPTION: str(SLOTS),
                    model.INPUT: '0', model.OUTPUT: str(int(direct_store)), speed.state.KV: '1', speed.state.GATHER: '0',
                    'STRATA_SYCL_KV_COPY_GROUPS': '0', 'STRATA_SYCL_PREFILL_SPLIT_SCRATCH_CHECK_ONCE': '0'})
        require('STRATA_TRACE' not in env and 'STRATA_DECODE_TIMING' not in env
                and env.get('STRATA_SYCL_CACHE_LOAN_AUDIT', '0') == '0', 'unexpected observer')
        if traced:
            env.update(STRATA_TRACE='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1')
        if profiled:
            require(env.get(host_profile.SELECTOR, '0') == '0', 'unexpected CPU accounting flag')
            env[host_profile.SELECTOR] = str(int(plan['index'] == 1))
        return env

    def save(row):
        if row['stage'] == 'prepared':
            for plan, mode in zip(row['plans'], order, strict=True):
                plan.update(mode=mode, trace_progress=traced)
                if profiled:
                    plan.update(role=host_profile.ROLES[plan['index']], cpu_wait_scopes=plan['index'] == 1)
                if direct_store:
                    plan['direct_output'] = True
            row.update(whole_bytes=len(order) * row['budgets']['total_cap'] + (8 << 20), plan=protocol)
        old_save(row)

    def current(row):
        require(row['configuration'] == row['observer_admission']['reference_configuration']
                and row['plan'] == protocol
                and [(p['index'], p['read'], p['mode'], p['trace_progress']) for p in row['plans']]
                    == [(i, READ, mode, traced) for i, mode in enumerate(order)],
                'frozen split-upload configuration, order or arm differs')
        if profiled:
            require([(p['role'], p['cpu_wait_scopes']) for p in row['plans']]
                    == [(role, i == 1) for i, role in enumerate(host_profile.ROLES)],
                    'frozen CPU accounting roles differ')
        if direct_store:
            require(all(p['direct_output'] is True for p in row['plans']), 'frozen direct output differs')
        return old_current(row)

    ctx.pins, ctx.environment, ctx.current, ctx.save = pins, environment, current, save
    return ctx


if __name__ == '__main__':
    phase, action = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and phase in KEYS,
            'use the benchmark environment and select admit, time, profile or store-admit')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment(phase)
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            require(action in map(str, range(len(ctx.ORDER))), 'select one planned process')
            basis.comparison.run(int(action), ctx)
