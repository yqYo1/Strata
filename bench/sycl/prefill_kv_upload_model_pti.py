"""Attribute V2 117K prefill work with a matched PTI overhead bracket."""

import fcntl
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_pti as fixture

bank, basis, require = fixture.bank, fixture.basis, fixture.require
profile, upload, pti = fixture.profile, fixture.capacity.upload, fixture.pti
model = basis.comparison
READ, RUNS = upload.READ, 2
ROLES = ('clean-before', 'trace', 'clean-after')
ORDER = [(READ, 1)] * len(ROLES)
KEY = 'prefill_kv_upload_model_pti_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
FIXTURE_EVIDENCE = '83da1a72abbf51fe5745c7781395b4e59a661567'
BUDGET = dict(wall=2400, cpu=28800, rss_cap=104 << 30, text_cap=16 << 20,
              file_cap=1280 << 20, total_cap=1600 << 20)
WHOLE_BYTES = 2 << 30
OVERHEAD = (0.8, 1.2)
PLAN = dict(roles=list(ROLES), read=READ, requests=RUNS,
    route='Fix batched upload, 128-slot split, GPU-local direct output, KV reuse and scalar gather.',
    purpose='Treat H2D/M2D activity outside observed kernel intervals as an attribution lead. '
            'Keep copy payload, interval overlap and host/event bounds distinct.',
    acceptance='Same two fresh thinking requests in each process, 24 generated tokens each. '
               'Exact work, clocks, ownership, runtime and current active-from-exec collector. '
               'Compare prompt and TTFT per request position to both clean controls.',
    capture='One trace for two full requests. The earlier two-request capture occupied 581101981 bytes. '
            'Allow a 1280 MiB file for the longer route and 2 GiB total. Reduce and retire it at closure.',
    stopping='Stop on the first correctness, device fault, ownership, clock or finite budget failure. '
             'Finish all eligible arms even if the observer is slow.')


def admitted_basis():
    native, prior, providers, extra = upload.store_basis(RUNS, False)
    gate = basis.validation.committed(bank.RESULT, (fixture.KEY,))
    frozen = bank.previous.historic(FIXTURE_EVIDENCE, fixture.KEY)
    trace = gate['outcomes']['trace']['trace']
    tools = pti.verify_tools(pti.committed_json(pti.RESULT, pti.CHECKOUT))
    inputs = gate['inputs']
    require(gate == frozen and upload.gather.closed(gate)
            and all(o['passed'] and o['ownership_closed'] and not o['fresh_faults'] for o in gate['outcomes'].values())
            and trace['gpu_events'] == 56 and trace['joined_native_events'] and trace['activated_from_exec']
            and inputs['fixture'] == native['binaries']['sycl_expert_bank_test']
            and inputs['engine'] == native['binaries']['strata']
            and inputs['native_source_commit'] == native['native_source_commit']
            and inputs['driver'] == native['driver'] and inputs['boot'] == prior['boot']
            and inputs['native_validation_probe'] == prior['probe'] and inputs['tools'] == tools,
            'exact closed V2 collector and fixture evidence required')
    artifacts = tools['artifacts']
    prior['observer_admission']['collector'] = dict(receipt=FIXTURE_EVIDENCE, sha256=bank.signature(gate),
        launcher=artifacts['unitrace'], collector=artifacts['libunitrace_tool.so'])
    return native, prior, providers, extra


def payload_window(events, bounds):
    low, high = bounds['begin_raw_ns'] / 1000, bounds['end_raw_ns'] / 1000
    clipped, boundary = [], 0
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') not in ('gpu_op', 'cpu_op'):
            continue
        start, stop = event['ts'], event['ts'] + event['dur']
        if start < high and stop > low:
            begin, end = max(start, low), min(stop, high)
            boundary += int(event['cat'] == 'gpu_op' and (begin != start or end != stop))
            clipped.append(dict(event, ts=begin, dur=end - begin))
    gpu = [e for e in clipped if e['cat'] == 'gpu_op']
    kernels = [e for e in gpu if not e['name'].startswith('zeCommandList')]
    uploads = [e for e in gpu if e['name'].startswith('zeCommandListAppendMemoryCopy')
               and any('(' + direction + ')' in e['name'] for direction in ('M2D', 'H2D'))]
    require(kernels and uploads, 'visible device kernels and host-to-device copy activity required')
    upload_us, kernel_us = profile.union_us(uploads), profile.union_us(kernels)
    overlap = upload_us + kernel_us - profile.union_us(uploads + kernels)
    require(-0.01 <= overlap <= min(upload_us, kernel_us) + 0.01, 'copy/kernel interval union failed')
    overlap = min(max(0., overlap), upload_us, kernel_us)
    return dict(clock=bounds, GPU_boundary_commands=boundary,
        host_to_device_union_us=upload_us, kernel_union_us=kernel_us,
        host_to_device_and_kernel_overlap_us=overlap,
        host_to_device_outside_kernel_intervals_us=upload_us - overlap,
        observed_services=profile.family_summary(clipped, bounds['begin_raw_ns'], bounds['end_raw_ns']))


def decision(processes):
    require(len(processes) == len(ROLES) and all(p['passed'] and p['ownership_closed'] and not p['fresh_faults']
        for p in processes) and [p['clean_speed'] for p in processes] == [True, False, True]
        and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 3,
        'three independent valid overhead bracket processes required')
    ratios = []
    for i in range(RUNS):
        observed = processes[1]['samples'][i]
        ratios.append(dict(request=i, scope='first-use' if i == 0 else 'second-request',
            **{key: [observed[key] / p['samples'][i][key] for p in (processes[0], processes[2])]
               for key in ('prompt_ms', 'ttft_s')}))
    eligible = all(OVERHEAD[0] <= value <= OVERHEAD[1] for sample in ratios
        for key in ('prompt_ms', 'ttft_s') for value in sample[key])
    return dict(observer_ratios=ratios, admissible_ratio=list(OVERHEAD),
        normal_route_attribution_qualified=eligible, clean_speedup_qualified=False,
        physical_traffic_qualified=False, prefill_critical_path_qualified=False)


def experiment():
    ctx = basis.core.source.device.load_module('v2_model_prefill_pti', basis.core.__file__)
    base = upload.experiment('store-time')
    ctx.THIS, ctx.ROOT, ctx.KEY = THIS, ROOT, KEY
    ctx.ORDER, ctx.READS, ctx.RUNS, ctx.CLEAN_FAST = ORDER, (READ,) * len(ROLES), RUNS, True
    ctx.basis, ctx.decision = admitted_basis, decision
    ctx.request_route, ctx.check_sample = upload.request_route, base.check_sample
    old_current, old_save, old_native_context = ctx.current, ctx.save, ctx.native_context
    clocks = []

    def pins(revision):
        return base.pins(revision) | bank.stage.source.pin(bank.P, [THIS, fixture.THIS])['source_sha256']

    def environment(row, plan):
        env = base.environment(row, dict(plan, index=1))
        if plan['index'] == 1:
            env.update(UNITRACE_ChromeKernelLogging='1', UNITRACE_ChromeEventBufferSize='128')
        return env

    def current(row):
        require(row['plan'] == PLAN and row['budgets'] == BUDGET and row['whole_bytes'] == WHOLE_BYTES
                and len(row['plans']) == len(ROLES), 'frozen model observer envelope differs')
        assets = {k: row['observer_admission']['collector'][k] for k in ('launcher', 'collector')}
        for i, plan in enumerate(row['plans']):
            require((plan['role'], plan['mode'], plan['direct_output'], plan['trace_progress']) ==
                    (ROLES[i], 'batched', True, False), 'frozen model route differs')
            prof = plan.get('native_profile')
            require(bool(prof) == (i == 1), 'collector enabled outside the middle arm')
            if prof:
                root = Path(plan['root'])
                profile.session_path(prof['session'])
                require({k:v for k,v in prof.items() if k != 'session'} == dict(assets, mode='aggregate',
                    environment=dict(UNITRACE_SystemTime='1', UNITRACE_FollowChildProcess='1',
                                     UNITRACE_TraceOutputDir=str(root / 'trace'), LD_DEBUG='libs')),
                    'frozen collector input differs')
        return old_current(row)

    def native_context(row, plan):
        value = old_native_context(row, plan)
        if 'native_profile' in plan:
            value['native_profile'] = plan['native_profile']
        return value

    def check_environment(observed, expected, plan):
        prof = plan.get('native_profile')
        actual = {k.decode(): v.decode() for k, v in observed.items()}
        if prof:
            profile.session_state(prof['session'], 0)
            native = bank.previous.read(Path(plan['root']) / 'plan.json')
            expected = profile.expected_target_environment(native,
                {k:v for k,v in expected.items() if k != 'LD_PRELOAD'}, prof)
            for key in ('UNITRACE_StartPaused', 'UNITRACE_HostTiming', 'UNITRACE_ChromeCallLogging'):
                expected.pop(key)
            expected['UNITRACE_Verbose'] = '1'
        profile.same_environment(actual, expected)

    def observe_lifetime(outcome, stage, raw):
        clocks.append(profile.clock_pair())

    def observe_gate(outcome, request, stage, raw):
        pid = outcome['native_identity']['pid']
        require(os.readlink(f'/proc/{pid}/ns/time') == os.readlink('/proc/self/ns/time'),
                'native/controller time namespace changed')
        clocks.append(profile.clock_pair())

    def after_execution(outcome, row, plan):
        prof = plan.get('native_profile')
        expected = upload.previous.direct_work(row['observer_admission']['expected_work'])
        require([upload.previous.split.work(r, s) for r, s in
                    zip(outcome['routes'], outcome['samples'], strict=True)] == expected
                and outcome['clean_speed'] == (prof is None), 'actual paced model work or observer differs')
        outcome.update(role=plan['role'], direct_output=True, mode='batched')
        if not prof:
            return
        root = Path(plan['root'])
        trace, = (root / 'trace').rglob('*.json')
        require(0 < trace.stat().st_size <= BUDGET['file_cap']
                and len(clocks) == 2 + 2 * RUNS, 'complete bounded trace and clocks required')
        log = (root / 'engine.stderr').read_text()
        require(not any(tag in log for tag in ('[WARNING]', '[ERROR]')),
                'collector diagnostic reported a warning or error')
        events = json.loads(trace.read_text())['traceEvents']
        whole = profile.trace_summary(events, outcome['native_identity']['pid'],
                    clocks[0]['raw_before_ns'], clocks[-1]['raw_after_ns'])
        results, = (root / 'bench').glob('*/results.json')
        samples = bank.previous.read(results)['results']['prefill_only']['requests']
        windows = [payload_window(events, ctx.profile_window(s['client_interval'], clocks)) for s in samples]
        outcome['resource_observation'] = dict(trace_bytes=trace.stat().st_size, trace_sha256=bank.sha(trace),
            brackets=clocks, full_GPU_events=whole['gpu_events'], full_host_events=whole['host_events'], windows=windows)

    def save(row):
        if row['stage'] == 'prepared':
            assets = {k: row['observer_admission']['collector'][k] for k in ('launcher', 'collector')}
            row.update(plan=PLAN, budgets=BUDGET, whole_bytes=WHOLE_BYTES)
            row['request_hashes'][str(READ)] = [w['request'] for w in row['observer_admission']['expected_work']]
            for plan, role in zip(row['plans'], ROLES, strict=True):
                plan.update(role=role, mode='batched', direct_output=True, trace_progress=False)
                root = Path(plan['root'])
                if role == 'trace':
                    (root / 'trace').mkdir(mode=0o700)
                    prof = plan['native_profile'] = dict(profile.profiling(root, assets), mode='aggregate')
                    path = root / 'plan.json'
                    bank.stage.write(path, dict(bank.previous.read(path), native_profile=prof))
                    plan['generated']['plan.json'] = bank.sha(path)
                plan['environment_sha256'] = model.fingerprint(environment(row, plan))
        if row['processes'] and row['processes'][-1]['index'] == 1:
            process = row['processes'][-1]
            plan = row['plans'][1]
            if 'collector_cleanup' not in process:
                try:
                    cleanup = ctx.close_profile_session(row, plan)
                    process['collector_cleanup'] = cleanup
                    require(cleanup['control_closed'] and cleanup['session_removed'] and not cleanup.get('error'),
                            'collector session cleanup failed')
                except BaseException as error:
                    process.update(passed=False, error=process.get('error', repr(error)), ownership_closed=False)
                    row.update(stage='failed', passed=False, complete=False, ownership_closed=False)
                    row.pop('decision', None)
        old_save(row)

    ctx.pins, ctx.environment, ctx.current, ctx.save = pins, environment, current, save
    ctx.native_context, ctx.check_native_environment = native_context, check_environment
    ctx.observe_lifetime, ctx.observe_gate, ctx.after_execution = observe_lifetime, observe_gate, after_execution
    ctx.native_extra_maps = lambda plan: {plan['native_profile']['collector']['path']} if 'native_profile' in plan else set()
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', '0', '1', '2'),
            'select one frozen observer stage with the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            model.host(ctx)
        else:
            plan = bank.previous.read(ROOT / 'record.json')['plans'][int(action)]
            if 'native_profile' in plan:
                require(not profile.session_path(plan['native_profile']['session']).exists(), 'collector already active')
            model.run(int(action), ctx)
