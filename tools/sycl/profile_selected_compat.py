#!/usr/bin/env python3
"""Qualify the owned collector on the selected production graph adapter.

Main-authored composition of this project's checked supervision, environment
and collector readers; repository LICENSE. Reuse the selected fixture ELF.
This first control keeps graph construction active: it cannot qualify a
paused-construction window, model phase capture, overhead or a speedup.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import sys
import time

sys.dont_write_bytecode = True
import cache_map_cpu_census as census
import cache_map_worker_scaling as scaling
import profile_split as profile

entry, base = census.entry, census.base
P, B, RESULT = census.P, census.B, census.RESULT
ROOT = entry.model.ROOT / 'compat-collector-control-v1'
KEY = 'cache_map_compat_collector_control_v1'
CONTROL_ROOT, CONTROL_KEY = ROOT, KEY
ACTIVE_FROM_EXEC = False
SOURCE = Path(__file__).resolve()
require, digest, write = base.require, base.digest, base.write
PHASES = ('before-runtime', 'runtime-ready')
BUDGET = dict(wall=180, cpu=180, rss_cap=4 << 30,
              text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)


def compact(row):
    out = {k: v for k, v in row.items()
           if k not in ('sources', 'environment_template')}
    for key in ('sources', 'environment_template'):
        out[key + '_sha256'] = hashlib.sha256(
            json.dumps(row[key], sort_keys=True).encode()).hexdigest()
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def basis():
    data = base.committed(RESULT)
    sweep = json.loads((scaling.ROOT / 'record.json').read_text())
    require(data[scaling.KEY] == scaling.maintained.compact(sweep)
            and sweep['passed'] and sweep['complete']
            and sweep['ownership_closed'] and not sweep['active']
            and len(sweep['processes']) == 3,
            'close and commit all three worker cases before profiling')
    native, _, providers = entry.proof()
    prior = json.loads((census.ROOT / 'record.json').read_text())
    require(data[census.KEY] == census.context().compact(prior)
            and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active']
            and prior['binary'] == native['binaries']['strata']
            and prior['native_source_commit'] == native['source_commit'],
            'closed unchanged selected-engine census qualification required')
    assets = profile.collector_assets(data)
    fixture = native['binaries']['sycl_submission_test']
    require(digest(fixture['path']) == fixture['sha256']
            and Path(fixture['path']).stat().st_size == fixture['bytes'],
            'selected compatibility fixture changed')
    if ACTIVE_FROM_EXEC:
        control = json.loads((CONTROL_ROOT / 'record.json').read_text())
        require(data[CONTROL_KEY] == compact(control) and control['passed']
                and control['complete'] and control['ownership_closed'] and not control['active']
                and control['binary'] == fixture and control['assets'] == assets,
                'commit the closed same-ELF active-construction control first')
        old = base.basis.build.git(P, 'show', control['source_commit'] + ':' + str(SOURCE.relative_to(P)))
        require(hashlib.sha256(old).hexdigest() == control['sources'][str(SOURCE.relative_to(P))],
                'original control policy is not recoverable')
    return native, prior, providers, assets


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row),
            'commit the exact compatibility collector plan')
    require(base.controls(row['source_commit'], controller=SOURCE) == row['sources'],
            'collector control source changed')
    native, prior, providers, assets = basis()
    require(native['source_commit'] == row['native_source_commit']
            and native['binaries']['strata'] == row['engine']
            and native['binaries']['sycl_submission_test'] == row['binary']
            and prior['native_validation_probe'] == row['native_validation_probe']
            and prior['environment'] == row['environment_template']
            and prior['boot'] == row['boot'] and prior['driver'] == row['driver']
            and assets == row['assets'], 'selected engine/runtime/collector changed')
    return providers


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT),
            'fresh compatibility control only; no retries')
    native, prior, providers, assets = basis()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = base.controls(revision, controller=SOURCE)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache', 'trace'):
        (ROOT / name).mkdir()
    row = dict(stage='prepared', source_commit=revision, sources=pins,
        native_source_commit=native['source_commit'], engine=native['binaries']['strata'],
        binary=native['binaries']['sycl_submission_test'],
        native_validation_probe=prior['native_validation_probe'],
        environment_template=prior['environment'], boot=prior['boot'], driver=prior['driver'],
        assets=assets, profile=profile.profiling(ROOT, assets), cases=[],
        plan=dict(question='Does the currently admitted owned collector cover the selected non-native-recording compatibility GraphExec with its real completion/lifetime rules?',
            native='Reuse selected479a/552b engine and same-build compatibility ELF;no rebuild. Run the no-argument compatibility graph case,not --native or the submission-only early return. cudaStreamBeginCapture uses normal command_graph recording;GraphExec finalizes without per-node profiling. STRATA_SYCL_VERIFY_NATIVE_CAPTURE remains0.',
            order='One fresh finite process. Start the named collector paused;resume before runtime initialization,then keep collection active through capture,replays,late launch,resource drain and normal native exit. Exact before-runtime/runtime-ready gates;no mid-run environment mutation.',
            acceptance='Verify committed source,original selected native inputs,actual engine/fixture/collector/runtime bytes,startup environment and provider mappings. Require the original numerical/event/copy/graph/host-handoff/lifetime pass,normal owned exit,quiet parameter validation,no fresh fault,all62 kernel instances with native host-flow joins and names,finite contained raw-clock intervals,closed controls and removed named session.',
            limits='Active-construction control only. No paused graph construction,lossless boundary,decode-only model capture,observer overhead,physical traffic,hardware saturation,performance gain or inference adoption claim.',
            next='Use the same actual compatibility route for a separate paused-construction/flush qualification before claiming an interior decode window. Reuse the existing launcher bootstrap and trace reducers.',
            stopping='Stop first source,ownership,provider,validation,fault,budget,numerical,count or join failure. No retry,hash refresh or changed criteria.',
            retention='Keep compact counts,clock/identity/qualification limits and diagnostic outcome. Retire successful full API trace,timeline,JIT and execution bookkeeping after joins and next-consumer needs are recorded. No new native generation.'),
        budgets=BUDGET, whole_bytes=128 << 20,
        passed=False, complete=False, active=False, ownership_closed=True)
    if ACTIVE_FROM_EXEC:
        row['profile']['mode'] = 'aggregate'
        row['plan'].update(
            question='Does the existing active-from-exec launcher mode preserve all compatibility graph/kernel flow joins with only kernel-activity Chrome logging?',
            order='One new same-ELF process,after the closed control. Use the existing aggregate launcher mode(device-timing/verbose,no --start-paused),plus immutable UNITRACE_ChromeKernelLogging1 and128-event buffer. No host-timing or full Chrome-call selector. Require the named session already active at both native gates and stay active through native exit;no in-application collection toggle.',
            limits='Collector-mode qualification only. Host API timeline coverage may be limited to kernel activities;its observed intervals are not total submission time or useful CPU time. No paused-construction,interior model window,observer overhead,clean speed,physical traffic,saturation or adoption claim.',
            next='A separately committed bounded fully active model capture can use existing benchmark client first-output/end monotonic timestamps with qualified raw-clock mapping to reduce only interior decode GPU services. Do not infer full model coverage or overhead from this fixture.')
    save(row)
    print('Prepared selected compatibility graph collector control;commit before execution.', flush=True)


def environment(row):
    old = row['environment_template']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(ROOT)) for k, v in row['environment_template'].items()}
    env.update(STRATA_TEST_MAIN_GATES='1',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(ROOT / 'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['EnableDirectSubmission'] == '0'
            and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0'
            and env['ZES_ENABLE_SYSMAN'] == '0', 'qualified safety/graph settings differ')
    if ACTIVE_FROM_EXEC:
        require(not any(k.startswith('UNITRACE_') for k in env),
                'unexpected inherited collector selector')
        env.update(UNITRACE_ChromeKernelLogging='1', UNITRACE_ChromeEventBufferSize='128')
    return env


def expected_environment(row, env, prof):
    expected = profile.expected_target_environment(row, env, prof)
    if ACTIVE_FROM_EXEC:
        for key in ('UNITRACE_StartPaused', 'UNITRACE_HostTiming', 'UNITRACE_ChromeCallLogging'):
            expected.pop(key)
        expected['UNITRACE_Verbose'] = '1'
    return expected


def check():
    row = json.loads((ROOT / 'record.json').read_text())
    providers = current(row)
    require(row['stage'] == 'prepared' and not row['cases']
            and not (ROOT / 'execution.json').exists(), 'fresh control only')
    frozen, env, prof = copy.deepcopy(row), environment(row), row['profile']
    require(not profile.session_path(prof['session']).exists(), 'preexisting collector session')
    target = profile.profile_target_environment(row, env, prof)
    expected = expected_environment(row, env, prof)
    owner = controls = None
    bounds, phases, mapped = {}, [], set()
    outcome = dict(passed=False, complete=False, host_probe_accepted=False)
    cursor = profile.cursor()

    def persist():
        write(ROOT / 'execution.json', dict(outcome,
            active=bool((owner and owner.active) or (controls and controls.active)),
            commands=owner.commands if owner else [], controls=controls.commands if controls else []))

    def control(action, state=None):
        command = [prof['launcher']['path'], '--' + action, prof['session']]
        command_entry, _, _ = controls.run(action, command, env, P,
            wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20,
            file_cap=32 << 20, total_cap=64 << 20)
        require(base.supervision.closed(command_entry) and command_entry['exit_code'] == 0,
                'collector session control failed')
        if state is not None:
            profile.session_state(prof['session'], state)

    def interact(io, native):
        for phase in PHASES:
            while True:
                line = io.line(45)
                if line.startswith('COMPONENT_READY '):
                    break
            require(line == 'COMPONENT_READY ' + phase, 'fixture gate order differs')
            require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole output budget')
            phases.append(phase)
            proc = Path('/proc', str(native['pid']))
            require((proc / 'exe').resolve() == Path(row['binary']['path']).resolve(),
                    'fixture executable differs')
            actual = dict(v.split('=', 1) for v in (proc / 'environ').read_text().split('\0') if '=' in v)
            profile.same_environment(actual, expected)
            seen = entry.template.shared.runtime.executable_mappings((proc / 'maps').read_text())
            require(seen <= set(providers) | {row['binary']['path'], prof['collector']['path']},
                    'unexpected fixture provider')
            mapped.update(seen)
            if phase == 'before-runtime':
                profile.session_state(prof['session'], 0 if ACTIVE_FROM_EXEC else 1)
                if not ACTIVE_FROM_EXEC:
                    bounds['begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    control('resume', 0)
            profile.session_state(prof['session'], 0)
            persist()
            io.send(b'X')
        io.proc.stdin.close()
        deadline = time.monotonic() + 45
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'fixture teardown deadline')
            require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole output budget')
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)

    try:
        owner = base.supervision.Owner(ROOT / 'owned', persist)
        controls = base.supervision.Owner(ROOT / 'controls', persist)
        command = entry.template.engine_once.profile_command(prof, [row['binary']['path']])
        bounds['begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        native, out, err = owner.run('compat', command, target, P, interaction=interact, **BUDGET)
        bounds['end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        require(base.supervision.closed(native) and native['exit_code'] == 0
                and phases == list(PHASES), 'compatibility fixture failed')
        stdout, stderr = out.read_text(), err.read_text()
        require(stdout.count('SYCL host adapter: events, profiling, graph replay, copies and lifecycle passed') == 1,
                'original graph/lifetime numerical check missing')
        activation = entry.template.shared.runtime.validation.preload_evidence(stderr, native['pid'])
        required = {row['native_validation_probe']['loader']['path'],
                    row['native_validation_probe']['library']['path'],
                    prof['collector']['path'], env['ZE_ENABLE_ALT_DRIVERS']}
        require(required <= mapped and activation['loader'] == row['native_validation_probe']['loader']['path']
                and not any(v in stderr for v in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]'))
                and (not (ROOT / 'validation.log').exists() or not (ROOT / 'validation.log').stat().st_size),
                'collector/validation/provider error')
        traces = list((ROOT / 'trace').rglob('*.json'))
        require(len(traces) == 1, 'one fully flushed compatibility trace required')
        events = json.loads(traces[0].read_text())['traceEvents']
        summary = profile.trace_summary(events, native['pid'], bounds['begin'], bounds['end'])
        require(summary['operation_counts'].get('kernel') == 62, 'compatibility kernel inventory differs')
        kernels = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op'
                   and not e.get('name', '').startswith('zeCommandList')]
        require(len(kernels) == 62 and all(type(e.get('name')) is str and e['name'].strip()
                and not any(v in e['name'].lower() for v in ('unknown', 'unnamed'))
                for e in kernels), 'named compatibility kernel coverage missing')
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0,
            validation=activation, trace=dict(expected_kernels=62, observed_kernels=62,
                native_PID=native['pid'], GPU_events=summary['gpu_events'],
                host_events=summary['host_events'], collection_active_from_exec=ACTIVE_FROM_EXEC,
                native_host_flow_joins_unique=True, all_kernel_names_present=True,
                timeline_sha256=digest(traces[0]),
                timeline_bytes=traces[0].stat().st_size), clocks=bounds)
        current(frozen)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner and owner.active is None and profile.session_path(prof['session']).exists():
            try:
                control('stop')
            except BaseException as error:
                outcome['control_close_error'] = repr(error)
        if owner:
            entry.template.shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        closed = bool(controls and controls.active is None
                      and all(base.supervision.closed(v) for v in controls.commands))
        outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and closed)
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']
            and outcome['ownership_closed'] and outcome.get('driver_after') == row['driver']
            and 'control_close_error' not in outcome and not profile.session_path(prof['session']).exists())
        row['cases'].append(outcome)
        row.update(stage='closed' if outcome['passed'] else 'failed',
            passed=outcome['passed'], complete=outcome['passed'],
            active=bool((owner and owner.active) or (controls and controls.active)),
            ownership_closed=outcome['ownership_closed'])
        save(row)
    require(outcome['passed'], outcome.get('error', 'compatibility collector control failed'))
    print('Selected compatibility graph collector control passed;paused/model windows remain unqualified.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) in (2, 3) and sys.argv[1] in ('prepare', 'check')
            and (len(sys.argv) == 2 or sys.argv[2] == 'active'), 'use prepare|check [active]')
    if len(sys.argv) == 3:
        ACTIVE_FROM_EXEC = True
        ROOT = entry.model.ROOT / 'compat-collector-active-v1'
        KEY = 'cache_map_compat_collector_active_v1'
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare': prepare, 'check': check}[sys.argv[1]]()
