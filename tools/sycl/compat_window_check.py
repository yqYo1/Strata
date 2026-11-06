#!/usr/bin/env python3
"""Bound the owned collector's pause/resume behavior on the actual SYCL adapter.

Main-authored; repository LICENSE. No inference changes or model execution.
Graph construction stays active; only fully drained replay groups are paused.
"""
import collections
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import sys
import time

sys.dont_write_bytecode = True
import compat_window_build as build

P, B, RESULT, base = build.P, build.B, build.RESULT, build.base
profile = build.collector.previous.profile
SELF = Path(__file__).resolve()
ROOT = build.ROOT / 'checks-v1'
KEY = 'compat_collector_window_check_v1'
MODES = ('smoke', 'prefix', 'window')
PHASES = ('before-runtime', 'runtime-ready', 'prefix-drained', 'gap-drained',
          'window-drained', 'suffix-drained', 'before-exit')
BUDGET = dict(wall=90, cpu=90, rss_cap=4 << 30, text_cap=8 << 20,
              file_cap=16 << 20, total_cap=64 << 20)
require, digest, write = build.require, build.digest, build.write


def compact(row):
    out = {k: v for k, v in row.items()
           if k not in ('sources', 'providers', 'environment_template')}
    out['inputs'] = {k: dict(count=len(row[k]), sha256=build.width.signature(row[k]))
                     for k in ('sources', 'providers', 'environment_template')}
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def basis():
    built = json.loads((build.ROOT / 'record.json').read_text())
    native, _ = build.current(built)
    require(built['stage'] == 'built' and built['passed'] and built['complete'] and
            built['ownership_closed'] and not built['active'] and
            not built['GPU_executed'] and not built['model_executed'],
            'commit the closed fixture build first')
    binary = built['outcome']['binary']
    require(digest(binary['path']) == binary['sha256'] and
            Path(binary['path']).stat().st_size == binary['bytes'], 'fixture executable changed')
    active = json.loads((build.collector.ROOT / 'record.json').read_text())
    checked, _, providers = build.width.short.model_proof()
    require(checked['binaries'] == native['binaries'] and
            active['engine'] == native['binaries']['strata'], 'native/runtime basis differs')
    assets = profile.collector_assets(base.committed(RESULT))
    require(assets == active['assets'], 'collector differs from the active62 control')
    return built, active, providers, assets


def coverage(events, clocks, mode):
    """Require every observed GPU operation inside an actually active interval."""
    require(mode in ('prefix', 'window'), 'invalid collector-window mode')
    gpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op']
    kernels = [e for e in gpu if not e['name'].startswith('zeCommandList')]
    intervals = {'prefix': (clocks['begin'], clocks['prefix-drained']['arrival'])}
    if mode == 'window':
        intervals['window'] = (clocks['gap-drained']['release'],
                               clocks['window-drained']['arrival'])
    counts = {k: collections.Counter() for k in intervals}
    tolerance_ns = 1_000_000  # Same clock tolerance as the owned whole-trace reader.
    for event in gpu:
        start, end = event['ts'] * 1000, (event['ts'] + event['dur']) * 1000
        matched = [k for k, (a, b) in intervals.items()
                   if a - tolerance_ns <= start <= end <= b + tolerance_ns]
        require(len(matched) == 1, 'GPU work outside an active replay interval')
        if event in kernels:
            names = [n for n in ('Produce', 'Consume')
                     if 'StrataCompatWindow' + n in event['name']]
            require(len(names) == 1, 'missing or unexpected fixture kernel name')
            counts[matched[0]][names[0]] += 1
    expected = dict(prefix=dict(Produce=2, Consume=2))
    if mode == 'window': expected['window'] = dict(Produce=3, Consume=3)
    require({k: dict(v) for k, v in counts.items()} == expected and
            len(kernels) == (4 if mode == 'prefix' else 10), 'fixture kernel coverage differs')
    return dict(executed_kernels=20, recorded_kernels=len(kernels), by_interval=expected,
                gap_and_suffix_excluded=True, all_GPU_operations_in_active_intervals=True,
                clock_tolerance_ns=tolerance_ns)


def reader_controls():
    clocks = dict(begin=1_000_000,
        **{'prefix-drained': dict(arrival=200_000_000),
           'gap-drained': dict(release=400_000_000),
           'window-drained': dict(arrival=600_000_000)})
    prefix = [dict(ph='X', cat='gpu_op', name='StrataCompatWindow' + name,
                   ts=10_000 + i * 10_000, dur=10)
              for i, name in enumerate(('Produce', 'Consume') * 2)]
    events = prefix + [dict(ph='X', cat='gpu_op', name='StrataCompatWindow' + name,
                           ts=410_000 + i * 10_000, dur=10)
                      for i, name in enumerate(('Produce', 'Consume') * 3)]
    coverage(prefix, clocks, 'prefix'); coverage(events, clocks, 'window')
    bad = [events[:-1], events + [copy.deepcopy(events[0])], copy.deepcopy(events),
           copy.deepcopy(events), copy.deepcopy(events), copy.deepcopy(events),
           events + [dict(ph='X', cat='gpu_op', name='zeCommandListAppendMemoryFill',
                          ts=300_000, dur=10)]]
    bad[2][0]['name'] = 'unknown'; bad[3][4]['ts'] = 300_000
    bad[4][4]['ts'] = 700_000; bad[5][4]['ts'] = 599_990; bad[5][4]['dur'] = 3000
    for candidate in bad:
        try: coverage(candidate, clocks, 'window')
        except RuntimeError: continue
        raise RuntimeError('corrupt collector-window reader control accepted')
    return dict(passed=True, positive=2, negative=len(bad), GPU_executed=False)


def prepare():
    built, active, providers, assets = basis()
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh window check only')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = base.controls(revision, controller=SELF)
    pins[build.CPP] = built['sources'][build.CPP]
    ROOT.mkdir(mode=0o700)
    profiles = {}
    for mode in MODES:
        root = ROOT / mode; root.mkdir()
        for name in ('tmp', 'cache', 'trace'): (root / name).mkdir()
        if mode != 'smoke': profiles[mode] = dict(profile.profiling(root, assets), mode='aggregate')
    save(dict(stage='prepared', source_commit=revision, sources=pins,
        fixture_build_source_commit=built['source_commit'], binary=built['outcome']['binary'],
        native_source_commit=built['native_source_commit'], providers=providers,
        native_validation_probe=active['native_validation_probe'], assets=assets,
        environment_template=active['environment_template'], profiles=profiles,
        boot=active['boot'], driver=active['driver'], budgets=BUDGET, whole_bytes=128 << 20,
        reader_controls=reader_controls(), cases=[], active=False, ownership_closed=True,
        passed=False, complete=False, model_executed=False,
        plan='Three separate owned processes in order:unprofiled smoke,prefix-only,prefix+resumed-window. All execute10two-kernel graph replays with numerical/guard checks after each. Collector active during construction;prefix2replays,paused gap3,resumed window3,paused suffix2. Prefix-only stays paused after its first2replays. Expected4/10 named kernels with unique native host-flow joins and raw-clock containment,all gap/suffix GPU work excluded. Gate only after actual completion and readback drain. A failed case stops progression;commit each closed outcome before the next. Keep strict before-main validation,Sysman0,nativecapture0 and immutable safety environment. No timing-speed or model/saturation qualification.',
        retention='Keep useful counts/names/clocks/validation/fault/closure outcomes. Retire full successful API/timeline/JIT/owner output and this disposable fixture after the current-runtime interior-window decision. Preserve cb7 and552b engines.'))
    print('Prepared smoke,prefix and resumed-window controls;commit before execution.', flush=True)


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row) and
            base.controls(row['source_commit'], controller=SELF) ==
            {k: v for k, v in row['sources'].items() if k != build.CPP},
            'window controller/committed plan changed')
    built, active, providers, assets = basis()
    require(row['binary'] == built['outcome']['binary'] and row['providers'] == providers and
            row['native_validation_probe'] == active['native_validation_probe'] and
            row['environment_template'] == active['environment_template'] and row['assets'] == assets and
            row['boot'] == active['boot'] and row['driver'] == active['driver'],
            'fixture/runtime/collector basis changed')
    return providers


def check(mode):
    row = json.loads((ROOT / 'record.json').read_text()); providers = current(row)
    require(len(row['cases']) < len(MODES) and mode == MODES[len(row['cases'])] and
            all(v['passed'] for v in row['cases']) and row['reader_controls']['passed'],
            'commit successful preceding case;no failed-case retry')
    root, frozen = ROOT / mode, copy.deepcopy(row)
    require(not (root / 'execution.json').exists(), 'one attempt per case')
    original = row['environment_template']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(original, str(root)) for k, v in row['environment_template'].items()}
    require(not any(k.startswith('UNITRACE_') or k == 'LD_PRELOAD' for k in env),
            'unexpected inherited profiler/preload')
    env.update(STRATA_TEST_MAIN_GATES='1', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
            env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0',
            'qualified runtime safety differs')
    prof = row['profiles'].get(mode)
    if prof:
        require(not profile.session_path(prof['session']).exists(), 'preexisting collector session')
        env.update(UNITRACE_ChromeKernelLogging='1', UNITRACE_ChromeEventBufferSize='128')
        target = profile.profile_target_environment(row, env, prof)
        expected = profile.expected_target_environment(row, env, prof)
        for k in ('UNITRACE_StartPaused', 'UNITRACE_HostTiming', 'UNITRACE_ChromeCallLogging'): expected.pop(k)
        expected['UNITRACE_Verbose'] = '1'
        command = profile.engine_once.profile_command(prof, [row['binary']['path']])
    else:
        target = expected = profile.engine_once.native_environment(row, env)
        command = [row['binary']['path']]
    owner = controls = None
    outcome = dict(mode=mode, passed=False, complete=False, host_probe_accepted=False)
    clocks, seen, mapped, cursor = {}, [], set(), None

    def persist():
        write(root / 'execution.json', dict(outcome,
            active=bool((owner and owner.active) or (controls and controls.active)),
            commands=owner.commands if owner else [], controls=controls.commands if controls else []))

    def control(action, state=None):
        receipt, _, _ = controls.run(str(len(controls.commands)) + '-' + action,
            [prof['launcher']['path'], '--' + action, prof['session']], env, P,
            wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20,
            file_cap=16 << 20, total_cap=64 << 20)
        require(base.supervision.closed(receipt) and receipt['exit_code'] == 0,
                'collector state change failed')
        if state is not None: profile.session_state(prof['session'], state)

    def interact(io, native):
        for phase in PHASES:
            while True:
                line = io.line(45)
                if line.startswith('COMPONENT_READY '): break
            require(line == 'COMPONENT_READY ' + phase, 'fixture gate order differs')
            clocks[phase] = dict(arrival=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
            seen.append(phase)
            proc = Path('/proc', str(native['pid']))
            require((proc / 'exe').resolve() == Path(row['binary']['path']).resolve(), 'fixture executable differs')
            actual = dict(v.split('=', 1) for v in (proc / 'environ').read_text().split('\0') if '=' in v)
            profile.same_environment(actual, expected)
            maps = build.collector.previous.entry.template.shared.runtime.executable_mappings((proc / 'maps').read_text())
            require(maps <= set(providers) | {row['binary']['path']} |
                    ({prof['collector']['path']} if prof else set()), 'unexpected fixture provider')
            mapped.update(maps)
            if prof:
                state = 0 if phase in PHASES[:3] or (mode == 'window' and phase == 'window-drained') else 1
                profile.session_state(prof['session'], state)
                if phase == 'prefix-drained' or (mode == 'window' and phase == 'window-drained'):
                    control('pause', 1)
                elif mode == 'window' and phase == 'gap-drained': control('resume', 0)
            require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole window output budget')
            # Host-side quiescence separates the clock-tolerance margins from
            # actual gap/suffix work;no device spin or process-env mutation.
            if phase in ('prefix-drained', 'gap-drained', 'window-drained'):
                deadline = time.monotonic() + .02
                while time.monotonic() < deadline: io.poll(); time.sleep(.005)
            clocks[phase]['release'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            persist(); io.send(b'X')
        io.proc.stdin.close()
        deadline = time.monotonic() + 30
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'fixture teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)

    try:
        with (root / 'execution.json').open('x') as stream: stream.write('{}\n')
        cursor = profile.cursor()
        owner = base.supervision.Owner(root / 'owned', persist)
        controls = base.supervision.Owner(root / 'controls', persist)
        clocks['begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        native, out, err = owner.run('fixture', command, target, P, interaction=interact, **BUDGET)
        clocks['end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        require(base.supervision.closed(native) and native['exit_code'] == 0 and seen == list(PHASES),
                'fixture execution/ownership failed')
        stdout, stderr = out.read_text(), err.read_text()
        samples = re.findall(r'^case phase=(\w+) replays=(\d+) kernels=(\d+) completed=(\d+) numerical=PASS guards=PASS$', stdout, re.M)
        require(samples == [('prefix','2','4','2'),('gap','3','6','5'),('window','3','6','8'),('suffix','2','4','10')] and
                stdout.count('fixture_status=PASS mode=compat-window replays=10 kernels=20') == 1,
                'cumulative numerical/guard/phase checks missing')
        validation = build.collector.previous.entry.template.shared.runtime.validation
        activation = validation.preload_evidence(stderr, native['pid'])
        required = {row['native_validation_probe']['loader']['path'],
                    row['native_validation_probe']['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']}
        if prof: required.add(prof['collector']['path'])
        require(required <= mapped and not any(v in stderr for v in
            ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]')) and
            not re.search(r'(?i)(?:dropped|lost)\s*[:=]\s*[1-9]', stderr) and
            (not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size),
            'validation/provider/collector error')
        traces = list((root / 'trace').rglob('*.json'))
        trace = None
        if prof:
            require(len(traces) == 1, 'one fully flushed timeline required')
            events = json.loads(traces[0].read_text())['traceEvents']
            summary = profile.trace_summary(events, native['pid'], clocks['begin'], clocks['end'])
            trace = dict(coverage(events, clocks, mode), unique_native_host_flow_joins=True,
                GPU_events=summary['gpu_events'], host_events=summary['host_events'],
                sha256=digest(traces[0]), bytes=traces[0].stat().st_size)
        else: require(not traces, 'unexpected smoke collector output')
        current(frozen)
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0,
                       numerical_guards_passed=True, validation=activation, trace=trace, clocks=clocks)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if prof and owner and owner.active is None and profile.session_path(prof['session']).exists():
            try: control('stop')
            except BaseException as error: outcome['control_close_error'] = repr(error)
        if owner and cursor:
            build.collector.previous.entry.template.shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and
            (controls is None or (controls.active is None and all(base.supervision.closed(c) for c in controls.commands))))
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and outcome['ownership_closed'] and
            outcome.get('driver_after') == row['driver'] and 'control_close_error' not in outcome and
            (not prof or not profile.session_path(prof['session']).exists()))
        persist(); row['cases'].append(outcome)
        row.update(stage='closed' if len(row['cases']) == len(MODES) and outcome['passed'] else
            'checked' if outcome['passed'] else 'failed', passed=all(v['passed'] for v in row['cases']),
            complete=len(row['cases']) == len(MODES) and outcome['passed'],
            active=bool((owner and owner.active) or (controls and controls.active)),
            ownership_closed=outcome['ownership_closed'])
        save(row)
    require(outcome['passed'], outcome.get('error', 'collector window failed'))
    print('Closed collector fixture ' + mode + ';no model attribution or speed claim.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] == ['prepare'] or
            (len(sys.argv) == 3 and sys.argv[1] == 'check' and sys.argv[2] in MODES),
            'use prepare|check smoke|prefix|window')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else check(sys.argv[2])
