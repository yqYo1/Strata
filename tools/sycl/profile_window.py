#!/usr/bin/env python3
"""Qualify a bounded collector window, without rebuilding the inference engine.

Main-authored from owned supervision/collector helpers; repository LICENSE.
Prepare/build/check are separate committed boundaries. This fixture does not
qualify a model request gate, observer overhead or physical resource saturation.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import sys
import time

sys.dont_write_bytecode = True
import pool_spin as cache
import profile_split as profile
import decode_window_trace as window

P, B, RESULT = cache.P, cache.B, cache.RESULT
ROOT = cache.model.adaptive.ROOT / 'decode-window-observer-v1'
KEY = 'decode_window_observer_v1'
CPP = 'tests/sycl/profile_window.cpp'
NATIVE = '20f2a971b5fa3c0b2c65ab08712c32febb894f93'
PHASES = ('before-runtime', 'runtime-ready', 'captured-paused', 'replay-drained',
          'late-enqueued', 'late-drained', 'before-exit')
require, digest, write = cache.require, cache.digest, cache.write
BUILD_BUDGET = dict(wall=600, cpu=1200, rss_cap=8 << 30,
                    text_cap=8 << 20, file_cap=64 << 20, total_cap=128 << 20)
DEVICE_BUDGET = dict(wall=180, cpu=180, rss_cap=4 << 30,
                     text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('sources', 'environment_template', 'build_environment')}
    for name in ('sources', 'environment_template', 'build_environment'):
        result[name + '_sha256'] = hashlib.sha256(json.dumps(row[name], sort_keys=True).encode()).hexdigest()
    return result


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def sources(revision):
    pins = cache.build.controls(revision, controller=Path(__file__).resolve())
    raw = cache.shared.gen.git('show', revision + ':' + CPP)
    require((P / CPP).read_bytes() == raw, 'fixture source changed')
    pins[CPP] = hashlib.sha256(raw).hexdigest()
    return pins


def basis():
    data = cache.build.committed(RESULT)
    cohort = data['adaptive_cache_generation_speed_v2']
    require(cohort['stage'] in ('closed', 'failed') and cohort['ownership_closed'] and not cohort['active']
            and all(p.get('ownership_closed') and p.get('fresh_faults') == [] for p in cohort['processes']),
            'finish the frozen clean cohort before observer work')
    cache.model.configure('adaptive-positive-throughput')
    native, _, providers = cache.model.proof()
    require(native['source_commit'] == NATIVE, 'fixed qualified native source required')
    prior = json.loads((cache.model.ROOT / 'record.json').read_text())
    require(data[cache.model.KEY] == cache.model.compact(prior) and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active'], 'closed fixed-engine model admission required')
    assets = profile.collector_assets(data)
    return native, prior, providers, assets


def current(row):
    require(cache.build.committed(RESULT)[KEY] == compact(row), 'commit the exact observer plan/results')
    require(sources(row['source_commit']) == row['sources'], 'observer source changed')
    native, prior, providers, assets = basis()
    require(native['compiler'] == row['compiler'] and native['sdk'] == row['sdk']
            and native['boot'] == row['boot'] and native['driver'] == row['driver']
            and native['binaries']['strata'] == row['engine']
            and prior['native_validation_probe'] == row['native_validation_probe']
            and prior['environment'] == row['environment_template'] and assets == row['assets'],
            'qualified compiler/runtime/collector/engine changed')
    inventory = row['runtime_manifest']
    require(digest(inventory['path']) == inventory['sha256']
            and json.loads(Path(inventory['path']).read_text()) == providers, 'runtime inventory changed')
    if 'dependencies' in row:
        inv = row['dependencies']; require(digest(inv['path']) == inv['sha256'], 'fixture dependency manifest changed')
        deps = json.loads(Path(inv['path']).read_text())
        require(len(deps) == inv['files'] and all(digest(p) == sha for p, sha in deps.items()),
                'fixture compiler input changed')
    if 'binary' in row:
        binary = row['binary']
        require(digest(binary['path']) == binary['sha256'] and Path(binary['path']).stat().st_size == binary['bytes'],
                'observer fixture ELF changed')
    return native, providers


def dependencies(path):
    text = path.read_text().replace('\\\n', ' ')
    paths = set()
    for line in text.splitlines():
        if not line.strip(): continue
        target, separator, body = line.partition(':')
        require(separator and target.strip(), 'malformed compiler dependency rule')
        paths.update(str(Path(name).resolve()) for name in shlex.split(body))
    require(str((P / CPP).resolve()) in paths, 'fixture missing from compiler dependencies')
    return paths


def prepare():
    require(not ROOT.exists() and KEY not in cache.build.committed(RESULT), 'fresh observer qualification only')
    revision = cache.shared.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    native, prior, providers, assets = basis()
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache', 'direct', 'trace'):
        (ROOT / name).mkdir()
    write(ROOT / 'runtime-inputs.json', providers)
    env = dict(native['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    flags = ['-O2', '-std=c++17', '-fsycl', '-fsycl-device-code-split=per_kernel', '-pthread']
    prefix = [native['compiler']['path'], *flags]
    row = dict(stage='preparing', source_commit=revision, sources=pins, compiler=native['compiler'], sdk=native['sdk'],
        boot=native['boot'], driver=native['driver'], engine=native['binaries']['strata'],
        native_source_commit=NATIVE, native_validation_probe=prior['native_validation_probe'], assets=assets,
        environment_template=copy.deepcopy(prior['environment']), build_environment=env,
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json')),
        preprocess=[*prefix, '-M', '-MT', 'fixture', '-MF', str(ROOT / 'headers.d'), str(P / CPP)],
        command=[*prefix, '-MD', '-MF', str(ROOT / 'build.d'), str(P / CPP), '-o', str(ROOT / 'fixture')],
        profile=profile.profiling(ROOT / 'trace', assets), passed=False, complete=False,
        active=False, ownership_closed=True, cases=[],
        limits='One observer fixture,not another engine generation. No model,observer-cost,full-context or saturation qualification.',
        consumer='Qualify paused graph construction/replay and pause-crossing metadata before the planned bounded decode capture. Retire raw capture after that question closes;retain exact source/recipe and compact counts.')
    owner = None
    def persist():
        write(ROOT / 'prepare-execution.json', dict(active=bool(owner and owner.active), commands=owner.commands if owner else []))
    try:
        owner = cache.build.supervision.Owner(ROOT / 'prepare-owned', persist)
        entry, _, _ = owner.run('headers', row['preprocess'], env, P, **BUILD_BUDGET)
        require(cache.build.supervision.closed(entry) and entry['exit_code'] == 0, 'fixture preprocessing failed')
        allowed = cache.model.adaptive.inputs(native)
        allowed[str((P / CPP).resolve())] = pins[CPP]
        paths = dependencies(ROOT / 'headers.d')
        require(all(name in allowed and digest(name) == allowed[name] for name in paths), 'unqualified fixture header input')
        deps = {name: allowed[name] for name in sorted(paths)}
        write(ROOT / 'compiler-inputs.json', deps)
        row.update(stage='prepared', dependencies=dict(path=str(ROOT / 'compiler-inputs.json'),
            sha256=digest(ROOT / 'compiler-inputs.json'), files=len(deps)))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=bool(owner and owner.active is None
            and all(cache.build.supervision.closed(x) for x in owner.commands)))
        save(row)
    require(row['stage'] == 'prepared' and row['ownership_closed'], row.get('error', 'preparation not closed'))
    print('Prepared observer fixture;commit before build.', flush=True)


def build():
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'build-execution.json').exists(), 'fresh committed build only')
    owner = None
    def persist():
        write(ROOT / 'build-execution.json', dict(active=bool(owner and owner.active), commands=owner.commands if owner else []))
    try:
        owner = cache.build.supervision.Owner(ROOT / 'build-owned', persist)
        entry, _, _ = owner.run('compile', row['command'], row['build_environment'], P, **BUILD_BUDGET)
        require(cache.build.supervision.closed(entry) and entry['exit_code'] == 0, 'fixture compilation failed')
        expected = json.loads(Path(row['dependencies']['path']).read_text())
        require(dependencies(ROOT / 'build.d') == set(expected) and
                all(digest(p) == sha for p, sha in expected.items()), 'actual compile dependencies differ')
        current(row)
        row.update(stage='built', binary=dict(path=str(ROOT / 'fixture'), sha256=digest(ROOT / 'fixture'),
                                             bytes=(ROOT / 'fixture').stat().st_size))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=bool(owner and owner.active is None
            and all(cache.build.supervision.closed(x) for x in owner.commands)))
        save(row)
    require(row['stage'] == 'built' and row['ownership_closed'], row.get('error', 'build not closed'))
    print('Built observer fixture;commit before traced/validated checks.', flush=True)


def device_environment(row, root):
    old = row['environment_template']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(root)) for k, v in row['environment_template'].items()}
    for name in ('tmp', 'cache'):
        (root / name).mkdir(exist_ok=True)
    env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and env['ZES_ENABLE_SYSMAN'] == '0',
            'qualified safety settings differ')
    return env


def check():
    row = json.loads((ROOT / 'record.json').read_text()); _, providers = current(row)
    require(row['stage'] == 'built' and not row['cases'], 'fresh ordered direct/trace qualification only')
    frozen = copy.deepcopy(row)
    for mode in ('direct', 'trace'):
        root = ROOT / mode
        require(not (root / 'execution.json').exists(), 'no fixture retry')
        prof = row['profile'] if mode == 'trace' else None
        env = device_environment(row, root)
        target = (profile.profile_target_environment(row, env, prof) if prof else
                  cache.model.engine_once.native_environment(row, env))
        expected = profile.expected_target_environment(row, env, prof) if prof else target
        owner, controls = None, None
        outcome = dict(mode=mode, passed=False, complete=False, host_probe_accepted=False)
        bounds, phases, mapped = {}, [], set()
        cursor = profile.cursor()
        def persist():
            write(root / 'execution.json', dict(outcome, active=bool((owner and owner.active) or (controls and controls.active)),
                commands=owner.commands if owner else [], controls=controls.commands if controls else []))
        def control(action, state=None):
            entry, _, _ = controls.run(action + '-' + str(len(controls.commands)),
                [prof['launcher']['path'], '--' + action, prof['session']], env, P,
                wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20, file_cap=32 << 20, total_cap=64 << 20)
            require(cache.build.supervision.closed(entry) and entry['exit_code'] == 0, 'collector control failed')
            if state is not None: profile.session_state(prof['session'], state)
        def interact(io, entry):
            for phase in PHASES:
                while True:
                    line = io.line(45)
                    if line.startswith('COMPONENT_READY '): break
                require(line == 'COMPONENT_READY ' + phase, 'fixture gate order differs')
                require(cache.build.supervision.artifact_bytes(ROOT) <= 256 << 20, 'observer whole-output budget')
                phases.append(phase)
                require(Path('/proc', str(entry['pid']), 'exe').resolve() == Path(row['binary']['path']).resolve(), 'fixture identity differs')
                actual = dict(v.split('=', 1) for v in Path('/proc', str(entry['pid']), 'environ').read_text().split('\0') if '=' in v)
                profile.same_environment(actual, expected)
                seen = cache.shared.runtime.executable_mappings(Path('/proc', str(entry['pid']), 'maps').read_text())
                require(seen <= set(providers) | {row['binary']['path']} | ({prof['collector']['path']} if prof else set()),
                        'fixture provider differs')
                mapped.update(seen)
                if prof:
                    if phase == 'before-runtime': profile.session_state(prof['session'], 1)
                    if phase == 'captured-paused':
                        control('resume', 0); bounds['begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    if phase == 'replay-drained':
                        bounds['end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW); control('pause', 1); control('resume', 0)
                    if phase == 'late-enqueued':
                        bounds['late_pause'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW); control('pause', 1)
                io.send(b'X')
            io.proc.stdin.close()
            deadline = time.monotonic() + 45
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll(); require(time.monotonic() < deadline, 'fixture teardown deadline')
                require(cache.build.supervision.artifact_bytes(ROOT) <= 256 << 20, 'observer whole-output budget')
                if select.select([io.proc.stdout], [], [], .02)[0]:
                    io.read(); io.pending.clear()
            io.exit(1)
        try:
            current(frozen)
            if prof: require(not profile.session_path(prof['session']).exists(), 'preexisting collector session')
            owner = cache.build.supervision.Owner(root / 'owned', persist)
            controls = cache.build.supervision.Owner(root / 'controls', persist)
            command = [row['binary']['path']]
            if prof: command = cache.model.engine_once.profile_command(prof, command)
            entry, out, err = owner.run(mode, command, target, P, interaction=interact, **DEVICE_BUDGET)
            require(cache.build.supervision.closed(entry) and entry['exit_code'] == 0 and phases == list(PHASES), 'fixture execution failed')
            stdout, stderr = out.read_text(), err.read_text()
            for marker in ('fixture_status=PASS mode=profile-window', 'late_graph_pending=1',
                           'late_replays=1 kernels=2 numerical=PASS guards=PASS timer_fallback=0',
                           'clean_replays=3 kernels=6 readbacks=3 numerical=PASS guards=PASS'):
                require(stdout.count(marker) == 1, 'fixture marker differs: ' + marker)
            activation = cache.shared.runtime.validation.preload_evidence(stderr, entry['pid'])
            required = {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']}
            if prof: required.add(prof['collector']['path'])
            require(required <= mapped and activation['loader'] == row['native_validation_probe']['loader']['path']
                    and not any(x in stderr for x in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]'))
                    and (not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size),
                    'fixture validation/provider error')
            if prof:
                traces = list((root / 'trace').rglob('*.json')); require(len(traces) == 1, 'one flushed fixture timeline required')
                events = json.loads(traces[0].read_text())['traceEvents']
                summary = window.summarize(events, native_pid=entry['pid'], begin_ns=bounds['begin'], end_ns=bounds['end'],
                    capture_begin_ns=bounds['begin'], capture_end_ns=bounds['end'])
                require(summary['operation_counts']['kernel'] == 6, 'paused-capture/replay kernel inventory differs')
                late = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op'
                        and ('StrataWindowProduce' in e.get('name', '') or 'StrataWindowConsume' in e.get('name', ''))
                        and e.get('ts', 0) >= bounds['late_pause'] / 1000]
                require(not any(e.get('ph') == 'X' and e.get('cat') == 'gpu_op'
                    and ('StrataWindowProduce' in e.get('name', '') or 'StrataWindowConsume' in e.get('name', ''))
                    and bounds['end'] / 1000 <= e.get('ts', 0) < bounds['late_pause'] / 1000 for e in events),
                    'late graph ran before its blocked host task was released')
                require(len(late) in (0, 2), 'partial/duplicate late graph inventory')
                outcome['trace'] = dict(expected_clean_kernels=6, observed_clean_kernels=6,
                    paused_boundary_late_kernels_observed=len(late),
                    gpu_interval_union_us=summary['observed_gpu_interval_union_us'],
                    kernel_interval_union_us=summary['observed_kernel_interval_union_us'],
                    host_api_interval_union_us=summary['observed_host_api_interval_union_us'],
                    window_wall_us=summary['window_wall_us'], included_instance_joins_unique=True,
                    timeline_sha256=digest(traces[0]), timeline_bytes=traces[0].stat().st_size,
                    lossless_boundary_qualified=False, model_observer_cost_qualified=False)
            current(frozen)
            outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=activation)
        except BaseException as error:
            outcome['error'] = repr(error)
        finally:
            if prof and owner and owner.active is None and profile.session_path(prof['session']).exists():
                try: control('stop')
                except BaseException as error: outcome['control_close_error'] = repr(error)
            if owner: cache.shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
            controls_closed = bool(controls and controls.active is None and all(cache.build.supervision.closed(x) for x in controls.commands))
            outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and controls_closed)
            outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and outcome['ownership_closed']
                and outcome.get('driver_after') == row['driver']
                and 'control_close_error' not in outcome and (not prof or not profile.session_path(prof['session']).exists()))
            row['cases'].append(outcome)
            row.update(stage='checked' if outcome['passed'] and mode == 'trace' else 'checking' if outcome['passed'] else 'failed',
                passed=bool(outcome['passed'] and mode == 'trace'), complete=bool(outcome['passed'] and mode == 'trace'),
                active=bool((owner and owner.active) or (controls and controls.active)), ownership_closed=outcome['ownership_closed'])
            # Keep the committed input plan unchanged during the two-case job.
            # Private partial results still prevent an interrupted case retry.
            write(ROOT / 'record.json', row)
            if not outcome['passed']: save(row)
        require(outcome['passed'], outcome.get('error', 'observer fixture qualification failed'))
    row['decision'] = dict(safe_fixture_window_qualified=True, model_capture_qualified=False,
        next='A new committed request-scoped decode gate,guard bands,clock/loss checks and model observer-cost plan remain required.')
    save(row)
    print('Window fixture checks passed;model attribution remains unqualified.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check'), 'profile_window.py prepare|build|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache.build.guards.no_model_owners([])
        {'prepare': prepare, 'build': build, 'check': check}[sys.argv[1]]()
