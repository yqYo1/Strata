#!/usr/bin/env python3
"""Qualify a bounded collector window, without rebuilding the inference engine.

Main-authored from owned supervision/collector helpers; repository LICENSE.
The original fixture build is recovered from the closed V1 check. This fixture does not
qualify a model request gate, observer overhead or physical resource saturation.
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
import pool_spin as cache
import profile_split as profile
import decode_window_trace as window

P, B, RESULT = cache.P, cache.B, cache.RESULT
OLD_ROOT = cache.model.adaptive.ROOT / 'decode-window-observer-v1'
OLD_KEY = 'decode_window_observer_v1'
ROOT = cache.model.adaptive.ROOT / 'decode-window-observer-v2'
KEY = 'decode_window_observer_v2'
CPP = 'tests/sycl/profile_window.cpp'
NATIVE = '20f2a971b5fa3c0b2c65ab08712c32febb894f93'
PHASES = ('before-runtime', 'runtime-ready', 'captured-paused', 'replay-drained',
          'late-enqueued', 'late-drained', 'before-exit')
require, digest, write = cache.require, cache.digest, cache.write
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


def original():
    old = json.loads((OLD_ROOT / 'record.json').read_text())
    require(cache.build.committed(RESULT)[OLD_KEY] == compact(old)
            and old['stage'] == 'failed' and old['ownership_closed'] and not old['active']
            and len(old['cases']) == 2 and old['cases'][0]['passed']
            and not old['cases'][1]['passed']
            and all(c['ownership_closed'] and c['fresh_faults'] == [] for c in old['cases'])
            and old['failure_diagnosis']['observed_native_exit_codes'] == dict(direct=0, trace=0)
            and old['failure_diagnosis']['named_graph_kernel_events'] == 0,
            'closed failed paused-construction fixture required')
    return old


def current(row):
    require(cache.build.committed(RESULT)[KEY] == compact(row), 'commit the exact observer plan/results')
    require(sources(row['source_commit']) == row['sources'], 'observer source changed')
    old = original()
    require(old['source_commit'] == row['fixture_reuse']['original_source_commit']
            and all(old[k] == row[k] for k in ('binary', 'dependencies', 'command', 'build_environment'))
            and old['sources'][CPP] == row['sources'][CPP], 'original fixture/reuse qualification changed')
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


def prepare():
    require(not ROOT.exists() and KEY not in cache.build.committed(RESULT), 'fresh observer qualification only')
    old = original()
    revision = cache.shared.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    recovered = {name: hashlib.sha256(raw).hexdigest() for name, raw in
                 cache.shared.gen.blobs(P, old['source_commit'], old['sources'])}
    require(recovered == old['sources'] and pins.keys() == recovered.keys()
            and {name for name in pins if pins[name] != recovered[name]} == {'tools/sycl/profile_window.py'},
            'only the declared controller delta may reuse the fixture')
    native, prior, providers, assets = basis()
    require(native['compiler'] == old['compiler'] and native['sdk'] == old['sdk']
            and native['boot'] == old['boot'] and native['driver'] == old['driver']
            and native['binaries']['strata'] == old['engine']
            and prior['environment'] == old['environment_template']
            and prior['native_validation_probe'] == old['native_validation_probe'] and assets == old['assets'],
            'original fixture/compiler/runtime/collector basis differs')
    inv = old['dependencies']
    require(digest(inv['path']) == inv['sha256'], 'original fixture dependency manifest changed')
    deps = json.loads(Path(inv['path']).read_text())
    require(len(deps) == inv['files'] == 638 and all(digest(p) == sha for p, sha in deps.items())
            and deps[str((P / CPP).resolve())] == pins[CPP], 'original638compiled inputs changed')
    inv = old['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == providers,
            'original runtime inventory changed')
    binary = old['binary']
    require(digest(binary['path']) == binary['sha256'] and Path(binary['path']).stat().st_size == binary['bytes'],
            'original fixture ELF changed')
    require(not profile.session_path(old['profile']['session']).exists(), 'original collector session remains')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache', 'trace'):
        (ROOT / name).mkdir()
    row = copy.deepcopy(old)
    row.pop('failure_diagnosis', None)
    row.update(stage='prepared', source_commit=revision, sources=pins, profile=profile.profiling(ROOT / 'trace', assets),
               passed=False, complete=False, active=False, ownership_closed=True, cases=[])
    row['fixture_reuse'] = dict(original_result_key=OLD_KEY, original_source_commit=old['source_commit'],
        changed_controls=['tools/sycl/profile_window.py'], compiled_inputs_unchanged=638,
        original_direct_numerical_check_reused=True,
        scope='Same original CPP,recipe,compiler638inputs,ELF and providers;no fixture or engine rebuild. V1 remains failed.')
    row['plan'] = dict(change='Resume at before-runtime and stay active through graph construction,replay,late completion,readback and native exit/flush. '
        'Do not pause before releasing the late host task;the original ELF gate names remain literal protocol labels. Persist every gate/control RAW clock bound.',
        acceptance='Exactly6named clean replay kernels and3copies inside the recorded inner window,plus2named late kernels,unique native flow joins,normal0,quiet validation,closed owners,no fresh fault. '
        'Any failure stops without retry;no model/lossless boundary/observer-cost claim.',
        whole_bytes=256 << 20, device_budget=DEVICE_BUDGET)
    save(row)
    print('Prepared controller-only active-construction check;commit before execution.', flush=True)


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
    require(row['stage'] == 'prepared' and not row['cases'], 'fresh active-construction qualification only')
    frozen = copy.deepcopy(row)
    for mode in ('trace',):
        root = ROOT / mode
        require(not (root / 'execution.json').exists(), 'no fixture retry')
        prof = row['profile'] if mode == 'trace' else None
        env = device_environment(row, root)
        target = (profile.profile_target_environment(row, env, prof) if prof else
                  cache.model.engine_once.native_environment(row, env))
        expected = profile.expected_target_environment(row, env, prof) if prof else target
        owner, controls = None, None
        bounds, phases, mapped = dict(phase_clocks={}, controls=[]), [], set()
        outcome = dict(mode=mode, passed=False, complete=False, host_probe_accepted=False,
                       clock_bounds=bounds, clock_domain='CLOCK_MONOTONIC_RAW')
        cursor = profile.cursor()
        def persist():
            write(root / 'execution.json', dict(outcome, active=bool((owner and owner.active) or (controls and controls.active)),
                commands=owner.commands if owner else [], controls=controls.commands if controls else []))
        def control(action, state=None):
            before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            entry, _, _ = controls.run(action + '-' + str(len(controls.commands)),
                [prof['launcher']['path'], '--' + action, prof['session']], env, P,
                wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20, file_cap=32 << 20, total_cap=64 << 20)
            require(cache.build.supervision.closed(entry) and entry['exit_code'] == 0, 'collector control failed')
            if state is not None: profile.session_state(prof['session'], state)
            after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            bounds['controls'].append(dict(action=action, before_ns=before, after_ns=after))
            persist()
        def interact(io, entry):
            for phase in PHASES:
                while True:
                    line = io.line(45)
                    if line.startswith('COMPONENT_READY '): break
                require(line == 'COMPONENT_READY ' + phase, 'fixture gate order differs')
                require(cache.build.supervision.artifact_bytes(ROOT) <= 256 << 20, 'observer whole-output budget')
                phases.append(phase)
                bounds['phase_clocks'][phase] = dict(raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                    monotonic_ns=time.monotonic_ns(), boottime_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME))
                require(Path('/proc', str(entry['pid']), 'exe').resolve() == Path(row['binary']['path']).resolve(), 'fixture identity differs')
                actual = dict(v.split('=', 1) for v in Path('/proc', str(entry['pid']), 'environ').read_text().split('\0') if '=' in v)
                profile.same_environment(actual, expected)
                seen = cache.shared.runtime.executable_mappings(Path('/proc', str(entry['pid']), 'maps').read_text())
                require(seen <= set(providers) | {row['binary']['path']} | ({prof['collector']['path']} if prof else set()),
                        'fixture provider differs')
                mapped.update(seen)
                if prof:
                    if phase == 'before-runtime':
                        profile.session_state(prof['session'], 1)
                        control('resume', 0); bounds['capture_begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    if phase == 'captured-paused':
                        bounds['begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    if phase == 'replay-drained':
                        bounds['end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    if phase == 'late-enqueued':
                        bounds['late_release'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    profile.session_state(prof['session'], 0)
                persist(); io.send(b'X')
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
            outcome['native_exit_code'] = entry['exit_code']
            bounds['capture_end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            persist()
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
                    capture_begin_ns=bounds['capture_begin'], capture_end_ns=bounds['capture_end'])
                require(summary['operation_counts'].get('kernel') == 6 and summary['operation_counts'].get('copy') == 3,
                        'active-construction replay inventory differs')
                named = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op'
                         and bounds['begin'] / 1000 <= e['ts'] < bounds['end'] / 1000
                         and 'StrataWindow' in e.get('name', '')]
                require(len(named) == 6 and all(sum(name in e['name'] for e in named) == 3
                    for name in ('StrataWindowProduce', 'StrataWindowConsume')),
                    'named producer/consumer replay coverage differs')
                clocks = list(bounds['phase_clocks'].values()); first = clocks[0]
                require(len(clocks) == len(PHASES) and all(abs((c['boottime_ns'] - first['boottime_ns']) -
                    (c['monotonic_ns'] - first['monotonic_ns'])) <= 50000000 for c in clocks),
                    'fixture clock/suspend discontinuity')
                late = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op'
                        and ('StrataWindowProduce' in e.get('name', '') or 'StrataWindowConsume' in e.get('name', ''))
                        and e.get('ts', 0) >= bounds['late_release'] / 1000]
                require(not any(e.get('ph') == 'X' and e.get('cat') == 'gpu_op'
                    and ('StrataWindowProduce' in e.get('name', '') or 'StrataWindowConsume' in e.get('name', ''))
                    and bounds['end'] / 1000 <= e.get('ts', 0) < bounds['late_release'] / 1000 for e in events),
                    'late graph ran before its blocked host task was released')
                require(len(late) == 2 and all(sum(name in e['name'] for e in late) == 1
                    for name in ('StrataWindowProduce', 'StrataWindowConsume')), 'partial/duplicate late graph inventory')
                ids = {window.identifier(e['args']['id']) for e in named}
                late_ids = {window.identifier(e['args']['id']) for e in late}
                require(len(late_ids) == 2 and ids.isdisjoint(late_ids)
                    and all(e['pid'] == named[0]['pid'] for e in late)
                    and all(sum(e.get('ph') == 's' and e.get('pid') == entry['pid']
                        and str(e.get('cat', '')).startswith('Flow_H2D_') and str(e.get('id')) == identity
                        for e in events) == 1 for identity in late_ids), 'late native flow joins differ')
                outcome['trace'] = dict(expected_clean_kernels=6, observed_clean_kernels=6,
                    active_late_graph_kernels_observed=len(late), collection_active_through_native_exit=True,
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
            # Keep the committed input plan unchanged during the observed job.
            # Private partial results still prevent an interrupted case retry.
            write(ROOT / 'record.json', row)
            if not outcome['passed']: save(row)
        require(outcome['passed'], outcome.get('error', 'observer fixture qualification failed'))
    row['decision'] = dict(fully_active_fixture_graph_coverage_qualified=True,
        paused_graph_construction_qualified=False, paused_boundary_qualified=False, model_capture_qualified=False,
        next='Separately qualify paused construction/flush and a request-scoped decode gate,guard bands,clock/loss checks and model observer cost.')
    save(row)
    print('Window fixture checks passed;model attribution remains unqualified.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'check'), 'profile_window.py prepare|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache.build.guards.no_model_owners([])
        {'prepare': prepare, 'check': check}[sys.argv[1]]()
