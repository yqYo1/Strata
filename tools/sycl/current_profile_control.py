#!/usr/bin/env python3
"""Qualify the current runtime with direct and active-from-exec collection.

Main-authored under the repository LICENSE. Reuse this fork's guarded collector
protocol and trace reducer; do not inherit the old fixture's runtime admission.
This is a measurement control, not model speed or resource-saturation evidence.
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
import current_profile_fixture as fixture_build
import profile_compat as adapter

P, B, RESULT = fixture_build.P, fixture_build.B, fixture_build.RESULT
ROOT, KEY = B / 'current-profile-control-v1', 'current_profile_control_v1'
THIS = 'tools/sycl/current_profile_control.py'
TEST = 'tools/sycl/test_current_profile_control.py'
EXTRA = {THIS, TEST}
MODES = ('direct', 'active')
require, digest, write = adapter.require, adapter.digest, adapter.write
signature = fixture_build.model.base.original.device.common.signature


def project_proof(proof, sources):
    """Remove only explicitly uncompiled new observers from an earlier proof."""
    value, inputs = copy.deepcopy(proof), dict(sources)
    require(value['observer_source_files'] == len(inputs) and
            value['observer_source_sha256'] == signature(inputs), 'observer source census differs')
    for path in EXTRA:
        require(value['declared_observers'].get(path) == inputs.get(path) and path in inputs
                and path in value['additional_observers'], 'missing explicit uncompiled observer')
        del value['declared_observers'][path]
        del inputs[path]
        value['additional_observers'].remove(path)
        value['observer_source_files'] -= 1
    value['observer_source_sha256'] = signature(inputs)
    return value


def fixture():
    built = json.loads((fixture_build.ROOT / 'record.json').read_text())
    require(adapter.validation.committed(RESULT, (fixture_build.KEY,)) == fixture_build.compact(built)
            and built['stage'] == 'built_not_executed' and built['passed'] and built['complete']
            and built['ownership_closed'] and not built['active'] and len(built['commands']) == 2
            and all(adapter.ownership.closed(c) and c['exit_code'] == 0 for c in built['commands']),
            'closed current-runtime fixture build required')
    n = fixture_build.model.native(observers=fixture_build.OBSERVERS | EXTRA)
    inputs = adapter.source.pin(P, fixture_build.model.base.original.device.common.PREFIXES)['source_sha256']
    require(project_proof(n['model_observer_admission'], inputs) == built['native_proof'],
            'fixture native source/proof changed')
    require(n['binaries'] == built['engines_and_fixtures'] and
            digest(P / fixture_build.THIS) == built['builder_sha256'] and
            hashlib.sha256(adapter.source.git(P, 'show', built['source_commit'] + ':' +
                fixture_build.THIS)).hexdigest() == built['builder_sha256'],
            'original builder or retained engine changed')
    for item in (built['binary'], built['runtime_archive']):
        require(digest(item['path']) == item['sha256'], 'fixture/archive bytes changed')
    require(all(digest(p) == h for p, h in built['runtime_objects'].items()) and
            digest(P / fixture_build.CPP) == built['fixture_source']['sha256'], 'fixture native inputs changed')
    dep = built['actual_compiler_inputs']['dependency_file']
    require(digest(dep['path']) == dep['sha256'], 'fixture dependency evidence changed')
    paths = shlex.split(Path(dep['path']).read_text().replace('\\\n', ' ').split(':', 1)[1])
    actual = {}
    for path in paths:
        p = Path(path)
        p = (p if p.is_absolute() else Path(built['build_root']) / p).resolve()
        actual[str(p)] = digest(p)
    require(len(actual) == built['actual_compiler_inputs']['files'] and
            signature(actual) == built['actual_compiler_inputs']['sha256'],
            'actual fixture compiler inputs changed')
    return built, n


def environment(native, mode, alt_driver):
    root = ROOT / mode
    env = dict(native['recipe']['environment'], TMPDIR=str(root / 'tmp'),
        XDG_CACHE_HOME=str(root / 'cache'), CUDA_MODULE_LOADING='EAGER', ZE_ENABLE_ALT_DRIVERS=alt_driver,
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    require(env.get('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD') == '1' and
            env.get('NEOReadDebugKeys') == '1' and env.get('EnableDirectSubmission') == '0' and
            env.get('STRATA_SYCL_PREFILL_SAME_QUEUE') == '1' and
            env.get('STRATA_SYCL_VERIFY_NATIVE_CAPTURE') == '0' and env.get('ZES_ENABLE_SYSMAN') == '0'
            and not any(k in env for k in ('LD_PRELOAD', 'UNITRACE_StartPaused', 'UNITRACE_Verbose')),
            'qualified immutable safety/selectors required')
    if mode == 'active':
        env.update(UNITRACE_HostTiming='1', UNITRACE_ChromeKernelLogging='1',
            UNITRACE_ChromeCallLogging='1', UNITRACE_ChromeEventBufferSize='128')
    return env


def expected_environment(row, case):
    prof, env = case['profile'], case['environment']
    if prof is None:
        return adapter.engine_once.native_environment(row, env)
    expected = adapter.profile.expected_target_environment(row, env, prof)
    expected.pop('UNITRACE_StartPaused')
    expected['UNITRACE_Verbose'] = '1'
    return expected


def next_case(row, mode):
    index = len(row['cases'])
    require(row['stage'] == 'ready' and row['host']['passed'] and index < len(MODES)
            and MODES[index] == mode and all(c['passed'] for c in row['cases']),
            'frozen direct/active order;stop on failure;no retry')
    return index


def compact(row):
    value = {k: v for k, v in row.items() if k not in ('source', 'runtime_files', 'plans')}
    value['source'] = {k: v for k, v in row['source'].items() if k != 'source_sha256'}
    value['runtime_files_sha256'] = signature(row['runtime_files'])
    value['plans'] = [{k: v for k, v in p.items() if k != 'environment'} |
                      dict(environment_sha256=signature(p['environment'])) for p in row['plans']]
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def current(row):
    require(adapter.validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact control plan/result')
    pin = row['source']
    require(signature(pin['source_sha256']) == pin['aggregate_sha256'] and
            len(pin['source_sha256']) == pin['count'] and
            all(digest(P / p) == h for p, h in pin['source_sha256'].items()) and
            {p: hashlib.sha256(raw).hexdigest() for p, raw in adapter.source.blobs(P,
                pin['revision'], pin['source_sha256'])} == pin['source_sha256'], 'control inputs changed')
    built, n = fixture()
    require(built['binary'] == row['binary'] and n['model_observer_admission'] == row['native_proof']
            and signature(fixture_build.compact(built)) == row['fixture_build_sha256'], 'fixture proof changed')
    require(adapter.profile.collector_assets(adapter.validation.committed(RESULT)) == row['assets'],
            'collector inputs/recipe/ELFs changed')
    probe, reuse = adapter.feature.current_probe()
    require(probe == row['native_validation_probe'] and reuse == row['runtime_reuse'], 'validation probe changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
            json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'] and
            all(digest(p) == h for p, h in row['runtime_files'].items()), 'runtime ELF inputs changed')
    require(adapter.guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')


def prepare():
    require(not ROOT.exists() and KEY not in adapter.validation.committed(RESULT), 'fresh control only')
    built, native = fixture()
    cohort = json.loads((B / 'cpu-mixed-cache-speed-v1/record.json').read_text())
    require(cohort['stage'] == 'closed' and cohort['ownership_closed'] and not cohort['active'],
            'close earlier model comparison first')
    assets = adapter.profile.collector_assets(adapter.validation.committed(RESULT))
    probe, reuse = adapter.feature.current_probe()
    require(probe == cohort['native_validation_probe'], 'current native probe differs')
    manifest = cohort['runtime_manifest']; files = json.loads(Path(manifest['path']).read_text())
    plans = []
    for mode in MODES:
        prof = dict(adapter.profile.profiling(ROOT / mode, assets), mode='aggregate') if mode == 'active' else None
        plans.append(dict(mode=mode, profile=prof,
            environment=environment(native, mode, cohort['environment_template']['ZE_ENABLE_ALT_DRIVERS'])))
    row = dict(stage='prepared', source=adapter.source.pin(P, ['tools/sycl', fixture_build.CPP]),
        native_source_commit=built['native_source_commit'], binary=built['binary'],
        native_proof=native['model_observer_admission'], fixture_build_sha256=signature(fixture_build.compact(built)),
        assets=assets, native_validation_probe=probe, runtime_reuse=reuse,
        runtime_manifest=manifest, runtime_files=files, boot=cohort['boot'], driver=cohort['driver'],
        plans=plans, cases=[], host=None, active=False, ownership_closed=True, passed=False, complete=False,
        budget=adapter.DEVICE_BUDGET, whole_bytes=128 << 20,
        question='Does current393 ordinary runtime retain exact guarded graph instances in direct and active-from-exec control?',
        order='Commit source/plan;four CPU controls and five existing trace controls;commit;direct;commit;active. Stop on first failed case.',
        acceptance='Exact source/build/13retainedELFs/665fixture compiler inputs;all five native gates;normal0,closed owners,quiet pre-main validation,no fresh fault. Active from exec with no pause/resume:inner6kernels/3copies,post2/1,whole8/5,unique native joins and contained RAW clocks.',
        limits='Control only;no model coverage,observer overhead,physical traffic/saturation,clean speed,task/full-context or ordinary-provider teardown claim.',
        retention='Keep compact source,counts,clocks,numerical/provider/fault/closure outcomes. Retire successful full trace/API/JIT/host output after the current collector question closes.')
    # Check all immutable evidence before allocating a fresh execution root.
    require(digest(manifest['path']) == manifest['sha256'] and all(digest(p) == h for p, h in files.items()),
            'runtime manifest changed')
    ROOT.mkdir(mode=0o700)
    for mode in MODES:
        for leaf in ('tmp', 'cache', 'trace'): (ROOT / mode / leaf).mkdir(parents=True, mode=0o700)
    save(row)
    print('Prepared current direct/active controls;commit before host stage.', flush=True)


def host():
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == 'prepared' and row['host'] is None, 'fresh host stage')
    owner = adapter.ownership.Owner(ROOT / 'host-owned')
    outcome = dict(passed=False, GPU_executed=False)
    try:
        entry, out, err = owner.run('controls', [sys.executable, '-B', '-m', 'unittest', '-v',
            'tools.sycl.test_current_profile_control', 'tools.sycl.test_profile_compat'],
            row['plans'][0]['environment'], P, wall=60, cpu=60, rss_cap=1 << 30,
            text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
        text = out.read_text() + err.read_text()
        require(adapter.ownership.closed(entry) and entry['exit_code'] == 0 and
                'Ran 9 tests' in text and '\nOK\n' in text and 'skipped' not in text, 'all nine CPU controls required')
        outcome.update(passed=True, tests=9, skipped=0, exit_code=0)
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        outcome['ownership_closed'] = owner.active is None and all(adapter.ownership.closed(c) for c in owner.commands)
        outcome['passed'] = bool(outcome['passed'] and outcome['ownership_closed'])
        row.update(host=outcome, stage='ready' if outcome['passed'] else 'failed',
            active=bool(owner.active), ownership_closed=outcome['ownership_closed']); save(row)
    require(outcome['passed'], outcome.get('error', 'host controls failed'))
    print('Nine host controls pass;commit before direct execution.', flush=True)


def check(mode):
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    index = next_case(row, mode); case = row['plans'][index]; root = ROOT / mode
    require(not (root / 'execution.json').exists(), 'one fresh process per mode')
    frozen = copy.deepcopy(row); env, prof = case['environment'], case['profile']
    target = adapter.profile.profile_target_environment(row, env, prof) if prof else expected_environment(row, case)
    expected = expected_environment(row, case)
    owner = controls = None; phases, mapped, bounds = [], set(), dict(phases={})
    outcome = dict(mode=mode, passed=False, complete=False, host_probe_accepted=False,
        clock_domain='CLOCK_MONOTONIC_RAW', clock_bounds=bounds)
    cursor = adapter.profile.cursor()

    def persist():
        write(root / 'execution.json', dict(outcome,
            active=bool((owner and owner.active) or (controls and controls.active)),
            commands=owner.commands if owner else [], controls=controls.commands if controls else []))

    def interact(io, entry):
        for phase in adapter.PHASES:
            while True:
                line = io.line(45)
                if line.startswith('COMPAT_PROFILE_GATE '): break
            require(line == 'COMPAT_PROFILE_GATE ' + phase, 'native gate order differs')
            require(adapter.ownership.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole control output budget')
            phases.append(phase)
            now = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            bounds['phases'][phase] = dict(raw_ns=now, monotonic_ns=time.monotonic_ns(),
                boottime_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME))
            if phase == 'before-runtime': bounds['capture_begin'] = now
            if phase == 'captured': bounds['begin'] = now
            if phase == 'window-drained': bounds['end'] = now
            proc = Path('/proc', str(entry['pid']))
            require((proc / 'exe').resolve() == Path(row['binary']['path']).resolve() and
                    (proc / 'cmdline').read_bytes().split(b'\0')[:-1] == [row['binary']['path'].encode()],
                    'actual executable/arguments differ')
            actual = dict(v.split('=', 1) for v in (proc / 'environ').read_text().split('\0') if '=' in v)
            adapter.profile.same_environment(actual, expected)
            seen = adapter.profile.d.runtime.executable_mappings((proc / 'maps').read_text())
            require(seen <= set(row['runtime_files']) | {row['binary']['path']} |
                    ({prof['collector']['path']} if prof else set()), 'actual provider differs')
            mapped.update(seen)
            if prof: adapter.profile.session_state(prof['session'], 0)
            persist(); io.send(b'X')
        io.proc.stdin.close(); deadline = time.monotonic() + 45
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'native teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)

    try:
        require(not prof or not adapter.profile.session_path(prof['session']).exists(), 'collector session exists')
        owner = adapter.ownership.Owner(root / 'owned', persist)
        controls = adapter.ownership.Owner(root / 'controls', persist)
        argv = adapter.engine_once.profile_command(prof, [row['binary']['path']]) if prof else [row['binary']['path']]
        entry, out, err = owner.run(mode, argv, target, P, interaction=interact, **row['budget'])
        outcome['native_exit_code'] = entry.get('exit_code')
        require(adapter.ownership.closed(entry) and entry['exit_code'] == 0 and phases == list(adapter.PHASES),
                'native closure/gates failed')
        bounds['capture_end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        stdout, stderr = out.read_text(), err.read_text()
        for marker in ('route=compat-ordinary profile=0', 'graph_nodes=2 native_recording=0 profiled_instantiate=0',
                'clean_replays=3 kernels=6 readbacks=3 numerical=PASS guards=PASS',
                'post_window_replays=1 kernels=2 readbacks=1 numerical=PASS guards=PASS',
                'fixture_status=PASS mode=compat-profile-window checked_teardown=PASS'):
            require(stdout.count(marker) == 1, 'native result differs: ' + marker)
        outcome['numerical_and_guard_check_passed'] = True
        activation = adapter.validation.preload_evidence(stderr, entry['pid'])
        required = {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'],
                    env['ZE_ENABLE_ALT_DRIVERS']} | ({prof['collector']['path']} if prof else set())
        require(required <= mapped and activation['loader'] == row['native_validation_probe']['loader']['path'] and
                not any(s in stderr for s in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]')) and
                (not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size),
                'validation/provider error')
        outcome['validation'] = activation
        clocks = list(bounds['phases'].values()); first = clocks[0]
        require(all(abs((c['boottime_ns'] - first['boottime_ns']) -
                (c['monotonic_ns'] - first['monotonic_ns'])) <= 50000000 for c in clocks), 'clock discontinuity')
        traces = list((root / 'trace').rglob('*.json'))
        require(len(traces) == (1 if prof else 0), 'unexpected flushed timeline count')
        if prof:
            outcome['trace'] = adapter.trace_evidence(json.loads(traces[0].read_text())['traceEvents'],
                entry['pid'], bounds, 'active')
            outcome['trace'].update(sha256=digest(traces[0]), bytes=traces[0].stat().st_size,
                collection_active_from_exec=True)
        current(frozen)
        outcome.update(complete=True, host_probe_accepted=True, mapped_ELF_files=len(mapped),
                       mapped_ELF_sha256=signature(sorted(mapped)))
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if prof and controls and owner and owner.active is None and adapter.profile.session_path(prof['session']).exists():
            try:
                e, _, _ = controls.run('stop', [prof['launcher']['path'], '--stop', prof['session']], env, P,
                    wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20, file_cap=32 << 20, total_cap=64 << 20)
                require(adapter.ownership.closed(e) and e['exit_code'] == 0, 'collector stop failed')
            except BaseException as error: outcome['control_close_error'] = repr(error)
        if owner: adapter.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and controls and controls.active is None
            and all(adapter.ownership.closed(c) for c in controls.commands))
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and outcome['ownership_closed']
            and outcome.get('driver_after') == row['driver'] and 'control_close_error' not in outcome
            and (not prof or not adapter.profile.session_path(prof['session']).exists()))
        row['cases'].append(outcome)
        row.update(stage='closed' if outcome['passed'] and index == 1 else 'ready' if outcome['passed'] else 'failed',
            passed=outcome['passed'] and index == 1, complete=outcome['passed'] and index == 1,
            active=bool((owner and owner.active) or (controls and controls.active)),
            ownership_closed=outcome['ownership_closed']); save(row)
    require(outcome['passed'], outcome.get('error', 'collector control failed'))
    print('Current ' + mode + ' control closed;commit outcome before advancing.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['host'], ['direct'], ['active']), 'use prepare|host|direct|active')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); adapter.guards.no_model_owners([])
        action = sys.argv[1]
        prepare() if action == 'prepare' else host() if action == 'host' else check(action)
