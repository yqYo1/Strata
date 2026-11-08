#!/usr/bin/env python3
"""Qualify collection through the ordinary production graph adapter.

Main-authored; repository LICENSE. Direct/active/paused-construction are separate
owned cases; commit each outcome before the next. No model or speed claim.
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
import build_compat_profile as build
import build_engine as source
import build_upstream_features as feature_build
import check_cache_loan_cli as feature
import measurement_guards as guards
import measurement_owner as ownership
import ur_validation as validation
import decode_window_trace as window
import engine_once
import profile_split as profile

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'compat-profile-window-v1', 'compat_profile_window_v1'
MODES = ('direct', 'active', 'paused-construction')
PHASES = ('before-runtime', 'runtime-ready', 'captured', 'window-drained', 'before-exit')
NAMES = ('StrataCompatWindowProduce', 'StrataCompatWindowConsume')
require, digest, write, signature = build.require, build.digest, build.write, build.signature
DEVICE_BUDGET = dict(wall=240, cpu=240, rss_cap=4 << 30, text_cap=16 << 20,
                     file_cap=32 << 20, total_cap=64 << 20)


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('source', 'environment_template', 'runtime_files')}
    out['source'] = {k: v for k, v in row['source'].items() if k != 'source_sha256'}
    out['environment_template_sha256'] = signature(row['environment_template'])
    out['runtime_files_sha256'] = signature(row['runtime_files'])
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def fixture():
    row = json.loads((build.ROOT / 'record.json').read_text()); build.current(row)
    require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'] and
        row['stage'] == 'built' and row['actual_inputs']['uncovered'] == 0 and
        row['actual_inputs']['translation_units'] == list(build.UNITS), 'closed actual-input fixture build required')
    for item in (row['binary'], row['actual_inputs']):
        require(digest(item['path']) == item['sha256'], 'fixture build output/census changed')
    census = json.loads(Path(row['actual_inputs']['path']).read_text())
    require(len(census['compiled']) == row['actual_inputs']['compiler_files'] and
        all(digest(p) == sha for group in ('compiled', 'generated', 'links')
            for p, sha in census[group].items()), 'actual fixture compiler/link inputs changed')
    return row


def environment(row, mode):
    root = ROOT / mode
    old = row['template_root']
    env = {k: v.replace(old, str(root)) for k, v in row['environment_template'].items()}
    env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
        env['NEOReadDebugKeys'] == '1' and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and
        env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0' and
        not env.get('LD_PRELOAD'), 'qualified immutable safety environment differs')
    return env


def current(row):
    require(validation.committed(RESULT, (KEY,)) == compact(row),
            'commit exact adapter-window plan/results first')
    pin = row['source']
    require(len(pin['source_sha256']) == pin['count'] and
        signature(pin['source_sha256']) == pin['aggregate_sha256'] and
        {p: digest(P / p) for p in pin['source_sha256']} == pin['source_sha256'], 'window controller source changed')
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        source.blobs(P, pin['revision'], pin['source_sha256'])}
    require(recovered == pin['source_sha256'], 'window controller source unavailable in Git')
    built = fixture()
    require(built['binary'] == row['binary'] and built['actual_inputs'] == row['actual_inputs'] and
        built['collector'] == row['assets'], 'fixture/collector identity changed')
    probe, reuse = feature.current_probe()
    require(probe == row['native_validation_probe'] and reuse == row['runtime_reuse'], 'current runtime proof changed')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and
        json.loads(Path(inv['path']).read_text()) == row['runtime_files'] and
        all(digest(p) == sha for p, sha in row['runtime_files'].items()), 'runtime inventory changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh finite window only')
    built = fixture(); probe, reuse = feature.current_probe()
    gen = validation.committed(feature_build.STATUS,
        ('current_sync', 'dflash2_cache_loan_generation_v1'))
    original = json.loads(source.git(P, 'show', gen['original_plan_commit'] + ':' +
        str(feature_build.STATUS.relative_to(P))))['current_sync']['dflash2_cache_loan_generation_v1']
    template = original['environment_template'] | original['plans'][0]['environment_overrides']
    manifest = B / 'decode-iq3-audit-speed-v2/runtime-inputs.json'
    pin = source.pin(P, ['tools/sycl', build.CPP])
    row = dict(stage='prepared', source=pin, binary=built['binary'], actual_inputs=built['actual_inputs'],
        assets=built['collector'], native_validation_probe=probe, runtime_reuse=reuse,
        environment_template=template, template_root=original['plans'][0]['root'],
        runtime_manifest=dict(path=str(manifest), sha256=digest(manifest)),
        runtime_files=json.loads(manifest.read_text()), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        driver=guards.driver_guard(), plans=[], cases=[], passed=False, complete=False,
        ownership_closed=True, active=False, whole_bytes=256 << 20,
        acceptance='Each separate owned process passes guarded exact integer results,ordinary unprofiled graph,'
            'checked teardown,pre-main validation,exact runtime/arguments/environment,normal0/reap/empty and no fresh fault.'
            ' Active and paused-construction independently require six inner named kernels/three copies,plus two'
            'post-window named kernels/one copy,unique native flow joins,whole eight named kernels and five/four'
            'copies respectively,drained clock bounds and final flush. Stop first failure;no retries or hash refresh.',
        limits='Known drained ordinary compatibility graph only. No pending-work/lossless pause boundary,'
            'SDK-native recording,model observer overhead,hardware saturation,quality,physical context or speed qualification.',
        retention='Retain compact counts,identities,clock/conservation checks,individual outcomes and decisions.'
            ' After evidence/source recovery,retire successful full API traces,timelines,cache and lifecycle output.'
            ' Keep a failed diagnostic only for a named distinct unresolved question.')
    for mode in MODES:
        prof = None if mode == 'direct' else profile.profiling(ROOT / mode, row['assets'])
        row['plans'].append(dict(mode=mode, profile=prof, environment_sha256=signature(environment(row, mode))))
    ROOT.mkdir(mode=0o700)
    for mode in MODES:
        root = ROOT / mode; root.mkdir(mode=0o700)
        for name in ('tmp', 'cache', 'trace'): (root / name).mkdir(mode=0o700)
    save(row)
    print('Prepared direct/active/paused-construction cases;commit before any device execution.', flush=True)


def trace_evidence(events, native_pid, bounds, mode):
    inner = window.summarize(events, native_pid=native_pid, begin_ns=bounds['begin'], end_ns=bounds['end'],
        capture_begin_ns=bounds['capture_begin'], capture_end_ns=bounds['capture_end'])
    whole = window.summarize(events, native_pid=native_pid, begin_ns=bounds['capture_begin'],
        end_ns=bounds['capture_end'], capture_begin_ns=bounds['capture_begin'], capture_end_ns=bounds['capture_end'])
    post_window = window.summarize(events, native_pid=native_pid, begin_ns=bounds['end'],
        end_ns=bounds['capture_end'], capture_begin_ns=bounds['capture_begin'], capture_end_ns=bounds['capture_end'])
    named = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op' and
             any(name in e.get('name', '') for name in NAMES)]
    inside = [e for e in named if bounds['begin'] / 1000 <= e['ts'] < bounds['end'] / 1000]
    post = [e for e in named if bounds['end'] / 1000 <= e['ts'] < bounds['capture_end'] / 1000]
    require(len(named) == 8 and len(inside) == 6 and len(post) == 2 and
        all(sum(name in e['name'] for e in inside) == 3 and sum(name in e['name'] for e in post) == 1
            for name in NAMES), 'named graph instance coverage differs')
    require(inner['operation_counts'].get('kernel') == 6 and inner['operation_counts'].get('copy') == 3 and
        whole['operation_counts'].get('kernel') == 8 and
        whole['operation_counts'].get('copy') == (5 if mode == 'active' else 4) and
        post_window['operation_counts'].get('kernel') == 2 and post_window['operation_counts'].get('copy') == 1 and
        not inner['boundary_crossing_counts'] and not whole['boundary_crossing_counts'] and
        not post_window['boundary_crossing_counts'],
        'inner/whole operation conservation or drained bounds differ')
    return dict(inner_kernels=6, inner_copies=3, post_kernels=2, post_copies=1,
        whole_kernels=8, whole_copies=whole['operation_counts']['copy'], unique_native_instance_joins=True,
        gpu_interval_union_us=inner['observed_gpu_interval_union_us'], window_wall_us=inner['window_wall_us'],
        collection_active_through_native_exit=True, lossless_pending_boundary_qualified=False,
        model_observer_overhead_qualified=False, physical_saturation_qualified=False)


def check(mode):
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    index = len(row['cases'])
    require(row['stage'] in ('prepared', 'checking') and index < len(MODES) and MODES[index] == mode and
        all(c['passed'] for c in row['cases']), 'fresh frozen case order only;stop first failure')
    root = ROOT / mode; require(not (root / 'execution.json').exists(), 'no duplicate fixture process')
    frozen = copy.deepcopy(row); prof = row['plans'][index]['profile']; env = environment(row, mode)
    require(signature(env) == row['plans'][index]['environment_sha256'], 'case environment changed')
    target = profile.profile_target_environment(row, env, prof) if prof else engine_once.native_environment(row, env)
    expected = profile.expected_target_environment(row, env, prof) if prof else target
    owner, controls = None, None
    phases, mapped, bounds = [], set(), dict(phases={}, controls=[])
    outcome = dict(mode=mode, complete=False, passed=False, host_probe_accepted=False,
        clock_domain='CLOCK_MONOTONIC_RAW', clock_bounds=bounds)
    cursor = profile.cursor()
    def persist():
        write(root / 'execution.json', dict(outcome, active=bool((owner and owner.active) or
            (controls and controls.active)), commands=owner.commands if owner else [],
            controls=controls.commands if controls else []))
    def control(action, state=None):
        before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        entry, _, _ = controls.run(action + '-' + str(len(controls.commands)),
            [prof['launcher']['path'], '--' + action, prof['session']], env, P,
            wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20, file_cap=32 << 20, total_cap=64 << 20)
        require(ownership.closed(entry) and entry['exit_code'] == 0, 'collector control failed')
        if state is not None: profile.session_state(prof['session'], state)
        bounds['controls'].append(dict(action=action, before_ns=before,
            after_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
        persist()
    def interact(io, entry):
        for phase in PHASES:
            while True:
                line = io.line(45)
                if line.startswith('COMPAT_PROFILE_GATE '): break
            require(line == 'COMPAT_PROFILE_GATE ' + phase, 'fixture gate order differs')
            require(ownership.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole fixture output budget')
            phases.append(phase)
            bounds['phases'][phase] = dict(raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                monotonic_ns=time.monotonic_ns(), boottime_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME))
            require(Path('/proc', str(entry['pid']), 'exe').resolve() == Path(row['binary']['path']).resolve(),
                    'actual fixture executable differs')
            require(Path('/proc', str(entry['pid']), 'cmdline').read_bytes().split(b'\0')[:-1] ==
                [row['binary']['path'].encode()], 'actual fixture arguments differ')
            actual = dict(v.split('=', 1) for v in Path('/proc', str(entry['pid']), 'environ').read_text().split('\0') if '=' in v)
            profile.same_environment(actual, expected)
            seen = profile.d.runtime.executable_mappings(Path('/proc', str(entry['pid']), 'maps').read_text())
            require(seen <= set(row['runtime_files']) | {row['binary']['path']} |
                ({prof['collector']['path']} if prof else set()), 'fixture provider differs')
            mapped.update(seen)
            if prof:
                if phase == 'before-runtime':
                    profile.session_state(prof['session'], 1)
                    if mode == 'active':
                        control('resume', 0); bounds['capture_begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                if phase == 'captured':
                    if mode == 'paused-construction':
                        profile.session_state(prof['session'], 1)
                        control('resume', 0); bounds['capture_begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    bounds['begin'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                if phase == 'window-drained': bounds['end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                profile.session_state(prof['session'], 0 if mode == 'active' or phase in
                    ('captured', 'window-drained', 'before-exit') else 1)
            persist(); io.send(b'X')
        io.proc.stdin.close(); deadline = time.monotonic() + 45
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'fixture teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        if prof: require(not profile.session_path(prof['session']).exists(), 'preexisting collector session')
        owner = ownership.Owner(root / 'owned', persist)
        controls = ownership.Owner(root / 'controls', persist)
        command = [row['binary']['path']]
        if prof: command = engine_once.profile_command(prof, command)
        entry, out, err = owner.run(mode, command, target, P, interaction=interact, **DEVICE_BUDGET)
        outcome['native_exit_code'] = entry.get('exit_code')
        require(ownership.closed(entry) and entry['exit_code'] == 0 and phases == list(PHASES),
                'fixture execution or ownership failed')
        outcome['native_execution_complete'] = True
        bounds['capture_end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        stdout, stderr = out.read_text(), err.read_text()
        for marker in ('route=compat-ordinary profile=0', 'graph_nodes=2 native_recording=0 profiled_instantiate=0',
            'clean_replays=3 kernels=6 readbacks=3 numerical=PASS guards=PASS',
            'post_window_replays=1 kernels=2 readbacks=1 numerical=PASS guards=PASS',
            'fixture_status=PASS mode=compat-profile-window checked_teardown=PASS'):
            require(stdout.count(marker) == 1, 'fixture marker differs: ' + marker)
        outcome['numerical_and_guard_check_passed'] = True
        activation = validation.preload_evidence(stderr, entry['pid'])
        required = {row['native_validation_probe']['loader']['path'],
            row['native_validation_probe']['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']}
        if prof: required.add(prof['collector']['path'])
        require(required <= mapped and activation['loader'] == row['native_validation_probe']['loader']['path'] and
            not any(s in stderr for s in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]')) and
            (not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size),
            'validation/provider warning or error')
        outcome['validation'] = activation
        clocks = list(bounds['phases'].values()); first = clocks[0]
        require(all(abs((c['boottime_ns'] - first['boottime_ns']) -
            (c['monotonic_ns'] - first['monotonic_ns'])) <= 50000000 for c in clocks), 'clock/suspend discontinuity')
        if prof:
            traces = list((root / 'trace').rglob('*.json')); require(len(traces) == 1, 'one flushed native timeline required')
            outcome['trace'] = trace_evidence(json.loads(traces[0].read_text())['traceEvents'], entry['pid'], bounds, mode)
            outcome['trace'].update(sha256=digest(traces[0]), bytes=traces[0].stat().st_size)
        current(frozen)
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=activation,
            mapped_ELF_files=len(mapped), mapped_ELF_sha256=signature(sorted(mapped)))
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if prof and owner and owner.active is None and profile.session_path(prof['session']).exists():
            try: control('stop')
            except BaseException as error: outcome['control_close_error'] = repr(error)
        if owner: validation.finish(outcome, owner, cursor, row['boot'])
        controls_closed = bool(controls and controls.active is None and
            all(ownership.closed(c) for c in controls.commands))
        outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and controls_closed)
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and outcome['ownership_closed'] and
            outcome.get('driver_after') == row['driver'] and 'control_close_error' not in outcome and
            (not prof or not profile.session_path(prof['session']).exists()))
        row['cases'].append(outcome)
        row.update(stage='failed' if not outcome['passed'] else 'closed' if index == 2 else 'checking',
            complete=bool(outcome['passed'] and index == 2), passed=bool(outcome['passed'] and index == 2),
            active=bool((owner and owner.active) or (controls and controls.active)),
            ownership_closed=outcome['ownership_closed'])
        save(row)
    require(outcome['passed'], outcome.get('error', 'adapter collection check failed'))
    print(mode + ' adapter case closed;commit before the next case. Model profiling remains unqualified.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] == ['prepare'] or len(sys.argv) == 3 and sys.argv[1] == 'check' and
        sys.argv[2] in MODES, 'profile_compat.py prepare|check direct|check active|check paused-construction')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else check(sys.argv[2])
