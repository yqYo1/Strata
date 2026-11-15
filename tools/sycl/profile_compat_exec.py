#!/usr/bin/env python3
"""Qualify current graph collection active from exec, without rebuilding.

Main-authored; repository LICENSE. The paused-construction attempt stays failed.
This separate mode uses the existing aggregate launcher plus explicit host/call
logging. It qualifies a fixture, not model coverage, overhead or saturation.
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
import profile_compat as adapter

P, B, RESULT = adapter.P, adapter.B, adapter.RESULT
ROOT, KEY = B / 'compat-profile-exec-v1', 'compat_profile_exec_v1'
require, digest, write, signature = adapter.require, adapter.digest, adapter.write, adapter.signature


def selection(proof):
    """Select the passing active arm; never relabel the failed comparison."""
    cases = proof['cases']
    require(proof['stage'] == 'failed' and not proof['passed'] and not proof['complete'] and
        proof['ownership_closed'] and not proof['active'] and
        [c['mode'] for c in cases] == list(adapter.MODES), 'preserve the closed three-case outcome')
    require(all(c['passed'] and c['complete'] and c['native_exit_code'] == 0 and
        c['ownership_closed'] and c['fresh_faults'] == [] for c in cases[:2]),
        'closed passing direct/active controls required')
    failed = cases[2]
    require(not failed['passed'] and not failed['complete'] and failed['native_exit_code'] == 0 and
        failed['native_execution_complete'] and failed['numerical_and_guard_check_passed'] and
        failed['ownership_closed'] and failed['fresh_faults'] == [] and
        failed['error'] == "ValueError('GPU kernels/host APIs missing from window')",
        'original paused-construction refusal changed')
    trace = cases[1]['trace']
    require([trace[k] for k in ('inner_kernels', 'inner_copies', 'post_kernels', 'post_copies',
        'whole_kernels', 'whole_copies')] == [6, 3, 2, 1, 8, 5] and
        trace['unique_native_instance_joins'] and trace['collection_active_through_native_exit'],
        'active control lacks complete guarded instance coverage')
    return dict(source_commit=proof['source']['revision'], binary=proof['binary'], assets=proof['assets'],
        active_trace_sha256=trace['sha256'], direct_and_active_passed=True,
        paused_construction_passed=False, paused_native_exit_code=0,
        original_evidence_commit='4d1528859a8dd65fb5b63fe3a5c3965516505e80')


def environment(seed):
    old = str(adapter.ROOT / 'active')
    env = {k: v.replace(old, str(ROOT)) for k, v in adapter.environment(seed, 'active').items()}
    require(not any(k in env for k in ('LD_PRELOAD', 'UNITRACE_StartPaused', 'UNITRACE_Verbose')),
        'fresh active-from-exec launcher environment required')
    env.update(UNITRACE_HostTiming='1', UNITRACE_ChromeKernelLogging='1',
        UNITRACE_ChromeCallLogging='1', UNITRACE_ChromeEventBufferSize='128')
    return env


def expected_environment(row):
    expected = adapter.profile.expected_target_environment(row, row['environment'], row['profile'])
    expected.pop('UNITRACE_StartPaused')
    expected['UNITRACE_Verbose'] = '1'
    return expected


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('source', 'environment', 'runtime_files')}
    out['source'] = {k: v for k, v in row['source'].items() if k != 'source_sha256'}
    out['environment_sha256'] = signature(row['environment'])
    out['runtime_files_sha256'] = signature(row['runtime_files'])
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def current(row):
    require(adapter.validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact exec-mode plan/result')
    pin = row['source']
    require(len(pin['source_sha256']) == pin['count'] and
        signature(pin['source_sha256']) == pin['aggregate_sha256'] and
        {n: digest(P / n) for n in pin['source_sha256']} == pin['source_sha256'] and
        {n: hashlib.sha256(raw).hexdigest() for n, raw in adapter.source.blobs(P,
            pin['revision'], pin['source_sha256'])} == pin['source_sha256'], 'exec controller source changed')
    require(selection(adapter.validation.committed(RESULT, (adapter.KEY,))) == row['adapter_control'] and
        adapter.host_qualification() == row['adapter_host_controls'], 'prior adapter proof changed')
    fixture = adapter.fixture()
    require(fixture['binary'] == row['binary'] and fixture['actual_inputs'] == row['actual_inputs'] and
        fixture['collector'] == row['assets'], 'actual fixture/collector identity changed')
    probe, reuse = adapter.feature.current_probe()
    require(probe == row['native_validation_probe'] and reuse == row['runtime_reuse'], 'runtime proof changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'] and
        all(digest(n) == sha for n, sha in row['runtime_files'].items()), 'runtime ELF inventory changed')
    require(adapter.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')


def prepare():
    require(not ROOT.exists() and KEY not in adapter.validation.committed(RESULT), 'fresh exec-mode plan only')
    control = selection(adapter.validation.committed(RESULT, (adapter.KEY,)))
    seed = json.loads((adapter.ROOT / 'record.json').read_text())
    require(adapter.compact(seed) == adapter.validation.committed(RESULT, (adapter.KEY,)), 'original private recipe differs')
    fixture = adapter.fixture(); probe, reuse = adapter.feature.current_probe()
    require(fixture['binary'] == control['binary'] and fixture['collector'] == control['assets'], 'same native assets required')
    profile = dict(adapter.profile.profiling(ROOT, fixture['collector']), mode='aggregate')
    row = dict(stage='prepared', source=adapter.source.pin(P, ['tools/sycl', adapter.build.CPP]),
        adapter_control=control, adapter_host_controls=adapter.host_qualification(),
        binary=fixture['binary'], actual_inputs=fixture['actual_inputs'], assets=fixture['collector'],
        native_validation_probe=probe, runtime_reuse=reuse, environment=environment(seed), profile=profile,
        runtime_manifest=seed['runtime_manifest'], runtime_files=seed['runtime_files'], boot=seed['boot'], driver=seed['driver'],
        host=None, outcome=None, active=False, ownership_closed=True, passed=False, complete=False,
        budget=adapter.DEVICE_BUDGET, whole_bytes=128 << 20,
        question='Does the existing aggregate launcher with host/call logging retain every current ordinary graph instance when active from exec?',
        order='Commit source/plan,run three CPU controls,commit their result,then one finite same-ELF process. Never resume a graph built paused.',
        acceptance='Same source/fixture/census/collector/runtime and immutable safety. Active named session at all five native gates;no pause/resume calls. Exact guarded results,normal0/reap/empty,quiet before-main validation,no fresh fault. Six inner kernels/three copies,two post kernels/one copy,whole eight kernels/five copies,unique native flow joins and contained RAW clocks through native exit/flush.',
        limits='Separate launcher-mode qualification only. The three-case V1 remains failed. No model coverage,observer overhead,physical traffic/saturation,clean speed,task quality,physical context or ordinary provider teardown qualification.',
        retention='Keep compact source,counts,clocks,numerical/provider/fault/closure evidence. Retire normal API/timeline/JIT/host output after verification. Reuse the named fixture generation;do not build another engine.')
    ROOT.mkdir(mode=0o700)
    for n in ('tmp', 'cache', 'trace'): (ROOT / n).mkdir(mode=0o700)
    save(row)
    print('Prepared active-from-exec fixture mode;commit before CPU controls.', flush=True)


def host():
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == 'prepared' and row['host'] is None, 'fresh CPU-control stage only')
    owner = adapter.ownership.Owner(ROOT / 'host-owned')
    outcome = dict(passed=False, GPU_executed=False)
    try:
        entry, out, err = owner.run('exec-host', [sys.executable, '-B', '-m', 'unittest', '-v',
            'tools.sycl.test_profile_compat_exec'], row['environment'], P,
            wall=60, cpu=60, rss_cap=1 << 30, text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
        text = out.read_text() + err.read_text()
        require(adapter.ownership.closed(entry) and entry['exit_code'] == 0 and
            'Ran 3 tests' in text and '\nOK\n' in text and 'skipped' not in text, 'all three CPU controls required')
        outcome.update(passed=True, tests=3, skipped=0, exit_code=0)
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        outcome['ownership_closed'] = owner.active is None and all(adapter.ownership.closed(c) for c in owner.commands)
        outcome['passed'] = bool(outcome['passed'] and outcome['ownership_closed'])
        row.update(host=outcome, stage='ready' if outcome['passed'] else 'failed',
            active=bool(owner.active), ownership_closed=outcome['ownership_closed']); save(row)
    require(outcome['passed'], outcome.get('error', 'CPU control failed'))
    print('Three CPU controls pass;commit before GPU execution.', flush=True)


def check():
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == 'ready' and row['host']['passed'] and row['outcome'] is None and
        not (ROOT / 'execution.json').exists(), 'one committed fresh native case only')
    frozen = copy.deepcopy(row); env = row['environment']; prof = row['profile']
    target = adapter.profile.profile_target_environment(row, env, prof)
    expected = expected_environment(row)
    owner = controls = None; phases = []; mapped = set(); bounds = dict(phases={})
    outcome = dict(passed=False, complete=False, host_probe_accepted=False, clock_domain='CLOCK_MONOTONIC_RAW', clock_bounds=bounds)
    cursor = adapter.profile.cursor()
    def persist():
        write(ROOT / 'execution.json', dict(outcome, active=bool((owner and owner.active) or
            (controls and controls.active)), commands=owner.commands if owner else [], controls=controls.commands if controls else []))
    def interact(io, entry):
        for phase in adapter.PHASES:
            while True:
                line = io.line(45)
                if line.startswith('COMPAT_PROFILE_GATE '): break
            require(line == 'COMPAT_PROFILE_GATE ' + phase, 'native gate order differs')
            require(adapter.ownership.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole exec-mode output budget')
            phases.append(phase)
            now = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            bounds['phases'][phase] = dict(raw_ns=now, monotonic_ns=time.monotonic_ns(),
                boottime_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME))
            if phase == 'before-runtime': bounds['capture_begin'] = now
            if phase == 'captured': bounds['begin'] = now
            if phase == 'window-drained': bounds['end'] = now
            proc = Path('/proc', str(entry['pid']))
            require((proc / 'exe').resolve() == Path(row['binary']['path']).resolve() and
                (proc / 'cmdline').read_bytes().split(b'\0')[:-1] == [row['binary']['path'].encode()], 'native executable/arguments differ')
            actual = dict(v.split('=', 1) for v in (proc / 'environ').read_text().split('\0') if '=' in v)
            adapter.profile.same_environment(actual, expected)
            seen = adapter.profile.d.runtime.executable_mappings((proc / 'maps').read_text())
            require(seen <= set(row['runtime_files']) | {row['binary']['path'], prof['collector']['path']}, 'native provider differs')
            mapped.update(seen); adapter.profile.session_state(prof['session'], 0)
            persist(); io.send(b'X')
        io.proc.stdin.close(); deadline = time.monotonic() + 45
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'native teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        require(not adapter.profile.session_path(prof['session']).exists(), 'preexisting collector session')
        owner = adapter.ownership.Owner(ROOT / 'owned', persist)
        controls = adapter.ownership.Owner(ROOT / 'controls', persist)
        entry, out, err = owner.run('exec-active', adapter.engine_once.profile_command(prof, [row['binary']['path']]),
            target, P, interaction=interact, **row['budget'])
        outcome['native_exit_code'] = entry.get('exit_code')
        require(adapter.ownership.closed(entry) and entry['exit_code'] == 0 and phases == list(adapter.PHASES), 'native closure/gates failed')
        outcome['native_execution_complete'] = True
        bounds['capture_end'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        stdout, stderr = out.read_text(), err.read_text()
        for marker in ('route=compat-ordinary profile=0', 'graph_nodes=2 native_recording=0 profiled_instantiate=0',
            'clean_replays=3 kernels=6 readbacks=3 numerical=PASS guards=PASS',
            'post_window_replays=1 kernels=2 readbacks=1 numerical=PASS guards=PASS',
            'fixture_status=PASS mode=compat-profile-window checked_teardown=PASS'):
            require(stdout.count(marker) == 1, 'native result marker differs: ' + marker)
        outcome['numerical_and_guard_check_passed'] = True
        activation = adapter.validation.preload_evidence(stderr, entry['pid'])
        required = {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'],
            env['ZE_ENABLE_ALT_DRIVERS'], prof['collector']['path']}
        require(required <= mapped and activation['loader'] == row['native_validation_probe']['loader']['path'] and
            not any(s in stderr for s in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]')) and
            (not (ROOT / 'validation.log').exists() or not (ROOT / 'validation.log').stat().st_size), 'validation/provider error')
        outcome['validation'] = activation
        clocks = list(bounds['phases'].values()); first = clocks[0]
        require(all(abs((c['boottime_ns'] - first['boottime_ns']) - (c['monotonic_ns'] - first['monotonic_ns'])) <= 50000000
            for c in clocks), 'clock/suspend discontinuity')
        traces = list((ROOT / 'trace').rglob('*.json')); require(len(traces) == 1, 'one flushed native timeline required')
        outcome['trace'] = adapter.trace_evidence(json.loads(traces[0].read_text())['traceEvents'], entry['pid'], bounds, 'active')
        outcome['trace'].update(sha256=digest(traces[0]), bytes=traces[0].stat().st_size, collection_active_from_exec=True)
        current(frozen)
        outcome.update(complete=True, host_probe_accepted=True, mapped_ELF_files=len(mapped),
            mapped_ELF_sha256=signature(sorted(mapped)))
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if controls and owner and owner.active is None and adapter.profile.session_path(prof['session']).exists():
            try:
                entry, _, _ = controls.run('stop', [prof['launcher']['path'], '--stop', prof['session']], env, P,
                    wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20, file_cap=32 << 20, total_cap=64 << 20)
                require(adapter.ownership.closed(entry) and entry['exit_code'] == 0, 'collector stop failed')
            except BaseException as error: outcome['control_close_error'] = repr(error)
        if owner: adapter.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and controls and controls.active is None and
            all(adapter.ownership.closed(c) for c in controls.commands))
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and outcome['ownership_closed'] and
            outcome.get('driver_after') == row['driver'] and 'control_close_error' not in outcome and
            not adapter.profile.session_path(prof['session']).exists())
        row.update(outcome=outcome, stage='closed' if outcome['passed'] else 'failed', passed=outcome['passed'],
            complete=outcome['complete'], active=bool((owner and owner.active) or (controls and controls.active)),
            ownership_closed=outcome['ownership_closed']); save(row)
    require(outcome['passed'], outcome.get('error', 'active-from-exec collector check failed'))
    print('Active-from-exec fixture mode closed;model coverage/overhead remain unqualified.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['host'], ['check']), 'profile_compat_exec.py prepare|host|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); adapter.guards.no_model_owners([])
        {'prepare': prepare, 'host': host, 'check': check}[sys.argv[1]]()
