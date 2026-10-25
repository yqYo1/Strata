#!/usr/bin/env python3
"""Current-runtime census probe using the retained original submission fixture.

No rebuild or model run. This first probe tests counts/bytes and clock/flow
coverage; it does NOT qualify per-operation identity or the complete model route.
The next adapter step consumes its bounded UR/PTI records. Repository LICENSE.
"""
import ast
import collections
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import runtime_probe as runtime
import unitrace_launch as launch

P, B = runtime.P, runtime.B
RESULT = runtime.RESULT.with_name('prefill-resources-current.json')
ROOT = B / 'prefill-event-census-current-runtime-v1'
KEY = ('event_reconciliation', 'native_observation')
require, digest = runtime.require, runtime.digest


def source_pins(revision):
    pending = [Path(__file__).resolve(), P / 'tests/core/test_work_ledger_fixture.py']
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in modules:
                local = P / 'tools/sycl' / ((name or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    pins = {}
    for path in sorted(seen):
        name = str(path.relative_to(P))
        raw = runtime.validation.git('show', revision + ':' + name)
        require(raw == path.read_bytes(), 'commit exact census input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def assets():
    row = runtime.validation.committed(runtime.RESULT)
    full = runtime.current(row)
    collector = runtime.validation.committed(launch.RESULT, ('unitrace_build',))
    owner = runtime.validation.committed(launch.RESULT, ('unitrace_session_owner',))
    require(all(r['passed'] and r['ownership_closed'] and not r['active'] for r in (collector, owner)),
            'closed collector/launcher build required')
    launch.validate(owner, collector)
    fixture = full['fixtures']['sycl_submission_test']
    for binary in (fixture, owner['binary'], collector['artifacts']['libunitrace_tool.so']):
        require(digest(binary['path']) == binary['sha256'], 'census executable changed')
    native = full.get('binary_origin_source_commit', full['source_commit'])
    require(native == '6e40afc1beddd059ee518617f6c1efce1da14aca', 'selected native source changed')
    # Recover the actual independently inspected route, not a nearby worktree.
    source = runtime.validation.git('show', native + ':tests/sycl/compat.cpp')
    require(source == (P / 'tests/sycl/compat.cpp').read_bytes(), 'fixture source differs from selected build')
    return dict(native_source_commit=native, fixture=fixture, native_fixture_source_sha256=hashlib.sha256(source).hexdigest(),
                runtime_result_sha256=digest(runtime.RESULT), collector=collector['artifacts']['libunitrace_tool.so'],
                launcher=owner['binary'], preload=row['preload'])


def source_census():
    # tests/sycl/compat.cpp --submission, inspected independently of any trace:
    # first kernel, five queued batches, three ordinary-RAM round trips.
    # Copy extents: five initial,32 default-stream,10 guarded,6 ordinary.
    return dict(kernel=1 + sum((1, 119, 120, 121, 241)) + 3, copy=5 + 32 + 10 + 6,
                fill=1, barrier=3 + 2 + 5 * 3, copy_bytes=(5 + 32 + 6) * 1028 + 10 * 1164,
                fill_bytes=1028)


def api_census(text):
    counts = collections.Counter()
    calls = collections.Counter()
    for name, args, status in re.findall(r'<--- (ur\w+)\((.*)\) -> (UR_RESULT_\w+);', text):
        if not name.startswith('urEnqueue'):
            require(not name.startswith('urCommandBuffer'), 'unexpected graph construction in submission route')
            continue
        require(status == 'UR_RESULT_SUCCESS', 'failed enqueue: ' + name)
        calls[name] += 1
        if name.startswith('urEnqueueKernelLaunch'):
            kind = 'kernel'
        elif name in ('urEnqueueUSMMemcpy', 'urEnqueueUSMFill'):
            kind = 'copy' if name.endswith('Memcpy') else 'fill'
            size = re.search(r'\.size = (\d+)', args)
            require(size is not None, 'missing requested extent')
            counts[kind + '_bytes'] += int(size[1])
        elif name in ('urEnqueueEventsWaitWithBarrier', 'urEnqueueEventsWaitWithBarrierExt'):
            kind = 'barrier'
        elif name == 'urEnqueueEventsWait':
            continue  # Host wait; never manufacture a device barrier from it.
        else:
            raise RuntimeError('unclassified enqueue: ' + name)
        counts[kind] += 1
    require(bool(counts), 'no successful submitted work')
    return dict(counts= dict(counts), enqueue_calls=dict(calls))


def device_census(events, native_pid, begin_ns, end_ns):
    gpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op']
    host = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op']
    counts = collections.Counter()
    ids = []
    for event in gpu:
        name = event.get('name', '')
        if name.startswith('zeCommandListAppendMemoryCopy'):
            kind = 'copy'
        elif name.startswith('zeCommandListAppendMemoryFill'):
            kind = 'fill'
        elif name == 'zeCommandListAppendBarrier':
            kind = 'barrier'
        else:
            require('main' in name, 'unclassified device work: ' + name)
            kind = 'kernel'
        if kind in ('copy', 'fill'):
            size = re.search(r'\[(\d+)\]$', name)
            require(size is not None, 'missing device requested extent')
            counts[kind + '_bytes'] += int(size[1])
        counts[kind] += 1
        ts, duration = event.get('ts'), event.get('dur')
        require(type(ts) in (int, float) and type(duration) in (int, float) and
                begin_ns / 1000 <= ts <= ts + duration <= end_ns / 1000 and
                (duration >= 0 if kind == 'barrier' else duration > 0), 'invalid device clock')
        ids.append(str(event.get('args', {}).get('id', '')))
    require(gpu and host and all(ids) and len(set(ids)) == len(ids), 'missing/duplicate device IDs or host events')
    sources = collections.Counter(str(e.get('id', '')) for e in events if e.get('ph') == 's' and
        e.get('pid') == native_pid and e.get('cat', '').startswith('Flow_H2D_'))
    require(all(sources[i] == 1 for i in ids), 'missing or duplicated actual host flow source')
    return dict(counts=dict(counts), host_events=len(host), device_events=len(gpu),
                device_instance_ids_unique=True, all_device_host_sources=True, raw_clock_bounds=True)


def prepare():
    require(not ROOT.exists(), 'inspect existing census owner; no retry')
    revision = runtime.validation.git('rev-parse', 'HEAD').decode().strip()
    pins, pinned = source_pins(revision), assets()
    container = runtime.validation.committed(RESULT)
    host = container['event_reconciliation']
    require('native_observation' not in host and host['host_controls']['tests'] == 18 and
            host['host_controls']['exit_code'] == 0, 'closed host ledger controls required')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache', 'trace'):
        (ROOT / name).mkdir()
    env = dict(runtime.validation.committed(runtime.SETTINGS)['environment'],
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'), LD_DEBUG='libs',
        STRATA_UNITRACE_TARGET_PRELOAD=pinned['preload']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=pinned['preload']['loader'],
        UNITRACE_FollowChildProcess='1', UNITRACE_SystemTime='1', UNITRACE_TraceOutputDir=str(ROOT / 'trace'),
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(ROOT / 'ur.log'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    session = 'StrataCensus' + secrets.token_hex(12)
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins, assets=pinned,
        expected_source_census=source_census(), environment=env, session=session,
        command=[pinned['launcher']['path'], '--session', session, '--host-timing', '--device-timing',
                 '--chrome-device-logging', '--chrome-call-logging', '--chrome-event-buffer-size', '128',
                 pinned['fixture']['path'], '--submission'],
        budgets=dict(wall=90, cpu=90, rss_cap=4 << 30, text_cap=8 << 20, file_cap=32 << 20, total_cap=64 << 20),
        gates='One already qualified original6e40 submission fixture on current NEO. Require606 kernels,53 copies/55844 requested bytes,one fill/1028 bytes,20 barriers from independent source census,actual successful API and PTI records; complete owner/provider/pre-main validation/numerical/drain/fault evidence. Any failed gate stops;no graph/model run. This count screen is NOT per-operation identity,shape,graph/lifetime coverage,observer overhead,production critical path or speed qualification.',
        retention='Main consumes bounded raw UR/PTI to implement current origin/geometry adapter. Retain at most64MiB until that analysis closes;then preserve compact counts,failure/coverage limits and delete raw/JIT/normal logs with manifest. No new binary generation.',
        active=False, ownership_closed=True, passed=False, model_qualified=False,
        per_operation_identity_qualified=False, observer_overhead_qualified=False)
    host['native_observation'] = row
    runtime.write(ROOT / 'record.json', row)
    runtime.write(RESULT, container)
    print('Prepared current-runtime count/flow probe; commit plan before run.', flush=True)


def run():
    row = runtime.validation.committed(RESULT, KEY)
    require(row['stage'] == 'prepared' and not (ROOT / 'target-owned').exists(), 'single-use committed census required')
    require(source_pins(row['source_commit']) == row['input_sha256'] and assets() == row['assets'] and
            source_census() == row['expected_source_census'], 'census inputs/assets changed')
    initial = copy.deepcopy(row)
    runtime_row = runtime.validation.committed(runtime.RESULT)
    session_path = Path('/dev/shm/uctrl' + str(os.getuid()) + '_' + row['session'])
    require(not session_path.exists(), 'preexisting collector session; no overwrite')
    journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(s.split(': ', 1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
    target = control = None

    def persist():
        row['active'] = any(o and o.active is not None for o in (target, control))
        runtime.write(ROOT / 'record.json', dict(row, target_commands=target.commands if target else [],
                                               control_commands=control.commands if control else []))

    try:
        target = runtime.supervision.Owner(ROOT / 'target-owned', persist)
        control = runtime.supervision.Owner(ROOT / 'control-owned', persist)
        row['stage'] = 'observing'
        seen = set()
        allowed = set(runtime_row['runtime_files']) | {str(Path(row['assets'][k]['path']).resolve())
            for k in ('collector', 'launcher', 'fixture')}

        def observe(io, entry):
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll()
                try:
                    maps = Path('/proc', str(entry['pid']), 'maps').read_text()
                except (FileNotFoundError, ProcessLookupError):
                    maps = ''
                for path in runtime.executable_mappings(maps):
                    require(path in allowed, 'unplanned executable provider: ' + path)
                    seen.add(path)
                if select.select([io.proc.stdout], [], [], .01)[0]:
                    io.read()
                    io.pending.clear()
            io.exit(1)

        row['begin_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        entry, out, err = target.run('submission', row['command'], row['environment'], P,
                                     interaction=observe, **row['budgets'])
        row['end_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        require(runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'native fixture owner/exit failed')
        row['pid'] = entry['pid']
        row['validation'] = runtime.validation.preload_evidence(err.read_text(), entry['pid'])
        require('fixture_status=PASS mode=submission' in out.read_text(), 'native numerical/drain footer missing')
        require(not (ROOT / 'validation.log').exists() or not (ROOT / 'validation.log').stat().st_size,
                'parameter validation warning')
        neo = next(path for path in runtime_row['runtime_files'] if 'libze_intel_gpu' in path)
        require({neo, row['assets']['collector']['path']} <= seen, 'updated NEO/collector not observed')
        row['actual_runtime_providers_verified'] = True
        row['api'] = api_census((ROOT / 'ur.log').read_text())
        traces = list((ROOT / 'trace').rglob('*.json'))
        require(len(traces) == 1, 'one complete native JSON timeline required')
        raw = traces[0].read_bytes()
        value = json.loads(raw)
        events = value['traceEvents'] if isinstance(value, dict) else value
        row['device'] = device_census(events, entry['pid'], row['begin_raw_ns'], row['end_raw_ns'])
        row['timeline'] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        require(row['api']['counts'] == row['device']['counts'] == row['expected_source_census'],
                'source/API/device census mismatch')
        require(assets() == initial['assets'] and source_pins(row['source_commit']) == row['input_sha256'],
                'post-closure native/runtime inputs changed')
        row.update(stage='count-screen-passed', passed=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = all(o is None or (o.active is None and
            all(runtime.supervision.closed(c) for c in o.commands)) for o in (target, control))
        if session_path.exists() and row['ownership_closed'] and control is not None:
            try:
                entry, _, _ = control.run('stop', [row['assets']['launcher']['path'], '--stop', row['session']],
                    row['environment'], P, wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20,
                    file_cap=32 << 20, total_cap=64 << 20)
                require(runtime.supervision.closed(entry) and entry['exit_code'] == 0 and not session_path.exists(),
                        'collector session closure failed')
            except BaseException as error:
                row.update(passed=False, cleanup_error=repr(error))
        row['session_removed'] = not session_path.exists()
        try:
            check = dict(host_probe_accepted=True)
            runtime.validation.finish(check, target, cursor, runtime_row['boot'])
            row.update(fresh_faults=check['fresh_faults'], boot_unchanged=check['boot_unchanged'],
                       driver_after=check['driver_after'])
            row['passed'] = bool(row['passed'] and check['passed'] and row['ownership_closed'] and
                                 row['session_removed'] and not row.get('cleanup_error'))
        except BaseException as error:
            row.update(passed=False, fault_check_error=repr(error))
        if not row['passed']:
            row['stage'] = 'failed'
        persist()
        container = json.loads(RESULT.read_text())
        container[KEY[0]][KEY[1]] = row
        runtime.write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'fresh_faults', 'error')}), flush=True)
    return 0 if row['passed'] else 1


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use work_ledger_fixture.py prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.guards.no_model_owners([])
        raise SystemExit(prepare() if sys.argv[1] == 'prepare' else run())
