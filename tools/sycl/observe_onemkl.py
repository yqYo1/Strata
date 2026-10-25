#!/usr/bin/env python3
"""Capture one production BF16 wrapper call; coverage remains a separate gate.

Main-authored, repository LICENSE. No model run, performance comparison,
runtime environment mutation or implementation copied from another engine.
"""
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import onemkl_work as native
import ur_call_subscriber as u

r, P, B, RESULT = u.r, u.P, u.B, u.RESULT
ROOT = native.ROOT / 'first-call-v1'
KEY = ('event_reconciliation', 'oneMKL_first_call')
BUFFER_KEY = ('event_reconciliation', 'oneMKL_buffer_call')
BUFFER_ROOT = native.ROOT / 'buffer-call-v1'
require, digest, write = r.require, r.digest, r.write


def pins(revision):
    result = u.source_pins(revision)
    for name in ('tools/sycl/observe_onemkl.py', 'tools/sycl/test_observe_onemkl.py',
                 'tools/sycl/onemkl_work.py', 'tools/sycl/gemm_device_probe.cpp'):
        raw = r.validation.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact observation input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def fixture():
    row = r.validation.committed(RESULT, ('event_reconciliation', native.KEY))
    private = json.loads((native.ROOT / 'record.json').read_text())
    require(row == native.summary(private) and row['stage'] == 'built-not-executed' and row['passed'] and
        row['ownership_closed'] and not row['active'] and not row['GPU_executed'], 'closed unexecuted wrapper build required')
    native.current(private)
    require(digest(row['binary']['path']) == row['binary']['sha256'], 'wrapper ELF changed')
    manifest = private['compiled_inputs']['manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'compiled-input manifest changed')
    for path, sha in json.loads(Path(manifest['path']).read_text()).items():
        require(digest(path) == sha, 'actual compiled input changed: ' + path)
    return row['binary'], dict(native_source_commit=row['source_commit'], build_result_key=native.KEY,
        compiled_inputs=manifest, source_groups=row['source_groups'])


def prepare(*, root=ROOT, key=KEY, buffers=False):
    container = r.validation.committed(RESULT)
    require(key[1] not in container[key[0]] and not root.exists(), 'fresh first-call capture only')
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    inputs = pins(revision); binary, proof = native.retained_fixture() if buffers else fixture()
    assets = u.probe.assets(); host, reuse = u.qualified_native(u.BUFFER_KEY if buffers else u.KEY)
    root.mkdir(mode=0o700)
    for name in ('tmp', 'cache', 'trace'): (root / name).mkdir()
    template = r.validation.committed(RESULT.with_name('phase-prefill-speed-current.json'))['environment_template']
    env = {k: v.replace(template['STRATA_MEASUREMENT_ROOT'], str(root)) for k, v in template.items()}
    env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'), LD_DEBUG='libs',
        STRATA_UNITRACE_TARGET_PRELOAD=assets['preload']['library']['path'] + ':' + host['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=assets['preload']['loader'],
        XPTI_FRAMEWORK_DISPATCHER=host['assets']['dispatcher']['path'], XPTI_SUBSCRIBERS=host['library']['path'],
        XPTI_TRACE_ENABLE='1', SYCL_UR_TRACE='0', UR_L0_ENABLE_SYSMAN_ENV_DEFAULT='0',
        STRATA_UR_CALL_LOG=str(root / 'calls.jsonl'),
        UNITRACE_FollowChildProcess='1', UNITRACE_SystemTime='1', UNITRACE_TraceOutputDir=str(root / 'trace'),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.log'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    require('LD_PRELOAD' not in env and env['ZES_ENABLE_SYSMAN'] == '0', 'native-only preload and qualified safety required')
    session = 'StrataMKL' + secrets.token_hex(12)
    row = dict(stage='prepared', source_commit=revision, input_sha256=inputs, binary=binary, build_proof=proof,
        assets=assets, observer_reuse=reuse, dispatcher=host['assets']['dispatcher'], environment=env,
        session=session, command=[assets['launcher']['path'], '--session', session, '--host-timing', '--device-timing',
            '--chrome-device-logging', '--chrome-call-logging', '--chrome-event-buffer-size', '128', binary['path'], '--one-bf16'],
        host_commands=[['/usr/bin/python3', '-B', str(P / p)] for p in
            ('tests/core/test_ur_call_subscriber.py', 'tools/sycl/test_observe_onemkl.py')],
        budgets=dict(wall=120, cpu=300, rss_cap=6 << 30, text_cap=8 << 20, file_cap=32 << 20, total_cap=64 << 20),
        criteria='One fresh native process after current committed CPU observer/parser and envelope/reuse controls. Actual production '
            'Gemm::bf16,64x320x10240,FP32 output stride323,beta0. Require full20480-value exact analytic oracle, '
            'end/stride/input guards,one BEGIN/RETURN/DRAIN envelope and normal owned closure. '
            'Capture every UR callback on every thread and the actual PTI device/native API records,not just '
            'known urEnqueueKernelLaunch. Validate paired calls,clock namespace,loaded pinned observer/dispatcher/ '
            'validation/collector/NEO/UR adapter/MKL providers,silent parameter validation and no fresh fault. '
            'Record optional/rejected UR calls without treating them as successful work. Stop first capture/ '
            'native/source/provider/owner/fault/budget failure;no retry. Existing diagnostic Sysman-default override only.',
        limits='Passed means native numerical/owner/provider/UR-pair capture gates only. No one-to-one UR/device '
            'assumption. Full independent operation-to-UR/native/PTI linkage,queue/list/event generations, '
            'host-task completion,observer cost,model coverage,hardware saturation,speed,decode preservation, '
            'physical262144 and adoption remain unqualified. Phase clocks are charged envelopes,not causal links.',
        retention='Main consumes bounded raw UR/PTI records for the actual oneMKL adapter decision. '
            'Retire normal traces/JIT/cache after that closed-data analysis;keep compact evidence. '
            'Original engine+12fixtures and qualified observer unchanged. Wrapper remains only for this coverage question.',
        active=False, ownership_closed=True, passed=False, complete_device_coverage_qualified=False,
        observer_cost_qualified=False, model_qualified=False, inference_changed=False)
    row['buffer_observer'] = buffers
    if buffers:
        row['limits'] = 'Typed direct joins and external API reference generations only;driver-internal command-list/event ownership,host-payload lifetime,independent internal library work,observer cost,model coverage,performance,physical262144 and adoption remain unqualified.'
        row['criteria'] += ' Additionally require all10 direct UR/PTI links,typed enqueue/transfer ranges,balanced serial queue/event/USM/buffer references,two4-byte buffers,and two actual graph queries returning false. Internal library expected-work and host-payload lifetime remain separate.'
    container[key[0]][key[1]] = row; write(RESULT, container); write(root / 'record.json', row)
    print('Prepared single-call native-work discovery;commit plan before execution.', flush=True)


def current(row):
    require(pins(row['source_commit']) == row['input_sha256'] and
        (native.retained_fixture() if row.get('buffer_observer') else fixture()) == (row['binary'], row['build_proof']) and
        u.probe.assets() == row['assets'] and
        u.qualified_native(u.BUFFER_KEY if row.get('buffer_observer') else u.KEY)[1] == row['observer_reuse'], 'native/observer/controller inputs changed')


def analyze(stdout, spans, events, pid, start, end):
    require(stdout.count('BEGIN type=') == stdout.count('PASS type=') == 1 and
        'BEGIN type=bf16 role=HCDown rows=64 N=320 K=10240 ld=323\n' in stdout and
        'PASS type=bf16 role=HCDown rows=64 values=20480 analytic=FULL_EXACT guards=PASS inputs=PASS device_bytes=7947264\n' in stdout and
        stdout.endswith('DONE cases=1 scope=synthetic_location_probe_not_model_performance\n'), 'native oracle/guard/footer mismatch')
    begin = re.findall(r'^OP_BEGIN invocation=1 pid=(\d+) tid=(\d+) raw_ns=(\d+)$', stdout, re.M)
    returned = re.findall(r'^OP_RETURN invocation=1 raw_ns=(\d+)$', stdout, re.M)
    drained = re.findall(r'^OP_DRAIN invocation=1 raw_ns=(\d+) checked=1$', stdout, re.M)
    require(len(begin) == len(returned) == len(drained) == 1 and int(begin[0][0]) == pid and
        int(begin[0][1]) > 0, 'missing/duplicate native envelope')
    first, last, done = int(begin[0][2]), int(returned[0]), int(drained[0])
    require(start <= first <= last <= done <= end, 'invalid native envelope clock')
    gpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op']
    require(gpu and all(type(e.get('ts')) in (float, int) and type(e.get('dur')) in (float, int) and
        math.isfinite(e['ts']) and math.isfinite(e['dur']) and e['dur'] >= 0 and
        start <= e['ts'] * 1000 <= (e['ts'] + e['dur']) * 1000 <= end for e in gpu), 'device clock/capture bounds invalid')
    inner = [e for e in gpu if first <= e['ts'] * 1000 and (e['ts'] + e['dur']) * 1000 <= done]
    calls = collections.Counter((s['name'], s['status']) for s in spans if s['name'].startswith('urEnqueue'))
    return dict(oracle_values=20480, device_bytes=7947264, wrapper_invocations=1,
        source_FMA_operations=64 * 320 * 10240, source_explicit_copy_bytes=2 * 7947264,
        outer_pid=pid, outer_tid=int(begin[0][1]), begin_ns=first, return_ns=last, drain_ns=done,
        PTI_device_records=len(gpu), device_records_inside_charged_envelope=len(inner),
        enqueue_outcomes=[dict(name=n, status=s, calls=c) for (n, s), c in sorted(calls.items())],
        enqueue_threads=len({s['tid'] for s in spans if s['name'].startswith('urEnqueue')}),
        graph_API_seen=any('Graph' in s['name'] or 'CommandBuffer' in s['name'] for s in spans),
        limits='Counts and charged envelope only;no internal operation association,device saturation,independent model ledger or performance qualification.')


def direct_coverage(spans, events, pid, expected_calls):
    """Closed-data direct joins; a capability query is not graph submission.

    The direct join independently rejects unknown,deferred,multi-append and
    unpaired work. Missing typed buffer metadata remains an explicit gap.
    """
    links = u.correlate_calls(spans, events, pid, expected_calls)
    queries = [s for s in spans if s['name'] == 'urQueueIsGraphCaptureEnabledExp']
    require(all(s['status'] == 0 for s in queries), 'graph capability query rejected')
    work = [s for s in spans if any(word in s['name'] for word in ('Graph', 'CommandBuffer', 'Capture'))
            and s['name'] != 'urQueueIsGraphCaptureEnabledExp']
    require(not work, 'actual graph lifecycle/work needs a separate adapter')
    missing = collections.Counter(s['name'] for s in spans if s['name'].startswith('urEnqueue')
                                  and not s.get('metadata', {}).get('queue'))
    return dict(links=links, graph_capture_query_calls=len(queries),
        graph_capture_query_wall_ns=sum(s['end_ns'] - s['begin_ns'] for s in queries),
        graph_lifecycle_or_submission_calls=0,
        missing_typed_enqueue_queue=dict(sorted(missing.items())),
        complete_device_coverage_qualified=False, observer_cost_qualified=False,
        limits='Observed successful direct UR/native/PTI identity joins only. Query return values,buffer extents/lifetimes,independent internal library work ledger,model-route coverage and clean speed remain unqualified.')


def run(*, root=ROOT, key=KEY):
    row = r.validation.committed(RESULT, key)
    require(row['stage'] == 'prepared' and not (root / 'target-owned').exists(), 'single-use committed capture only')
    current(row); rt = r.validation.committed(r.RESULT)
    session_path = Path('/dev/shm/uctrl' + str(os.getuid()) + '_' + row['session'])
    require(not session_path.exists(), 'preexisting collector session;no overwrite')
    before = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(v.split(': ', 1)[1] for v in before.splitlines() if v.startswith('-- cursor: '))
    target = control = None
    def persist():
        row['active'] = any(o and o.active is not None for o in (target, control))
        write(root / 'record.json', dict(row, target_commands=target.commands if target else [], control_commands=control.commands if control else []))
    try:
        target = r.supervision.Owner(root / 'target-owned', persist); control = r.supervision.Owner(root / 'control-owned', persist)
        for i, command in enumerate(row['host_commands']):
            entry, _, _ = target.run('host-controls-' + str(i), command,
                dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1'), P,
                wall=45, cpu=45, rss_cap=2 << 30, text_cap=1 << 20, file_cap=1 << 20, total_cap=64 << 20)
            require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'host observer controls failed;no GPU launch')
        required = {row['observer_reuse']['library']['path'], row['dispatcher']['path'], row['assets']['collector']['path'],
            row['assets']['preload']['library']['path']}
        for token in ('libze_intel_gpu', 'libur_adapter_level_zero_v2', 'libmkl_sycl_blas'):
            found = {p for p in rt['runtime_files'] if token in p}; require(found, 'missing qualified provider: ' + token); required.update(found)
        allowed = set(rt['runtime_files']) | required | {row['binary']['path'], row['assets']['launcher']['path']}
        seen = set(); namespace = os.readlink('/proc/self/ns/time'); namespace_seen = False
        def observe(io, entry):
            nonlocal namespace_seen
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll()
                try:
                    require(os.readlink('/proc/' + str(entry['pid']) + '/ns/time') == namespace, 'native/controller time namespace differs')
                    namespace_seen = True; maps = Path('/proc', str(entry['pid']), 'maps').read_text()
                except (FileNotFoundError, ProcessLookupError): maps = ''
                for p in r.executable_mappings(maps):
                    require(p in allowed, 'unplanned executable provider: ' + p); seen.add(p)
                if select.select([io.proc.stdout], [], [], .01)[0]: io.read(); io.pending.clear()
            io.exit(1)
        row['start_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        entry, out, err = target.run('one-bf16', row['command'], row['environment'], P, interaction=observe, **row['budgets'])
        row['end_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'native wrapper owner/exit failed')
        require(required <= seen and namespace_seen, 'required actual providers/namespace not observed')
        row['activation'] = r.validation.preload_evidence(err.read_text(), entry['pid'])
        require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size, 'parameter validation warning')
        calls = u.parse_calls((root / 'calls.jsonl').read_text(), entry['pid'], intervals=True)
        spans = calls.pop('intervals'); paths = list((root / 'trace').rglob('*.json')); require(len(paths) == 1, 'one complete PTI JSON required')
        value = json.loads(paths[0].read_text()); events = value['traceEvents'] if isinstance(value, dict) else value
        row['discovery'] = analyze(out.read_text(), spans, events, entry['pid'], row['start_raw_ns'], row['end_raw_ns'])
        if row.get('buffer_observer'):
            expected = {s['name']: s['calls'] for s in row['discovery']['enqueue_outcomes']}
            row['direct_links'] = direct_coverage(spans, events, entry['pid'], expected)
            row['typed'] = u.metadata_census(spans); row['external_lifetimes'] = u.lifetime_census(spans)
            require(row['direct_links']['links']['linked_UR_instances'] == 10 and
                    not row['direct_links']['missing_typed_enqueue_queue'] and
                    row['external_lifetimes']['counts']['buffer_created'] == 2 and
                    row['external_lifetimes']['counts']['buffer_released'] == 2 and
                    row['typed']['counts']['buffer_copy_bytes'] == 8, 'actual buffer/UR coverage differs')
            queries = [s['metadata'] for s in spans if s['name'] == 'urQueueIsGraphCaptureEnabledExp']
            require(len(queries) == 2 and all(q['capture_result_valid'] == 1 and q['capture_enabled'] == 0 for q in queries),
                    'actual graph query results differ')
            require(all(s['metadata']['bytes'] == 4 for s in spans if s['name'] == 'urMemBufferCreate'),
                    'actual internal buffer extents differ')
            row['actual_typed_buffer_and_external_refs_qualified'] = True
        row['callback_capture'] = {k: calls[k] for k in ('call_pairs', 'source_threads')}
        current(row)
        row.update(stage='native-passed-coverage-pending', passed=True, host_probe_accepted=True, pid=entry['pid'],
            actual_required_providers_observed=True, environment_mutation_guard_not_triggered=True,
            analysis_inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
                for p in (out, err, root / 'calls.jsonl', root / 'ur.log', paths[0])])
    except BaseException as error: row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = all(o is None or o.active is None and all(r.supervision.closed(c) for c in o.commands) for o in (target, control))
        if session_path.exists() and row['ownership_closed'] and control:
            try:
                entry, _, _ = control.run('stop', [row['assets']['launcher']['path'], '--stop', row['session']], row['environment'], P,
                    wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20, file_cap=32 << 20, total_cap=64 << 20)
                require(r.supervision.closed(entry) and entry['exit_code'] == 0 and not session_path.exists(), 'collector session not closed')
            except BaseException as error: row.update(passed=False, cleanup_error=repr(error))
        row['ownership_closed'] = all(o is None or o.active is None and all(r.supervision.closed(c) for c in o.commands)
                                       for o in (target, control))
        try:
            done = dict(host_probe_accepted=row.get('host_probe_accepted', False))
            r.validation.finish(done, target, cursor, rt['boot'])
            row.update(fresh_faults=done['fresh_faults'], boot_unchanged=done['boot_unchanged'], driver_after=done.get('driver_after'))
            row['passed'] = bool(row['passed'] and done['passed'] and row['ownership_closed'] and not session_path.exists() and not row.get('cleanup_error'))
        except BaseException as error: row.update(passed=False, fault_check_error=repr(error))
        if not row['passed']: row['stage'] = 'failed'
        row['session_removed'] = not session_path.exists(); persist()
        container = json.loads(RESULT.read_text()); container[key[0]][key[1]] = row; write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'fresh_faults', 'discovery', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['run'], ['prepare-buffers'], ['run-buffers']),
            'prepare|run|prepare-buffers|run-buffers')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([]); r.guards.driver_guard()
        if sys.argv[1:] == ['prepare']: prepare()
        elif sys.argv[1:] == ['prepare-buffers']: prepare(root=BUFFER_ROOT, key=BUFFER_KEY, buffers=True)
        elif sys.argv[1:] == ['run-buffers']: sys.exit(run(root=BUFFER_ROOT, key=BUFFER_KEY))
        else: sys.exit(run())
