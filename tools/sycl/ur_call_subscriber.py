#!/usr/bin/env python3
"""Build and qualify a pre-exec UR-call subscriber on the CPU before GPU use.

This diagnostic has no production engine changes. Retained original6e40 and
its12 fixtures stay selected; one small observer ELF serves the next GPU probe.
Repository LICENSE; no external implementation copied.
"""
import collections
import copy
import bisect
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import select
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import work_ledger_fixture as probe

r = probe.runtime
P, B, RESULT = probe.P, probe.B, probe.RESULT
ROOT = B / 'ur-call-subscriber-v2'
SOURCE = 'bench/sycl/ur_call_subscriber.cpp'
DRIVER_SOURCE = 'bench/sycl/ur_call_guard_test.cpp'
LEGACY_KEY = ('event_reconciliation', 'ur_subscriber')
DIRECT_KEY = ('event_reconciliation', 'ur_subscriber_gpu')
KEY = ('event_reconciliation', 'ur_subscriber_metadata')
GPU_KEY = ('event_reconciliation', 'ur_metadata_gpu')
GPU_ROOT = ROOT / 'gpu-direct-v1'
SDK_KEY = ('UR_PTI_identity_bridge_20261025', 'exact_SDK_inspection')
require, digest = r.require, r.digest


def source_pins(revision):
    pins = probe.source_pins(revision)
    for name in (SOURCE, DRIVER_SOURCE, 'tools/sycl/ur_call_subscriber.py', 'tests/core/test_ur_call_subscriber.py'):
        raw = r.validation.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact subscriber input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def parse_calls(text, pid, *, intervals=False):
    """Validate complete begin/end instance pairs without inventing PTI links."""
    require(len(text.encode()) <= 16 << 20, 'subscriber text budget')
    rows = [json.loads(line) for line in text.splitlines()]
    require(rows and rows[0]['kind'] == 'load', 'missing subscriber load')
    require(all(type(e.get('pid')) is int and e['pid'] == pid and type(e.get('raw_ns')) is int and
                e['raw_ns'] >= rows[0]['raw_ns'] for e in rows), 'invalid subscriber owner/clock')
    require(sum(e['kind'] == 'load' for e in rows) == 1, 'duplicate subscriber load')
    subscribed = finished = False
    pending, seen, stacks, functions = {}, set(), collections.defaultdict(list), {}
    counts = collections.Counter()
    spans = []
    for event in rows[1:]:
        kind = event['kind']
        if kind in ('subscribe', 'finish'):
            require(event.get('stream') == 'ur.call' and not pending, 'invalid stream boundary')
            if kind == 'subscribe':
                require(not subscribed and not finished, 'duplicate UR subscription')
                subscribed = True
            else:
                require(subscribed and not finished, 'invalid UR finish')
                finished = True
            continue
        require(subscribed and not finished and kind in ('begin', 'end'), 'unclassified/out-of-stream record')
        ident, tid, function = event.get('instance'), event.get('tid'), event.get('function_id')
        require(type(ident) is int and 0 <= ident < (1 << 64) and type(tid) is int and tid > 0 and
                type(function) is int and 0 <= function < (1 << 32) and
                isinstance(event.get('name'), str) and re.fullmatch(r'ur[A-Za-z0-9_]+', event['name']),
                'invalid UR call identity')
        require(function not in functions or functions[function] == event['name'], 'unstable UR function identity')
        functions[function] = event['name']
        if kind == 'begin':
            require(ident not in seen and event.get('status') is None, 'duplicate/invalid UR begin')
            seen.add(ident)
            pending[ident] = event
            stacks[tid].append(ident)
        else:
            begin = pending.pop(ident, None)
            require(begin is not None and stacks[tid] and stacks[tid][-1] == ident,
                    'unmatched/cross-thread/nonnested UR end')
            require((begin['tid'], begin['function_id'], begin['name']) == (tid, function, event['name']) and
                    begin['raw_ns'] <= event['raw_ns'] and type(event.get('status')) is int and
                    -(1 << 31) <= event['status'] < (1 << 31),
                    'UR call end identity/clock/status mismatch')
            stacks[tid].pop()
            counts[(event['name'], event['status'])] += 1
            if intervals:
                span = dict(instance=ident, tid=tid, name=event['name'], function_id=function,
                    begin_ns=begin['raw_ns'], end_ns=event['raw_ns'], status=event['status'])
                if 'metadata' in event:
                    require(isinstance(event['metadata'], dict), 'invalid typed UR metadata')
                    span['metadata'] = event['metadata']
                spans.append(span)
    require(subscribed and seen and not pending and all(not stack for stack in stacks.values()),
            'missing UR subscription/call/end')
    result = dict(call_pairs=len(seen), source_threads=len(stacks),
                outcomes=[dict(name=name, status=status, calls=count) for (name, status), count in sorted(counts.items())],
                ordinary_runtime_teardown_qualified=False, device_work_qualified=False)
    if intervals:
        result['intervals'] = spans
    return result


def metadata_fixture(text, pid):
    """Native CPU-generated callbacks; never actual device work."""
    parsed = parse_calls(text, pid, intervals=True)
    spans = parsed.pop('intervals')
    require(parsed['call_pairs'] == 26 and parsed['source_threads'] == 1, 'host metadata callback census differs')
    calls = collections.defaultdict(list)
    rejected = []
    for span in spans:
        require('metadata' in span, 'missing native typed metadata')
        (rejected if span['status'] else calls[span['name']]).append(span)
    require(len(rejected) == 2 and all(s['status'] == 46 for s in rejected), 'missing rejected callback controls')
    require(rejected[0]['metadata']['created_queue'] == 0 and rejected[1]['metadata']['produced_event'] == 0 and
            rejected[1]['metadata']['wait_count'] == 1 and 'waits' not in rejected[1]['metadata'],
            'failed call exposed an invalid output/list pointee')
    require([s['metadata']['created_queue'] for s in calls['urQueueCreate']] == [0x101, 0x101], 'queue reuse not recorded')
    require([s['metadata']['allocation'] for s in calls['urUSMDeviceAlloc']] == [0x100000, 0x100000] and
            all(s['metadata']['bytes'] == 1028 for s in calls['urUSMDeviceAlloc']), 'allocation reuse/extent missing')
    kernels = calls['urEnqueueKernelLaunchWithArgsExp'] + calls['urCommandBufferAppendKernelLaunchExp']
    require(len(kernels) == 3 and all(s['metadata']['work_dim'] == 1 and s['metadata']['global'] == [257] and
            s['metadata']['local'] == s['metadata']['offset'] == [] for s in kernels), 'kernel geometry differs')
    children = calls['urCommandBufferAppendKernelLaunchExp']
    require([s['metadata']['sync_point'] for s in children] == [1, 2] and
            [s['metadata']['command'] for s in children] == [0x501, 0x502] and
            all(s['metadata']['graph'] == 0x301 for s in children), 'graph children lost parent/command identity')
    replay = calls['urEnqueueCommandBufferExp']
    require(len(replay) == 2 and all(s['metadata']['graph'] == 0x301 and s['metadata']['queue'] == 0x101 and
            s['metadata']['produced_event'] == 0x201 for s in replay) and replay[0]['instance'] != replay[1]['instance'],
            'replay/event reuse identity differs')
    require(len(calls['urEventRetain']) == 1 and len(calls['urEventRelease']) == 4 and
            len(calls['urQueueRelease']) == 2 and len(calls['urUSMFree']) == 2 and
            len(calls['urCommandBufferFinalizeExp']) == len(calls['urCommandBufferReleaseExp']) == 1,
            'lifecycle callback control missing')
    return dict(simulated_calls=26, rejected_calls=2, successful_calls=24, recycled_queues=2,
                recycled_allocations=2, graph_children=2, successful_replays=2,
                failed_output_and_list_not_read=True, device_work_qualified=False,
                limits='Fabricated CPU XPTI payloads only;no UR/LevelZero device calls or actual lifetime/graph execution qualification.')


def metadata_census(spans):
    """Check typed capture extents; counts alone do not qualify lifetimes."""
    counts = collections.Counter()
    queue_handles = set()
    for span in spans:
        name, status = span['name'], span['status']
        meta = span.get('metadata')
        require(isinstance(meta, dict), 'metadata absent from native callback')
        for key, value in meta.items():
            values = value if isinstance(value, list) else [value]
            require(len(values) <= 32 and all(type(v) is int and 0 <= v < 1 << 64 for v in values),
                    'invalid typed scalar/list')
        if name.startswith('urEnqueue'):
            require(meta.get('queue', 0) > 0, 'unclassified enqueue metadata')
            queue_handles.add(meta['queue'])
            counts['enqueue_calls'] += 1
            if status == 0:
                require('waits' in meta and len(meta['waits']) == meta.get('wait_count') and 'produced_event' in meta,
                        'missing successful wait/output metadata')
        if status:
            counts['rejected_calls'] += 1
            continue
        if name.startswith('urEnqueueKernelLaunch') or name == 'urCommandBufferAppendKernelLaunchExp':
            dim = meta.get('work_dim')
            require(type(dim) is int and 1 <= dim <= 3 and meta.get('kernel', 0) > 0 and
                    len(meta.get('global', [])) == dim and all(n > 0 for n in meta['global']) and
                    'local' in meta and len(meta['local']) in (0, dim) and all(n > 0 for n in meta['local']) and
                    'offset' in meta and len(meta['offset']) in (0, dim), 'invalid/missing kernel geometry')
            counts['kernel_geometries'] += 1
        if name in ('urEnqueueUSMMemcpy', 'urEnqueueUSMFill'):
            require(meta.get('dst', 0) > 0 and meta.get('bytes', 0) > 0, 'missing transfer address/extent')
            if name == 'urEnqueueUSMMemcpy':
                require(meta.get('src', 0) > 0, 'missing source address')
            counts['copy_bytes' if name.endswith('Memcpy') else 'fill_bytes'] += meta['bytes']
        for api, field in (('urQueueCreate', 'created_queue'), ('urCommandBufferCreateExp', 'created_graph'),
                           ('urUSMDeviceAlloc', 'allocation'), ('urUSMHostAlloc', 'allocation'),
                           ('urUSMSharedAlloc', 'allocation')):
            if name == api:
                require(meta.get(field, 0) > 0, 'successful create missing output')
                counts[api] += 1
    require(counts['enqueue_calls'] and counts['kernel_geometries'], 'missing typed work')
    return dict(counts=dict(counts), enqueue_queue_handles=len(queue_handles),
                lifetime_generations_qualified=False, graph_children_qualified=False,
                limits='Typed callback capture/geometry/requested-byte census only. Handles are raw values;reference generations,retirement,actual launch rounding,graph child mapping and application host-memory lifetime still need independent checks.')


def correlate_calls(spans, events, pid, expected_calls):
    """Join independent UR instances to actual PTI sources; no ordinal join.

    This bounded direct fixture has one device append per successful enqueue.
    Graph parents, deferred/multiple appends and unknown work fail this contract.
    """
    internal = probe.trace_links(events, pid)
    names = {'urEnqueueKernelLaunch': 'kernel', 'urEnqueueKernelLaunchWithArgsExp': 'kernel',
             'urEnqueueUSMMemcpy': 'copy', 'urEnqueueUSMFill': 'fill',
             'urEnqueueEventsWaitWithBarrier': 'barrier', 'urEnqueueEventsWaitWithBarrierExt': 'barrier'}
    enqueue = [s for s in spans if s['name'].startswith('urEnqueue')]
    require(enqueue and len(enqueue) <= 4096 and all(s['name'] in names and s['status'] == 0 for s in enqueue),
            'unknown/failed direct enqueue')
    require(dict(collections.Counter(s['name'] for s in enqueue)) == expected_calls,
            'callback/text enqueue census mismatch')
    require(len({s['instance'] for s in enqueue}) == len(enqueue), 'duplicate UR execution instance')
    by_thread = collections.defaultdict(list)
    for span in enqueue:
        require(span['end_ns'] > span['begin_ns'], 'invalid enqueue interval')
        by_thread[span['tid']].append(span)
    starts = {}
    for tid, rows in by_thread.items():
        rows.sort(key=lambda s: s['begin_ns'])
        require(all(a['end_ns'] < b['begin_ns'] for a, b in zip(rows, rows[1:])), 'ambiguous enqueue intervals')
        starts[tid] = [s['begin_ns'] for s in rows]
    sources = {str(e['id']): e for e in events if e.get('ph') == 's' and e.get('pid') == pid and
               e.get('cat', '').startswith('Flow_H2D_')}
    used = set()
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') != 'gpu_op':
            continue
        source = sources[str(event['args']['id'])]
        tid, stamp = source['tid'], source['ts'] * 1000
        require(tid in by_thread, 'device source has no UR caller thread')
        index = bisect.bisect_right(starts[tid], stamp) - 1
        require(index >= 0, 'device source precedes UR call')
        span = by_thread[tid][index]
        require(span['begin_ns'] <= stamp <= span['end_ns'] and span['instance'] not in used,
                'unmapped/duplicated device-to-UR link')
        name = event['name']
        kind = 'copy' if name.startswith('zeCommandListAppendMemoryCopy') else (
            'fill' if name.startswith('zeCommandListAppendMemoryFill') else (
            'barrier' if name == 'zeCommandListAppendBarrier' else 'kernel'))
        require(names[span['name']] == kind, 'UR/device operation kind mismatch')
        used.add(span['instance'])
    require(len(used) == len(enqueue) == internal['linked_device_operations'], 'unlinked UR enqueue or GPU operation')
    return dict(linked_UR_instances=len(used), source_threads=len(by_thread),
                independent_UR_to_PTI_links=True, collector_internal_links=internal,
                shape_lifetime_graph_qualified=False, observer_cost_qualified=False,
                limits='Exact successful-direct-enqueue interval/thread/instance joins on this fixture only. No ordinal inference,tolerance fallback,graph expansion,queue/event generations,model completeness,speed or hardware-saturation qualification.')


def runtime_assets():
    row = r.validation.committed(r.RESULT)
    r.current(row)
    sdk = r.validation.committed(r.validation.INDEX, SDK_KEY)
    loader = row['preload']['loader']
    require(digest(loader) == sdk['runtime_assets']['libur_loader.so.0.12.0']['installed_sha256'],
            'inspected SDK loader differs from qualified runtime')
    dispatcher = Path(loader).parent / 'libxptifw.so.1.0.1'
    require(digest(dispatcher) == sdk['runtime_assets'][dispatcher.name]['installed_sha256'], 'dispatcher changed')
    for name, sha in sdk['source_inputs'].items():
        require(digest(Path(sdk['source_root']) / name) == sha, 'inspected source changed')
    for name in ('xpti_data_types.h', 'xpti_trace_framework.h'):
        require(digest(dispatcher.parent.parent / 'include/xpti' / name) ==
                sdk['source_inputs']['xpti/include/xpti/' + name], 'installed XPTI header/source mismatch')
    return dict(runtime_result_sha256=digest(r.RESULT), preload=row['preload'],
                dispatcher=dict(path=str(dispatcher), sha256=digest(dispatcher)), sdk=sdk)


def admit(row, binary=False):
    require(source_pins(row['source_commit']) == row['input_sha256'], 'subscriber controller source changed')
    admit_native(row, binary)


def admit_native(row, binary=False):
    require(runtime_assets() == row['assets'], 'subscriber runtime assets changed')
    for name, sha in row['compiler_tools'].items():
        require(digest(name) == sha, 'compiler/linker provider changed')
    for field in ('compiler_inputs', 'driver_compiler_inputs'):
        manifest = Path(row[field]['path'])
        require(digest(manifest) == row[field]['sha256'], 'header census changed')
        inputs = json.loads(manifest.read_text())
        require(len(inputs) == row[field]['files'], 'header census count differs')
        for name, sha in inputs.items():
            path = Path(name)
            if path.is_relative_to(P):
                local = str(path.relative_to(P))
                require(local in (SOURCE, DRIVER_SOURCE), 'unplanned project compiler input')
                revision = row.get('library_origin', {}).get('source_commit', row['source_commit']) if local == SOURCE else row['source_commit']
                require(hashlib.sha256(r.validation.git('show', revision + ':' + local)).hexdigest() == sha,
                        'original native project source does not recover: ' + local)
            else:
                require(digest(name) == sha, 'compiler source/header changed: ' + name)
    if binary:
        require(digest(row['library']['path']) == row['library']['sha256'], 'observer ELF changed')
        if 'driver' in row:
            require(digest(row['driver']['path']) == row['driver']['sha256'], 'native control ELF changed')


def qualified_native(key=KEY):
    row = r.validation.committed(RESULT, key)
    require(row['stage'] == 'host-checked' and row['passed'] and row['ownership_closed'] and not row['fresh_faults'] and
            len(row['outcomes']) == (5 if key == LEGACY_KEY else 6) and row['outcomes'][-1]['calls']['call_pairs'] == 2,
            'closed native CPU qualification required')
    admit_native(row, True)
    for name, raw in r.build.blobs(P, row['source_commit'], row['input_sha256']):
        require(hashlib.sha256(raw).hexdigest() == row['input_sha256'][name], 'original host producer source does not recover')
    for source, revision, census in ((SOURCE, row.get('library_origin', {}).get('source_commit', row['source_commit']), row['compiler_inputs']),
                                    (DRIVER_SOURCE, row['source_commit'], row['driver_compiler_inputs'])):
        inputs = json.loads(Path(census['path']).read_text())
        require(hashlib.sha256(r.validation.git('show', revision + ':' + source)).hexdigest() == inputs[str(P / source)],
                'qualified native source/compiled-input identity mismatch')
    raw = json.dumps(row, sort_keys=True, separators=(',', ':')).encode()
    proof = dict(host_component_sha256=hashlib.sha256(raw).hexdigest(), producer_source_commit=row['source_commit'],
                 original_project_inputs_recovered=len(row['input_sha256']), library=row['library'], driver=row['driver'],
                 reuse='Original observer/driver native sources and recipes recover;all recorded compiled inputs and ELF hashes verified. CPU qualification reuse does not establish new GPU coverage.')
    return row, proof


def gpu_prepare():
    """Freeze one small direct GPU join, reusing the host-qualified native ELF."""
    container = r.validation.committed(RESULT)
    require(not GPU_ROOT.exists() and GPU_KEY[1] not in container[GPU_KEY[0]], 'inspect existing GPU owner;no retry')
    prior = container[DIRECT_KEY[0]][DIRECT_KEY[1]]
    require(prior['stage'] == 'direct-join-passed' and prior['passed'] and prior['ownership_closed'] and
            not prior['fresh_faults'], 'closed independent direct-identity gate required')
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    pins, assets = source_pins(revision), probe.assets()
    host, native = qualified_native()
    GPU_ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache', 'trace'):
        (GPU_ROOT / name).mkdir()
    env = dict(r.validation.committed(r.SETTINGS)['environment'],
        TMPDIR=str(GPU_ROOT / 'tmp'), XDG_CACHE_HOME=str(GPU_ROOT / 'cache'), LD_DEBUG='libs',
        STRATA_UNITRACE_TARGET_PRELOAD=assets['preload']['library']['path'] + ':' + host['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=assets['preload']['loader'],
        XPTI_FRAMEWORK_DISPATCHER=host['assets']['dispatcher']['path'],
        XPTI_SUBSCRIBERS=host['library']['path'], XPTI_TRACE_ENABLE='1', SYCL_UR_TRACE='0',
        UR_L0_ENABLE_SYSMAN_ENV_DEFAULT='0',
        STRATA_UR_CALL_LOG=str(GPU_ROOT / 'calls.jsonl'),
        UNITRACE_FollowChildProcess='1', UNITRACE_SystemTime='1', UNITRACE_TraceOutputDir=str(GPU_ROOT / 'trace'),
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(GPU_ROOT / 'ur.log'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(GPU_ROOT / 'validation.log'))
    require('LD_PRELOAD' not in env, 'target-only preload required')
    session = 'StrataUR' + secrets.token_hex(12)
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins, assets=assets, native_reuse=native,
        host_controls=host['host_controls'],
        diagnostic_startup_override=dict(name='UR_L0_ENABLE_SYSMAN_ENV_DEFAULT', value='0',
            source_diagnosis=prior['diagnostic_startup_override']['source_diagnosis'],
            scope='Only this bounded diagnostic process. Keep all existing safety settings and immutable-environment guard. Do not infer disabled zesInit,hardware-counter capability,clean speed or production environment qualification.'),
        dispatcher=host['assets']['dispatcher'], expected_source_census=probe.source_census(), environment=env,
        session=session, command=[assets['launcher']['path'], '--session', session, '--host-timing', '--device-timing',
            '--chrome-device-logging', '--chrome-call-logging', '--chrome-event-buffer-size', '128',
            assets['fixture']['path'], '--submission'],
        budgets=dict(wall=90, cpu=90, rss_cap=4 << 30, text_cap=8 << 20, file_cap=32 << 20, total_cap=64 << 20),
        gates='One unchanged original6e40 direct-submission fixture,current NEO. Source/API/device606 kernels,53 copies/55844 requested bytes,one fill/1028 bytes,20 barriers. Every actual PTI GPU event needs one exact internal host/device flow and one distinct successful same-thread UR enqueue interval containing its actual append timestamp,with matching kind. Typed metadata additionally needs680 queues/wait/output snapshots,606 valid launch geometries and matching55844 copy/1028 fill requested bytes. No ordinal/tolerance fallback,ambiguous/unknown/unmapped/failed enqueue. Require complete subscriber pairs,actual pinned providers,pre-main validation,matching time namespace,numerical/drain footer,normal owners/session closure and no fresh faults. Stop first failure;no graph or model work.',
        retention='Main consumes bounded raw records for this direct join only;after verified compact evidence,retire normal traces/JIT/output with a deletion manifest. Reuse current13 native artifacts and the small observer;no new binary generation.',
        active=False, ownership_closed=True, passed=False, gpu_work=True, model_qualified=False,
        graph_lifetime_qualified=False, observer_cost_qualified=False, clean_speed_qualified=False)
    container[GPU_KEY[0]][GPU_KEY[1]] = row
    r.write(RESULT, container)
    r.write(GPU_ROOT / 'record.json', row)
    print('Prepared single-use GPU UR/PTI join;commit plan before execution.', flush=True)


def gpu_check():
    row = r.validation.committed(RESULT, GPU_KEY)
    require(row['stage'] == 'prepared' and not (GPU_ROOT / 'target-owned').exists(), 'single-use committed GPU plan required')

    def admit_gpu():
        require(source_pins(row['source_commit']) == row['input_sha256'] and probe.assets() == row['assets'] and
                qualified_native()[1] == row['native_reuse'] and probe.source_census() == row['expected_source_census'],
                'GPU observer source/native/runtime changed')

    admit_gpu()
    runtime = r.validation.committed(r.RESULT)
    session_path = Path('/dev/shm/uctrl' + str(os.getuid()) + '_' + row['session'])
    require(not session_path.exists(), 'preexisting collector session;no overwrite')
    journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(s.split(': ', 1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
    target = control = None

    def persist():
        row['active'] = any(o and o.active is not None for o in (target, control))
        r.write(GPU_ROOT / 'record.json', dict(row, target_commands=target.commands if target else [],
                                             control_commands=control.commands if control else []))

    try:
        target = r.supervision.Owner(GPU_ROOT / 'target-owned', persist)
        control = r.supervision.Owner(GPU_ROOT / 'control-owned', persist)
        row['stage'] = 'observing'
        seen = set()
        required = {str(Path(row['native_reuse']['library']['path']).resolve()),
                    str(Path(row['dispatcher']['path']).resolve()), str(Path(row['assets']['collector']['path']).resolve()),
                    next(p for p in runtime['runtime_files'] if 'libze_intel_gpu' in p)}
        allowed = set(runtime['runtime_files']) | required | {str(Path(row['assets'][k]['path']).resolve())
            for k in ('launcher', 'fixture')}
        namespace = os.readlink('/proc/self/ns/time')
        namespace_seen = False

        def observe(io, entry):
            nonlocal namespace_seen
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll()
                try:
                    require(os.readlink('/proc/' + str(entry['pid']) + '/ns/time') == namespace,
                            'native/controller time namespace mismatch')
                    namespace_seen = True
                    maps = Path('/proc', str(entry['pid']), 'maps').read_text()
                except (FileNotFoundError, ProcessLookupError):
                    maps = ''
                for path in r.executable_mappings(maps):
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
        row['pid'] = entry['pid']
        require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'native observer owner/exit failed')
        row['validation'] = r.validation.preload_evidence(err.read_text(), entry['pid'])
        require('fixture_status=PASS mode=submission' in out.read_text(), 'native numerical/drain footer missing')
        require(not (GPU_ROOT / 'validation.log').exists() or not (GPU_ROOT / 'validation.log').stat().st_size,
                'parameter validation warning')
        require(required <= seen and namespace_seen, 'observer/dispatcher/collector/current NEO or time namespace not observed')
        row.update(actual_runtime_providers_verified=True, time_namespace=namespace)
        row['api'] = probe.api_census((GPU_ROOT / 'ur.log').read_text())
        traces = list((GPU_ROOT / 'trace').rglob('*.json'))
        require(len(traces) == 1, 'one complete native JSON timeline required')
        raw = traces[0].read_bytes()
        value = json.loads(raw)
        events = value['traceEvents'] if isinstance(value, dict) else value
        row['device'] = probe.device_census(events, entry['pid'], row['begin_raw_ns'], row['end_raw_ns'])
        require(row['api']['counts'] == row['device']['counts'] == row['expected_source_census'],
                'source/API/device census mismatch')
        calls = parse_calls((GPU_ROOT / 'calls.jsonl').read_text(), entry['pid'], intervals=True)
        spans = calls.pop('intervals')
        row['links'] = correlate_calls(spans, events, entry['pid'], row['api']['enqueue_calls'])
        row['typed_metadata'] = metadata_census(spans)
        typed = row['typed_metadata']['counts']
        require(typed['enqueue_calls'] == 680 and typed['kernel_geometries'] == 606 and
                typed['copy_bytes'] == 55844 and typed['fill_bytes'] == 1028, 'typed fixture census differs')
        row['subscriber'] = dict(call_pairs=calls['call_pairs'], source_threads=calls['source_threads'],
                                 complete_pairs=True, environment_guard_exit_not_triggered=True)
        row['analysis_inputs'] = [dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)
                                  for p in (traces[0], GPU_ROOT / 'ur.log', GPU_ROOT / 'calls.jsonl')]
        admit_gpu()
        row.update(stage='direct-join-passed', passed=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = all(o is None or (o.active is None and all(r.supervision.closed(c) for c in o.commands))
                                      for o in (target, control))
        if session_path.exists() and row['ownership_closed'] and control is not None:
            try:
                entry, _, _ = control.run('stop', [row['assets']['launcher']['path'], '--stop', row['session']],
                    row['environment'], P, wall=15, cpu=15, rss_cap=2 << 30, text_cap=1 << 20,
                    file_cap=32 << 20, total_cap=64 << 20)
                require(r.supervision.closed(entry) and entry['exit_code'] == 0 and not session_path.exists(),
                        'collector session closure failed')
            except BaseException as error:
                row.update(passed=False, cleanup_error=repr(error))
        row['ownership_closed'] = all(o is None or (o.active is None and all(r.supervision.closed(c) for c in o.commands))
                                      for o in (target, control))
        row['session_removed'] = not session_path.exists()
        try:
            check = dict(host_probe_accepted=True)
            r.validation.finish(check, target, cursor, runtime['boot'])
            row.update(fresh_faults=check['fresh_faults'], boot_unchanged=check['boot_unchanged'], driver_after=check['driver_after'])
            row['passed'] = bool(row['passed'] and check['passed'] and row['ownership_closed'] and
                                 row['session_removed'] and not row.get('cleanup_error'))
        except BaseException as error:
            row.update(passed=False, fault_check_error=repr(error))
        if not row['passed']:
            row['stage'] = 'failed'
        persist()
        container = json.loads(RESULT.read_text())
        container[GPU_KEY[0]][GPU_KEY[1]] = row
        r.write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'fresh_faults', 'error')}), flush=True)
    return 0 if row['passed'] else 1


def driver_plan(compiler, env):
    sdk = Path(runtime_assets()['dispatcher']['path']).parent.parent
    options = [compiler, '-std=c++17', '-O2', '-I' + str(sdk / 'include')]
    raw = subprocess.check_output([*options, '-M', str(P / DRIVER_SOURCE)], env=env, timeout=30)
    require(len(raw) <= 1 << 20 and b':' in raw, 'finite driver header census required')
    names = shlex.split(raw.decode().split(':', 1)[1].replace('\\\n', ' '))
    require(0 < len(names) < 4096, 'driver header population cap')
    manifest = ROOT / 'driver-inputs.json'
    headers = {str(Path(name).resolve()): digest(name) for name in names}
    r.write(manifest, headers)
    return dict(driver_command=[*options, str(P / DRIVER_SOURCE), '-L' + str(sdk / 'lib'),
                '-Wl,-rpath,' + str(sdk / 'lib'), '-lxptifw', '-o', str(ROOT / 'ur_call_guard_test')],
                driver_compiler_inputs=dict(path=str(manifest), sha256=digest(manifest), files=len(headers)))


def repair():
    """New native-driver plan, preserving the closed failed Python experiment."""
    old = r.validation.committed(RESULT, KEY)
    require(old['stage'] == 'failed' and old['ownership_closed'] and not old['fresh_faults'] and
            old['error'] == "RuntimeError('mutation control failed')" and not old['outcomes'] and
            old['failure_evidence']['exit_code'] == 87 and not old['failure_evidence']['test_marker_reached'] and
            not (ROOT / 'driver-inputs.json').exists(), 'only the closed unadvanced Python control may be superseded')
    require(runtime_assets() == old['assets'] and digest(old['library']['path']) == old['library']['sha256'],
            'unchanged original subscriber required')
    for name, sha in old['compiler_tools'].items():
        require(digest(name) == sha, 'original compiler provider changed')
    manifest = Path(old['compiler_inputs']['path'])
    require(digest(manifest) == old['compiler_inputs']['sha256'], 'original native census changed')
    inputs = json.loads(manifest.read_text())
    for name, sha in inputs.items():
        require(digest(name) == sha, 'original native compiled input changed')
    native_source = r.validation.git('show', old['source_commit'] + ':' + SOURCE)
    require(hashlib.sha256(native_source).hexdigest() == inputs[str(P / SOURCE)] and
            native_source == (P / SOURCE).read_bytes(), 'original observer source does not recover')
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    pins = source_pins(revision)
    row = copy.deepcopy(old)
    for name in ('outcomes', 'error', 'failure_evidence', 'next_action'):
        row.pop(name, None)
    row.update(stage='driver-prepared', source_commit=revision, input_sha256=pins,
        original_failed_result=dict(commit=revision, path=str(RESULT.relative_to(P)), sha256=digest(RESULT)),
        library_origin=dict(source_commit=old['source_commit'], cpp_sha256=inputs[str(P / SOURCE)],
            library_sha256=old['library']['sha256'], compiled_inputs_verified=len(inputs),
            reuse='No observer rebuild or hash refresh. Original source/recipe and254 compiled inputs recover;only the native host test driver is new.'),
        gates=old['gates'].replace('Four CPU mutation controls', 'Four NATIVE CPU mutation controls') +
            ' New plan replaces Python startup with a source-pinned native driver;all original failed outcomes remain in the archived receipt.',
        **driver_plan(old['command'][0], row['environment']))
    container = json.loads(RESULT.read_text())
    container[KEY[0]][KEY[1]] = row
    r.write(RESULT, container)
    r.write(ROOT / 'record.json', row)
    print('Prepared native mutation controls;original failure/library preserved. Commit before driver build.', flush=True)


def prepare():
    require(not ROOT.exists(), 'inspect existing subscriber owner; no retry')
    container = r.validation.committed(RESULT)
    require(KEY[1] not in container[KEY[0]], 'preserve existing subscriber result')
    _, fallback = qualified_native(LEGACY_KEY)
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    pins, assets = source_pins(revision), runtime_assets()
    sdk = Path(assets['dispatcher']['path']).parent.parent
    compiler = str(Path('/usr/bin/g++').resolve())
    options = [compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-pthread',
               '-DXPTI_CALLBACK_API_EXPORTS', '-I' + str(sdk / 'include')]
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
    tools = {compiler: digest(compiler)}
    for name in ('cc1plus', 'collect2', 'as', 'ld'):
        value = subprocess.check_output([compiler, '-print-prog-name=' + name], env=env, text=True, timeout=10).strip()
        path = Path(value) if '/' in value else Path(shutil.which(value, path=env['PATH']))
        tools[str(path.resolve())] = digest(path)
    raw = subprocess.check_output([*options, '-M', str(P / SOURCE)], env=env, timeout=30)
    require(len(raw) <= 1 << 20 and b':' in raw, 'finite compiler header census required')
    dependencies = shlex.split(raw.decode().split(':', 1)[1].replace('\\\n', ' '))
    require(0 < len(dependencies) < 4096, 'header population cap')
    headers = {str(Path(name).resolve()): digest(name) for name in dependencies}
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    manifest = ROOT / 'compiler-inputs.json'
    r.write(manifest, headers)
    library = ROOT / 'libstrata_ur_calls.so'
    command = [*options, str(P / SOURCE), '-L' + str(sdk / 'lib'), '-Wl,-rpath,' + str(sdk / 'lib'),
               '-Wl,--no-undefined', '-lxptifw', '-o', str(library)]
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins, assets=assets, qualified_fallback=fallback,
        compiler_tools=tools, compiler_inputs=dict(path=str(manifest), sha256=digest(manifest), files=len(headers)),
        command=command, environment=dict(env, TMPDIR=str(ROOT / 'tmp')),
        budgets=dict(wall=120, cpu=120, rss_cap=4 << 30, text_cap=4 << 20, file_cap=32 << 20, total_cap=64 << 20),
        gates='Compile one Main-authored observer ELF and native driver;commit binary hashes before checks. Four CPU mutation controls reach markers then exit87. One synthetic CPU XPTI metadata fixture yields26 complete pairs (24 success,two rejection),same-address queue/allocation/event reuse,two graph children,two replay instances and unreached invalid failed-output/list pointees. No UR or LevelZero device calls. Original CPU UR canary exits0 with pre-main validation and exactly two rejected urPlatformGetApiVersion pairs,status46. Stop first failure. No GPU/model/performance or ordinary teardown qualification.',
        retention='Original6e40 plus12 fixtures retained unchanged. Keep this one small observer ELF,exact source/recipe and header manifest only for the imminent GPU coverage fixture after host checks;retire normal build/API/loader output after compact evidence.',
        active=False, ownership_closed=True, passed=False, gpu_work=False, model_qualified=False)
    row.update(driver_plan(compiler, row['environment']))
    container[KEY[0]][KEY[1]] = row
    r.write(RESULT, container)
    r.write(ROOT / 'record.json', row)
    print('Prepared host-only UR subscriber;commit recipe before build.', flush=True)


def execute(action):
    row = r.validation.committed(RESULT, KEY)
    expected = {'build': ('prepared',), 'build-driver': ('driver-prepared',), 'check': ('built', 'driver-built')}
    require(row['stage'] in expected[action], 'committed single-use component stage required')
    admit(row, action != 'build')
    owner = None
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(x.split(': ', 1)[1] for x in cursor_text.splitlines() if x.startswith('-- cursor: '))
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def persist():
        row['active'] = bool(owner and owner.active is not None)
        r.write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))

    try:
        label = 'check-native' if action == 'check' else action
        require(not (ROOT / (label + '-owned')).exists(), 'inspect original component process;no retry')
        owner = r.supervision.Owner(ROOT / (label + '-owned'), persist)
        row['stage'] = action + 'ing'
        if action in ('build', 'build-driver'):
            if action == 'build':
                entry, _, _ = owner.run('compile', row['command'], row['environment'], P, **row['budgets'])
                require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'subscriber compile failed')
                library = ROOT / 'libstrata_ur_calls.so'
                row['library'] = dict(path=str(library), sha256=digest(library), bytes=library.stat().st_size)
            entry, _, _ = owner.run('compile-driver', row['driver_command'], row['environment'], P, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'native driver compile failed')
            driver = ROOT / 'ur_call_guard_test'
            row.update(stage='built' if action == 'build' else 'driver-built',
                driver=dict(path=str(driver), sha256=digest(driver), bytes=driver.stat().st_size))
        else:
            base = r.validation.committed(r.SETTINGS)['environment']
            row['outcomes'] = []
            for kind in ('setenv', 'unsetenv', 'putenv', 'clearenv', 'metadata', 'canary'):
                folder = ROOT / ('native-' + kind)
                folder.mkdir()
                env = dict(base, STRATA_UR_CALL_LOG=str(folder / 'calls.jsonl'),
                    XPTI_FRAMEWORK_DISPATCHER=row['assets']['dispatcher']['path'],
                    XPTI_SUBSCRIBERS=row['library']['path'], XPTI_TRACE_ENABLE='1', SYCL_UR_TRACE='0',
                    LD_PRELOAD=row['library']['path'], TMPDIR=str(ROOT / 'tmp'), LD_DEBUG='libs',
                    UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'ur.log'))
                command = [row['driver']['path'], kind]
                if kind == 'canary':
                    env.update(LD_PRELOAD=row['assets']['preload']['library']['path'] + ':' + row['library']['path'],
                               STRATA_UR_PROBE_EXPECTED_LOADER=row['assets']['preload']['loader'])
                    command = [row['assets']['preload']['canary']['path']]
                entry, out, err = owner.run(kind, command, env, P, wall=45, cpu=45, rss_cap=2 << 30,
                                           text_cap=4 << 20, file_cap=32 << 20, total_cap=64 << 20)
                require(r.supervision.closed(entry), 'component owner closure failed')
                if kind == 'metadata':
                    require(entry['exit_code'] == 0 and json.loads(out.read_text()) ==
                            dict(metadata_fixture='PASS', simulated_calls=26), 'native metadata driver failed')
                    result = dict(kind=kind, exit_code=0, metadata=metadata_fixture((folder / 'calls.jsonl').read_text(), entry['pid']))
                elif kind != 'canary':
                    require(entry['exit_code'] == 87 and out.read_text() == 'mutation-case-begin:' + kind + '\n' and
                            'strata UR subscriber: environment mutation blocked' in err.read_text(), 'mutation control failed')
                    result = dict(kind=kind, exit_code=87, mutation_blocked=True)
                else:
                    require(entry['exit_code'] == 0, 'CPU UR subscriber canary failed')
                    validation = r.validation.preload_evidence(err.read_text(), entry['pid'])
                    require(json.loads(out.read_text()) == dict(pid=entry['pid'], initialized=0, checked=46,
                        expected=46, version=(1 << 32) - 1, closed=0), 'original UR canary result changed')
                    calls = parse_calls((folder / 'calls.jsonl').read_text(), entry['pid'])
                    require(calls['call_pairs'] == 2 and calls['source_threads'] == 1 and calls['outcomes'] ==
                        [dict(name='urPlatformGetApiVersion', status=46, calls=2)], 'CPU UR callback contract differs')
                    result = dict(kind=kind, exit_code=0, validation=validation, calls=calls)
                row['outcomes'].append(result)
                persist()
            row.update(stage='host-checked', passed=True)
        admit(row, True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or (owner.active is None and all(r.supervision.closed(e) for e in owner.commands))
        try:
            check = dict(host_probe_accepted=True)
            r.validation.finish(check, owner, cursor, boot)
            row.update(fresh_faults=check['fresh_faults'], boot_unchanged=check['boot_unchanged'])
            require(check['passed'] and row['ownership_closed'], 'component fault/closure check failed')
        except BaseException as error:
            row.update(stage='failed', passed=False, fault_check_error=repr(error))
        persist()
        container = json.loads(RESULT.read_text())
        container[KEY[0]][KEY[1]] = row
        r.write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'fresh_faults', 'error')}), flush=True)
    return 1 if row['stage'] == 'failed' else 0


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'repair', 'build-driver', 'check', 'gpu-prepare', 'gpu-check'),
            'use ur_call_subscriber.py prepare|build|repair|build-driver|check|gpu-prepare|gpu-check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.guards.no_model_owners([])
        raise SystemExit({'prepare': prepare, 'repair': repair, 'gpu-prepare': gpu_prepare, 'gpu-check': gpu_check}
                         .get(sys.argv[1], lambda: execute(sys.argv[1]))())
