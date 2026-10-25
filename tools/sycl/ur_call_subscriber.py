#!/usr/bin/env python3
"""Read qualified UR subscriber artifacts and reconcile captured calls.

The selected buffer-aware observer has a separate native CPU/GPU qualification.
Closed one-use build/check controllers recover from05f0f7af4 and7f45b88e0;
current admission uses their original inputs rather than an obsolete ELF.
Repository LICENSE; no external implementation copied.
"""
import collections
import bisect
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import work_ledger_fixture as probe

r = probe.runtime
P, B, RESULT = probe.P, probe.B, probe.RESULT
SOURCE = 'bench/sycl/ur_call_subscriber.cpp'
DRIVER_SOURCE = 'bench/sycl/ur_call_guard_test.cpp'
LEGACY_KEY = ('event_reconciliation', 'ur_subscriber')
BUFFER_KEY = ('event_reconciliation', 'ur_subscriber_buffers')
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


def metadata_census(spans, *, require_kernel=True):
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
        if name in ('urEnqueueMemBufferRead', 'urEnqueueMemBufferWrite'):
            require(meta.get('buffer', 0) > 0 and meta.get('host', 0) > 0 and
                    meta.get('bytes', 0) > 0 and 'offset' in meta and meta.get('blocking') in (0, 1) and
                    meta['host'] + meta['bytes'] <= 1 << 64, 'invalid buffer transfer extent')
            counts['buffer_copy_bytes'] += meta['bytes']
        if name == 'urQueueIsGraphCaptureEnabledExp':
            require(meta.get('queue', 0) > 0 and meta.get('capture_result_valid') == 1 and
                    meta.get('capture_enabled') in (0, 1), 'missing graph query result')
            counts['graph_query_calls'] += 1
        if name == 'urMemGetNativeHandle':
            require(meta.get('buffer', 0) > 0 and meta.get('native_result_valid') == 1 and
                    'native_handle' in meta, 'missing native buffer identity')
        for api, field in (('urQueueCreate', 'created_queue'), ('urCommandBufferCreateExp', 'created_graph'),
                           ('urUSMDeviceAlloc', 'allocation'), ('urUSMHostAlloc', 'allocation'),
                           ('urUSMSharedAlloc', 'allocation'), ('urMemBufferCreate', 'created_buffer')):
            if name == api:
                require(meta.get(field, 0) > 0, 'successful create missing output')
                if name == 'urMemBufferCreate':
                    require(meta.get('bytes', 0) > 0, 'missing buffer allocation extent')
                counts[api] += 1
    require(counts['enqueue_calls'] and (not require_kernel or counts['kernel_geometries']), 'missing typed work')
    return dict(counts=dict(counts), enqueue_queue_handles=len(queue_handles),
                lifetime_generations_qualified=False, graph_children_qualified=False,
                limits='Typed callback capture/geometry/requested-byte census only. Handles are raw values;reference generations,retirement,actual launch rounding,graph child mapping and application host-memory lifetime still need independent checks.')


def lifetime_census(spans):
    """External API reference generations in a serial,fully captured fixture.

    Internal driver references,USM payload readers,drained GPU completion and
    application host buffers are different lifetimes;none is inferred here.
    """
    live = {kind: {} for kind in ('queue', 'event', 'allocation', 'graph', 'buffer')}
    generations = collections.Counter()
    counts = collections.Counter()
    tracked = [s for s in spans if s.get('metadata')]
    require(tracked and len({s['tid'] for s in tracked}) == 1, 'external lifetime adapter requires one caller thread')
    tracked.sort(key=lambda s: s['begin_ns'])
    require(all(a['end_ns'] <= b['begin_ns'] for a, b in zip(tracked, tracked[1:])), 'overlapping lifetime callbacks')

    def current(kind, handle):
        require(handle in live[kind], 'unknown/retired ' + kind + ' generation')
        return live[kind][handle]

    def create(kind, handle, **extra):
        require(handle not in (0, None) and handle not in live[kind], 'duplicate/live ' + kind + ' create')
        key = (kind, handle)
        generations[key] += 1
        require(len(generations) <= 16384, 'external generation budget')
        live[kind][handle] = dict(refs=1, generation=generations[key], **extra)
        counts[kind + '_created'] += 1
        counts[kind + '_reused_values'] += generations[key] > 1

    def reference(kind, handle, retain):
        value = current(kind, handle)
        value['refs'] += 1 if retain else -1
        counts[kind + ('_retained' if retain else '_released')] += 1
        if not value['refs']:
            del live[kind][handle]

    for span in tracked:
        name, meta = span['name'], span['metadata']
        if span['status']:
            # Rejection must neither create nor retire any external generation.
            counts['rejected_without_transition'] += 1
            continue
        if name == 'urQueueCreate':
            create('queue', meta['created_queue'], context=meta['context'])
        elif name == 'urMemBufferCreate':
            require(meta['bytes'] > 0, 'invalid buffer allocation extent')
            create('buffer', meta['created_buffer'], context=meta['context'], bytes=meta['bytes'])
        elif name in ('urMemRetain', 'urMemRelease'):
            reference('buffer', meta['buffer'], name == 'urMemRetain')
        elif name == 'urMemGetNativeHandle':
            current('buffer', meta['buffer'])
            require(meta['native_result_valid'] == 1, 'missing native buffer result')
            counts['buffer_native_queries'] += 1
        elif name == 'urQueueIsGraphCaptureEnabledExp':
            current('queue', meta['queue'])
            require(meta['capture_result_valid'] == 1 and meta['capture_enabled'] in (0, 1),
                    'missing graph query result')
            counts['graph_query_calls'] += 1
        elif name in ('urUSMDeviceAlloc', 'urUSMHostAlloc', 'urUSMSharedAlloc'):
            require(meta['bytes'] > 0 and meta['allocation'] > 0, 'invalid successful allocation extent')
            create('allocation', (meta['context'], meta['allocation']), bytes=meta['bytes'])
        elif name == 'urUSMFree':
            reference('allocation', (meta['context'], meta['allocation']), False)
        elif name == 'urCommandBufferCreateExp':
            create('graph', meta['created_graph'], context=meta['context'], finalized=False, children=0)
        elif name in ('urQueueRetain', 'urQueueRelease', 'urEventRetain', 'urEventRelease',
                      'urCommandBufferRetainExp', 'urCommandBufferReleaseExp'):
            kind = 'queue' if name.startswith('urQueue') else 'event' if name.startswith('urEvent') else 'graph'
            reference(kind, meta[kind], name.endswith('Retain') or name.endswith('RetainExp'))
        elif name == 'urCommandBufferFinalizeExp':
            graph = current('graph', meta['graph'])
            require(not graph['finalized'], 'duplicate graph finalize')
            graph['finalized'] = True
        elif name == 'urQueueFinish':
            current('queue', meta['queue'])
            counts['queue_finish_calls'] += 1
        if name.startswith('urEnqueue'):
            queue = current('queue', meta['queue'])
            if name in ('urEnqueueMemBufferRead', 'urEnqueueMemBufferWrite'):
                buffer = current('buffer', meta['buffer'])
                require(buffer['context'] == queue['context'] and 0 <= meta['offset'] <= buffer['bytes'] and
                        0 < meta['bytes'] <= buffer['bytes'] - meta['offset'],
                        'wrong-context or out-of-bounds buffer transfer')
                counts['buffer_transfers'] += 1
            if name == 'urEnqueueCommandBufferExp':
                graph = current('graph', meta['graph'])
                require(graph['finalized'] and graph['context'] == queue['context'], 'unfinalized/wrong-context graph replay')
                counts['graph_replays'] += 1
                counts['declared_replay_children'] += graph['children']
        if name.startswith('urCommandBufferAppend'):
            graph = current('graph', meta['graph'])
            require(not graph['finalized'], 'append after graph finalize')
            graph['children'] += 1
            counts['graph_children'] += 1
        if 'waits' in meta:
            require(len(meta['waits']) == meta['wait_count'], 'incomplete event dependency snapshot')
            for handle in meta['waits']:
                current('event', handle)
        if meta.get('produced_event'):
            create('event', meta['produced_event'])
    require(all(not objects for objects in live.values()), 'unretired external fixture generations')
    return dict(counts=dict(counts), external_generations_closed=True, source_threads=1,
        limits='Complete serial fixture external API reference generations only. Not driver-internal ownership,buffer-reader completion,host-memory lifetime,ordinary runtime teardown or exact graph-child/device identity.')


def buffer_fixture(text, pid):
    """Validate CPU-generated buffer/query callbacks,not actual GPU lifetimes."""
    parsed = parse_calls(text, pid, intervals=True); spans = parsed.pop('intervals')
    rejected = [s for s in spans if s['status']]
    require(parsed['call_pairs'] == 21 and parsed['source_threads'] == 1 and
            len(rejected) == 5 and all(s['status'] == 46 for s in rejected), 'buffer callback census differs')
    failed = {s['name']: s['metadata'] for s in rejected}
    require(failed['urMemBufferCreate']['created_buffer'] == 0 and
            failed['urQueueIsGraphCaptureEnabledExp']['capture_result_valid'] == 0 and
            failed['urMemGetNativeHandle']['native_result_valid'] == 0, 'failed output pointee was read')
    for name in ('urEnqueueMemBufferRead', 'urEnqueueMemBufferWrite'):
        require(failed[name]['produced_event'] == 0 and failed[name]['wait_count'] == 1 and
                'waits' not in failed[name], 'failed buffer dependency pointee was read')
    calls = collections.defaultdict(list)
    for s in spans:
        if not s['status']: calls[s['name']].append(s['metadata'])
    require([m['bytes'] for m in calls['urMemBufferCreate']] == [32, 16] and
            all(m['created_buffer'] == 0x601 for m in calls['urMemBufferCreate']) and
            [m['capture_enabled'] for m in calls['urQueueIsGraphCaptureEnabledExp']] == [1, 0] and
            [m['native_handle'] for m in calls['urMemGetNativeHandle']] == [0x800, 0x900],
            'buffer generation/query values differ')
    for name, host, offset, blocking, event in (('urEnqueueMemBufferWrite', 0x701, 4, 0, 0x201),
                                               ('urEnqueueMemBufferRead', 0x702, 8, 1, 0)):
        require(calls[name] == [dict(queue=0x101, buffer=0x601, host=host, bytes=8, offset=offset,
                                   blocking=blocking, wait_count=0, waits=[], produced_event=event)],
                'buffer typed arguments differ')
    typed = metadata_census(spans, require_kernel=False); lifetime = lifetime_census(spans)
    require(lifetime['counts']['buffer_created'] == 2 and lifetime['counts']['buffer_reused_values'] == 1 and
            lifetime['counts']['buffer_transfers'] == 2, 'buffer generation conservation differs')
    return dict(simulated_calls=21, rejected_calls=5, typed=typed, external_lifetimes=lifetime,
        failed_pointees_not_read=True, gpu_work=False,
        limits='CPU XPTI simulation only;no UR/L0 device work,actual host-payload ownership,performance or model coverage.')


def correlate_calls(spans, events, pid, expected_calls):
    """Join independent UR instances to actual PTI sources; no ordinal join.

    This bounded direct fixture has one device append per successful enqueue.
    Graph parents, deferred/multiple appends and unknown work fail this contract.
    """
    internal = probe.trace_links(events, pid)
    names = {'urEnqueueKernelLaunch': 'kernel', 'urEnqueueKernelLaunchWithArgsExp': 'kernel',
             'urEnqueueUSMMemcpy': 'copy', 'urEnqueueMemBufferRead': 'copy',
             'urEnqueueMemBufferWrite': 'copy', 'urEnqueueUSMFill': 'fill',
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


def qualified_native(key=BUFFER_KEY):
    row = r.validation.committed(RESULT, key)
    require(row['stage'] == 'host-checked' and row['passed'] and row['ownership_closed'] and not row['fresh_faults'] and
            len(row['outcomes']) == (5 if key == LEGACY_KEY else 7 if key == BUFFER_KEY else 6) and row['outcomes'][-1]['calls']['call_pairs'] == 2,
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
