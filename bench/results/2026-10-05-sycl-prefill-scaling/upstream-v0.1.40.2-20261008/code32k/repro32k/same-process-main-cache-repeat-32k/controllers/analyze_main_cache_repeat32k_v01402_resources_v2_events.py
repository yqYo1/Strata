"""Offline successful native-resource lifetimes for three full32K rereads.

API requested sizes and native object counts are not resident VRAM sizes.
No GPU or recovery operation is performed by this script.
"""
from pathlib import Path
import collections, datetime, hashlib, json, re

base = Path(__file__).parent
job = base / 'owned-main-vmm-full-ram-repeat32k-v01402-code32k-diagnostic-r1'
sequence = json.loads((base / 'main-cache-repeat32k-v01402-state-sequence/record.json').read_text())
record = json.loads((job / 'record.json').read_text())
assert sequence['passed'] and not sequence['active']
assert record['healthy'] and record['math_gate_passed'] and not record['active']
assert record['exit_code'] == 0 and not record['new_fault_messages']
out = base / 'main-cache-repeat32k-v01402-resource-analysis-v2-events'
out.mkdir(mode=0o700)

apis = {
    'zeEventPoolCreate': ('event_pool', 'create', 'phEventPool'),
    'zeEventPoolDestroy': ('event_pool', 'destroy', 'hEventPool'),
    'zeModuleCreate': ('module', 'create', 'phModule'),
    'zeModuleDestroy': ('module', 'destroy', 'hModule'),
    'zeKernelCreate': ('kernel', 'create', 'phKernel'),
    'zeKernelDestroy': ('kernel', 'destroy', 'hKernel'),
    'zeCommandListCreate': ('command_list', 'create', 'phCommandList'),
    'zeCommandListCreateImmediate': ('command_list', 'create', 'phCommandList'),
    'zeCommandListDestroy': ('command_list', 'destroy', 'hCommandList'),
    'zePhysicalMemCreate': ('physical_memory', 'create', 'phPhysicalMemory'),
    'zePhysicalMemDestroy': ('physical_memory', 'destroy', 'hPhysicalMemory'),
    'zeMemAllocDevice': ('device_usm', 'create', 'pptr'),
    'zeMemFree': ('device_usm', 'free_if_device', 'ptr'),
    'zeMemFreeExt': ('device_usm', 'free_if_device', 'ptr'),
    'zeEventCreate': ('event', 'create', 'phEvent'),
    'zeEventDestroy': ('event', 'destroy', 'hEvent'),
    'zeVirtualMemMap': ('virtual_map', 'map', 'ptr'),
    'zeVirtualMemUnmap': ('virtual_map', 'unmap', 'ptr'),
}
success = re.compile(rb'SUCCESS \(ZE_RESULT_SUCCESS\) in (' + b'|'.join(k.encode() for k in apis) + rb')\(')
memory = re.compile(rb'^strata prefill memory: (.*?); free=(\d+) total=(\d+) bytes')
size_pattern = re.compile(rb'\bsize=(\d+)')
live = {k: {} for k in {v[0] for v in apis.values()}}
counts = collections.defaultdict(collections.Counter)
requested = collections.defaultdict(collections.Counter)
phases = []
examples = {}
request_number = 0
phase = 'startup'
analysis = {'active': True, 'passed': False, 'gpu_access': False,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'job_record_sha256': hashlib.sha256((job/'record.json').read_bytes()).hexdigest(),
            'scope': 'Three complete32768-token rereads in one validated/logged normal-MTP process; successful native object handles, requested allocation sizes and phase markers only. No speed, resident-size attribution, proven leak or steady-state conclusion.'}
def save():
    (out/'record.json').write_text(json.dumps(analysis, indent=2)+'\n')
def live_summary():
    return {kind: {'objects': len(values),
                   'requested_bytes': sum(values.values()) if kind in ['physical_memory', 'device_usm', 'virtual_map'] else None}
            for kind, values in live.items()}
save()
try:
    log = job/'debugger/inferior.stderr'
    assert log.stat().st_size == record['engine_log_bytes']
    with log.open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            if line.startswith(b'strata prefill memory:'):
                m = memory.match(line)
                assert m, (line_number, line[:500])
                phase = m[1].decode()
                if phase == 'before decode cache release': request_number += 1
                phases.append({'line': line_number, 'request': request_number, 'phase': phase,
                               'free_bytes': int(m[2]), 'total_bytes': int(m[3]), 'live_native': live_summary()})
            elif b'SUCCESS (ZE_RESULT_SUCCESS) in ze' in line:
                m = success.search(line)
                if not m: continue
                api = m[1].decode()
                kind, operation, field = apis[api]
                h = re.search(rb'\b' + field.encode() + rb'=(0x[0-9a-f]+)', line)
                assert h, (line_number, line[:1200])
                handle = h[1].decode()
                values = live[kind]
                s = size_pattern.search(line)
                size = int(s[1]) if s else 0
                key = f'read-{request_number}/{phase}'
                counts[key][api] += 1
                if operation in ['create', 'map']:
                    assert handle not in values, (api, handle, line_number)
                    if kind in ['physical_memory','device_usm','virtual_map']: assert size > 0
                    values[handle] = size
                    if size: requested[key][api] += size
                elif operation == 'free_if_device':
                    # zeMemFree also receives host/shared pointers, tracked elsewhere.
                    values.pop(handle, None)
                else:
                    assert handle in values, (api, handle, line_number)
                    if operation == 'unmap': assert values[handle] == size
                    del values[handle]
                examples.setdefault(key, {}).setdefault(api, {'line': line_number, 'raw': line.decode().rstrip()})
    assert request_number == 3 and len(phases) == 24
    assert not any(live.values()), live_summary()
    analysis.update(memory_phases=phases, counts={p: dict(c) for p,c in counts.items()},
                    requested_sizes={p: dict(c) for p,c in requested.items()}, first_examples=examples,
                    live_native_at_exit=live_summary(), passed=True)
    memory_controls = []
    for phase_name, repetition in [('diagnostic',1),('state',1),('state',2)]:
        p=base/f'owned-main-vmm-full-ram-repeat32k-v01402-code32k-{phase_name}-r{repetition}'
        r=json.loads((p/'record.json').read_text())
        assert r['healthy'] and r['math_gate_passed'] and not r['active']
        entries=[]; n=0
        for line in (p/'project-messages.txt').read_text().splitlines():
            m=memory.match(line.encode())
            if not m: continue
            if m[1]==b'before decode cache release': n+=1
            entries.append({'request': n, 'phase': m[1].decode(), 'free_bytes': int(m[2])})
        assert n==3 and len(entries)==24
        memory_controls.append({'phase': phase_name, 'repetition': repetition,
                                'record_sha256':hashlib.sha256((p/'record.json').read_bytes()).hexdigest(),
                                'markers': entries})
    analysis.update(memory_controls=memory_controls,
                    limitations=['Native handles are tracked from successful API returns through process exit, including zeMemFreeExt and handle reuse. Host/shared frees do not count as device frees.',
                                 'Only explicit zeMemAllocDevice/zePhysicalMemCreate sizes are summed. Runtime-private heaps, residency, caches and allocation granules may affect reported free memory.',
                                 'Three requests do not establish steady state or explain the older full-context capacity failure. Captured/validated durations are excluded from throughput comparisons.'])
except BaseException as error:
    analysis.update(error=repr(error), passed=False, partial_live_native=live_summary())
finally:
    analysis.update(active=False, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
print(json.dumps({'passed': analysis['passed'], 'error': analysis.get('error'),
                  'memory_phases': analysis.get('memory_phases'), 'counts': analysis.get('counts'),
                  'live_native_at_exit': analysis.get('live_native_at_exit')}, indent=2))
if not analysis['passed']: raise SystemExit(1)
