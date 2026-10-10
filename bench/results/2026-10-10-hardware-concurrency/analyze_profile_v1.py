"""Offline PTI event reconciliation; no device work or clock-domain subtraction.

Run under the experiment lock with the closed profile directory as argument.
The compact operation table keeps every GPU interval and its host append call,
but omits the full successful API/poll timeline.
"""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib, json, sys

def identity(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest())

def ns(value):
    value = Decimal(value) * 1000
    assert value.is_finite() and value >= 0 and value == value.to_integral_value()
    return int(value)

def union(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        assert hi >= lo
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged

def intersection(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total

p = Path(sys.argv[1])
r = json.loads((p / 'record.json').read_text())
assert r['complete'] and r['passed'] and not r['active']
assert r['profiling_perturbation'] and not r['model_executed'] and not r['adopted']
assert r['visible_kernel_GPU_entries'] == [] and not r['devcoredump_after']
for command in r['commands']:
    assert command['normal_exit'] and command['session_empty']
    assert command['observation_complete'] and command['exit_code'] == 0
    assert command['direct_child_reaped'] and not command['cleanup']
    assert not command['errors'] and not command['survivors']
    for owner in command['owners']:
        proc = Path('/proc') / str(owner['pid']) / 'stat'
        if proc.exists():
            fields = proc.read_text().rsplit(')', 1)[1].split()
            assert int(fields[19]) != owner['start_ticks'], 'original owner remains'
for filename, pinned in r['profile_files'].items():
    assert identity(Path(filename)) == pinned, filename
trace_paths = list(p.glob('hardware-concur.*.json'))
assert len(trace_paths) == 1
trace = trace_paths[0]
events = json.loads(trace.read_text(), parse_float=Decimal)['traceEvents']
gpu = [e for e in events if e.get('cat') == 'gpu_op']
cpu = [e for e in events if e.get('cat') == 'cpu_op']
assert len(gpu) == 3330 and len(cpu) == 16516
assert all(e['ph'] == 'X' for e in gpu + cpu)
metadata = [e for e in events if e['ph'] == 'M']
lanes = {(e['pid'], e['tid']): e['args']['name'] for e in metadata
         if e['name'] == 'thread_name'}
ids = [int(e['args']['id']) for e in gpu]
assert len(set(ids)) == len(ids) and set(ids) == set(range(1, 3331))
flows = defaultdict(list)
host = defaultdict(list)
for e in events:
    if e.get('cat', '').startswith(('Flow_H2D_', 'Flow_D2H_')):
        fields = e['cat'].split('_')
        assert len(fields) == 4 and int(fields[2]) == int(e['id'])
        flows[(fields[1], int(e['id']))].append(e)
for e in cpu:
    host[(e['pid'], e['tid'], ns(e['ts']))].append(e)
assert len(flows) == 6660
origin = min(ns(e['ts']) for e in gpu)
rows, copy_intervals, gemm_intervals = [], [], []
for e in gpu:
    ident = int(e['args']['id'])
    start = ns(e['ts']); end = start + ns(e['dur'])
    assert end > start
    h2d = flows[('H2D', ident)]; d2h = flows[('D2H', ident)]
    assert Counter(x['ph'] for x in h2d) == {'s': 1, 't': 1}
    assert Counter(x['ph'] for x in d2h) == {'s': 1, 't': 1}
    append = next(x for x in h2d if x['ph'] == 's')
    begin = next(x for x in h2d if x['ph'] == 't')
    finish = next(x for x in d2h if x['ph'] == 's')
    assert (begin['pid'], begin['tid']) == (e['pid'], e['tid'])
    assert (finish['pid'], finish['tid']) == (e['pid'], e['tid'])
    # Pinned PTI StringifyDeviceEvent deliberately anchors both flow directions
    # at command_start. Flow_D2H is not a command_end/completion timestamp.
    assert ns(begin['ts']) == start and ns(finish['ts']) == start
    apis = host[(append['pid'], append['tid'], ns(append['ts']))]
    assert len(apis) == 1, (ident, len(apis))
    api = apis[0]
    assert api['name'] in ('zeCommandListAppendMemoryCopy',
                           'zeCommandListAppendLaunchKernelWithArguments')
    role = ('copy8MiB' if e['name'] == 'zeCommandListAppendMemoryCopy(H2D)[8388608]'
            else 'GEMM' if e['name'].startswith('gemm_') else 'setup_or_validation')
    if role == 'copy8MiB':
        assert api['name'] == 'zeCommandListAppendMemoryCopy'
        copy_intervals.append([start, end])
    if role == 'GEMM':
        assert api['name'] == 'zeCommandListAppendLaunchKernelWithArguments'
        gemm_intervals.append([start, end])
    rows.append([ident, role, e['name'], e['pid'], e['tid'],
                 start - origin, end - origin, api['name'],
                 ns(api['ts']) - origin, ns(api['dur'])])
assert len(copy_intervals) == 144 and len(gemm_intervals) == 3168
copy_union = union(copy_intervals); gemm_union = union(gemm_intervals)
overlap = intersection(copy_union, gemm_union)
assert overlap == 0
ordered = sorted([(a,b,'copy') for a,b in copy_union] +
                 [(a,b,'GEMM') for a,b in gemm_union])
cross_gaps = [right[0]-left[1] for left,right in zip(ordered, ordered[1:])
              if left[2] != right[2]]
assert cross_gaps and min(cross_gaps) >= 1000
table = dict(schema=1, trace_identity=identity(trace), origin_epoch_ns=origin,
             columns=['operation_id', 'role', 'device_name', 'device_pid', 'device_tid',
                      'device_start_relative_ns', 'device_end_relative_ns',
                      'host_append_api', 'host_append_start_relative_ns', 'host_append_duration_ns'],
             rows=rows)
target = p / 'compact-operation-intervals.json'
assert not target.exists()
target.write_text(json.dumps(table, separators=(',', ':')) + '\n')
summary = dict(schema=1, passed=True, device_executed=False,
    receipt=dict(path=str(p / 'record.json'), **identity(p / 'record.json')),
    analyzer_identity=identity(Path(__file__)), trace=dict(path=str(trace), **identity(trace)),
    compact_operations=dict(path=str(target), **identity(target)),
    GPU_operations=len(gpu), host_API_calls=len(cpu), host_append_joins=len(rows),
    target_copy_calls=144, target_copy_logical_bytes=144*8388608,
    target_GEMM_kernels=3168,
    expected_source_counts=dict(copy='6 shapes * 2 passes * 3 modes * 4 copies',
         gemm='4 small shapes * 2 passes * 3 modes * 128 + 2 large shapes * 2 passes * 3 modes * 8'),
    device_name_counts=dict(Counter(e['name'] for e in gpu)),
    target_lane_counts={name: dict(Counter(str(e['tid']) for e in gpu if
        (e['name'] == 'zeCommandListAppendMemoryCopy(H2D)[8388608]' if name == 'copy'
         else e['name'].startswith('gemm_')))) for name in ('copy', 'GEMM')},
    lane_names=[dict(pid=pid, tid=tid, label=name) for (pid, tid), name in lanes.items()],
    device_copy_union_ns=sum(b-a for a,b in copy_union),
    device_GEMM_union_ns=sum(b-a for a,b in gemm_union),
    device_copy_GEMM_intersection_ns=overlap,
    closest_cross_role_gap_ns=min(cross_gaps),
    serialization_guard_ns=1000,
    scope='All target GPU intervals, including warm, solo, forced serial and concurrent modes; zero intersection over all implies zero over the concurrent subset.',
    limitations=['Instrumented qualification process; excluded from clean capacity timings.',
      'Independent next-slot copies, not exact Strata operands/rings/routes/dependencies.',
      'PTI engine label is recorded ordinal/index, not a hardware utilization counter.',
      'Both PTI device flow directions anchor command_start, not command_end/completion.',
      'Serialized epoch-microsecond timestamps have submicrosecond precision; closest cross-role gap exceeds the conservative 1us guard.',
      'No production oneMKL returned-event/internal-kernel span join is established.',
      'Device GPU timestamps are compared only within this PTI trace; no host steady-clock join.',
      'Probe validates both copy buffers and final surviving products, not overwritten products.',
      'No model correctness, clean performance or full physical 262144-context qualification.'])
out = p / 'profile-analysis.json'
assert not out.exists()
out.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(dict(passed=True, compact_bytes=target.stat().st_size,
                     copy_calls=144, GEMM_kernels=3168, intersection_ns=overlap)))
