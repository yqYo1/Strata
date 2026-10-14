#!/usr/bin/env python3
"""Verify the closed probe's complete operation sequence without UTC alignment."""
import collections
import datetime
import fcntl
import hashlib
import json
import re
import statistics
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = D / 'unitrace-gemm-collection-v1.json'
        r = json.loads(original.read_text())
        assert r['passed'] and r['complete'] and r['collected'] and not r['active']
        assert not r['fresh_faults'] and not r['driver_dump_exists'] and r['boot_unchanged']
        assert all(c['exit_code'] == 0 and c['direct_child_reaped'] and c['session_empty']
                   for c in r['commands'])
        assert len(r['trace_files']) == 1
        trace = r['trace_files'][0]
        path = Path(trace['path'])
        assert path.stat().st_size == trace['bytes'] and digest(path) == trace['sha256']
        events = json.loads(path.read_bytes())['traceEvents']
        gpu = sorted((e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op'),
                     key=lambda e: int(e['args']['id']))
        cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op']
        native = r['real_native_exit']['native']['pid']
        assert r['real_native_exit']['exit_code'] == 0
        assert len(gpu) == 739 and len(cpu) == 10242
        assert [int(e['args']['id']) for e in gpu] == list(range(1, 740))
        assert all(e['pid'] == native for e in cpu) and len({e['pid'] for e in gpu}) == 1
        assert any(e.get('ph') == 'M' and 'Arc(TM) B570' in e.get('args', {}).get('name', '')
                   for e in events)
        tolerance_us = .5  # Epoch-microsecond double ULP is .25us here.
        assert all(e['dur'] >= 0 for e in gpu + cpu)
        assert all(a['ts'] + a['dur'] <= b['ts'] + tolerance_us for a, b in zip(gpu, gpu[1:]))
        flows = collections.defaultdict(list)
        for e in events:
            if e.get('cat', '').startswith('Flow_H2D_'):
                flows[e['cat']].append(e)
        assert len(flows) == len(gpu)
        enqueue_delay = {}
        for e in gpu:
            i = int(e['args']['id'])
            f = flows[f'Flow_H2D_{i}_{native}']
            assert len(f) == 2 and {v['ph'] for v in f} == {'s', 't'}
            start = next(v for v in f if v['ph'] == 's')
            end = next(v for v in f if v['ph'] == 't')
            assert start['pid'] == native and end['pid'] == e['pid']
            assert start['ts'] <= e['ts'] + tolerance_us
            assert abs(end['ts'] - e['ts']) <= tolerance_us
            enqueue_delay[i] = e['ts'] - start['ts']

        names, calls, copies, bootstrap = [], [], [], []
        cursor = 0

        def take():
            nonlocal cursor
            e = gpu[cursor]
            cursor += 1
            return e

        def copy(direction, size, purpose, cell):
            e = take()
            assert e['name'] == f'zeCommandListAppendMemoryCopy({direction})[{size}]', (cursor, e['name'], direction, size)
            copies.append(dict(operation_id=int(e['args']['id']), cell=cell, direction=direction,
                               bytes=size, purpose=purpose, duration_us=e['dur']))
            return e

        for ne in [1, 12, 428, 1490, 1792, 3245, 3855, 4096]:
            for role, n, k in [('GU', 1280, 2560), ('Down', 2560, 640)]:
                for ld in [n, n + 17]:
                    cell = f'{ne}/{role}/{ld}'
                    xb, wb, yb = (ne * k + 32) * 2, (n * k + 32) * 2, (ne * ld + 32) * 4
                    copy('M2D', xb, 'initial_X', cell)
                    copy('M2D', wb, 'initial_W', cell)
                    copy('M2D', yb, 'initial_Y', cell)
                    for beta in [0, .25]:
                        for arm in [0, 1, 2]:
                            reset = copy('M2D', yb, 'reset_Y', cell)
                            if cursor == 4:
                                copy('M2D', 4, 'library_bootstrap', cell)
                                e = take()
                                assert e['name'] == 'binary_test[SIMD16 {1; 1; 1} {4; 2; 1}]'
                                bootstrap.append(dict(operation_id=int(e['args']['id']), name=e['name'], duration_us=e['dur']))
                                copy('D2M', 4, 'library_bootstrap', cell)
                            kernel = take()
                            assert kernel['name'].startswith('gemm_f16f16f32_nocopy_tn_')
                            if kernel['name'] not in names:
                                names.append(kernel['name'])
                            read = copy('D2M', yb, 'read_Y', cell)
                            calls.append(dict(ne=ne, role=role, N=n, K=k, ld=ld, beta=beta, arm=arm,
                                reset_operation_id=int(reset['args']['id']), kernel_operation_id=int(kernel['args']['id']),
                                read_operation_id=int(read['args']['id']), kernel_name_index=names.index(kernel['name']),
                                kernel_duration_us=kernel['dur'],
                                submission_flow_to_kernel_start_us=enqueue_delay[int(kernel['args']['id'])]))
                    copy('D2M', xb, 'check_X', cell)
                    copy('D2M', wb, 'check_W', cell)
        assert cursor == len(gpu) and len(calls) == 192 and len(copies) == 546 and len(bootstrap) == 1
        comparisons = []
        for i in range(0, len(calls), 3):
            group = calls[i:i+3]
            assert [c['arm'] for c in group] == [0, 1, 2]
            assert len({c['kernel_name_index'] for c in group}) == 1
            comparisons.append({k: group[0][k] for k in ['ne', 'role', 'ld', 'beta', 'kernel_name_index']})
        copy_summary = []
        for direction, purpose in sorted({(c['direction'], c['purpose']) for c in copies}):
            group = [c for c in copies if (c['direction'], c['purpose']) == (direction, purpose)]
            copy_summary.append(dict(direction=direction, purpose=purpose, count=len(group),
                bytes=sum(c['bytes'] for c in group), duration_us=sum(c['duration_us'] for c in group)))
        first = min(e['ts'] for e in gpu + cpu)
        last = max(e['ts'] + e['dur'] for e in gpu + cpu)
        wall_start = datetime.datetime.fromisoformat(r['real_native_exit']['started_utc']).timestamp() * 1e6
        wall_end = datetime.datetime.fromisoformat(r['real_native_exit']['finished_utc']).timestamp() * 1e6
        external = first >= wall_start - 1000 and last <= wall_end + 1000
        result = dict(passed=True, original_collection_sha256=digest(original), trace=trace,
            extraction_recipe_sha256=digest(Path(__file__)), source_pins=r['source_pins'],
            binary_sha256=r['binary_sha256'], native=r['real_native_exit']['native'],
            kernel_visibility_qualified=True, complete_operation_sequence_and_internal_flows_qualified=True,
            external_UTC_correlation_qualified=external, performance_qualified=False, model_profile_executed=False,
            quality_qualified=False, adopted=False, GPU_events=len(gpu), CPU_events=len(cpu),
            GEMM_kernels=192, bootstrap_kernels=bootstrap, copies=546, numeric_tolerance_us=tolerance_us,
            trace_first_minus_wrapper_start_us=first-wall_start, trace_last_minus_wrapper_end_us=last-wall_end,
            all_64_row_column_wrapper_groups_select_same_device_kernel=True,
            kernel_names=names, calls=calls, copy_summary=copy_summary,
            copy_duration_samples=copies,
            API_event_counts=dict(collections.Counter(e['name'] for e in cpu)), fresh_GPU_faults=0,
            limits='Synthetic hot FP16 validation with beta0/.25, both strides, per-call waits/readbacks, profiler overhead. Kernel durations and flow delays are not clean model throughput or a critical-path attribution. M2D/D2M reflect profiler classification of malloc host pointers, not pinned H2D bandwidth. No external UTC alignment or model markers are claimed. Original collection outcome remains unchanged.')
        dest = D / 'unitrace-gemm-evidence-v1.json'
        assert not dest.exists()
        dest.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(passed=True, GEMM_kernels=192, identical_device_kernel_groups=64,
            external_UTC_correlation_qualified=external, evidence_bytes=dest.stat().st_size)))


if __name__ == '__main__':
    main()
