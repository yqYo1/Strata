#!/usr/bin/env python3
"""Extract closed collection evidence; keep a failed wall-clock gate explicit."""
import collections
import datetime
import fcntl
import gzip
import hashlib
import json
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
O = B/'main-primary-unitrace-runtime-root-v1'


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads((O/'record.json').read_text())
        assert not record['active'] and record['passed'] and record['complete']
        assert record['collected'] and not record['fresh_faults']
        assert not record['driver_dump_exists'] and record['boot_unchanged']
        receipt = record['trace_files'][0]
        raw = Path(receipt['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt['sha256']
        events = json.loads(raw)['traceEvents']
        native = record['real_runtime_exit']['native']['pid']
        gpu = sorted((e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op'),
                     key=lambda e:int(e['args']['id']))
        cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op']
        assert len(gpu) == 195 and len(cpu) > 0
        assert all(e['pid'] == native for e in cpu)
        assert len({e['pid'] for e in gpu}) == 1
        assert [int(e['args']['id']) for e in gpu] == list(range(1, 196))
        for e in gpu+cpu:
            assert isinstance(e['ts'], (float, int)) and e['dur'] >= 0
        # Epoch-microsecond doubles have 0.25us ULP here. Preserve this explicit
        # numeric tolerance; do not compensate the unexplained wall-clock shift.
        tolerance_us = .5
        for i in range(64):
            h, k, d = gpu[3*i:3*i+3]
            assert h['name'] == 'zeCommandListAppendMemoryCopy(H2D)[16388]'
            assert d['name'] == 'zeCommandListAppendMemoryCopy(D2H)[16388]'
            assert 'MemoryCopy' not in k['name']
            assert h['ts']+h['dur'] <= k['ts']+tolerance_us
            assert k['ts']+k['dur'] <= d['ts']+tolerance_us
            if i < 63:
                assert d['ts']+d['dur'] <= gpu[3*i+3]['ts']+tolerance_us
        for i, e in enumerate(gpu, 1):
            flows = [a for a in events if a.get('cat') == f'Flow_H2D_{i}_{native}']
            assert len(flows) == 2
            start = next(a for a in flows if a['ph'] == 's')
            end = next(a for a in flows if a['ph'] == 't')
            assert start['pid'] == native and end['pid'] == e['pid']
            assert start['ts'] <= e['ts']+tolerance_us
            assert abs(end['ts']-e['ts']) <= tolerance_us
        phases = gpu+cpu
        first = min(e['ts'] for e in phases)
        last = max(e['ts']+e['dur'] for e in phases)
        runtime = record['real_runtime_exit']
        wall_start = datetime.datetime.fromisoformat(runtime['started_utc']).timestamp()*1e6
        wall_end = datetime.datetime.fromisoformat(runtime['finished_utc']).timestamp()*1e6
        wall_window_passed = first >= wall_start-1000 and last <= wall_end+1000
        result = dict(original_record_sha256=hashlib.sha256((O/'record.json').read_bytes()).hexdigest(),
            native=runtime['native'], trace=receipt, original_collection_passed=True,
            actual_native_exit_code=runtime['exit_code'], collector_exit_normal=True,
            internal_API_device_lanes_and_dependency_flows_passed=True,
            all_64_H2D_kernel_D2H_dependencies_ordered=True,
            API_event_counts=dict(collections.Counter(e['name'] for e in cpu)),
            GPU_events=195, H2D_copies=64, D2H_copies=64, kernels=66, barriers=1,
            count_note='64 data kernels+zero-byte Runtime.copy device task+bounded-wait device task; isolated host-task error uses one barrier.',
            flow_note='Flow_H2D/Flow_D2H name causal flow direction, not payload direction. Only the195 complete host-to-device pairs are qualified here.',
            numeric_tolerance_us=tolerance_us, trace_first_us=first, trace_last_us=last,
            wrapper_wall_start_us=wall_start, wrapper_wall_end_us=wall_end,
            trace_first_minus_wrapper_start_us=first-wall_start,
            trace_last_minus_wrapper_end_us=last-wall_end,
            external_wall_clock_window_passed=wall_window_passed,
            full_timeline_capability_qualified=wall_window_passed,
            prior_validator_status='Rejected an external wall-clock assertion; collection and original outcome are not rewritten.',
            model_profile_executed=False, GPU_faults=0,
            retained_raw_question='Explain unitrace timestamp offset from actual child UTC and establish clock mapping before external model-marker attribution.',
            retained_raw_owner='Main unitrace clock qualification',
            retained_raw_byte_budget=1<<20,
            retained_raw_consumer='Clock-source/epoch calibration analysis using this500709-byte trace; no model collector yet.',
            next_review_point='Resolve clock mapping or retire this profiler before any model timeline attribution')
        (D/'unitrace-runtime-evidence.json').write_text(json.dumps(result, indent=2)+'\n')
        for name in ('main-primary-unitrace-fetch-root-v1', 'main-primary-unitrace-build-root-v1',
                     'main-primary-unitrace-runtime-root-v1'):
            original = B/name/'record.json'
            source = original.read_bytes()
            original_record = json.loads(source)
            assert not original_record['active']
            destination = D/'receipts'/name
            destination.mkdir(parents=True, exist_ok=True)
            archive = gzip.compress(source, mtime=0)
            assert gzip.decompress(archive) == source
            (destination/'record.json.gz').write_bytes(archive)
            (destination/'index.json').write_text(json.dumps(dict(
                original=str(original), raw_bytes=len(source), raw_sha256=hashlib.sha256(source).hexdigest(),
                archive_sha256=hashlib.sha256(archive).hexdigest(), status=original_record['passed']), indent=2)+'\n')
        print(json.dumps(dict(internal_timeline_passed=True,
            external_wall_clock_window_passed=wall_window_passed,
            full_timeline_capability_qualified=wall_window_passed,
            model_profile_executed=False, retained_raw_bytes=len(raw))))


if __name__ == '__main__':
    main()
