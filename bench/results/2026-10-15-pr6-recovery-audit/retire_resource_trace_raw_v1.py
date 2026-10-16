#!/usr/bin/env python3
"""Retire four explicitly superseded closed profiler artifacts under the run lock.
Repository LICENSE. Preserve every original result and the current sample fixture.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def pin(p):
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest)


def main():
    target = A / 'resource-trace-retirement-v1.json'
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not target.exists()
        receipts = {}
        expected = {2: 'f83e96724fbe5370c6c801d10ca04c77bf154acc514c81466dc4f99d486b7489',
            3: 'df26c8605a701fede97c1e50510b9c5ce3668d76b3f7245f695b3344ecaa208e',
            4: 'a6e8d8c6f16482b0db8a29012017659746f0f877a7c25c41f4616df5be5bbee2'}
        for n, digest in expected.items():
            p = A / f'own-vllm-resource-timeline-model-v{n}.json'
            assert pin(p)['sha256'] == digest
            receipt = json.loads(p.read_text())
            assert not receipt['active'] and receipt['fresh_faults'] == [] and not receipt['driver_dump_exists']
            assert receipt['status'] == ('PASS' if n == 4 else 'FAIL')
            for command in receipt['commands']:
                for old in command.get('owners', []) + [command.get('direct_identity')]:
                    if old is None:
                        continue
                    path = Path('/proc', str(old['pid']), 'stat')
                    if path.exists():
                        assert int(path.read_text().rsplit(')', 1)[1].split()[19]) != old['start_ticks']
            receipts[n] = receipt
        compact = A / 'hardware-resource-profile-v1.json'
        failed = A / 'own-resource-failed-trace-analysis-v3.json'
        assert pin(compact)['sha256'] == 'ae8140e87bd4ebfa059f9864d6fbc7e8c390093adfd9c0de7dcc0295ec36c410'
        assert pin(failed)['sha256'] == '5c3fa70b4f636212f7bf7eedc1c1205bda643cd1210f205fd8c9d77b38a6d5c0'
        current = json.loads(compact.read_text())
        old = json.loads(failed.read_text())
        sample = Path(current['sample_pin']['path'])
        assert pin(sample)['sha256'] == 'f0593196a2167882ad15c7fe86a5672d3a808f235a7d5181eddad5aa442dcd23'
        assert old['counts'] == current['counts']
        assert old['requested_copy_payload_bytes_by_allocation_type'] == current['requested_copy_payload_bytes']
        retired = []
        for n in (2, 3, 4):
            value, = receipts[n]['collector_sides']
            p = Path(value['path'])
            assert p.parent == B / f'main-vllm-resource-timeline-model-v{n}/collector'
            assert pin(p) == {k: value[k] for k in ('path', 'bytes', 'sha256')}
            reason = {2: 'Raw-budget stop is resolved by the predeclared v3 budget; original FAIL/budget/exit/fault receipt retained.',
                3: 'Verbose-summary timeout question resolved for current capture by nonverbose summary; scalar/count/payload/QSA evidence retained. Original FAIL is unchanged; exact historical stop location remains unproved.',
                4: 'Normal complete capture verified against every summary count/sum/min/max; canonical interned individual table and compact hardware report replace duplicate successful raw text.'}[n]
            retired.append(dict(**pin(p), allocated_bytes=p.stat().st_blocks * 512, reason=reason,
                replacement=[str(A / f'own-vllm-resource-timeline-model-v{n}.json'), str(compact), str(failed)], deleted=False))
        old_sample = Path(old['individual_samples']['path'])
        assert pin(old_sample) == old['individual_samples']
        retired.append(dict(**pin(old_sample), allocated_bytes=old_sample.stat().st_blocks * 512,
            reason='Failed emitted-row comparison is closed: counts and requested payload match complete current capture. Compact failed-run quantities/QSA samples retained; current complete table is the sole next counter/gap comparison fixture.',
            replacement=[str(failed), str(sample), str(compact)], deleted=False))
        # Keep the actual incomplete summary as representative reporting-failure evidence.
        raw = Path(retired[1]['path'])
        excerpt = A / 'profiler-verbose-summary-timeout-excerpt-v3.txt'
        assert not excerpt.exists()
        parts = []
        with raw.open() as stream:
            selected = False
            for line in stream:
                if line.strip() == '=== API Timing Summary ===':
                    selected = True
                if selected:
                    parts.append(line)
        text = ''.join(parts)
        assert 0 < len(text.encode()) <= 256 << 10
        assert '=== Device Timing Summary ===' in text and '== L0 Backend ==' in text
        excerpt.write_text(text)
        before = os.statvfs(B)
        result = dict(schema='strata-artifact-retirement-v1', owner='Main', cleanup_active=True,
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=pin(Path(__file__)),
            GPU_or_build_test_executed=False, original_result_statuses={str(n): r['status'] for n, r in receipts.items()},
            original_receipts_unchanged=True, owned_PID_start_absence_verified=True,
            retained=[pin(compact), pin(failed), pin(excerpt), pin(sample)],
            canonical_consumer='Main next GPU-internal counters/component comparison and host-submission-gap diagnosis',
            canonical_byte_budget=64 << 20, canonical_review_utc='2026-10-17T03:00:00Z',
            unrelated_active_worker_inputs_untouched=True, deleted_files=retired)
        target.write_text(json.dumps(result, indent=2) + '\n')
        for row in retired:
            p = Path(row['path'])
            assert pin(p) == {k: row[k] for k in ('path', 'bytes', 'sha256')}
            p.unlink()
            row['deleted'] = True
            target.write_text(json.dumps(result, indent=2) + '\n')
        after = os.statvfs(B)
        result.update(cleanup_active=False, complete=True,
            deleted_logical_bytes=sum(r['bytes'] for r in retired),
            deleted_file_allocated_bytes=sum(r['allocated_bytes'] for r in retired),
            available_bytes_before=before.f_bavail * before.f_frsize,
            available_bytes_after=after.f_bavail * after.f_frsize,
            available_space_limit='Filesystem allocation/snapshots and unrelated activity can make observed free-space change differ from deleted logical/file allocation.')
        for n, digest in expected.items():
            assert pin(A / f'own-vllm-resource-timeline-model-v{n}.json')['sha256'] == digest
        assert pin(sample)['sha256'] == result['retained'][-1]['sha256']
        target.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(manifest=pin(target), deleted_files=len(retired),
            deleted_logical_bytes=result['deleted_logical_bytes'], original_statuses=result['original_result_statuses'])))


if __name__ == '__main__':
    main()
