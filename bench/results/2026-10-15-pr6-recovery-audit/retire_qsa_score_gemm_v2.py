#!/usr/bin/env python3
"""Retire only closed successful GEMM probe text logs and duplicate records."""
import datetime, fcntl, hashlib, importlib.util, json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    out = A / 'qsa-score-gemm-retention-v2.json'
    assert not out.exists()
    cohort_path = A / 'qsa-score-gemm-cohort-v2.json'
    cohort = json.loads(cohort_path.read_text())
    assert cohort['passed'] and cohort['complete'] and not cohort['active']
    resource_path = A / 'qsa-score-gemm-resource-results-v1.json'
    resource = json.loads(resource_path.read_text())
    assert resource['passed']
    # Complete allocation/command-history question was resolved by this single
    # large-shape trace; retain the bracket, transfer records and all counts in
    # the immutable receipts/summary, rather than the duplicate full log.
    assert len(resource['whole_fixture_tiny_transfers']) == 2
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(owner_path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = module('gemm_retirement_owner', owner_path)
    package = A / 'own-vllm-xe2-qualification-v1'
    meta = json.loads((package / 'manifest.json').read_text())
    helper = package / 'main_controller.py'
    assert sha(helper) == meta['package_files']['main_controller.py']['sha256']
    guards = module('gemm_retirement_guards', helper)
    receipts = sorted(A.glob('qsa-score-gemm-probe-*-v[12].json'))
    receipts += sorted(A.glob('qsa-score-gemm-resource-gemm-trace78453-r0-v1.json'))
    rec = dict(schema=2, passed=False, complete=False,
               owner='Main / closed QSA GEMM v1/v2 component comparison',
               lock=str(B / 'owned-v0141-measurement.lock'),
               scope='Only successful closed probe/build raw text and byte-identical duplicate records; canonical source, receipts, samples, binaries, build trees and named source snapshots remain.',
               canonical_receipts=[], deleted=[], retained_exception=[],
               extraction='Useful API bracket, transfer sizes/counts, numerical and selected-set results, every timing sample, source/toolchain/runtime, owners and fault results are in canonical receipts and compact results; preserve all meaningful warning lines below.')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners(meta['known_model_files'])
        guards.driver_guard()
        planned = []
        for receipt in receipts:
            r = json.loads(receipt.read_text())
            assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
            if r.get('GPU_executed'):
                assert r['fresh_faults'] == [] and not r['driver_dump_entries'] and r['boot_unchanged']
            root = B / ('main-' + receipt.stem)
            assert root.is_dir()
            rec['canonical_receipts'].append(dict(path=str(receipt), sha256=sha(receipt), bytes=receipt.stat().st_size))
            for c in r['commands']:
                assert owned.closed(c) and c['exit_code'] == 0
                for old in c['owners'] + [c['direct_identity']]:
                    now = owned.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
                for name, metadata in c['logs'].items():
                    p = root / 'owned' / name
                    assert p.is_file() and p.stat().st_size == metadata['bytes'] and sha(p) == metadata['sha256']
                    value = p.read_text()
                    warnings = [x for x in value.splitlines() if re.search(r'(?i)\b(warning|error|fatal|exception)\b|UR_RESULT_ERROR_|ZE_RESULT_ERROR_', x)]
                    if sum(len(x.encode()) + 1 for x in warnings) > 256 << 10:
                        rec['retained_exception'].append(dict(path=str(p), reason='Extracted diagnostic text exceeds256KiB; review separately.'))
                        continue
                    planned.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p), retained=str(receipt), warnings=warnings, reason='Closed successful raw log; canonical receipt and compact result preserve the evidence.'))
            duplicate = root / 'record.json'
            if duplicate.exists():
                assert duplicate.read_bytes() == receipt.read_bytes()
                planned.append(dict(path=str(duplicate), bytes=duplicate.stat().st_size, sha256=sha(duplicate), retained=str(receipt), warnings=[], reason='Byte-identical duplicate of canonical receipt.'))
        assert len({x['path'] for x in planned}) == len(planned)
        rec.update(deleted=planned, file_count=len(planned), logical_bytes=sum(x['bytes'] for x in planned),
                   canonical_bytes_unchanged=True, passed=True, complete=True,
                   closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        out.write_text(json.dumps(rec, indent=2) + '\n')
        for item in planned:
            p = Path(item['path'])
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
            p.unlink()
        for item in rec['canonical_receipts']:
            assert sha(Path(item['path'])) == item['sha256']
    print(json.dumps(dict(receipts=len(receipts), retired_files=rec['file_count'], logical_bytes=rec['logical_bytes'], retained_exceptions=len(rec['retained_exception']))))

if __name__ == '__main__':
    main()
