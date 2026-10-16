#!/usr/bin/env python3
"""Retire closed successful BF16-prefix benchmark logs after committed analysis."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def committed(p):
    data = subprocess.check_output(['git', 'show', 'HEAD:'+str(p.relative_to(P))], cwd=P)
    assert hashlib.sha256(data).hexdigest() == sha(p)


def main():
    dst = A / 'retention-qsa-bf16-prefix-closed-v1.json'
    assert not dst.exists()
    owner_path = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(owner_path) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('prefix_retention_owner', owner_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cases = [('diagnostic', 'bf16-prefix', 0), ('profile', 'bf16-prefix', 0)]
    cases += [('clean', arm, n) for arm in ['current', 'bf16', 'bf16-prefix'] for n in range(3)]
    replacements = [A / n for n in ['qsa-bf16-prefix-clean-summary-v1.json',
                                  'qsa-bf16-prefix-profile-summary-v1.json']]
    record = dict(schema='strata-bf16-prefix-closed-retention-v1', deleted=[], receipts={},
                  logical_bytes=0, allocated_file_bytes=0, original_statuses_unchanged=True,
                  retained_current_engine_components_and_sources=True,
                  actual_free_space_reclaimed='Not inferred; filesystem snapshots may retain old extents.')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p in replacements:
            committed(p)
        record['replacements'] = {str(p): sha(p) for p in replacements}
        frozen = json.loads((A/'own-qsa-bf16-prefix-engine-build-v1.json').read_text())['frozen_files']
        paths = {}
        for phase, arm, n in cases:
            p = A / f'own-qsa-bf16-prefix-{phase}-{arm}-r{n}-v1.json'
            r = json.loads(p.read_text())
            committed(p)
            assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
            assert r['fresh_faults'] == [] and r['boot_unchanged'] and not r['driver_dump_exists']
            record['receipts'][str(p)] = sha(p)
            for c in r['commands']:
                assert m.closed(c) and c['exit_code'] == 0
                for old in c['owners'] + [c['direct_identity']]:
                    now = m.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            root = B / f'main-qsa-bf16-prefix-{phase}-{arm}-r{n}-v1'
            assert sha(root/'record.json') == sha(p)
            assert json.loads((root/'result.json').read_text()) == r['result']
            for f in (root/'stock-results').glob('*/results.json'):
                assert json.loads(f.read_text()) == r['benchmark_result']
            assert r['request_journal_verification']['valid'] and r['request_journal_verification']['complete']
            assert sha(root/'owned/engine.stderr') == r['native_log_sha256']
            if phase == 'profile':
                raw = (root/'owned/engine.stderr').read_text().splitlines()
                for key in ['coarse_profile_lines', 'qsa_profile_lines', 'cpu_wait_lines', 'cpu_host_lines']:
                    assert r[key] and all(line in raw for line in r[key])
                summary = json.loads(replacements[1].read_text())
                assert all(line in raw for line in summary['host_lifecycle_lines'])
            assert not (root/'cache').exists()
            allowed = {'record.json', 'result.json', 'config.json', 'plan.json', 'engine-wrapper.py',
                       'engine-start.json', 'engine-exit.json', 'expert-profile.bin'}
            for f in root.rglob('*'):
                if not f.is_file():
                    continue
                rel = f.relative_to(root)
                assert not f.is_symlink() and str(f) not in frozen
                assert (len(rel.parts) == 1 and rel.name in allowed) or (
                    rel.parts[0] == 'owned' and f.suffix in ['.stdout', '.stderr']) or (
                    rel.parts[0] == 'stock-results' and f.name in ['journal-state.json', 'results.json', 'README.md', 'requests.jsonl'])
                paths[f] = dict(replacement=str(p), reason='Closed successful protocol, verified duplicate result/receipt, private profile copy or reproducible generated recipe; individual samples, exact commands/environment, request identity, profile tables and fault/ownership outcomes are committed. No current fixture/RESTORE consumer.')
        for f, why in paths.items():
            st = f.stat()
            row = dict(path=str(f), sha256=sha(f), bytes=st.st_size,
                       allocated_file_bytes=st.st_blocks*512, **why)
            record['deleted'].append(row)
            record['logical_bytes'] += st.st_size
            record['allocated_file_bytes'] += st.st_blocks*512
        for row in record['deleted']:
            f = Path(row['path'])
            assert sha(f) == row['sha256'] and f.stat().st_size == row['bytes']
            f.unlink()
        for p, digest in record['receipts'].items():
            assert sha(p) == digest
        dst.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(manifest=str(dst), files=len(record['deleted']),
                         logical_bytes=record['logical_bytes'], allocated_file_bytes=record['allocated_file_bytes'])))


if __name__ == '__main__':
    main()
