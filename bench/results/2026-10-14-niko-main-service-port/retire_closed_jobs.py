#!/usr/bin/env python3
"""Retire only this report's closed job roots after committed evidence checks."""
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-niko-main-service-port'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_commit = subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip()
        assert receipt_commit == '06da66fe052934be4d1f66ca755458edc905bf7a'
        receipts = json.loads((D / 'receipts-index.json').read_text())
        saved = {}
        for row in receipts:
            f = D / row['saved_path']
            blob = subprocess.check_output(['git', '-C', str(P), 'show', 'HEAD:' + str(f.relative_to(P))])
            assert blob == f.read_bytes() and sha(blob) == row['saved_sha256']
            data = gzip.decompress(blob)
            assert sha(data) == row['source_sha256'] and data == Path(row['source_path']).read_bytes()
            saved[row['source_path']] = row['saved_path']
        raw = {r['path']: r for r in json.loads((D / 'closed-log-retirement-plan.json').read_text())['files']}
        outcomes = json.loads((D / 'outcomes.json').read_text())
        rows, roots = [], []
        for item in outcomes:
            root = B / item['root']
            assert root.parent == B and root.name.startswith('main-niko-') and not root.is_symlink()
            record = json.loads((root / 'record.json').read_text())
            assert record['active'] is False
            assert all(c['direct_child_reaped'] and c['session_empty'] for c in record['commands'])
            roots.append(root)
            for directory, dirs, files in os.walk(root, followlinks=False):
                for name in dirs + files:
                    f = Path(directory) / name
                    if f.is_symlink():
                        target = os.readlink(f)
                        rows.append(dict(path=str(f), type='symlink', target=target,
                                         sha256=sha(target.encode()), bytes=0, reason='closed generated build link'))
                    elif f.is_file():
                        data = f.read_bytes()
                        if str(f) in saved:
                            evidence, reason = saved[str(f)], 'byte-identical receipt committed; redundant original'
                        elif str(f) in raw:
                            r = raw[str(f)]
                            assert sha(data) == r['sha256'] and len(data) == r['bytes']
                            evidence, reason = r['evidence'], 'closed raw log; useful results/error windows committed'
                        else:
                            evidence = 'source-pins, exact intermediate sources and compile/run command receipts'
                            reason = 'closed CPU fixture/build artifact; no active consumer; source fixtures retained'
                        rows.append(dict(path=str(f), type='file', sha256=sha(data), bytes=len(data),
                                         reason=reason, evidence=evidence))
        manifest = D / 'deletion-manifest.json.gz'
        manifest.write_bytes(gzip.compress((json.dumps(rows, indent=2) + '\n').encode(), mtime=0))
        assert len(json.loads(gzip.decompress(manifest.read_bytes()))) == len(rows)
        result = dict(complete=False, evidence_commit=receipt_commit, manifest_sha256=sha(manifest.read_bytes()),
                      files=len(rows), logical_bytes=sum(r['bytes'] for r in rows), deleted_roots=[],
                      original_result_status_preserved=True, GPU_model_executed=False,
                      retained_dependency_target=dict(path=str(B / 'niko-service-test-deps-v1'),
                      budget=32 << 20, consumer='subsequent mandatory Niko/native feature tests',
                      review_point='main reconciliation completion or dependency supersession'))
        def persist():
            (D / 'retention-results.json').write_text(json.dumps(result, indent=2) + '\n')
        persist()
        for root in roots:
            shutil.rmtree(root)
            assert not root.exists()
            result['deleted_roots'].append(str(root))
            persist()
        # Historical paths in the untouched receipts are interpreted using this
        # manifest. No validation assertion is removed to accommodate deletion.
        for row in receipts:
            data = gzip.decompress((D / row['saved_path']).read_bytes())
            assert sha(data) == row['source_sha256']
        result['complete'] = True
        persist()
        print(json.dumps(dict(complete=True, roots=len(roots), files=len(rows), logical_bytes=result['logical_bytes'])), flush=True)

if __name__ == '__main__':
    main()
