#!/usr/bin/env python3
"""Materialize byte-verified historical dependencies, without a build or GPU.

Source Git objects are read from Strata and the independent ghq llama.cpp clone.
No checkout, branch, installed dependency or historical source is modified.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

A = Path(__file__).resolve().parent
P = A.parents[2]
C = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-pr6-prefill-restore')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
LLAMA = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp')
REV = '3cf03257f219afbe7334045ff7c6a06ac68c627d'


def git(repo, *argv, data=None):
    p = subprocess.run(['/usr/bin/git', '-C', str(repo), *argv], input=data,
                       capture_output=True, timeout=30)
    assert p.returncode == 0, (argv, p.stderr[:2048])
    return p.stdout


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p.read_bytes()))


def copy_objects(repo, entries, destination):
    assert len(entries) <= 2048
    raw = git(repo, 'cat-file', '--batch', data=b''.join(
        (v['blob'] + '\n').encode('ascii') for v in entries))
    assert len(raw) <= 128 << 20
    result = []
    offset = 0
    for item in entries:
        end = raw.index(b'\n', offset)
        blob, kind, size = raw[offset:end].decode('ascii').split()
        assert blob == item['blob'] and kind == 'blob'
        size = int(size)
        assert 0 <= size <= 32 << 20
        start = end + 1
        payload = raw[start:start + size]
        assert len(payload) == size and raw[start + size:start + size + 1] == b'\n'
        assert hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + payload).hexdigest() == blob
        if 'sha256' in item:
            assert item['bytes'] == size and item['sha256'] == digest(payload)
        rel = Path(item['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        target = destination / rel
        assert not target.exists() and not target.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        mode = item.get('mode', '100644')
        assert mode in ('100644', '100755')
        target.chmod(0o700 if mode == '100755' else 0o600)
        assert target.read_bytes() == payload
        result.append(dict(path=str(rel), blob=blob, mode=mode, bytes=size, sha256=digest(payload)))
        offset = start + size + 1
    assert offset == len(raw)
    return result


def main():
    root = B / 'main-pr6-historical-dependencies-v1'
    out = A / 'own-pr6-dependency-preparation-v1.json'
    assert not root.exists() and not out.exists()
    closure = C / 'bench/results/2026-10-16-pr6-faithful-restore/historical-closure.json'
    before = pin(closure)
    spec = json.loads(closure.read_text())
    assert spec['external_llama_commit'] == REV and len(spec['external_vendor_files']) == 49
    assert git(LLAMA, 'status', '--porcelain=v1', '--untracked-files=no') == b''
    assert git(LLAMA, 'cat-file', '-t', REV).strip() == b'commit'
    rows = git(LLAMA, 'ls-tree', '-r', '-z', '--full-tree', REV, 'ggml', 'cmake', 'LICENSE').split(b'\0')
    entries = []
    for row in rows:
        if not row:
            continue
        info, path = row.split(b'\t')
        mode, kind, blob = info.decode('ascii').split()
        assert kind == 'blob' and mode in ('100644', '100755')
        entries.append(dict(path=path.decode('utf-8'), mode=mode, blob=blob))
    assert 1000 < len(entries) <= 2048
    started = time.monotonic()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        record = dict(schema='strata-pr6-dependency-preparation-v1', passed=False,
            generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_closure=before, llama_repository=str(LLAMA), llama_revision=REV,
            llama_root_HEAD=git(LLAMA, 'rev-parse', 'HEAD').decode().strip(),
            llama_root_clean=True, GPU_executed=False, configured=False, built=False,
            vendor_root=str(root / 'vendor'), llama_snapshot=str(root / 'llama'),
            byte_budget=128 << 20, retention=dict(owner='Main',
                consumer='PR6 isolated configure/build input; not a resume backup',
                next_review='At replacement or retirement of the PR6 validation build'))
        try:
            record['vendor_files'] = copy_objects(P, spec['external_vendor_files'], root / 'vendor')
            record['llama_files'] = copy_objects(LLAMA, entries, root / 'llama')
            assert pin(closure) == before
            record['total_source_bytes'] = sum(v['bytes'] for group in ('vendor_files', 'llama_files') for v in record[group])
            assert record['total_source_bytes'] <= 128 << 20
            record.update(passed=True, complete=True,
                scope='49 historical vendor blobs and pinned ggml/cmake/LICENSE only; complete per-file Git blob and SHA256 verification. No compiler or runtime qualification. Git build-commit macro in the private snapshot is not a source-identity proof.')
        except BaseException as e:
            record['error'] = repr(e)
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
            out.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=pin(out), passed=record['passed'],
            vendor_files=len(record['vendor_files']), llama_files=len(record['llama_files']),
            total_source_bytes=record['total_source_bytes'], GPU_executed=False)))


if __name__ == '__main__':
    main()
