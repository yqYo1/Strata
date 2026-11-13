#!/usr/bin/env python3
"""Check every report worker input inside its actual filesystem namespace.

Main-authored; repository LICENSE. This proves filesystem access only. The
worker must separately prove its own Orca IPC and normal completion lifecycle.
No source is modified: the write probe opens with APPEND and never writes.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

OUTPUT = 'worker-filesystem-proof.json'
MAX_INPUT = 64 << 20


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inspect_input(item):
    path = item['path']
    row = dict(path=path, expected_sha256=item['sha256'], passed=False)
    fd = None
    try:
        if not Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('absolute normalized input path required')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= MAX_INPUT:
            raise ValueError('bounded regular input required')
        with os.fdopen(os.dup(fd), 'rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError('input changed while reading')
        readonly = bool(os.fstatvfs(fd).f_flag & os.ST_RDONLY)
        denied = None
        try:
            write_fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as error:
            denied = error.errno
        else:
            os.close(write_fd)
        row.update(bytes=before.st_size, sha256=actual, uid=before.st_uid,
            mode=stat.S_IMODE(before.st_mode), readonly_mount=readonly, write_open_errno=denied,
            no_source_write_performed=True,
            passed=actual == item['sha256'] and readonly and denied in (13, 30))
    except (OSError, ValueError) as error:
        row['error'] = repr(error)
    finally:
        if fd is not None:
            os.close(fd)
    return row


def main():
    if len(sys.argv) != 3:
        raise ValueError('use worker_input_check.py MANIFEST EXPECTED_MANIFEST_SHA256')
    manifest_path = Path(sys.argv[1])
    expected = sys.argv[2]
    if not manifest_path.is_absolute() or manifest_path.stat().st_size > 1 << 20:
        raise ValueError('bounded absolute manifest required')
    if len(expected) != 64 or digest(manifest_path) != expected:
        raise ValueError('manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    report = Path(manifest['report_directory']).resolve(strict=True)
    if Path.cwd().resolve() != report or os.geteuid() == 0:
        raise ValueError('actual non-root report workspace required')
    inputs = manifest['protected_inputs']
    if not 8 <= len(inputs) <= 511:
        raise ValueError('8..511 protected inputs required')
    paths = [v['path'] for v in inputs]
    if len(set(paths)) != len(paths) or str(manifest_path) in paths:
        raise ValueError('unique inputs; manifest is checked automatically')
    for item in inputs:
        if not isinstance(item['sha256'], str) or len(item['sha256']) != 64:
            raise ValueError('explicit SHA256 required for every input')
    rows = [inspect_input(v) for v in [dict(path=str(manifest_path), sha256=expected), *inputs]]
    canary = report / '.worker-check-canary'
    with canary.open('xb') as f:
        f.write(b'report-write-check\n')
    writable = canary.read_bytes() == b'report-write-check\n'
    canary.unlink()
    no_gpu_device = not os.path.lexists('/dev/dri')
    passed = all(r['passed'] for r in rows) and writable and no_gpu_device
    proof = dict(schema=1, classification='filesystem-only', pid=os.getpid(),
        uid=os.getuid(), euid=os.geteuid(), report_directory=str(report),
        manifest_sha256=expected, required_inputs=len(rows), checked_inputs=len(rows),
        passed_inputs=sum(r['passed'] for r in rows), inputs=rows,
        report_write_check=writable, no_dev_dri=no_gpu_device,
        worker_side_orca_status_qualified=False, passed=passed)
    output = report / OUTPUT
    with output.open('x') as f:
        json.dump(proof, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps(dict(passed=passed, checked=len(rows), passed_inputs=proof['passed_inputs'],
        proof=str(output), sha256=digest(output)), sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('worker filesystem check failed: ' + repr(error), file=sys.stderr)
        sys.exit(2)
