"""Retire only verified, closed T success captures and empty gprofng data."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A = W / 'bench/results/2026-10-09-sycl-layer-major-lease-capacity'
T = B / 'owned-decode-pool-phase-timing-v0141-code32k-tasks6-diagnostic-r1'
G = B / 'gprofng-software-protocol-smoke-v1'
PLAN = A / 'closed-T-artifact-retirement-plan-v1.json'
RESULT = A / 'closed-T-artifact-retirement-result-v1.json'
T_SHA = 'e770766de708a5ab8842b4aa3ea80a6d1453765059e8b7816e36aed09331148c'
G_SHA = '9acfc1557407e7085c65b6d271e7095a0b9e82d43cf84c4daa54e9087599621e'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def same_process(pid, ticks):
    try:
        return int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]) == ticks
    except (FileNotFoundError, ProcessLookupError):
        return False


def guards():
    assert digest(T / 'record.json') == T_SHA
    assert digest(G / 'record.json') == G_SHA
    t = json.loads((T / 'record.json').read_text())
    g = json.loads((G / 'record.json').read_text())
    assert not t['active'] and t['completed'] and t['healthy'] and t['math_gate_passed']
    assert t['exit_code'] == 0 and not t['new_fault_messages'] and not any(t['cleanup'].values())
    for name in ['inferior', 'debugger']:
        assert not same_process(t[name]['pid'], t[name]['start_ticks'])
    assert not g['active'] and not g['passed'] and not g['cleanup'] and not g['survivors']
    for row in g['owners'].values():
        assert not same_process(row['pid'], row['start_ticks'])
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / 'comm').read_text().strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        assert comm not in ['strata', 'strata-xe-health', 'gdb', 'ninja', 'vtune', 'gprofng', 'gp-collect-app'], (proc.name, comm)
    return t


def verify_state(meta):
    p = Path(meta['file'])
    assert p.stat().st_size == meta['bytes']
    whole = hashlib.sha256()
    assert len(meta['parts']) == 66
    with p.open('rb') as stream:
        for ordinal, part in enumerate(meta['parts']):
            header = stream.read(8)
            assert len(header) == 8 and struct.unpack('<Q', header)[0] == part['bytes']
            assert ordinal == part['index'] and stream.tell() == part['offset']
            whole.update(header)
            section = hashlib.sha256()
            left = part['bytes']
            while left:
                chunk = stream.read(min(left, 1024 * 1024))
                assert chunk
                whole.update(chunk)
                section.update(chunk)
                left -= len(chunk)
            assert section.hexdigest() == part['sha256']
        assert stream.tell() == meta['bytes'] and not stream.read(1)
    return whole.hexdigest()


def file_row(p, reason, expected=None):
    info = p.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    h = digest(p)
    if expected is not None:
        assert h == expected
    return {'path': str(p), 'bytes': info.st_size, 'allocated_bytes': info.st_blocks * 512,
            'device': info.st_dev, 'inode': info.st_ino, 'mtime_ns': info.st_mtime_ns,
            'sha256': h, 'reason': reason}


lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
t = guards()
assert sys.argv[1:] in [['plan'], ['execute']]
if sys.argv[1] == 'plan':
    assert not PLAN.exists() and not RESULT.exists()
    rows = []
    for request in t['requests']:
        assert request['math_gate_passed']
        assert all(request['head_and_live_state_comparison'].values())
        assert all(request['qualified_physical_reference_output_comparison'].values())
        state_meta = request['prefill_state']
        rows.append(file_row(Path(state_meta['file']), 'All 66 payload hashes and live comparisons verified; no current T candidate fixture/RESTORE consumer. Canonical baseline fixtures remain.', verify_state(state_meta)))
        head = request['first_head']
        rows.append(file_row(Path(head['file']), 'Exact canonical head comparison and original head fingerprint retained in immutable receipt.', head['sha256']))
    for name in ['project-messages.txt', 'events.jsonl', 'protocol.stdout.raw', 'input-A-tokens.txt', 'input-B-tokens.txt']:
        rows.append(file_row(T / name, 'Successful progress, phase counters, individual outputs/LP/MTP and fixture identities retained in terminal receipt/controller; no additional causal evidence.'))
    for name in ['map.xml', 'dyntext', 'data.frameinfo', 'log.xml', 'overview']:
        rows.append(file_row(G / 'result.er' / name, 'Collector initialization failed, frame data empty; complete meaningful log text and failure receipt retained, map/resource data has no current attribution consumer.'))
    for name in ['T-live-launch-v1.json', 'report-registry-v18.json', 'report-registry-v19.json']:
        p = A / name
        tracked = subprocess.run(['git', '-C', str(W), 'ls-files', '--error-unmatch', str(p.relative_to(W))], capture_output=True)
        assert tracked.returncode == 1
        rows.append(file_row(p, 'Untracked duplicate progress snapshot superseded by final T receipt and complete research registry v20; originals not touched.'))
    plan = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'script_sha256': digest(Path(__file__)), 'T_original_receipt_sha256': T_SHA,
            'gprofng_original_receipt_sha256': G_SHA, 'all_owned_processes_absent': True,
            'exclusive_lock': str(B / 'owned-v0141-measurement.lock'),
            'retained': [{'path': str(p), 'sha256': digest(p)} for p in [A / 'T-four-fresh32k-phase-counter-record.json', A / 'gprofng-software-smoke-rejected-record.json', A / 'gprofng-software-smoke-rejection.txt', A / 'terminal-evidence-retention-review-v1.md', A / 'report-registry-v20.json']],
            'canonical_baseline_and_C_failure_captures_retained': True,
            'files': rows, 'logical_bytes': sum(x['bytes'] for x in rows),
            'allocated_bytes': sum(x['allocated_bytes'] for x in rows),
            'original_status_changed': False}
    PLAN.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'files': len(rows), 'logical_bytes': plan['logical_bytes'], 'allocated_bytes': plan['allocated_bytes']}))
else:
    assert not RESULT.exists()
    plan = json.loads(PLAN.read_text())
    assert plan['script_sha256'] == digest(Path(__file__))
    for entry in [*plan['retained'], {'path': str(PLAN), 'sha256': digest(PLAN)}]:
        p = Path(entry['path'])
        archived = subprocess.check_output(['git', '-C', str(W), 'show', 'HEAD:' + str(p.relative_to(W))])
        assert hashlib.sha256(archived).hexdigest() == entry['sha256'] == digest(p)
    for row in plan['files']:
        p = Path(row['path'])
        info = p.lstat()
        assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        assert (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size) == (row['device'], row['inode'], row['mtime_ns'], row['bytes'])
        assert digest(p) == row['sha256']
    before = os.statvfs(B)
    for row in plan['files']:
        Path(row['path']).unlink()
    (G / 'result.er').rmdir()
    after = os.statvfs(B)
    guards()
    result = {'completed': True, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'plan_sha256': digest(PLAN), 'script_sha256': digest(Path(__file__)),
              'removed_files': len(plan['files']), 'removed_logical_bytes': plan['logical_bytes'],
              'removed_file_allocation_bytes': plan['allocated_bytes'],
              'filesystem_available_bytes_before': before.f_bavail * before.f_frsize,
              'filesystem_available_bytes_after': after.f_bavail * after.f_frsize,
              'all_planned_paths_absent': all(not Path(x['path']).exists() for x in plan['files']),
              'original_receipts_unchanged': True, 'canonical_baseline_and_C_failure_captures_retained': True,
              'note': 'Available-space change includes concurrent filesystem activity/snapshots and is distinct from removed file allocation.'}
    RESULT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
