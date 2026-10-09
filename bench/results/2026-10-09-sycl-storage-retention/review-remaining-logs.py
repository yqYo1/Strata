#!/usr/bin/env python3
"""One-off, root-managed follow-up to the authorized 2026-10-09 cleanup.

plan records and verifies replacement evidence; apply unlinks only pinned files.
No original receipt, source, binary, fixture, service or GPU state is changed.
"""
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

STATE = Path('/home/yayoi/.local/state/strata-sycl')
BASE = STATE / 'post-reboot-tuning-20261007'
OLD = STATE / 'storage-audit-20261009-v1'
OUT = OLD / 'remaining-log-review'
ARCHIVE = Path(__file__).resolve().parent / 'remaining-log-review'
RELEVANT = re.compile(r'\bxe\b|\bdrm\b|\bi915\b|gpu|05:00|fault|reset|wedg|hang|timestamp stuck|guc|device lost|memory.cat', re.I)
SYNC = re.compile(rb'^strata prefill sync: mark (\d+) phase (.*?) done \(waited ([\d.]+) ms\)')
ISSUE = re.compile(rb'warning|\[warn\]|\[error\]|ERROR \(|UR_RESULT_ERROR|exception|terminate|cannot|failed|failure|ERR ', re.I)
SAVED_BY_DIGEST = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(relative, data):
    raw = data if isinstance(data, bytes) else (json.dumps(data, indent=2) + '\n').encode()
    assert len(raw) <= 256 * 1024, (relative, len(raw))
    digest = hashlib.sha256(raw).hexdigest()
    if digest in SAVED_BY_DIGEST:
        return SAVED_BY_DIGEST[digest]
    for root in (OUT, ARCHIVE):
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.open('xb').write(raw)
    saved = {'path': str(OUT / relative), 'archive': str(ARCHIVE / relative),
             'sha256': digest, 'bytes': len(raw)}
    SAVED_BY_DIGEST[digest] = saved
    return saved


def absent(identity):
    if not isinstance(identity, dict) or not isinstance(identity.get('pid'), int):
        return True
    p = Path('/proc') / str(identity['pid']) / 'stat'
    if not p.exists():
        return True
    current = int(p.read_text().rsplit(')', 1)[1].split()[19])
    return identity.get('start_ticks') is not None and current != identity['start_ticks']


def closed(r):
    assert r.get('active') is False
    assert absent(r.get('inferior')) and absent(r.get('debugger'))
    job = r.get('job') or {}
    assert not job.get('active') and not job.get('survivors')
    assert absent(job.get('root'))
    for identity in job.get('owned', []):
        assert absent(identity)
    cleanup = r.get('cleanup') or {}
    assert not cleanup.get('inferior_survived') and not cleanup.get('gdb_survived')


def normal(r):
    return (r.get('active') is False and r.get('completed') is True
            and r.get('healthy') is True and r.get('exit_code') == 0
            and not r.get('exit_signal') and not r.get('error')
            and not r.get('new_fault_messages')
            and not any((r.get('cleanup') or {}).values())
            and r.get('math_gate_passed') is not False)


def pin(path, reason, replacements, receipt=None):
    assert path.is_relative_to(STATE) and not path.is_symlink()
    s = path.stat()
    assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1
    a = {'path': str(path), 'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512,
         'device': s.st_dev, 'inode': s.st_ino, 'mtime_ns': s.st_mtime_ns,
         'sha256': sha(path), 'reason': reason, 'replacements': replacements}
    if receipt:
        closed(json.loads(receipt.read_text()))
        a.update(receipt=str(receipt), receipt_sha256=sha(receipt))
    return a


def summarize_text(path, key, bounded_window=False):
    selected = []; phases = {}; count = 0
    with path.open('rb') as f:
        for line in f:
            count += 1
            m = SYNC.match(line)
            if m:
                name = m[2].decode(errors='replace'); value = float(m[3])
                a = phases.setdefault(name, {'count': 0, 'wait_ms_sum': 0., 'wait_ms_min': value, 'wait_ms_max': value})
                a['count'] += 1; a['wait_ms_sum'] += value
                a['wait_ms_min'] = min(a['wait_ms_min'], value)
                a['wait_ms_max'] = max(a['wait_ms_max'], value)
            elif line.startswith(b'strata ') or ISSUE.search(line):
                selected.append(line)
    messages = b''.join(selected)
    assert len(messages) <= 128 * 1024, (key, len(messages))
    replacements = []
    if phases:
        replacements.append(save(key + '.json', {'original_path': str(path), 'original_sha256': sha(path),
            'original_bytes': path.stat().st_size, 'lines': count,
            'sync_phases': phases, 'selected_lines': len(selected),
            'timing_eligible': False, 'note': 'Logged synchronization counts/waits only; not a clean speed measurement or a new pass.'}))
    if messages:
        replacements.append(save(key + '.messages.txt', messages))
    if bounded_window:
        with path.open('rb') as f:
            start = f.read(64 * 1024); f.seek(max(0, path.stat().st_size - 64 * 1024)); end = f.read()
        replacements.append(save(key + '.window.txt', b'First 64 KiB:\n' + start + b'\nLast 64 KiB:\n' + end))
    return replacements


def plan():
    assert not (OUT / 'plan.json').exists()
    OUT.mkdir(parents=True, exist_ok=True); ARCHIVE.mkdir(parents=True, exist_ok=True)
    # Preserve exact original records without creating another complete record backup.
    records = {str(p): sha(p) for p in STATE.rglob('record.json') if '.git' not in p.parts}
    actions = []
    owned = []
    for p in sorted(BASE.glob('owned-*/record.json')):
        r = json.loads(p.read_text())
        if normal(r):
            closed(r); owned.append((p, r))
    probes = [p.parent / 'probes/health.stderr' for p, r in owned if (p.parent / 'probes/health.stderr').is_file()]
    representative = max(probes, key=lambda p: p.stat().st_mtime_ns)
    for receipt, r in owned:
        d = receipt.parent
        p = d / 'probes/kernel-gap.stdout'
        if p.is_file():
            entries = []; count = 0
            for line in p.read_text().splitlines():
                if not line.strip(): continue
                row = json.loads(line); count += 1
                if RELEVANT.search(str(row.get('MESSAGE', ''))):
                    entries.append(row)
            replacement = save(d.name + '/kernel-gap.json', {
                'original_path': str(p), 'original_sha256': sha(p), 'original_bytes': p.stat().st_size,
                'receipt_sha256': sha(receipt), 'boot_id': r.get('boot_id'),
                'capture_steps': [s for s in r.get('steps', []) if s.get('label') == 'kernel-gap'],
                'filter': RELEVANT.pattern, 'line_count': count, 'matched_count': len(entries),
                'entries': entries, 'original_new_fault_messages': r.get('new_fault_messages'),
                'note': 'Deterministic relevant-entry extraction; original result is unchanged.'})
            actions.append(pin(p, 'Repeated cumulative kernel window; relevant entries/counts/cursors retained', [replacement], receipt))
        p = d / 'probes/health.stderr'
        if p.is_file() and p != representative:
            stdout = p.with_name('health.stdout'); environment = p.with_name('health-environment.json')
            assert 'PASS 0000:05:00.0: 3 rounds, 16384 exact words each' in stdout.read_text()
            assert environment.is_file()
            replacement = summarize_text(p, d.name + '/health')
            for q in (stdout, environment):
                replacement.append({'path': str(q), 'sha256': sha(q), 'bytes': q.stat().st_size})
            actions.append(pin(p, 'Healthy repeated probe; exact PASS/environment and fault/exit receipt retained; one full representative remains', replacement, receipt))
        for relative in ('debugger/gdb-mi.stdout', 'debugger/engine-and-gdb.stderr'):
            p = d / relative
            if p.is_file():
                replacement = summarize_text(p, d.name + '/' + p.name)
                actions.append(pin(p, 'Normally exited debugger; exact process/protocol/math result and snapshots retained; repeated MI acknowledgements/startup chatter are unnecessary', replacement, receipt))
    names = ['expert-phase-baseline-repeat-sync', 'unitrace-model-2048-suffix-profile',
             'unitrace-model-2048-suffix-clean', 'dequant-refresh-2048-candidate',
             'dequant-refresh-2048-control', 'unitrace-model-2048-clean',
             'unitrace-model-short-profile', 'unitrace-model-short-clean']
    for name in names:
        d = BASE / name; receipt = d / 'record.json'; closed(json.loads(receipt.read_text()))
        p = d / 'collect/stderr'
        replacements = summarize_text(p, name + '/stderr')
        actions.append(pin(p, 'Closed old diagnostic; project/validation messages and full sync-phase aggregates retained; repeated per-step lines discarded; original pass/failure unchanged', replacements, receipt))
    d = BASE / 'full-context-entry-capacity'; receipt = d / 'record.json'
    r = json.loads(receipt.read_text()); closed(r)
    assert r['completed'] is False and r['healthy'] is False and not r['new_fault_messages']
    assert len(r['snapshots']) == 3 and all(Path(s['path']).is_file() for s in r['snapshots'])
    p = d / 'debugger/engine-and-gdb.stderr'
    actions.append(pin(p, 'Earlier stalled diagnostic: three owned ioctl stacks/counters and bounded API/project window retain stop evidence; no current full-history consumer', summarize_text(p, d.name + '/stderr', True), receipt))
    for p in sorted((OLD / 'log-context').glob('*.txt')):
        original_archive = Path(__file__).resolve().parent / 'failure-contexts' / p.name
        assert sha(p) == sha(original_archive)
        # Old derived head/tail windows, not an immutable original capture.
        with p.open('rb') as f:
            first = f.read(64 * 1024); f.seek(max(0, p.stat().st_size - 64 * 1024)); last = f.read()
        selected = b''.join(line for line in p.read_bytes().splitlines(keepends=True) if ISSUE.search(line))
        selected = selected[-32 * 1024:]
        replacement = save('failure-excerpts/' + p.name,
            ('Derived excerpt of ' + str(p) + '\nOriginal derived-window SHA256: ' + sha(p) + '\nFirst/last 64 KiB; selected issue lines at end.\n').encode()
            + first + b'\nLast 64 KiB:\n' + last + b'\nSelected issue lines (last 32 KiB):\n' + selected)
        actions.append(pin(p, 'Previously retained arbitrary 16-MiB windows contain repeated API polling; smaller error/progress excerpts plus original stacks/records are sufficient', [replacement]))
    # Record three explicit exceptions, not a keep-all-failures rule.
    traces = json.loads((OLD / 'cleanup-result.json').read_text())['kept_full_traces']
    purposes = [
        ('CPU SIGSEGV path through memcpy: compact representative startup/call history alongside exact source/stack; cause review remains open', 'Review after CPU crash localization is resolved or a minimal reproducer supersedes it'),
        ('Representative xe engine-reset allocation/command history for lifetime investigation; error tail alone omits queued work', 'Review after the device-lifetime mechanism is resolved or isolated'),
        ('Current C repeated-prefill VRAM capacity failure: compare allocations/frees across first A and second B; whole history is the current consumer', 'Review immediately after retained-allocation attribution and lease capacity experiment')]
    for t, (question, review) in zip(traces, purposes):
        assert sha(Path(t['path'])) == t['archive_sha256']
        t.update(question=question, owner='root / Strata SYCL investigation',
                 consumer='Source/stack and API allocation-lifetime review of the named experiment',
                 next_review=review, historical_captured_raw_bytes=t['original_bytes'],
                 prospective_default_raw_log_budget=64 * 1024 * 1024)
    retained = save('full-trace-exceptions.json', {
        'traces': traces, 'full_health_representative': {'path': str(representative), 'bytes': representative.stat().st_size, 'sha256': sha(representative)},
        'note': 'These three small archives are current explicit exceptions. They are not a permanent quota or a justification to retain future repeats.'})
    plan = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'actions': actions, 'original_records': records, 'explicit_exceptions': retained,
            'scope': 'Closed redundant logs and oversized derived excerpts only; no binary/tensor/source/model/service/GPU changes'}
    for root in (OUT, ARCHIVE):
        (root / 'plan.json').open('x').write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'actions': len(actions), 'removed_bytes': sum(a['bytes'] for a in actions),
                      'removed_allocated_bytes': sum(a['allocated_bytes'] for a in actions), 'original_records': len(records)}, indent=2))


def validate(a):
    p = Path(a['path']); s = p.lstat()
    assert not p.is_symlink() and stat.S_ISREG(s.st_mode) and s.st_nlink == 1
    assert (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == (a['device'], a['inode'], a['bytes'], a['mtime_ns'])
    assert sha(p) == a['sha256']
    if a.get('receipt'):
        q = Path(a['receipt']); assert sha(q) == a['receipt_sha256']; closed(json.loads(q.read_text()))
    for replacement in a['replacements']:
        assert sha(Path(replacement['path'])) == replacement['sha256']
        if replacement.get('archive'):
            assert sha(Path(replacement['archive'])) == replacement['sha256']
    return p


def apply():
    assert sha(OUT / 'plan.json') == sha(ARCHIVE / 'plan.json')
    repo = Path(__file__).resolve().parents[3]
    tracked = subprocess.check_output(['git', '-C', str(repo), 'show', 'HEAD:' + str((ARCHIVE / 'plan.json').relative_to(repo))])
    assert hashlib.sha256(tracked).hexdigest() == sha(OUT / 'plan.json'), 'Plan must be committed before deletion'
    data = json.loads((OUT / 'plan.json').read_text())
    for a in data['actions']: validate(a)
    for a in data['actions']:
        p = validate(a)
        with (OUT / 'journal.jsonl').open('a') as f:
            f.write(json.dumps({'path': str(p), 'sha256': a['sha256'], 'unlink_pending': True}) + '\n'); f.flush(); os.fsync(f.fileno())
            p.unlink()
            f.write(json.dumps({'path': str(p), 'deleted': True}) + '\n'); f.flush(); os.fsync(f.fileno())
    assert all(sha(Path(p)) == digest for p, digest in data['original_records'].items())
    result = {'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
              'deleted_files': len(data['actions']), 'removed_bytes': sum(a['bytes'] for a in data['actions']),
              'removed_allocated_bytes': sum(a['allocated_bytes'] for a in data['actions']),
              'original_records_byte_identical': len(data['original_records']),
              'new_evidence_bytes': sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
              'no_inference_build_compression_service_or_gpu_actions': True}
    for root in (OUT, ARCHIVE):
        (root / 'result.json').open('x').write(json.dumps(result, indent=2) + '\n')
    (ARCHIVE / 'journal.jsonl').open('xb').write((OUT / 'journal.jsonl').read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    lock = (BASE / 'owned-v0141-measurement.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = subprocess.run(['ps', '-C', 'strata', '-C', 'gdb', '-C', 'ninja', '-C', 'icpx', '-C', 'zstd', '-o', 'comm='], capture_output=True, text=True)
    assert not active.stdout.strip(), active.stdout
    assert len(sys.argv) == 2 and sys.argv[1] in ('plan', 'apply')
    globals()[sys.argv[1]]()
