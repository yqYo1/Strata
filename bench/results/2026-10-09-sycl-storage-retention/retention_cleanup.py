#!/usr/bin/python3
"""One recorded, root-managed cleanup. Never modifies source, models or receipts."""
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

AUDIT = Path(__file__).resolve().parent
STATE = AUDIT.parent
BASE = STATE / 'post-reboot-tuning-20261007'
ARCHIVE = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009/bench/results/2026-10-09-sycl-storage-retention')
PLAN = AUDIT / 'cleanup-plan.json'
if len(sys.argv) > 2:
    PLAN = AUDIT / sys.argv[2]
    assert PLAN.parent == AUDIT

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def identity_absent(r):
    for field in ('inferior', 'debugger'):
        old = r.get(field)
        if not isinstance(old, dict) or not isinstance(old.get('pid'), int):
            continue
        p = Path('/proc') / str(old['pid']) / 'stat'
        if p.exists():
            now = p.read_text().rsplit(')', 1)[1].split()
            if old.get('start_ticks') is None or int(now[19]) == old['start_ticks']:
                return False
    return True

def normal(r):
    return (r.get('active') is False and r.get('completed') is True
            and r.get('healthy') is True and r.get('exit_code') == 0
            and not r.get('exit_signal') and not r.get('new_fault_messages')
            and not r.get('error') and not any(r.get('cleanup', {}).values())
            and identity_absent(r))

def load_record(p):
    if not p.is_file() or p.stat().st_size > 16 * 1024 * 1024:
        return {}
    return json.loads(p.read_text())

def plan():
    d = json.loads((AUDIT / 'inventory.json').read_text())
    subset = load_record(BASE / 'upstream-v0141-tuned-fourfresh-numerical-subset-20261009-v1.json')
    protected = set()
    for req in subset['requests']:
        for key in ('first_head', 'prefill_state'):
            protected.add(req[key]['file'])
    protected.add(str(BASE / 'owned-native-expert-copy-v0141-full256k-diagnostic-r1/control32k.session.bin'))
    protected_tree = BASE / 'owned-upstream-v0141-integrated-full256k-diagnostic-r2'
    actions = []
    retained = []
    for f in d['files']:
        p = Path(f['path'])
        if not p.is_relative_to(BASE) or f['bytes'] < 1024 * 1024 or 'hardlink_duplicate_of' in f:
            continue
        # Only exact engine log names and raw captured tensors/session results.
        is_log = p.name == 'inferior.stderr'
        is_dump = p.parent.parent == BASE and (p.name.endswith(('.state.bin', '.session.bin')) or p.name == 'prefill-state.bin')
        if not (is_log or is_dump):
            continue
        parent = p.parent.parent if is_log else p.parent
        receipt = parent / 'record.json'
        r = load_record(receipt)
        if not r or r.get('active') is not False or not identity_absent(r):
            retained.append({'path': str(p), 'reason': 'No terminal owned-process proof; not automatically eligible'})
            continue
        proof = {'receipt': str(receipt), 'receipt_sha256': sha(receipt)}
        if is_log:
            messages = parent / 'project-messages.txt'
            if normal(r) and r.get('math_gate_passed') is not False and messages.is_file():
                actions.append(dict(f, **proof, action='delete-log', reason='Terminal no-fault engine run; project messages and exact protocol/math/timing receipt retained; verbose API history is redundant', project_messages=str(messages), project_messages_sha256=sha(messages), original_sha256=r.get('engine_log_sha256')))
            else:
                actions.append(dict(f, **proof, action='compress-log', reason='Failure/unknown numerical result: preserve exact API evidence losslessly, remove raw copy only after stream SHA256 and size verification', original_sha256=r.get('engine_log_sha256')))
        elif str(p) in protected or p.is_relative_to(protected_tree):
            retained.append({'path': str(p), 'reason': 'Current concrete32K comparison or physical262144 qualification fixture; still used by test controllers'})
        elif normal(r) and r.get('math_gate_passed') is not False:
            actions.append(dict(f, **proof, action='delete-dump', reason='Completed old/candidate capture; current baseline supersedes it; compact tensor fingerprints/comparisons and measurement receipt retained'))
        else:
            retained.append({'path': str(p), 'reason': 'Failure evidence; preserve raw tensor/session data pending separate causal review'})
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'Only listed closed engine traces and obsolete captured results; user authorized removal of unnecessary measurement files', 'protected_files': sorted(protected), 'protected_tree': str(protected_tree), 'actions': actions, 'retained': retained, 'receipts_changed': False, 'models_source_research_builds_changed': False}
    PLAN.open('x').write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    groups = collections.defaultdict(lambda: [0, 0])
    for a in actions:
        groups[a['action']][0] += 1
        groups[a['action']][1] += a['allocated_bytes']
    print(json.dumps({k: {'files': v[0], 'allocated_GiB': v[1]/2**30} for k, v in groups.items()}, indent=2), flush=True)

def validate(a):
    p = Path(a['path'])
    assert p.is_relative_to(BASE) and not p.is_symlink()
    s = p.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1
    assert (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == (a['device'], a['inode'], a['bytes'], a['mtime_ns'])
    receipt = Path(a['receipt'])
    assert sha(receipt) == a['receipt_sha256']
    assert identity_absent(load_record(receipt))
    return p

def journal(a, outcome):
    with (AUDIT / 'cleanup-journal.jsonl').open('a') as f:
        f.write(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **a, **outcome}, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def run(mode):
    lock = (BASE / 'owned-v0141-measurement.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # The audit is scheduled while root builds/tests/GPU execution are idle.
    active = subprocess.run(['ps', '-C', 'strata', '-C', 'gdb', '-C', 'ninja', '-C', 'icpx', '-o', 'comm='], capture_output=True, text=True)
    assert not active.stdout.strip(), active.stdout
    d = json.loads(PLAN.read_text())
    assert (ARCHIVE / PLAN.name).is_file() and sha(ARCHIVE / PLAN.name) == sha(PLAN)
    already = set()
    if (AUDIT / 'cleanup-journal.jsonl').exists():
        for line in (AUDIT / 'cleanup-journal.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row.get('done'):
                already.add(row['path'])
    groups = {'delete': {'delete-log', 'delete-dump', 'delete-log-reviewed'}, 'compress': {'compress-log'}}
    total = 0
    for a in d['actions']:
        if a['action'] not in groups[mode] or a['path'] in already:
            continue
        p = validate(a)
        if a['action'] in ('delete-log', 'delete-log-reviewed'):
            if a.get('project_messages'):
                assert sha(Path(a['project_messages'])) == a['project_messages_sha256']
            context = AUDIT / 'log-context' / (p.parent.parent.name + '.txt')
            context.parent.mkdir(exist_ok=True)
            if a.get('keep_context', True):
                context_bytes = a.get('context_bytes', 16384)
                with p.open('rb') as f, context.open('xb') as dest:
                    dest.write(('First ' + str(context_bytes) + ' bytes of retired log:\n').encode())
                    dest.write(f.read(context_bytes))
                    dest.write(('\nLast ' + str(context_bytes) + ' bytes of retired log:\n').encode())
                    f.seek(max(context_bytes, a['bytes'] - context_bytes))
                    dest.write(f.read(context_bytes))
            else:
                context = None
            journal(a, {'done': False, 'operation': 'unlink pending', 'context': str(context)})
            validate(a).unlink()
            journal(a, {'done': True, 'operation': 'deleted', 'context': str(context)})
        elif a['action'] == 'delete-dump':
            journal(a, {'done': False, 'operation': 'unlink pending'})
            validate(a).unlink()
            journal(a, {'done': True, 'operation': 'deleted'})
        else:
            packed = p.with_name(p.name + '.zst')
            assert not packed.exists() and not packed.with_suffix('.partial').exists()
            tmp = packed.with_suffix('.partial')
            print('Compressing', p, round(a['bytes']/2**30, 3), 'GiB', flush=True)
            original_hash = hashlib.sha256()
            original_bytes = 0
            with p.open('rb') as source, tmp.open('xb') as dest:
                proc = subprocess.Popen(['/usr/bin/zstd', '-q', '-1', '-T2', '-c'], stdin=subprocess.PIPE, stdout=dest)
                try:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        original_hash.update(chunk)
                        original_bytes += len(chunk)
                        proc.stdin.write(chunk)
                    proc.stdin.close()
                    assert proc.wait() == 0
                    dest.flush()
                    os.fsync(dest.fileno())
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
            verified_hash = hashlib.sha256()
            verified_bytes = 0
            proc = subprocess.Popen(['/usr/bin/zstd', '-q', '-d', '-c', str(tmp)], stdout=subprocess.PIPE)
            try:
                for chunk in iter(lambda: proc.stdout.read(4 * 1024 * 1024), b''):
                    verified_hash.update(chunk)
                    verified_bytes += len(chunk)
                assert proc.wait() == 0
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            assert original_bytes == verified_bytes == a['bytes']
            assert original_hash.digest() == verified_hash.digest()
            if a['original_sha256']:
                assert original_hash.hexdigest() == a['original_sha256']
            validate(a)
            assert tmp.stat().st_size < p.stat().st_size
            os.replace(tmp, packed)
            result = {'operation': 'lossless archive verified; raw unlink', 'archive': str(packed), 'archive_sha256': sha(packed), 'archive_bytes': packed.stat().st_size, 'archive_allocated_bytes': packed.stat().st_blocks*512, 'verified_original_sha256': original_hash.hexdigest(), 'verified_original_bytes': verified_bytes}
            journal(a, dict(result, done=False))
            validate(a).unlink()
            journal(a, dict(result, done=True))
        total += 1
        print('Done', a['action'], p.parent.parent.name if a['action'].endswith('log') else p.parent.name, total, flush=True)
    print('Finished', mode, 'actions', total, flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'plan':
        plan()
    else:
        run(sys.argv[1])
