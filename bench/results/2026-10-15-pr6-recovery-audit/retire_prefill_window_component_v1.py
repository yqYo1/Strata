#!/usr/bin/env python3
"""Close only the failed v1 diagnostic's own named collector object."""
import fcntl
import json
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id


def main():
    os.umask(0o077)
    module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = A/'prefill-window-profile-component-v1.json'
        old = json.loads(original.read_text())
        require(not old['passed'] and not old['complete'] and not old['active'] and
                old['boot_id'] == boot_id() and old['boot_unchanged'] and
                not old['fresh_faults'] and not old['driver_dump_exists'], 'exact failed component')
        model = [c for c in old['commands'] if c['label'] == 'request-profile']
        require(len(model) == 1 and model[0]['error'] == "RuntimeError('native ready protocol')" and
                model[0]['exit_code'] == -15, 'known startup receiver failure')
        require(all(c['direct_child_reaped'] and c['session_empty'] and not c.get('survivors') and
                    not c.get('errors') and c['ownership_status'] == 'observed' for c in old['commands']), 'terminal identities required')
        for command in old['commands']:
            for prior in command['owners']:
                now = module.identity(prior['pid'])
                require(now is None or now['start_ticks'] != prior['start_ticks'], 'old identity still live')
        argv = old['argv']
        session = argv[argv.index('--session')+1]
        require(session.startswith('StrataPrefill') and session.isalnum(), 'owned unique session name')
        shm = Path('/dev/shm', 'uctrl'+session)
        tool = json.loads((B/'main-primary-unitrace-build-root-v1/record.json').read_text())
        exe = Path(tool['build_path'])/'unitrace'
        require(str(exe) == argv[0] and sha(exe) == tool['binaries']['unitrace'], 'collector pin')
        root = B/'main-prefill-window-component-v1-retirement-root-v1'
        receipt = A/'prefill-window-component-v1-retirement.json'
        require(not root.exists() and not receipt.exists(), 'fresh retirement receipt')
        root.mkdir(mode=0o700)
        record = dict(passed=False, complete=False, active=True, GPU_executed=False,
            original_receipt_sha256=sha(original), original_passed=False,
            failure='Receiver expected READY before the normal INFO configuration line; no GEN was sent.',
            source_recipe_sha256=sha(Path(__file__)), commands=[], session=session, old_identity_absence_checked=True)
        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        try:
            if shm.exists():
                before = shm.lstat()
                require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and
                        stat.S_IMODE(before.st_mode) == 0o600 and before.st_size == 1024, 'owned private session object')
                record['session_object'] = dict(path=str(shm), device=before.st_dev, inode=before.st_ino,
                                               bytes=before.st_size, uid=before.st_uid)
                persist()
                env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), LANG='C.UTF-8',
                           LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
                c, out, err = owner.run('stop-owned-session', [str(exe), '--stop', session], env, P,
                    wall=10, cpu=10, rss_cap=1 << 30, text_cap=1 << 20,
                    total_cap=8 << 20, file_cap=1 << 20)
                require(module.closed(c) and c['exit_code'] == 0 and
                        f'[INFO] Session {session} is stopped' in out.read_text()+err.read_text(), 'public stop confirmation')
            require(not shm.exists(), 'session object was not released')
            record.update(passed=True, complete=True, session_released=True)
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with receipt.open('x') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], original_remains_failed=True,
                             session_released=record.get('session_released'))), flush=True)


if __name__ == '__main__':
    main()
