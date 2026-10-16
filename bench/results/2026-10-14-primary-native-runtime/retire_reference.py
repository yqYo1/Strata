#!/usr/bin/env python3
"""Retire the closed reference checkout; never follow its toolchain symlink."""
import fcntl
import gzip
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-primary-native-runtime'
Z = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/feat-layer-major-bulk-upload-20261013')
BRANCH = 'feat/layer-major-bulk-upload-20261013'
TIP = '28a7b9ae7ca5eba28eecd65f6453a7479c04fce8'


def git(*argv, cwd=P):
    return subprocess.check_output(['git', *argv], cwd=cwd, text=True).strip()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert git('branch', '--show-current') == 'feature/sycl'
        assert git('rev-parse', BRANCH) == TIP
        assert git('rev-parse', 'HEAD', cwd=Z) == TIP
        assert not git('status', '--porcelain', '--untracked-files=all', cwd=Z)
        subprocess.run(['git', 'merge-base', '--is-ancestor', TIP, 'HEAD'], cwd=P, check=True)
        subprocess.run(['git', 'merge-base', '--is-ancestor', TIP, 'origin/feature/sycl'], cwd=P, check=True)
        remote = git('ls-remote', '--heads', 'origin', BRANCH)
        assert remote.split()[0] == TIP, remote
        # The exact compact comparison and recipes restore from the reachable tip.
        source_tree = git('rev-parse', TIP + '^{tree}')
        proof = git('show', TIP + ':bench/results/2026-10-14-layer-bulk-stock-repeat/README.md')
        assert proof
        target = (Z / '.lint').resolve()
        assert target.is_dir() and (Z / '.lint').is_symlink()
        target_identity = target.stat()
        items = []
        for root, directories, names in os.walk(Z, followlinks=False):
            parent = Path(root)
            for name in list(directories):
                path = parent / name
                if path.is_symlink():
                    directories.remove(name)
                    link = os.readlink(path)
                    items.append(dict(path=str(path), type='symlink', target=link,
                                      sha256=sha(os.fsencode(link)), bytes=0,
                                      replacement=TIP, reason='closed reference; unlink only, target remains'))
            for name in names:
                path = parent / name
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    link = os.readlink(path)
                    items.append(dict(path=str(path), type='symlink', target=link,
                                      sha256=sha(os.fsencode(link)), bytes=0, replacement=TIP,
                                      reason='closed reference; unlink only, target remains'))
                elif stat.S_ISREG(info.st_mode):
                    items.append(dict(path=str(path), type='file', bytes=info.st_size,
                                      sha256=sha(path.read_bytes()), replacement=TIP,
                                      reason='source/report recoverable from reachable commit; closed build reproducible from preserved recipes'))
                else:
                    raise RuntimeError('unexpected file type: ' + str(path))
        manifest = b''.join((json.dumps(item) + '\n').encode() for item in items)
        compressed = gzip.compress(manifest, mtime=0)
        (D / 'reference-deletion-manifest.jsonl.gz').write_bytes(compressed)
        assert gzip.decompress(compressed) == manifest
        record = dict(complete=False, branch=BRANCH, original_tip=TIP, source_tree=source_tree,
                      source_and_reports_restore_command=['git', 'show', TIP + ':<repository-relative-path>'],
                      inference_adopted=False, source_donor_retained_as_commit=True,
                      current_task_uses_reference_checkout=False,
                      owner='Main', decision='No current port or measurement consumes this checkout; use archived source if selected later.',
                      manifest=dict(path='reference-deletion-manifest.jsonl.gz', records=len(items),
                                    sha256=sha(compressed), decoded_sha256=sha(manifest),
                                    logical_bytes=sum(item['bytes'] for item in items)), commands=[])
        def persist():
            (D / 'reference-retirement.json').write_text(json.dumps(record, indent=2) + '\n')
        persist()
        for command in [['git', 'push', 'origin', '--delete', BRANCH],
                        ['git', 'worktree', 'remove', str(Z)],
                        ['git', 'branch', '-d', BRANCH]]:
            entry = subprocess.run(command, cwd=P, capture_output=True, text=True)
            record['commands'].append(dict(argv=command, exit_code=entry.returncode,
                                          stdout=entry.stdout, stderr=entry.stderr))
            persist()
            assert entry.returncode == 0, entry.stderr
        assert not Z.exists()
        assert target.is_dir() and target.stat().st_ino == target_identity.st_ino
        subprocess.run(['git', 'merge-base', '--is-ancestor', TIP, 'HEAD'], cwd=P, check=True)
        assert not git('ls-remote', '--heads', 'origin', BRANCH)
        record.update(complete=True, toolchain_symlink_target_preserved=True,
                      local_branches=git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads'),
                      remote_heads=git('ls-remote', '--heads', 'origin'),
                      worktrees=git('worktree', 'list', '--porcelain'))
        persist()
    print(json.dumps(dict(retired=BRANCH, files=len(items), logical_bytes=record['manifest']['logical_bytes'],
                          only_primary_and_default_remain=True)), flush=True)


if __name__ == '__main__':
    main()
