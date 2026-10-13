#!/usr/bin/env python3
"""Retire individually archived, idle experiment worktrees and branches."""
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
G = Path('/home/yayoi/ghq/github.com/yqYo1/Strata')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
O = P / 'bench/results/2026-10-14-primary-experiment-retirement'
Z = G / '.worktree/feat-layer-major-bulk-upload-20261013'

def git(*args, cwd=G):
    p = subprocess.run(['git', '-C', str(cwd), *args], text=True, capture_output=True)
    assert p.returncode == 0, (args, p.stdout[-2000:], p.stderr[-2000:])
    return p.stdout

def digest_file(path):
    if path.is_symlink():
        raw = os.fsencode(os.readlink(path))
        return {'kind': 'symlink', 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'target_not_deleted': True}
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return {'kind': 'file', 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}

def write(name, data):
    (O / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def main():
    saved = json.loads((O / 'preservation.json').read_text())
    assert saved['complete'] and saved['history_only'] and not saved['adopted_into_primary']
    assert not (O / 'retirement.json').exists(), 'Do not rerun destructive retirement'
    result = {'complete': False, 'GPU_build_test_executed': False, 'removed_local': [], 'removed_remote': [],
              'kept': saved['kept_reference'], 'default_promotion_complete': False, 'main_function_reconciliation_complete': False}
    manifest = []
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not git('status', '--porcelain', cwd=P).strip()
        head = git('rev-parse', 'HEAD', cwd=P).strip()
        remote_lines = git('ls-remote', '--heads', 'origin').splitlines()
        remote = {ref.removeprefix('refs/heads/'): sha for sha, ref in (line.split() for line in remote_lines)}
        assert remote['feature/sycl'] == head, 'Primary archives must be remote-durable before deletion'
        roots = [Path(x['worktree']) for x in saved['source_archives']]
        busy = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdecimal() or int(proc.name) == os.getpid():
                continue
            try:
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
                cwd = (proc / 'cwd').resolve()
            except (OSError, RuntimeError):
                continue
            for root in roots:
                if cwd == root or root in cwd.parents or any(str(root).encode() in arg for arg in argv):
                    busy.append({'pid': int(proc.name), 'worktree': str(root)})
        assert not busy, ('Do not delete an active consumer', busy)
        for row in saved['source_archives']:
            branch, root = row['branch'], Path(row['worktree'])
            assert git('rev-parse', branch).strip() == row['original_head']
            git('merge-base', '--is-ancestor', row['archive_commit'], head, cwd=P)
            current = [x[3:] for x in git('status', '--porcelain', '--untracked-files=all', cwd=root).splitlines()]
            assert set(current) == {x['path'] for x in row['files']}
            for item in row['files']:
                assert digest_file(root / item['path'])['sha256'] == item['sha256']
            raw = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--others', '--ignored', '--exclude-standard', '-z'])
            paths = [x.decode() for x in raw.split(b'\0') if x]
            assert all(x.startswith(('build/', '.lint/')) or x in ['.lint', 'LOCAL.md'] for x in paths), (branch, 'unexpected ignored artifact')
            for path in paths:
                manifest.append({'root': str(root), 'path': path, **digest_file(root / path),
                    'reason': 'closed reproducible build/test output or duplicate local host-policy note; relevant proof archived'})
            row['ignored_artifact_count'] = len(paths)
            row['restoration'] = ['git', 'show', row['archive_commit'] + ':<repository-relative-path>']
        z_status = git('status', '--porcelain', '--untracked-files=all', cwd=Z)
        z_paths = [x[3:] for x in z_status.splitlines()]
        assert all(x in ['AGENTS.md', 'docs/ARTIFACT_RETENTION.md'] or x.startswith((
            'skills/strata-upstream-sync/', '.claude/skills/strata-upstream-sync/',
            'bench/results/2026-10-14-primary-branch-workflow/')) for x in z_paths)
        for path in z_paths:
            p = Z / path
            if p.exists():
                manifest.append({'root': str(Z), 'path': path, **digest_file(p),
                    'reason': 'superseded Main-owned draft metadata; corrected actual-primary skill/audit and failed status preserved'})
        payload = ''.join(json.dumps(x, sort_keys=True, ensure_ascii=False) + '\n' for x in manifest).encode()
        compressed = gzip.compress(payload, mtime=0)
        (O / 'artifact-deletion-manifest.jsonl.gz').write_bytes(compressed)
        result['manifest'] = {'path': 'artifact-deletion-manifest.jsonl.gz', 'records': len(manifest),
            'sha256': hashlib.sha256(compressed).hexdigest(), 'decoded_sha256': hashlib.sha256(payload).hexdigest(),
            'logical_bytes': sum(x['bytes'] for x in manifest if x['kind'] == 'file'),
            'note': 'Symlink targets are not deleted; tracked source/report contents restore from archive commits.'}
        # Commit the complete reviewable plan before the first deletion.
        write('retirement-plan.json', {'archives': saved['source_archives'], 'default_promotion': 'pending functional reconciliation',
            'original_main': saved['main_reconciliation_inputs'], 'manifest': result['manifest'],
            'remote_heads_before': remote, 'kept_reference': saved['kept_reference']})
        (O / 'retire_source.py').write_bytes(Path(__file__).read_bytes())
        git('add', '--', str(O.relative_to(P)), cwd=P)
        git('diff', '--cached', '--check', cwd=P)
        git('commit', '-m', 'docs(sycl): preserve experiment retirement plan and artifact hashes', cwd=P)
        git('push', 'origin', 'feature/sycl', cwd=P)
        args = ['push', '--atomic', 'origin']
        remote_rows = [x for x in saved['source_archives'] if x['branch'] in remote]
        for row in remote_rows:
            assert remote[row['branch']] == row['original_head']
            args.append('--force-with-lease=refs/heads/' + row['branch'] + ':' + row['original_head'])
        args += [':refs/heads/' + x['branch'] for x in remote_rows]
        if remote_rows:
            git(*args)
            result['removed_remote'] = [x['branch'] for x in remote_rows]
        for row in saved['source_archives']:
            git('worktree', 'remove', '--force', row['worktree'])
            git('branch', '-D', row['branch'])
            result['removed_local'].append(row['branch'])
            print(json.dumps({'retired': row['branch']}), flush=True)
        git('restore', '--source=HEAD', '--worktree', '--', 'AGENTS.md',
            'skills/strata-upstream-sync/SKILL.md', 'skills/strata-upstream-sync/agents/openai.yaml',
            'skills/strata-upstream-sync/references/evaluation.md', 'skills/strata-upstream-sync/references/integration.md', cwd=Z)
        for path in [Z / '.claude/skills/strata-upstream-sync', Z / 'bench/results/2026-10-14-primary-branch-workflow']:
            assert path.is_dir() and not path.is_symlink()
            shutil.rmtree(path)
        (Z / 'docs/ARTIFACT_RETENTION.md').unlink()
        assert not git('status', '--porcelain', cwd=Z).strip()
        for row in saved['source_archives']:
            git('merge-base', '--is-ancestor', row['archive_commit'], 'HEAD', cwd=P)
            for item in row['files']:
                raw = subprocess.check_output(['git', '-C', str(P), 'show', row['archive_commit'] + ':' + item['path']])
                assert hashlib.sha256(raw).hexdigest() == item['sha256']
        git('fsck', '--connectivity-only', '--no-dangling')
        result.update({'complete': True, 'branch_only_source_restored_after_retirement': True,
            'git_connectivity': 'passed', 'reference_worktree_clean': True,
            'remaining_local': git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads').splitlines(),
            'remaining_remote': git('ls-remote', '--heads', 'origin').splitlines(),
            'primary_engine_modified': False, 'product_function_adoption': False})
        write('retirement.json', result)
        git('add', '--', str(O.relative_to(P)), cwd=P)
        git('diff', '--cached', '--check', cwd=P)
        git('commit', '-m', 'chore(sycl): retire eight archived reference experiment branches', cwd=P)
        git('push', 'origin', 'feature/sycl', cwd=P)
        print(json.dumps({'complete': True, 'local_removed': len(result['removed_local']),
                          'remote_removed': len(result['removed_remote']), 'remaining_branches': len(result['remaining_local'])}), flush=True)

if __name__ == '__main__':
    main()
