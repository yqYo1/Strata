#!/usr/bin/env python3
"""Archive reviewed experiment source without adopting its product tree."""
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
G = Path('/home/yayoi/ghq/github.com/yqYo1/Strata')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
O = P / 'bench/results/2026-10-14-primary-experiment-retirement'
Z = G / '.worktree/feat-layer-major-bulk-upload-20261013'
EXPECTED = {
 'diag/bounded-decode-route-ledger-20261013': ('97f8b40ca99ec298c24d7ecc7b270e530d43194d', 6),
 'diag/cpu-job-and-raw-bench-20261012': ('d787a19b20c023dbcc2d1d05b404340465f9b895', 0),
 'diag-decode-service-budget-20261014': ('97f8b40ca99ec298c24d7ecc7b270e530d43194d', 32),
 'diag-layer-route-union-census-20261013': ('db088ff24e67e6a4566566dfc4762c63c55bf343', 6),
 'feat/layer-major-phase-full-context-probe-20261013': ('8b48d17c99b70d32781f9f5b056602a598a37b1c', 18),
 'feat/layer-major-phase4k-20261013': ('9c83ea6ba02a562d55cd90ae678aadebc2b88ae2', 0),
 'feat/serve-first-row-adapter-20261013': ('88ebaae7ee98b908226eb880e1ba4f500dfb2c5a', 11),
 'feat/strata-upstream-sync-skill-20261013': ('97f8b40ca99ec298c24d7ecc7b270e530d43194d', 0)}

def git(*args, cwd=G, env=None, data=None):
    p = subprocess.run(['git', '-C', str(cwd), *args], env=env, input=data, text=True, capture_output=True)
    assert p.returncode == 0, (args, p.stdout[-2000:], p.stderr[-2000:])
    return p.stdout

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def write(name, data):
    (O / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def source_tree():
    return sha(git('ls-tree', '-r', 'HEAD', '--', 'CMakeLists.txt', 'cmake', 'include', 'src', 'serve', 'setup.py', cwd=P).encode())

def main():
    O.mkdir(parents=True, exist_ok=True)
    assert not (O / 'preservation.json').exists(), 'Do not rerun archival mutation'
    record = {'adopted_into_primary': False, 'history_only': True, 'complete': False,
              'main_function_reconciliation_complete': False, 'default_promotion_complete': False,
              'upstream_update_checked': False, 'source_archives': [], 'commands': []}
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert git('branch', '--show-current', cwd=P).strip() == 'feature/sycl'
        assert not git('status', '--porcelain', cwd=P).strip()
        assert not git('diff', '--cached', '--name-only').strip()
        identity = subprocess.run(['git', '-C', str(G), 'config', '--local', '--get-regexp', '^user\\.'], capture_output=True)
        assert identity.returncode == 1, 'Use global identity, no local override'
        before_tree = source_tree()
        for block in git('worktree', 'list', '--porcelain').strip().split('\n\n'):
            wt = dict(line.split(' ', 1) for line in block.splitlines() if ' ' in line)
            branch = wt['branch'].removeprefix('refs/heads/')
            if branch not in EXPECTED:
                continue
            revision, count = EXPECTED[branch]
            assert wt['HEAD'] == revision
            root = Path(wt['worktree'])
            status = git('status', '--porcelain', '--untracked-files=all', cwd=root)
            files = [line[3:] for line in status.splitlines()]
            assert len(files) == count, (branch, files)
            assert all(x == 'CMakeLists.txt' or x.startswith(('src/', 'include/', 'tests/', 'tools/', 'docs/')) for x in files)
            assert not git('diff', '--cached', '--name-only', cwd=root).strip()
            evidence = [{'path': x, 'sha256': sha((root / x).read_bytes()), 'bytes': (root / x).stat().st_size} for x in files]
            snapshot = revision
            if files:
                index = B / ('archive-index-' + branch.replace('/', '-') + '.tmp')
                assert not index.exists()
                env = {**os.environ, 'GIT_INDEX_FILE': str(index)}
                try:
                    git('read-tree', revision, cwd=root, env=env)
                    git('add', '--', *files, cwd=root, env=env)
                    tree = git('write-tree', cwd=root, env=env).strip()
                    message = ('chore(sycl): archive unadopted reference proposal ' + branch + '\n\n'
                               'Source preservation only. This Xe-derived proposal is not integrated into feature/sycl.\n'
                               'See primary source audit for qualification limits and Main-owned tests.\n')
                    snapshot = git('commit-tree', tree, '-p', revision, cwd=root, env=env, data=message).strip()
                    assert git('status', '--porcelain', '--untracked-files=all', cwd=root) == status
                    for item in evidence:
                        blob = subprocess.check_output(['git', '-C', str(G), 'show', snapshot + ':' + item['path']])
                        assert sha(blob) == item['sha256']
                finally:
                    index.unlink(missing_ok=True)
            record['source_archives'].append({'branch': branch, 'original_head': revision,
                'archive_commit': snapshot, 'worktree': str(root), 'qualification': 'unadopted reference proposal',
                'files': evidence, 'retirement': 'planned after primary push and restoration checks'})
        assert len(record['source_archives']) == len(EXPECTED)
        record['kept_reference'] = {'branch': 'feat/layer-major-bulk-upload-20261013',
            'head': git('rev-parse', 'HEAD', cwd=Z).strip(),
            'consumer': 'component source donor and existing closed benchmark provenance for primary adaptation',
            'review_point': 'first primary-side lease/diagnostic port and matched comparison boundary',
            'adopted': False, 'new_implementation_children': 'must start from current feature/sycl'}
        record['main_reconciliation_inputs'] = {'feature_before_archive': git('rev-parse', 'HEAD', cwd=P).strip(),
            'main_before_archive': git('rev-parse', 'main').strip(),
            'original_common_ancestor': 'd988955e141190bc079b58ab3d3038da637ad070',
            'warning': 'Main ancestry is preserved, not functionally reconciled. Use these pinned trees and the prior merge preview; an already-up-to-date merge is not functional adoption.'}
        parents = [record['kept_reference']['head'], *[x['archive_commit'] for x in record['source_archives']]]
        parents = list(dict.fromkeys(parents))
        for parent in parents:
            git('cat-file', '-e', parent + '^{commit}')
        args = ['merge', '-s', 'ours', '--no-ff', '--allow-unrelated-histories', '-m',
                'chore(sycl): preserve reference experiments without inference adoption', *parents]
        result = git(*args, cwd=P)
        record['commands'].append({'argv': ['git', *args], 'result': result})
        record['preservation_merge'] = git('rev-parse', 'HEAD', cwd=P).strip()
        assert source_tree() == before_tree
        record['primary_engine_tree_unchanged_sha256'] = before_tree
        for row in record['source_archives']:
            git('merge-base', '--is-ancestor', row['archive_commit'], 'HEAD', cwd=P)
            git('merge-base', '--is-ancestor', row['original_head'], 'HEAD', cwd=P)
        git('merge-base', '--is-ancestor', '8927eb6991f01f8dcca34aa2f0b31fb2549923e9', 'HEAD', cwd=P)
        sample = '1a7b95e336c4dda84e6beef544fe5acd3fabdc92:bench/results/2026-10-13-layer-major-full-phase-lease/README.ja.md'
        restored = subprocess.check_output(['git', '-C', str(P), 'show', sample])
        assert sha(restored) == '73dcd271d248782e15d37de086659863f8b604eef8acdde0f2de7a74fd0f1aef'
        record['historical_report_restoration'] = {'spec': sample, 'bytes': len(restored), 'sha256': sha(restored), 'passed': True}
        record['complete'] = True
        write('preservation.json', record)
        (O / 'preserve_source.py').write_bytes(Path(__file__).read_bytes())
        git('add', '--', str(O.relative_to(P)), cwd=P)
        git('diff', '--cached', '--check', cwd=P)
        git('commit', '-m', 'docs(sycl): record unadopted source preservation and retirement gates', cwd=P)
        git('push', 'origin', 'feature/sycl', cwd=P)
        print(json.dumps({'complete': True, 'archive_count': len(record['source_archives']), 'preservation_merge': record['preservation_merge'], 'primary_source_unchanged': True}))

if __name__ == '__main__':
    main()
