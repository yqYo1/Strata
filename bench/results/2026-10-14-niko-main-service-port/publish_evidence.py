#!/usr/bin/env python3
"""Preserve closed Main-owned receipts; no tests, GPU or Git mutations."""
import fcntl
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = P / 'bench/results/2026-10-14-niko-main-service-port'
PRE = '36cc848a0eb68b96a1ed2ec007c5d86cfb3b13a7'
MAIN = 'beddab2fd6b61631d73b39b7484bc9a585ebe1de'
Z = '28a7b9ae7ca5eba28eecd65f6453a7479c04fce8'
ROOTS = [
    *['main-niko-service-port-tests-root-v' + str(i) for i in (1, 2, 3)],
    *['main-niko-primary-host-qualification-root-v' + str(i) for i in (1, 2)],
    *['main-niko-primary-syntax-finish-root-v' + str(i) for i in (1, 2)],
    *['main-niko-native-setup-tests-root-v' + str(i) for i in (1, 2, 3)],
    *['main-niko-setup-isolation-root-v' + str(i) for i in (1, 2)],
]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write(name, data):
    (D / name).write_text(json.dumps(data, indent=2) + '\n')

def git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args])

def preserve(path, target):
    data = path.read_bytes()
    dest = D / 'receipts' / target
    dest.parent.mkdir(parents=True, exist_ok=True)
    packed = gzip.compress(data, mtime=0)
    dest.write_bytes(packed)
    assert gzip.decompress(dest.read_bytes()) == data
    return dict(source_path=str(path), source_sha256=sha(data), source_bytes=len(data),
                saved_path=str(dest.relative_to(D)), saved_sha256=sha(packed), saved_bytes=len(packed))

def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts, outcomes, raw = [], [], []
        for name in ROOTS:
            root = B / name
            record = json.loads((root / 'record.json').read_text())
            assert record['active'] is False, name
            for command in record['commands']:
                assert command['direct_child_reaped'] and command['session_empty'], (name, command['label'])
            receipts.append(preserve(root / 'record.json', name + '/record.json.gz'))
            outcomes.append(dict(root=name, passed=record['passed'], complete=record['complete'],
                                 error=record.get('error'), commands=[dict(label=c['label'], exit_code=c['exit_code'],
                                 errors=c['errors'], cleanup=c['cleanup'], observed=c['observation_complete'],
                                 reaped=c['direct_child_reaped'], session_empty=c['session_empty']) for c in record['commands']]))
            for f in sorted(root.glob('*.json')):
                if f.name != 'record.json':
                    receipts.append(preserve(f, name + '/' + f.name + '.gz'))
            for f in sorted((root / 'owned').glob('*')):
                if not f.is_file():
                    continue
                data = f.read_bytes()
                row = dict(path=str(f), sha256=sha(data), bytes=len(data), job=name)
                # Preserve short error/progress/compile windows. Repetitive setup
                # output is represented by all cases plus the isolation proof.
                if data and len(data) <= 16384:
                    row['evidence'] = preserve(f, name + '/' + f.name + '.gz')['saved_path']
                elif data:
                    row['evidence'] = 'individual-result receipts; repeated full successful/test output retired'
                else:
                    row['evidence'] = 'empty'
                raw.append(row)

        failed = B / 'main-niko-native-setup-tests-root-v1/owned/setup-unit.stderr'
        text = failed.read_text()
        blocks = re.split(r'^={60,}\n', text, flags=re.M)
        failures = []
        for block in blocks:
            if not block.startswith(('FAIL:', 'ERROR:')):
                continue
            lines = block.splitlines()
            end = next((i + 1 for i, s in enumerate(lines) if s.startswith('AssertionError:')), len(lines))
            failures.append(dict(case=lines[0], relevant_trace='\n'.join(lines[:end])))
        assert len(failures) == 48
        write('initial-setup-failures.json', dict(original_status='failed', original_failure_count=48,
              source_path=str(failed), source_sha256=sha(failed.read_bytes()), cases=failures,
              causes=dict(golden_linux_normalization=46, windows_fixture_executable=1, missing_mcp_dependency=1),
              isolation='main-niko-setup-isolation-root-v2 receipts reproduce first two causes on unchanged main'))
        write('receipts-index.json', receipts)
        write('outcomes.json', outcomes)
        write('closed-log-retirement-plan.json', dict(execution='pending commit, hash verification and serial cleanup', files=raw))

        qualified = json.loads((B / 'main-niko-primary-syntax-finish-root-v2/record.json').read_text())
        for path, digest in qualified['source_pins'].items():
            assert sha((P / path).read_bytes()) == digest, path
        setup = json.loads((B / 'main-niko-native-setup-tests-root-v3/record.json').read_text())
        for path, digest in setup['source_pins'].items():
            assert sha((P / path).read_bytes()) == digest, path
        assert setup['passed'] and qualified['passed']
        sources = sorted(set(git('diff', '--name-only', PRE).decode().splitlines()) |
                         {p for p in git('ls-files', '--others', '--exclude-standard').decode().splitlines()
                          if not p.startswith('bench/results/')})
        write('source-pins.json', dict(pre_revision=PRE, imported_main_revision=MAIN,
              files=[dict(path=f, sha256=sha((P / f).read_bytes())) for f in sources]))

        dirs = ['src/kernels', 'src/prefill', 'src/core', 'src/spec',
                'include/strata/kernels', 'include/strata/prefill', 'include/strata/core', 'include/strata/spec']
        paths = git('ls-tree', '-r', '--name-only', PRE, '--', *dirs).decode().splitlines()
        math = []
        for path in paths:
            before = git('show', PRE + ':' + path)
            after = (P / path).read_bytes()
            math.append(dict(path=path, original_sha256=sha(before), current_sha256=sha(after), unchanged=before == after))
        changed = [r['path'] for r in math if not r['unchanged']]
        assert changed == ['include/strata/core/expert_source.hpp'], changed
        write('inference-preservation.json', dict(compared_revision=PRE, files=math,
              changed_source=changed, changed_reason='reads() adds override only; function body and dispatch unchanged',
              separate_changes=['SYCL checked teardown', 'generator tail-role checkpoint selection'],
              primary_inference_performance_measured=False, Xe_engine_adopted=False))
        write('branch-state.json', dict(primary=dict(branch='feature/sycl', pre_commit=PRE),
              fork_default=dict(branch='main', revision=MAIN, promotion='pending native engine functional reconciliation'),
              reference=dict(branch='feat/layer-major-bulk-upload-20261013', revision=Z,
              role='optional Xe-derived donor and closed comparison provenance; not primary',
              consumer='first primary lease/diagnostic port and matched comparison',
              review_point='after that port/comparison or candidate rejection'),
              worktrees=git('worktree', 'list', '--porcelain').decode(),
              local_branches=git('branch', '--format=%(refname:short) %(objectname)').decode(),
              no_branch_created=True, upstream_remote_checked=False))
        rows = []
        for f in sorted((D / 'research').iterdir()):
            rows.append(dict(path=str(f.relative_to(P)), sha256=sha(f.read_bytes()),
                             role='implementation proposal, Main reviewed/tested separately' if f.suffix == '.txt'
                             else 'read-only Luna research; Main adoption decisions separate'))
        write('research-index.json', dict(reports=rows,
              previous_reports=['bench/results/2026-10-14-primary-strata-source-audit/research-index.json',
                                str(B / 'research-20261009')],
              next_round='share this index and previous paths; scopes separated; no interim agent intervention'))
        print(json.dumps(dict(closed_jobs=len(ROOTS), preserved_receipts=len(receipts),
              initial_failures_preserved=len(failures), inference_files=len(math), changed_source=changed)), flush=True)

if __name__ == '__main__':
    main()
