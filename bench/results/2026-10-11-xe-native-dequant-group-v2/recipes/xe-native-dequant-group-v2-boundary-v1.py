import ast
import fcntl
import importlib.util
import json
import os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
X = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
D = W / 'bench/results/2026-10-11-xe-native-dequant-group-v2'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-native-dequant-group-v2-boundary-root-v1'
    m.require(not out.exists(), 'immutable boundary')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), gpu_executed=False, model_executed=False, deleted=[])

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()

    def run(label, args):
        entry, stdout, stderr = owner.run(label, args, ENV, W, wall=60, cpu=120,
                                          text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(entry) and entry['exit_code'] == 0, 'normal closed ' + label)
        return stdout.read_text().strip()

    copies, plans, receipt_pins = {}, [], {}

    def copy(src, name):
        target = D / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        m.require(m.sha(target) == m.sha(src), 'byte-identical archive')
        copies[str(target.relative_to(W))] = dict(original_path=str(src), bytes=src.stat().st_size, sha256=m.sha(src))
        return target

    try:
        m.require(not run('report-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean report branch')
        m.require(not D.exists(), 'new evidence archive')
        D.mkdir()
        roots = ['xe-native-dequant-group-api-integrate-root-v2',
                 'xe-e32-lints-baseline-compare-root-v13',
                 'xe-native-dequant-group-byte-gpu-root-v4-free-smoke',
                 'xe-native-dequant-group-byte-gpu-root-v4-free-full',
                 'xe-native-dequant-group-component-analysis-root-v2',
                 'xe-native-dequant-group-source-publish-root-v2']
        for mode in ['free', 'contrib-llvm']:
            roots += ['xe-e32-' + mode + '-full-build-root-v18',
                      'xe-e32-' + mode + '-asan-full-build-root-v15',
                      'xe-native-dequant-group-byte-build-root-v3-' + mode,
                      'xe-native-dequant-group-component-build-root-v2-' + mode]
        roots += ['xe-native-dequant-group-component-gpu-root-v2-free-' + phase
                  for phase in ['smoke-host', 'full-host', 'smoke-tags', 'full-tags']]
        for name in roots:
            src = B / name / 'record.json'
            value = json.loads(src.read_text())
            m.require(value['passed'] and value['complete'] and not value['active'] and
                      all(m.closed(c) for c in value['commands']), 'closed qualified receipt ' + name)
            for command in value['commands']:
                for identity in command.get('owners', []):
                    live = m.identity(identity['pid'])
                    m.require(not live or live['start_ticks'] != identity['start_ticks'], 'no active owner')
            receipt_pins[name] = dict(path=str(src), sha256=m.sha(src))
            copy(src, 'receipts/' + name + '/record.json')
            for filename in ['byte-comparison-ledger.txt', 'source.patch']:
                if (src.parent / filename).is_file():
                    copy(src.parent / filename, 'receipts/' + name + '/' + filename)
            for path in (src.parent / 'commands').glob('*'):
                if path.suffix in ['.stdout', '.stderr']:
                    st = path.stat()
                    plans.append(dict(path=str(path), bytes=st.st_size, allocated_bytes=st.st_blocks * 512,
                                      sha256=m.sha(path), retained_receipt='receipts/' + name + '/record.json',
                                      reason='Closed successful run: exact commands/environment/source/binary/ownership/status, CPU tests and compile recipes, lint findings, all byte case identities and individual timing samples retained. Repeated API/copy/drain/journal output has no unresolved complete-history consumer.'))
            object_file = src.parent / 'probe.o'
            if object_file.is_file():
                st = object_file.stat()
                plans.append(dict(path=str(object_file), bytes=st.st_size, allocated_bytes=st.st_blocks * 512,
                                  sha256=m.sha(object_file), retained_receipt='receipts/' + name + '/record.json',
                                  reason='Standalone intermediate object: final executable, compiler/link recipe and source/archive identities retained. No independent object consumer.'))
        analysis = json.loads((B / 'xe-native-dequant-group-component-analysis-root-v2/record.json').read_text())
        for file in ['xe-native-dequant-group-api-main-review-v2.json',
                     'xe-native-dequant-group-api-applied-main-review-v2.json',
                     'xe-native-dequant-group-v2-byte-main-review-v1.json',
                     'xe-native-dequant-group-v2-component-main-review-v1.json',
                     'xe-native-dequant-group-v2-mode-identity-v1.json']:
            copy(B / file, 'reviews/' + file)
        for file in ['xe-native-dequant-group-component-analyse-v2.py',
                     'xe-native-dequant-group-source-publish-v2.py', Path(__file__).name,
                     'xe-native-dequant-group-api-integrate-v2.py']:
            ast.parse((B / file).read_text())
            copy(B / file, 'recipes/' + file)
        for directory in ['xe-native-dequant-group-v2-byte-controller-source-v1',
                          'xe-native-dequant-group-v2-component-controller-source-v1']:
            for path in sorted((B / directory).iterdir()):
                if path.is_file():
                    if path.suffix == '.py':
                        ast.parse(path.read_text())
                    copy(path, 'controller-source/' + directory + '/' + path.name)
        for directory in ['xe-e32-full-build-v18', 'xe-e32-host-asan-build-v15',
                          'xe-e32-lints-baseline-compare-v13']:
            path = B / directory / 'run_root.py'
            ast.parse(path.read_text())
            copy(path, 'recipes/' + directory + '/run_root.py')
        copy(B / 'xe-native-dequant-group-api-source-v2/candidate.patch', 'source/candidate.patch')
        copy(B / 'xe-native-dequant-group-v2-environment-v1.json', 'environment.json')
        previous_path = W / 'bench/results/2026-10-11-xe-native-dequant-group/report-registry-v128.json'
        previous = json.loads(previous_path.read_text())
        m.require(previous['research_completed'] == 506, 'previous fully reviewed report count')
        reports = []
        for suffix, review in [('byte', 'xe-native-dequant-group-v2-byte-main-review-v1.json'),
                              ('component', 'xe-native-dequant-group-v2-component-main-review-v1.json')]:
            path = B / 'research-20261009' / ('implementation-xe-native-dequant-group-v2-' + suffix + '-controllers-v1.txt')
            qualified = json.loads((B / review).read_text())
            m.require(qualified['reviewer'] == 'Main' and qualified['all_manifest_dependencies_verified'] and
                      qualified['syntax_ast_passed'] and m.sha(path) == qualified['report_pin']['sha256'],
                      'complete unchanged Main report review')
            m.require(all(m.sha(Path(p)) == h for p, h in qualified['source_artifact_pins'].items()),
                      'unchanged reviewed controller artifacts')
            dst = copy(path, path.name)
            reports.append(dict(path=str(path), archive=str(dst.relative_to(W)), sha256=m.sha(path),
                                bytes=path.stat().st_size, model='gpt-6.1-sol', read_only=False,
                                qualification='Source-only delivery fully reviewed by Main; tests/builds/GPU controlled serially by Main. No research-model substitution.',
                                main_review=str((D / 'reviews' / review).relative_to(W))))
        registry = dict(registry_version=129, research_completed=508, created_utc=m.utc(),
                        registry_format='incremental; follow previous_committed_registry',
                        previous_committed_registry=dict(path=str(previous_path.relative_to(W)), sha256=m.sha(previous_path), reports=506),
                        new_reports=reports, reports=reports,
                        dependency='Requested gpt-6-luna research quota until Oct14 2026 12:33PM; no replacement research model. These concrete implementation reports are not research.',
                        live_agent_snapshot=dict(implementation_agents='Both new source-only controller agents returned; complete Main review and fresh serialized validation recorded.', research_agents='No running research agent under selected-model quota'),
                        next_research_scopes=['Official independent prefill/decode parameter selection after safe qualification', 'Measured whole-model phase attribution and hardware ceilings; avoid repeated past proposals'])
        (D / 'report-registry-v129.json').write_text(json.dumps(registry, indent=2) + '\n')
        m.require((B / 'xe-native-dequant-group-v2-results-ja.md').is_file(), 'Main final result explanation')
        copy(B / 'xe-native-dequant-group-v2-results-ja.md', 'RESULTS_JA.md')
        (D / 'archive-manifest.json').write_text(json.dumps(dict(byte_identical_copies=copies,
                                                               source_revision=json.loads((B / 'xe-native-dequant-group-source-publish-root-v2/record.json').read_text())['committed_revision'],
                                                               individual_samples_preserved=True, original_statuses_preserved=True,
                                                               model_executed=False, production_caller_adopted=False), indent=2) + '\n')
        m.require(sum(p.stat().st_size for p in D.rglob('*') if p.is_file()) < 16 << 20,
                  'compact full-case identity archive16MiB')
        run('archive-secret-check', [str(X / '.lint/bin/gitleaks'), 'detect', '--no-git', '--redact', '--no-banner', '--exit-code', '1', '--source', str(D)])
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        run('stage-evidence', ['/usr/bin/git', 'add', '--', str(D.relative_to(W))])
        run('commit-evidence', ['/usr/bin/git', 'commit', '-m', 'docs(bench): qualify grouped dequant constant pointer selection'])
        r['evidence_revision'] = run('evidence-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        retention = D / 'retention-v1.json'
        consumers = dict(standalone_binaries='Current APIv2 Main byte/caller admission and component follow-up use; keep mode-specific executables after deleting intermediate objects, review at caller adoption/rejection.',
                         optimized_ir='Existing named free-v1/free-v2 LLVM IR consumer remains for final lowering question; existing5MiB cap, review at next implementation decision. No duplicate IR copy added.')
        retention.write_text(json.dumps(dict(retained_commit=r['evidence_revision'], deletions=plans,
                                             logical_bytes=sum(x['bytes'] for x in plans), allocated_file_bytes=sum(x['allocated_bytes'] for x in plans),
                                             available_space_not_measured=True, current_consumers=consumers,
                                             original_statuses_unchanged=True), indent=2) + '\n')
        run('retention-secret-check', [str(X / '.lint/bin/gitleaks'), 'detect', '--no-git', '--redact', '--no-banner', '--exit-code', '1', '--source', str(retention)])
        for item in plans:
            path = Path(item['path'])
            m.require(m.sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], 'exact retired path')
            path.unlink()
            r['deleted'].append(item['path'])
        m.require(all(m.sha(Path(v['path'])) == v['sha256'] for v in receipt_pins.values()), 'original statuses unchanged')
        run('stage-retention', ['/usr/bin/git', 'add', '--', str(retention.relative_to(W))])
        run('commit-retention', ['/usr/bin/git', 'commit', '-m', 'docs(bench): retire closed grouped dequant validation traces'])
        r['committed_revision'] = run('final-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('publish-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean reports')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'evidence_revision', 'committed_revision']}))
    raise SystemExit(0 if r['passed'] else 1)
