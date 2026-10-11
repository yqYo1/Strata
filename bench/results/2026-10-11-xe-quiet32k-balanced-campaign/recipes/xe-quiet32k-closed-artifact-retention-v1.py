"""Main: retire closed successful logs and reproducible synthetic CPU fixtures."""
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-quiet32k-balanced-campaign'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
with O.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']


def closed_tree(value):
    if isinstance(value, dict):
        if 'direct_child_reaped' in value:
            m.require(m.closed(value) and value.get('session_anchored') is True, 'actual anchored terminal session')
            for expected in value.get('owners', []):
                actual = m.identity(expected['pid'])
                m.require(actual is None or actual['start_ticks'] != expected['start_ticks'], 'owned PID/start identity absent')
        for child in value.values():
            closed_tree(child)
    elif isinstance(value, list):
        for child in value:
            closed_tree(child)


def file_identity(path):
    s = path.lstat()
    if stat.S_ISLNK(s.st_mode):
        target = os.readlink(path)
        return dict(kind='symlink', bytes=s.st_size, allocated_bytes=s.st_blocks*512,
                    sha256=hashlib.sha256(os.fsencode(target)).hexdigest(), link_target=target,
                    hash_scope='link target text; target never followed', device=s.st_dev,
                    inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
    m.require(stat.S_ISREG(s.st_mode), 'only selected regular file or generated link')
    return dict(kind='regular', bytes=s.st_size, allocated_bytes=s.st_blocks*512,
                sha256=m.sha(path), device=s.st_dev, inode=s.st_ino,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-quiet32k-closed-artifact-retention-root-v1'
    m.require(not out.exists(), 'immutable retention root')
    out.mkdir(mode=0o700)
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, reset_executed=False, deletion_executed=False,
             original_results_unchanged=True)
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')
    def save():
        write(out / 'record.json', r)
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv):
        c, stdout, stderr = owner.run(label, argv, ENV, W, wall=60, cpu=120,
                                      text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0,
                  'normally closed ' + label)
        return stdout.read_bytes()
    def commit(label, message):
        run(label+'-diff-check', ['/usr/bin/git', 'diff', '--check'])
        g = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
        scan = out / (label+'-gitleaks-redacted.json')
        run(label+'-secret-scan', [str(g), 'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'zero secret findings')
        run(label+'-stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W))])
        run(label+'-staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run(label+'-commit', ['/usr/bin/git', 'commit', '-m', message])
        head = run(label+'-head', ['/usr/bin/git', 'rev-parse', 'HEAD']).decode().strip()
        run(label+'-push', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run(label+'-status', ['/usr/bin/git', 'status', '--porcelain']).strip(), 'clean own-fork publication')
        return head
    try:
        pub_path = B / 'xe-quiet32k-balanced-results-publish-root-v1/record.json'
        pub = json.loads(pub_path.read_text())
        m.require(pub['passed'] and pub['complete'] and not pub['active']
                  and pub['committed_revision'] == '9da5f683b14a376282975037eda600634500f2e8', 'actual published sample evidence')
        closed_tree(pub)
        m.require(not run('initial-status', ['/usr/bin/git', 'status', '--porcelain']).strip(), 'clean evidence checkout')
        head = run('initial-head', ['/usr/bin/git', 'rev-parse', 'HEAD']).decode().strip()
        m.require(head == pub['committed_revision'], 'fixed committed evidence revision')
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$']).decode()
        m.require(identity and all(line.startswith('file:/home/yayoi/.config/git/config') for line in identity.splitlines()), 'global identity only')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']).decode().strip() == 'git@github.com:yqYo1/Strata.git', 'own fork only')
        retained = []
        for name in ('archive-manifest-v1.json', 'archive-manifest-v2.json'):
            p = R / name
            m.require(run('committed-'+name, ['/usr/bin/git', 'show', head+':'+str(p.relative_to(W))]) == p.read_bytes(), 'committed exact manifest')
            manifest = json.loads(p.read_text())
            for entry in manifest['byte_exact_copies']:
                source, archive = Path(entry['source']), W / entry['archive']
                m.require(m.sha(source) == m.sha(archive) == entry['sha256'], 'unchanged original and archive')
                retained.append(dict(path=str(source), archive=entry['archive'], sha256=entry['sha256']))
            for entry in manifest['lossless_encoded_copies']:
                archive = W / entry['archive']
                raw = base64.b64decode(json.loads(archive.read_text())['data'], validate=True)
                m.require(m.sha(archive) == entry['archive_sha256'] and hashlib.sha256(raw).hexdigest() == entry['decoded_sha256'], 'lossless archived research')
        sources, evidence_by_source = [], {}
        for p in sorted((R / 'samples').glob('sample-*.json')):
            value = json.loads(p.read_text())
            m.require(value['passed'] and value['complete'] and not value['active'], 'closed actual model sample')
            closed_tree(value)
            original = next(Path(x['path']) for x in retained if x['archive'] == str(p.relative_to(W)))
            sources.append((original, value))
            evidence_by_source[str(original)] = str(p.relative_to(W))
        cpu_archive = R / 'receipts/xe-native-dequant-prefill-quiet-timing-controller-cpu-root-v3.json'
        cpu_path = B / 'xe-native-dequant-prefill-quiet-timing-controller-cpu-root-v3/record.json'
        cpu = json.loads(cpu_path.read_text())
        m.require(cpu['passed'] and cpu['complete'] and not cpu['active'] and cpu['gpu_executed'] is False
                  and cpu['synthetic_fixture_receipts_are_model_evidence'] is False
                  and m.sha(cpu_path) == m.sha(cpu_archive), 'actual closed CPU qualification archive')
        closed_tree(cpu)
        sources.append((cpu_path, cpu))
        evidence_by_source[str(cpu_path)] = str(cpu_archive.relative_to(W))
        files = {}
        compact = dict(actual_model_stdout_reconstructed=[], retained_model_stderr=[],
                       cpu_fixture_stdout_equal_to_parsed_record=False,
                       rule='Retain exact individual receipts, protocol, health/error/fault outcomes and small useful startup stderr. Retire cumulative successful journals/API logs and regenerated synthetic inputs.')
        def add(path, receipt, reason, expected_log=None):
            path = Path(path)
            m.require(path.is_absolute() and path.is_relative_to(B) and str(path) not in files, 'unique task-local deletion path')
            for parent in path.parents:
                m.require(not parent.is_symlink(), 'no ancestor symlink for deletion')
            item = file_identity(path)
            if expected_log is not None:
                m.require(item['kind'] == 'regular' and item['bytes'] == expected_log['bytes'] and item['sha256'] == expected_log['sha256'], 'closed log hash/size')
            files[str(path)] = dict(path=str(path), **item, evidence=evidence_by_source[str(receipt)],
                                   original_receipt_path=str(receipt), original_receipt_sha256=m.sha(receipt),
                                   reason=reason, unlink_status='PENDING')
        for receipt, value in sources:
            command_root = receipt.parent / 'commands'
            for command in value['commands']:
                for name, log in command['logs'].items():
                    m.require(Path(name).name == name and Path(name).suffix in ('.stdout', '.stderr'), 'bounded stdout/stderr name')
                    add(command_root / name, receipt, 'Closed successful command log; useful source/configuration/process/protocol/health/fault evidence retained in the exact receipt.', log)
            if receipt != cpu_path:
                stdout = command_root / 'model.stdout'
                m.require(stdout.read_text().splitlines() == value['startup'] + value['request_protocol'], 'entire model stdout reconstructed exactly')
                stderr = command_root / 'model.stderr'
                m.require(stderr.stat().st_size <= 32 << 10, 'bounded useful startup/model stderr')
                compact['actual_model_stdout_reconstructed'].append(dict(receipt=str(receipt), sha256=m.sha(stdout), all_lines_retained=True))
                compact['retained_model_stderr'].append(dict(receipt=str(receipt), path=str(stderr), sha256=m.sha(stderr), text=stderr.read_text()))
        expanded = cpu_path.parent / 'commands/expanded-protocol-fixtures.stdout'
        m.require(json.loads(expanded.read_text()) == cpu['fixtures'], 'full CPU fixture result retained in actual receipt')
        compact['cpu_fixture_stdout_equal_to_parsed_record'] = True
        fixtures = cpu_path.parent / 'fixtures'
        for path in sorted(fixtures.rglob('*')):
            if path.is_symlink():
                add(path, cpu_path, 'Generated symlink-rejection fixture; target text and actual rejection remain recorded. Unlink only the fixture link.')
            elif path.is_file():
                m.require(path.suffix in ('.json', '.stdout', '.stderr'), 'generated fixture types only')
                if path.suffix == '.json':
                    value = json.loads(path.read_text())
                    m.require(value.get('fixture_synthetic') is True, 'synthetic-only receipt, never original model data')
                    reason = 'Generated synthetic admission input; actual CPU case/result and reproduction source are committed. Original real diagnostic/model fixtures remain.'
                else:
                    reason = 'Closed synthetic CPU QUIT-child log; actual owner closure and case results remain in CPU receipt.'
                add(path, cpu_path, reason)
        m.require(0 < len(files) <= 400 and sum(x['bytes'] for x in files.values()) <= 64 << 20, 'finite reviewed deletion budget')
        plan = dict(status='PLANNED', created_utc=m.utc(), evidence_revision=head,
                    publication_receipt_sha256=m.sha(pub_path), count=len(files), files=list(files.values()),
                    deletion_executed=False, original_result_statuses_unchanged=True,
                    kept=retained, total_logical_bytes=sum(x['bytes'] for x in files.values()),
                    total_allocated_bytes=sum(x['allocated_bytes'] for x in files.values()),
                    sources_models_current_required_fixtures_services_and_gpu_untouched=True)
        for name in ('retention-plan-v1.json', 'retention-evidence-v1.json', 'retention-executed-v1.json'):
            m.require(not (R / name).exists(), 'new retention artifact')
        write(R / 'retention-plan-v1.json', plan)
        write(R / 'retention-evidence-v1.json', compact)
        target_recipe = R / 'recipes' / Path(__file__).name
        m.require(not target_recipe.exists(), 'new retention recipe')
        target_recipe.write_bytes(Path(__file__).read_bytes())
        r['plan_commit'] = commit('plan', 'docs(sycl): record closed quiet32k artifact retention plan')
        plan_hash = m.sha(R / 'retention-plan-v1.json')
        m.require(run('verify-committed-plan', ['/usr/bin/git', 'show', r['plan_commit']+':'+str((R / 'retention-plan-v1.json').relative_to(W))]) == (R / 'retention-plan-v1.json').read_bytes(), 'committed exact finite deletion plan')
        m.require(run('verify-committed-extraction', ['/usr/bin/git', 'show', r['plan_commit']+':'+str((R / 'retention-evidence-v1.json').relative_to(W))]) == (R / 'retention-evidence-v1.json').read_bytes(), 'committed useful log extraction')
        for receipt, value in sources:
            closed_tree(value)
        before = os.statvfs(B)
        before_bytes = before.f_bavail * before.f_frsize
        r['files'] = list(files.values())
        save()
        for i, item in enumerate(r['files']):
            path = Path(item['path'])
            need_identity = {k:item[k] for k in file_identity(path)}
            m.require(file_identity(path) == need_identity, 'final exact file/link identity before unlink')
            m.require(m.sha(Path(item['original_receipt_path'])) == item['original_receipt_sha256'], 'original status and evidence unchanged')
            path.unlink()
            item['unlink_status'] = 'REMOVED'
            r['deletion_executed'] = True
            if i % 16 == 0:
                save()
        m.require(all(not os.path.lexists(x['path']) for x in r['files']), 'all selected files/links absent')
        for item in retained:
            m.require(m.sha(Path(item['path'])) == m.sha(W / item['archive']) == item['sha256'], 'all exact retained source/evidence unchanged')
        after = os.statvfs(B)
        execution = dict(status='EXECUTED', passed=True, recipe_sha256=m.sha(__file__), created_utc=m.utc(),
                         original_plan_sha256=plan_hash, plan_commit=r['plan_commit'], evidence_revision=head,
                         count=len(files), files=r['files'], removed_count=len(files),
                         removed_logical_bytes=sum(x['bytes'] for x in r['files']),
                         removed_allocated_bytes=sum(x['allocated_bytes'] for x in r['files']),
                         available_bytes_before=before_bytes, available_bytes_after=after.f_bavail*after.f_frsize,
                         observed_available_delta_bytes=after.f_bavail*after.f_frsize-before_bytes,
                         filesystem_observation_is_not_exclusive_reclamation_proof=True,
                         original_result_statuses_unchanged=True, all_required_retained_evidence_hashes_verified=True,
                         owned_processes_absent=True, serialization_lock=str(B / 'owned-v0141-measurement.lock'),
                         gpu_executed=False, model_executed=False, reset_executed=False)
        evidence = out / 'execution-evidence.json'
        write(evidence, execution)
        (R / 'retention-executed-v1.json').write_bytes(evidence.read_bytes())
        r['execution_evidence_sha256'] = m.sha(evidence)
        r.update({k:execution[k] for k in ('removed_count', 'removed_logical_bytes', 'removed_allocated_bytes', 'observed_available_delta_bytes')})
        readme = R / 'README.md'
        readme.write_text(readme.read_text()+'\nClosed-artifact review: successful cumulative journal/API/protocol logs and\nregenerated CPU admission fixtures were retired after exact evidence was committed.\nThe original receipts/statuses and current real fixtures remain unchanged.\n[The executed deletion manifest](retention-executed-v1.json) records each path,\nhash, retained replacement and reason, with logical/allocation/available-space\nobservations separated. No GPU/service/source/model operation was part of cleanup.\n')
        r['committed_revision'] = commit('executed', 'chore(sycl): retire closed quiet32k logs and synthetic fixtures')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed', 'error', 'removed_count', 'removed_logical_bytes',
                                         'removed_allocated_bytes', 'observed_available_delta_bytes', 'committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
