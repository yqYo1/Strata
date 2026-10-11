import base64, fcntl, hashlib, importlib.util, json, os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-preload-disabled-diagnostic'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner pin')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def closed_tree(value):
    if isinstance(value, dict):
        if 'direct_child_reaped' in value:
            m.require(value['direct_child_reaped'] and value['session_empty'] and not value['survivors'], 'terminal owner session')
            for expected in value.get('owners', []):
                actual = m.identity(expected['pid'])
                m.require(actual is None or actual['start_ticks'] != expected['start_ticks'], 'owned identity absent')
        for child in value.values():
            closed_tree(child)
    elif isinstance(value, list):
        for child in value:
            closed_tree(child)

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-preload-disabled-log-retention-root-v1'
    m.require(not out.exists(), 'immutable retention execution')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, reset_executed=False, original_results_unchanged=True,
             files=[], deletion_executed=False)
    def save():
        write(out / 'record.json', r)
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv):
        c, so, se = owner.run(label, argv, ENV, W, wall=60, cpu=120,
                             text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c['exit_code'] == 0, 'normal closed ' + label)
        return so.read_bytes()
    try:
        publication = B / 'xe-preload-disabled-evidence-publish-root-v2/record.json'
        p = json.loads(publication.read_bytes())
        m.require(p['passed'] and p['complete'] and not p['active'] and
                  p['committed_revision'] == 'f0124f19a5bf4e35c428912952d03773caf09ff8', 'actual published archive')
        closed_tree(p)
        r['publication_receipt_sha256'] = m.sha(publication)
        r['evidence_revision'] = p['committed_revision']
        m.require(not run('initial-status', ['/usr/bin/git', 'status', '--porcelain']).rstrip(b'\n'), 'clean evidence checkout')
        head = run('evidence-head', ['/usr/bin/git', 'rev-parse', 'HEAD']).decode().strip()
        m.require(head == p['committed_revision'], 'fixed evidence commit')
        for name in ['archive-manifest.json', 'retention-v1.json', 'README.md', 'main-paired-comparison.json',
                     'api-log-budget-excerpt.json', 'api-log-budget-expanded-progress.json']:
            data = run('verify-committed-' + name.replace('.', '-'),
                       ['/usr/bin/git', 'show', head + ':' + str((R / name).relative_to(W))])
            m.require(data == (R / name).read_bytes(), 'committed exact evidence')
        manifest = json.loads((R / 'archive-manifest.json').read_bytes())
        m.require(m.sha(R / 'archive-manifest.json') == p['manifest_sha256'], 'published manifest identity')
        for entry in manifest['byte_exact_copies']:
            m.require(m.sha(W / entry['archive']) == entry['sha256'] == m.sha(Path(entry['source'])), 'retained exact evidence')
        for entry in manifest['lossless_encoded_copies']:
            archive = W / entry['archive']
            raw = base64.b64decode(json.loads(archive.read_bytes())['data'], validate=True)
            m.require(m.sha(archive) == entry['archive_sha256'] and hashlib.sha256(raw).hexdigest() == entry['decoded_sha256']
                      and raw == Path(entry['source']).read_bytes(), 'retained lossless evidence')
        # All raw logs in the plan belong to these byte-exact terminal original receipts.
        for q in (R / 'receipts').glob('*.json'):
            data = json.loads(q.read_bytes())
            m.require(not data['active'], 'no active original receipt')
            closed_tree(data)
        plan_path = R / 'retention-v1.json'
        plan = json.loads(plan_path.read_bytes())
        m.require(m.sha(plan_path) == p['retention_sha256'] and plan['status'] == 'PLANNED' and
                  plan['count'] == 402 and not plan['deletion_executed'], 'committed exact unexecuted plan')
        r['retention_plan_sha256'] = m.sha(plan_path)
        r['files'] = [dict(entry, unlink_status='PENDING') for entry in plan['files']]
        m.require(len({v['path'] for v in r['files']}) == 402, 'unique finite paths')
        for entry in r['files']:
            path = Path(entry['path'])
            m.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(B.resolve()) and
                      path.suffix in ['.stdout', '.stderr'], 'raw log scope')
            s = path.stat()
            m.require(s.st_size == entry['bytes'] and s.st_blocks * 512 == entry['allocated_bytes'] and
                      m.sha(path) == entry['sha256'], 'exact raw log identity before deletion')
            evidence = W / entry['evidence']
            m.require(m.sha(evidence) == entry['terminal_receipt_sha256'], 'individual immutable terminal evidence')
        v = os.statvfs(B)
        r['filesystem_before'] = dict(available_bytes=v.f_bavail * v.f_frsize, free_bytes=v.f_bfree * v.f_frsize,
                                     observation='Filesystem snapshot; no exclusive attribution')
        save()
        for i, entry in enumerate(r['files']):
            path = Path(entry['path'])
            m.require(path.stat().st_size == entry['bytes'] and m.sha(path) == entry['sha256'], 'final unlink identity')
            path.unlink()
            entry['unlink_status'] = 'REMOVED'
            r['deletion_executed'] = True
            if i % 16 == 0:
                save()
        m.require(all(not Path(v['path']).exists() for v in r['files']), 'all planned raw logs absent')
        for entry in manifest['byte_exact_copies']:
            m.require(m.sha(W / entry['archive']) == entry['sha256'] == m.sha(Path(entry['source'])), 'original and retained evidence unchanged')
        v = os.statvfs(B)
        r['filesystem_after'] = dict(available_bytes=v.f_bavail * v.f_frsize, free_bytes=v.f_bfree * v.f_frsize,
                                    observation='Filesystem snapshot; no exclusive attribution')
        r['removed_count'] = len(r['files'])
        r['removed_logical_bytes'] = sum(v['bytes'] for v in r['files'])
        r['removed_allocated_bytes'] = sum(v['allocated_bytes'] for v in r['files'])
        r['observed_available_delta_bytes'] = r['filesystem_after']['available_bytes'] - r['filesystem_before']['available_bytes']
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
        executed = dict(status='EXECUTED' if r['passed'] else 'INCOMPLETE', controller_sha256=m.sha(__file__),
                        execution_receipt=str(out / 'record.json'), execution_receipt_sha256=m.sha(out / 'record.json'),
                        original_plan_sha256=r.get('retention_plan_sha256'), evidence_revision=r.get('evidence_revision'),
                        created_utc=m.utc(), original_result_statuses_unchanged=True, files=r['files'],
                        removed_count=r.get('removed_count'), removed_logical_bytes=r.get('removed_logical_bytes'),
                        removed_allocated_bytes=r.get('removed_allocated_bytes'),
                        observed_available_delta_bytes=r.get('observed_available_delta_bytes'),
                        filesystem_observation_is_not_exclusive_reclamation_proof=True, error=r.get('error'))
        write(R / 'retention-executed-v1.json', executed)
    print(json.dumps({k:r.get(k) for k in ['passed','error','removed_count','removed_logical_bytes','removed_allocated_bytes','observed_available_delta_bytes']}))
    raise SystemExit(0 if r['passed'] else 1)
