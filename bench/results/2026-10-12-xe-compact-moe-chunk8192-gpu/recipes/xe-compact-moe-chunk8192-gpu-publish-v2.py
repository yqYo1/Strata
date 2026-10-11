"""Main: preserve a report byte-exact while correcting its uncommitted Git representation."""
import base64
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
R = W / 'bench/results/2026-10-12-xe-compact-moe-chunk8192-gpu'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('chunk_repr_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(B / 'xe-compact-moe-chunk8192-controller-source-v1'))
import admission as a
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-chunk8192-gpu-publish-root-v2'
    m.require(not out.exists(), 'new correction result')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             owner_sha256=m.sha(O), recipe_sha256=m.sha(__file__), gpu_executed=False,
             model_executed=False, original_results_modified=False, original_reports_modified=False)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands

    def run(label, argv):
        c, so, _ = owner.run(label, argv, ENV, W, wall=60, cpu=120, rss_cap=2 << 30,
                            text_cap=8 << 20, total_cap=32 << 20, file_cap=64 << 20)
        m.require(a.strong_closed_command(c) and c['exit_code'] == 0, 'normal closed ' + label)
        return so.read_text().strip()

    def write(p, value):
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        failed_path = B / 'xe-compact-moe-chunk8192-gpu-publish-root-v1/record.json'
        failed = json.loads(failed_path.read_text())
        m.require(not failed['active'] and not failed['complete'] and failed['passed'] is False and
                  failed['error'] == "RuntimeError('normal closed staged-check')" and
                  all(a.strong_closed_command(c) for c in failed['commands']), 'unchanged closed original failure')
        prefix = str(R.relative_to(W)) + '/'
        status = run('status', ['/usr/bin/git', 'status', '--porcelain'])
        m.require(status and all(x.startswith('A  ' + prefix) for x in status.splitlines()), 'only own staged report additions')
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()), 'global identity')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) == 'git@github.com:yqYo1/Strata.git', 'own fork')
        m.require(json.loads((failed_path.parent / 'gitleaks-plaintext.json').read_text()) == [] and
                  json.loads((failed_path.parent / 'gitleaks-archive.json').read_text()) == [], 'actual original plaintext/archive scans passed')
        name = 'research-xe-gdn-bounded-row-scratch-liveness-capacity-seam-round155-v1.txt'
        original = B / 'research-20261009' / name
        direct = R / 'research' / name
        digest = 'e468a9b734b4ea674819f6d5b4686bfc26a9be462d201c3d67dac084a42c908f'
        m.require(m.sha(original) == m.sha(direct) == digest, 'exact original report unchanged')
        raw = direct.read_bytes()
        archive = direct.with_name(name + '.lossless.json')
        m.require(not archive.exists(), 'new lossless representation')
        write(archive, dict(encoding='gzip+base64', source=str(original), decoded_bytes=len(raw), decoded_sha256=digest,
              data_base64=base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii')))
        m.require(gzip.decompress(base64.b64decode(json.loads(archive.read_text())['data_base64'], validate=True)) == raw,
                  'byte-exact decoded original')
        reg_path = R / 'report-registry-v142.json'
        reg = json.loads(reg_path.read_text())
        reg_before = m.sha(reg_path)
        entries = [x for x in reg['new_reports'] if x['path'] == str(original)]
        m.require(len(entries) == 1 and entries[0]['sha256'] == digest and
                  entries[0]['archive'] == str(direct.relative_to(W)) and reg['research_completed'] == 600,
                  'original uncommitted registry entry')
        entries[0].update(archive=str(archive.relative_to(W)), encoding='gzip+base64',
                          decoded_bytes=len(raw), archive_sha256=m.sha(archive))
        write(reg_path, reg)
        receipt_target = R / 'receipts/original-gpu-publication-failed-v1.json'
        receipt_target.write_bytes(failed_path.read_bytes())
        m.require(m.sha(receipt_target) == m.sha(failed_path), 'unchanged failure archive')
        evidence = failed_path.parent / 'commands/staged-check.stdout'
        (R / 'receipts/original-staged-check.stdout').write_bytes(evidence.read_bytes())
        write(R / 'report-representation-correction.json', dict(original_failure_path=str(failed_path),
            original_failure_sha256=m.sha(failed_path), original_passed=False, original_result_modified=False,
            original_report_path=str(original), original_report_sha256=digest, original_report_modified=False,
            prior_uncommitted_registry_sha256=reg_before, current_registry_sha256=m.sha(reg_path),
            removed_uncommitted_representation=dict(path=str(direct), bytes=len(raw), sha256=digest,
                retained_replacement=str(archive), replacement_sha256=m.sha(archive),
                reason='Original report ends with an extra blank line rejected by git diff --check. Preserve original bytes in verified lossless archive rather than altering the report or relaxing the check.'),
            decoded_bytes_verified=True, substantive_measurement_results_modified=False))
        direct.unlink()
        m.require(m.sha(original) == digest, 'private original still byte-exact')
        (R / 'recipes' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        scan = out / 'gitleaks-archive.json'
        scanner = str(S.parent / 'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
        run('secret-scan-archive', [scanner, 'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'full corrected archive scan')
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W))])
        run('staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): record compact8192 and64K GPU parity'])
        r['committed_revision'] = run('head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published boundary')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
