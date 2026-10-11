import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-quiet32k-balanced-campaign'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
with O.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-quiet32k-campaign-publish-root-v1'
    m.require(not out.exists() and not R.exists(), 'immutable new publication roots')
    out.mkdir(mode=0o700)
    R.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, cleanup_executed=False)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv, expected=(0,)):
        c, so, se = owner.run(label, argv, ENV, W, wall=60, cpu=120,
                             text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] in expected, 'normal anchored ' + label)
        return so.read_text().rstrip('\n')
    try:
        manifest = dict(byte_exact_copies=[], lossless_encoded_copies=[])
        def copy(source, relative):
            target = R / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            entry = dict(source=str(source), archive=str(target.relative_to(W)), bytes=target.stat().st_size, sha256=m.sha(source))
            m.require(m.sha(target) == entry['sha256'], 'exact new archive copy')
            manifest['byte_exact_copies'].append(entry)
            return entry
        def encode(source, relative):
            data = source.read_bytes()
            target = R / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            value = dict(encoding='base64', source=str(source), decoded_bytes=len(data), decoded_sha256=m.sha(source),
                         data=base64.b64encode(data).decode())
            target.write_text(json.dumps(value, indent=2) + '\n')
            entry = dict(source=str(source), archive=str(target.relative_to(W)), decoded_bytes=len(data),
                         decoded_sha256=m.sha(source), archive_sha256=m.sha(target))
            manifest['lossless_encoded_copies'].append(entry)
            return entry
        source = B / 'xe-native-dequant-prefill-quiet-timing-controller-source-v3'
        for p in sorted(source.glob('*')):
            if p.is_file():
                copy(p, 'controller-v3/' + p.name)
        for name in ['xe-native-dequant-prefill-quiet-timing-controller-main-cpu-v3.py',
                     'xe-native-dequant-prefill-quiet-timing-controller-cpu-fixtures-v3.py',
                     'xe-quiet32k-campaign-main-prepare-v3.py', Path(__file__).name]:
            copy(B / name, 'recipes/' + name)
        for name in ['xe-native-dequant-prefill-quiet-timing-controller-main-review-v3.json',
                     'xe-quiet32k-main-closed-diagnostic-preflight-v3.json',
                     'xe-quiet32k-publication-main-review-v3.json',
                     'xe-quiet32k-diagnostic-pair-main-review-v3.json',
                     'xe-quiet32k-campaign-qualification-v3.json']:
            copy(B / name, 'reviews/' + name)
        for name in ['xe-native-dequant-prefill-quiet-timing-controller-cpu-root-v3',
                     'xe-quiet32k-campaign-main-prepare-root-v3']:
            p = B / name / 'record.json'
            v = json.loads(p.read_text())
            m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) and c.get('session_anchored') is True for c in v['commands']), 'actual closed CPU preparation')
            copy(p, 'receipts/' + name + '.json')
        previous = W / 'bench/results/2026-10-11-xe-preload-disabled-diagnostic/report-registry-v134.json'
        registry = json.loads(previous.read_text())
        m.require(registry['research_completed'] == 553, 'previous registry')
        reports = [
            'research-xe-counterbalanced-v2-order-and-numerical-gates-next-v1.txt',
            'research-xe-counterbalanced-v2-runtime-publication-predecessor-gates-next-v1.txt',
            'research-xe-native-group-final-igc-spill-observability-next-v1.txt',
            'research-xe-prefill-owned-device-byte-formula-current-receipt-next-v1.txt',
            'implementation-xe-quiet32k-controller-cpu-qualification-source-v3.txt',
            'research-xe-prefill-8192-owned-byte-delta-low-precision-candidates-round135-v1.txt',
            'research-xe-prefill-pass-reduction-scheduling-round135-v1.txt']
        new_reports = []
        for name in reports:
            e = encode(B / 'research-20261009' / name, 'research/' + name + '.lossless.json')
            new_reports.append(dict(path=e['source'], archive=e['archive'], sha256=e['decoded_sha256'],
                                    archive_sha256=e['archive_sha256'], archive_encoding='base64',
                                    main_full_report_review=True, implemented=False))
        (R / 'report-registry-v135.json').write_text(json.dumps(dict(research_completed=560,
            prior_registry_path=str(previous.relative_to(W)), prior_registry_sha256=m.sha(previous), new_reports=new_reports,
            scope='Seven new fully read reports. Source-only implementation report remains unchanged; separate actual CPU PASS is in receipts.'), indent=2) + '\n')
        (R / 'archive-manifest-v1.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (R / 'README.md').write_text('''# Quiet 32K balanced campaign

Main reviewed and fixed the pending v2 timing controller, then executed CPU
qualification v3. It passed: 9 positive and 35 negative original protocol/trace
fixtures, 39 additional cases (26 rejected inputs), and 4 malformed CLI cases.
Both OFF-first and ON-first admission, complete prior-pair numerical joins,
exact health vectors, anchored process ownership, core/compiler mapping changes,
symlink outputs, actual CPU maps/build IDs and normal QUIT were checked.
The immutable receipts and reproduction source are kept here.

The resolved launch manifest binds the published 0740b434 source, historical
fa3e20e8 + exact five-line patch build, 38 content hashes, actual free release
binary and closed diagnostic32K OFF/ON receipts. The pending template remains
pending. No new GPU/model timing has been collected at this publication boundary.

The plan is four independent process pairs at 32768 input tokens, chunk6144,
G16, physical context262144, KV INT8/resident32768, cache request128/actual144,
workers5, MTP4/min-p0, FUSED0 and preload0. Order is OFF/ON, ON/OFF, OFF/ON,
ON/OFF. Warning/error logs remain; API/validation/transfer/phase traces are absent.
Mapping hashes occur before GEN and after DONE; cold cache/JIT/temperature
effects are admitted. Each sample needs strict16 T/LP5 parity and fresh health,
kernel, normal close and source/input identity checks. No automatic retry/reset.
Individual prefill and cold-decode values will be evaluated independently.
Full-model ability, RESTORE and full262144 completion remain unqualified.

Research v135 contains 560 completed reports. The source-derived Prefill owned
requests are 2952728320 B at T6144; T8192 adds765.672 MiB. These are requested
extents, not measured physical VRAM. A redundant GDN FP32 output-copy candidate
could save192 MiB at8192 if full call-graph review confirms no consumer. It is
not implemented or tested. The width2 grouped schedule is an existing
conditional candidate selected against the measured full-stream pass budget;
its 256K capacity and state parity remain unqualified.

Main correction: the older byte-formula report's 2.815941 GiB wording confuses
MiB and GiB; 2952728320 B is2815.941 MiB, about2.750 GiB. Original reports remain
lossless. The R working copies of the earlier H2D/workspace reports were updated
after their prior publication; the previous committed bytes/hashes remain the
authoritative historical versions and are not relabelled.

The upstream parameter audit is still open: explicit historical settings prove
that testing was not box defaults only, but do not prove an independent search
for the best official prefill/decode settings. Follow the committed audit TODO.
''')
        for p in R.rglob('*.json'):
            json.loads(p.read_text())
        identity = run('identity-origin', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(all(line.startswith('file:/home/yayoi/.config/git/config') for line in identity.splitlines()), 'global identity only')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) == 'git@github.com:yqYo1/Strata.git', 'own fork remote')
        status = run('status', ['/usr/bin/git', 'status', '--porcelain', '--untracked-files=normal'])
        m.require(status and all(line[3:].startswith(str(R.relative_to(W))) for line in status.splitlines()), 'only campaign archive changes')
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        g = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
        scan = out / 'gitleaks-redacted.json'
        run('secret-scan', [str(g), 'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'zero actual secret findings')
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W))])
        run('staged-diff-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'test(sycl): qualify balanced quiet 32k measurement controller'])
        r['committed_revision'] = run('committed-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published campaign')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed', 'error', 'committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
