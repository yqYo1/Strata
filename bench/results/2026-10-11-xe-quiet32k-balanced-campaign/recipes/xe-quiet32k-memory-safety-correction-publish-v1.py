"""Main: correct traffic-versus-capacity interpretation and preserve official safety audit."""
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
with O.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-quiet32k-memory-safety-correction-publish-root-v1'
    m.require(not out.exists(), 'immutable correction publication')
    out.mkdir(mode=0o700)
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), gpu_executed=False, model_executed=False)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2)+'\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    def run(label, argv):
        c, stdout, stderr = owner.run(label, argv, ENV, W, wall=60, cpu=120,
                                      text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0, 'closed '+label)
        return stdout.read_text().strip()
    save()
    try:
        m.require(not run('initial-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean prior boundary')
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()), 'global identity only')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) == 'git@github.com:yqYo1/Strata.git', 'own fork')
        reports = [
            'research-xe-layer-major-width2-residual-budget-reconciliation-round140-v1.txt',
            'research-official-v0142-long-prefill-source-identity-safety-admission-round140-v1.txt']
        reviewed = []
        for name in reports:
            p = B / 'research-20261009' / name
            target = R / 'research' / (name+'.lossless.json')
            m.require(not target.exists(), 'new report archive')
            raw = p.read_bytes()
            value = dict(encoding='base64', source=str(p), decoded_bytes=len(raw), decoded_sha256=m.sha(p), data=base64.b64encode(raw).decode())
            target.write_text(json.dumps(value, indent=2)+'\n')
            m.require(base64.b64decode(value['data'], validate=True) == raw, 'lossless original report')
            reviewed.append(dict(path=str(p), archive=str(target.relative_to(W)), sha256=m.sha(p),
                                 archive_sha256=m.sha(target), archive_encoding='base64', main_full_report_review=True, implemented=False))
        samples = []
        for p in sorted((R / 'samples').glob('sample-*.json')):
            value = json.loads(p.read_text())
            m.require(value['passed'] and value['complete'] and not value['active']
                      and value['initial_memory']['MemTotal'] == 134986948608, 'actual matching memory receipt')
            samples.append(dict(path=str(p.relative_to(W)), sha256=m.sha(p), initial_memory=value['initial_memory']))
        m.require(len(samples) == 8, 'all eight actual host-memory observations')
        correction = dict(main_reviewed=True, original_reports_modified=False, original_measurements_modified=False,
            actual_memory=dict(MemTotal_bytes=134986948608, total_gib=134986948608/(1<<30),
                               representative_initial_MemAvailable_bytes=97499279360, samples=samples,
                               scope='Kernel-reported physical total and pre-run available memory, not allocated phase headroom.'),
            corrections=[
                'Earlier64GB machine assumptions are incorrect for these actual campaign receipts.',
                'R277 width2 residual57.498/114.998/459.998GiB are cumulative D2H+H2D over46 plane transfers, not simultaneous RAM allocations.',
                'One FP32 residual plane of HC4*N2560 is2.49996GiB at65535 batched rows and9.99996GiB at262143; other owners must be added independently.',
                'The recorded expert arena is default owned host memory, not mmap. A true mmap branch has dynamic file-page residency and must not be counted as permanently resident or zero-cost.',
            ],
            decision='Restore width2/layer-major to conditional candidates; capacity interpretation no longer rules it out. Streamed-QSA-KV incompatibility, session/MTP/state parity, actual phase fit and full262144 remain unresolved.',
            full262144_qualified=False, official_pristine_long_prefill_admitted=False,
            official_scope='Tracked original0.1.42 source with mandatory volatile host-USM Stager/Ple polls remains inadmissible on the recorded non-concurrent host-USM capability; settings alone do not repair those protocols.',
            source_only_candidate_worktree='/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011',
            candidate_baseline='0740b43488d812117f021ccbe4bd0e15d223f955',
            candidate_scope='Separate Sol source-only compact scratch implementation in progress; Main owns every build/test/GPU qualification.')
        write_path = R / 'main-memory-traffic-correction-v3.json'
        m.require(not write_path.exists(), 'new Main correction')
        write_path.write_text(json.dumps(correction, indent=2)+'\n')
        previous = R / 'report-registry-v136.json'
        m.require(json.loads(previous.read_text())['research_completed'] == 568, 'previous report count')
        (R / 'report-registry-v137.json').write_text(json.dumps(dict(research_completed=570,
            prior_registry_path=str(previous.relative_to(W)), prior_registry_sha256=m.sha(previous), new_reports=reviewed,
            scope='Two additional read-only reports fully read by Main; separate explicit correction preserves original reports and samples.'), indent=2)+'\n')
        for source, name in [(Path(__file__), Path(__file__).name),
                             (B / 'xe-prefill-row-scratch-worktree-create-v1.py', 'xe-prefill-row-scratch-worktree-create-v1.py')]:
            target = R / 'recipes' / name
            m.require(not target.exists(), 'new recipe archive')
            target.write_bytes(source.read_bytes())
            m.require(m.sha(target) == m.sha(source), 'exact recipe')
        p = B / 'xe-prefill-row-scratch-worktree-create-root-v1/record.json'
        value = json.loads(p.read_text())
        m.require(value['passed'] and value['complete'] and not value['active']
                  and all(m.closed(c) and c.get('session_anchored') is True for c in value['commands']), 'actual source-only worktree preparation')
        (R / 'receipts/xe-prefill-row-scratch-worktree-create-root-v1.json').write_bytes(p.read_bytes())
        readme = R / 'README.md'
        readme.write_text(readme.read_text()+'\nMain corrected the host-memory interpretation in [a separate record](main-memory-traffic-correction-v3.json).\nThe measured kernel MemTotal is125.715GiB, not64GB. R277 width2 values are cumulative\nresidual traffic, not peak RAM: one plane is2.5GiB/64K or10GiB/262K. Layer-major\nremains a conditional candidate; streamed-KV compatibility and phase admission\nstill need implementation and validation. Registry v137 contains570 reviewed reports.\n')
        todo = W / 'bench/results/2026-10-11-upstream-parameter-audit/TODO.md'
        todo.write_text(todo.read_text()+'\n追加の[R140 source安全監査](../2026-10-11-xe-quiet32k-balanced-campaign/research/research-official-v0142-long-prefill-source-identity-safety-admission-round140-v1.txt.lossless.json)では、本家0.1.42のtracked sourceを確認した。現在のHOST_USM/ATOMIC_HOST_USM capabilityでは、長いprefillのstagerに残るvolatileポーリングを設定だけで避けられないため、未修正本家の長入力は未実行。新コンパイラを用意したことはこのプロトコルを修正した証拠にならず、設定探索完了とも扱わない。\n')
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        g = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
        scan = out / 'gitleaks-redacted.json'
        run('secret-scan', [str(g), 'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'zero secret findings')
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W)), str(todo.relative_to(W))])
        run('staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): distinguish residual traffic from live memory'])
        r['committed_revision'] = run('head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published correction')
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
