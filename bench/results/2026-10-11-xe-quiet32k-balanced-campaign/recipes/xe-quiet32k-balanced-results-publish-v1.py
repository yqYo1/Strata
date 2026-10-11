"""Main: preserve actual individual quiet32K samples and separately reviewed research."""
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
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
REPORTS = [
    'research-xe-moe-gu-hh-reusable-scratch-capacity-round136-v1.txt',
    'research-xe-decode-cache-expansion-post-prefill-owner-contract-round136-v1.txt',
    'research-xe-fallback-xs-reusable-gather-capacity-round137-v1.txt',
    'research-xe-serve-gen-prefill-phase-retirement-cache-contract-round137-v1.txt',
    'research-xe-fallback-one-expert-gemm-descriptor-lifetime-round138-v1.txt',
    'research-xe-serve-prompt-to-decode-prefill-retirement-order-round138-v1.txt',
    'research-xe-fallback-row-capped-group-scratch-schedule-round139-v1.txt',
    'research-xe-long-context-weight-reuse-memory-bounded-schedules-round139-v1.txt',
]

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-quiet32k-balanced-results-publish-root-v1'
    m.require(not out.exists() and R.is_dir(), 'new results publication root')
    out.mkdir(mode=0o700)
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, cleanup_executed=False)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv, expected=(0,)):
        c, stdout, stderr = owner.run(label, argv, ENV, W, wall=60, cpu=120,
                                      text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] in expected,
                  'normal anchored ' + label)
        return stdout.read_text().rstrip('\n')
    try:
        a_path = B / 'xe-quiet32k-balanced-main-analysis-root-v1/record.json'
        analysis = json.loads(a_path.read_text())
        m.require(analysis['passed'] and analysis['complete'] and not analysis['active']
                  and analysis['timing_comparison_qualified'] is True and len(analysis['samples']) == 8,
                  'actual complete Main analysis PASS')
        manifest = dict(byte_exact_copies=[], lossless_encoded_copies=[], original_sample_status_preserved=True)
        def copy(source, relative):
            source = Path(source)
            target = R / relative
            m.require(not target.exists(), 'immutable new archive path ' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            item = dict(source=str(source), archive=str(target.relative_to(W)), bytes=target.stat().st_size,
                        sha256=m.sha(source))
            m.require(m.sha(target) == item['sha256'], 'exact result copy')
            manifest['byte_exact_copies'].append(item)
            return item
        for sample in analysis['samples']:
            original = Path(sample['receipt']['path'])
            m.require(m.sha(original) == sample['receipt']['sha256'], 'original individual sample stable')
            value = json.loads(original.read_text())
            m.require(value['passed'] and value['complete'] and not value['active']
                      and all(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0 for c in value['commands']),
                      'closed actual sample before publication')
            copy(original, f"samples/sample-{sample['sample_index']:03d}-{sample['arm']}.json")
        copy(a_path, 'main-comparison-v1.json')
        copy(B / 'xe-quiet32k-balanced-main-analysis-v1.py', 'recipes/xe-quiet32k-balanced-main-analysis-v1.py')
        copy(Path(__file__), 'recipes/' + Path(__file__).name)
        previous = R / 'report-registry-v135.json'
        m.require(json.loads(previous.read_text())['research_completed'] == 560, 'exact prior report registry')
        reviewed = []
        for name in REPORTS:
            source = B / 'research-20261009' / name
            data = source.read_bytes()
            target = R / ('research/' + name + '.lossless.json')
            m.require(not target.exists(), 'new unmodified research artifact')
            payload = dict(encoding='base64', source=str(source), decoded_bytes=len(data),
                           decoded_sha256=m.sha(source), data=base64.b64encode(data).decode())
            target.write_text(json.dumps(payload, indent=2) + '\n')
            m.require(base64.b64decode(payload['data'], validate=True) == data, 'lossless report')
            item = dict(source=str(source), archive=str(target.relative_to(W)), decoded_bytes=len(data),
                        decoded_sha256=m.sha(source), archive_sha256=m.sha(target))
            manifest['lossless_encoded_copies'].append(item)
            reviewed.append(dict(path=str(source), archive=item['archive'], sha256=item['decoded_sha256'],
                                 archive_sha256=item['archive_sha256'], archive_encoding='base64',
                                 main_full_report_review=True, implemented=False))
        (R / 'report-registry-v136.json').write_text(json.dumps(dict(research_completed=568,
            prior_registry_path=str(previous.relative_to(W)), prior_registry_sha256=m.sha(previous),
            new_reports=reviewed, scope='Eight additional complete read-only reports fully read by Main; proposals remain unimplemented.'), indent=2) + '\n')
        correction = dict(source_revision=analysis['published_source_revision'], original_reports_modified=False,
            corrections=[
                dict(report=REPORTS[1], scope='Non-serve only; stack Prefill around generate.cpp10118 is not the actual --serve persistent sp route. Round137 and138 cover serve explicitly.'),
                dict(report=REPORTS[3], scope='Reinitializing immediately after retirement restores workspace; a decode-memory interval needs retirement before decode and initialization before a later GEN, as clarified in round138.'),
                dict(report=REPORTS[4], scope='rows_gpu is bounded by T*K, not T. Only one expert has cnt[e]<=T. Current full-route Xs/GU/Hh/Dm span T*K rows; a fused multi-expert group does not necessarily fit a T-row scratch. Apply explicit group row-sum admission or process individual experts. Source around prefill.cpp2065/2086/2365 confirms global grouped row bounds.'),
            ],
            capacity=dict(scope='Combined Xs/GU/Hh reusable fallback scratch source request only, no physical VRAM or performance measurement.',
                T8192_old_owned_bytes=3755593472, T8192_new_owned_bytes=2999273728,
                T8192_owned_decrease_bytes=756319744, T6144_old_owned_bytes=2952728320,
                T8192_new_minus_T6144_old_bytes=46545408,
                arithmetic_precision_changed=False, implemented=False))
        (R / 'main-research-corrections-v2.json').write_text(json.dumps(correction, indent=2) + '\n')
        (R / 'archive-manifest-v2.json').write_text(json.dumps(manifest, indent=2) + '\n')
        lines = ['# Quiet 32K balanced campaign', '',
                 'Main completed the planned four independent fresh-process pairs. All eight',
                 'samples passed strict GEN16/token/selected LP5 checks, normal QUIT, stable',
                 'source/runtime/compiler mapping checks and fresh before/after GPU health and',
                 'kernel fault gates. No retry, reset or warmup was used.', '',
                 'The source is 0740b43488d812117f021ccbe4bd0e15d223f955, with the actual free',
                 'release built from fa3e20e8 plus the content-identical published five-line',
                 'optional preload guard. The original historical build identity is preserved.', '',
                 'Each arm uses the same32768-token input, actual chunk6144, G16, context262144,',
                 'KV INT8/resident32768, cache request128/actual144, workers5, MTP4/min-p0,',
                 'FUSED0 and preload0. Only STRATA_PREFILL_NATIVE_DEQUANT_BATCH differs.',
                 'Warning/error logs remain; API/validation/transfer/census/GPU timers are absent.',
                 'Engine DONE phase times are used. Mapping hashes occur before GEN and after',
                 'DONE; their CPU/I/O/cache effects are not calibrated.', '',
                 '| Pair | Order | OFF prefill tok/s | ON prefill tok/s | OFF cold decode tok/s | ON cold decode tok/s |',
                 '| --- | --- | ---: | ---: | ---: | ---: |']
        for index in range(4):
            p = {s['arm']:s for s in analysis['samples'] if s['sample_index'] == index}
            first = p['off']['first_arm'].upper()
            second = 'ON' if first == 'OFF' else 'OFF'
            lines.append(f"| {index+1} | {first} / {second} | {p['off']['prefill_tps']:.3f} | {p['on']['prefill_tps']:.3f} | {p['off']['cold_decode_tps']:.3f} | {p['on']['cold_decode_tps']:.3f} |")
        lines += ['', '| Phase | OFF mean | OFF sample SD | ON mean | ON sample SD |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for label, key in [('Prefill tok/s', 'prefill_tps'), ('Cold decode tok/s', 'cold_decode_tps')]:
            off, on = analysis['summaries']['off'][key], analysis['summaries']['on'][key]
            lines.append(f"| {label} | {off['mean']:.3f} | {off['sample_sd']:.3f} | {on['mean']:.3f} | {on['sample_sd']:.3f} |")
        lines += ['',
            'Keep native dequant batching default OFF. Prefill and decode are evaluated',
            'separately; these four pairs do not establish a robust benefit. This flag does',
            'not change decode kernels. All individual phase times, paired differences,',
            'medians, SD/ranges, exact command/environment/input and process/fault outcomes',
            'are in [main-comparison-v1.json](main-comparison-v1.json) and the byte-exact samples.',
            'The raw receipts retain performance_eligible=false; Main qualifies only this',
            'complete recipe-specific timing comparison in a separate derived record.', '',
            'GEN16 cold decode is a short sample, not sustained warm or restored decode.',
            'One canonical prompt and selected LP5 equality do not qualify overall model',
            'ability, RESTORE or a completed262144-token context. Those remain open.', '',
            'Actual CPU controller qualification v3 passed 9 original positive and35',
            'negative fixtures,39 additional cases (26 rejections), and4 malformed CLI cases.',
            'The immutable launch manifest and the unchanged pending template remain here.', '',
            'Research registry v136 contains568 completed reports. Original reports are',
            'lossless; [main-research-corrections-v2.json](main-research-corrections-v2.json)',
            'records the serve/non-serve scope and route-row-bound corrections. Reusable',
            'FP16 Xs/FP32 GU/FP16 Hh scratch could lower the T8192 owned request by721.283MiB',
            'after the shared max(GDN,QSA,MoE) limit, leaving the new T8192 request only',
            '44.389MiB above the current old T6144 request. This is unimplemented and does',
            'not measure physical VRAM released or added dispatch costs. Row-capped groups',
            'can preserve grouping conditionally; descriptor lifetime and all callers',
            'need review before execution. The earlier GDN y-removal capacity claim remains',
            'rejected: its view lies inside an MoE-dominated shared arena, so owned bytes',
            'saved are zero in this recipe. Source-owned T6144 is2815.941MiB (about2.750GiB).', '',
            'The [upstream settings audit TODO](../2026-10-11-upstream-parameter-audit/TODO.md)',
            'remains open: past settings were explicit, but an independent search for the',
            'best official prefill/decode settings and its verification is not established.',
            'Current quiet samples are the Xe candidate, not new official-pristine timings.', '',
            'Next: implement and qualify bounded fallback scratch reuse, then measure actual',
            '8K capacity and64K pass cost. Do not weaken the512MiB READY reserve or full-context',
            'requirements. A separate serve workspace-retirement candidate preserves session',
            'and decoder state; cache enlargement still needs its own pointer/graph contract.', '']
        (R / 'README.md').write_text('\n'.join(lines))
        for path in R.rglob('*.json'):
            json.loads(path.read_text())
        identity = run('identity-origin', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(all(line.startswith('file:/home/yayoi/.config/git/config') for line in identity.splitlines()), 'global identity only')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) == 'git@github.com:yqYo1/Strata.git', 'own fork')
        status = run('status', ['/usr/bin/git', 'status', '--porcelain', '--untracked-files=normal'])
        m.require(status and all(line[3:].startswith(str(R.relative_to(W))) for line in status.splitlines()), 'only result archive changes')
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        gitleaks = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
        scan = out / 'gitleaks-redacted.json'
        run('secret-scan', [str(gitleaks), 'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'zero secret findings')
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W))])
        run('staged-diff-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'bench(sycl): record four balanced quiet 32k pairs'])
        r['committed_revision'] = run('committed-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published results')
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
