import fcntl
import importlib.util
import json
import os
import re
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
XE = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-prefill-diagnostics32k-boundary-root-v1'
    m.require(not out.exists(), 'immutable publication')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), gpu_executed=False, model_executed=False, deleted=[])
    def save():
        (out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
    owner = m.Owner(out/'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv, wall=60):
        e, so, se = owner.run(label, argv, ENV, W, wall=wall, cpu=int(wall)*2,
                             text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0, 'closed '+label)
        return so.read_text().strip()
    try:
        m.require(not run('initial-status', ['/usr/bin/git','status','--porcelain']), 'clean report tree')
        roots = {
            'free-build':'xe-e32-free-full-build-root-v14',
            'contrib-llvm-build':'xe-e32-contrib-llvm-full-build-root-v14',
            'free-host-asan':'xe-e32-free-asan-full-build-root-v11',
            'contrib-llvm-host-asan':'xe-e32-contrib-llvm-asan-full-build-root-v11',
            'lints':'xe-e32-lints-baseline-compare-root-v9',
            'source-publication':'xe-prefill-diagnostics-source-publish-root-v1',
            'parser-cpu':'xe-prefill-diagnostic-observer-cpu-root-v1',
            'controller-cpu-v1':'xe-source-llvm-prefill-diagnostic32k-cpu-root-v1',
            'controller-cpu-v2':'xe-source-llvm-prefill-diagnostic32k-cpu-root-v2',
            'counts-observer-failed':'xe-source-llvm-prefill-diagnostic32k-root-v1/counts',
            'counts-capture-review':'xe-prefill-counts-closed-capture-review-root-v1',
            'timer':'xe-source-llvm-prefill-diagnostic32k-root-v2/timer',
        }
        results = {}
        for key, root in roots.items():
            v = json.loads((B/root/'record.json').read_text())
            m.require(not v['active'] and all(m.closed(c) for c in v['commands']), 'closed run '+key)
            for c in v['commands']:
                for ident in c['owners']:
                    now = m.identity(ident['pid'])
                    m.require(not now or now['start_ticks'] != ident['start_ticks'], 'owned identity absent '+key)
            if key == 'counts-observer-failed':
                m.require(not v['passed'] and not v['complete'] and v['model_normal_close'] and
                          v['error'] == "ValidationError('line 38: wrong mode, issuer or format')" and not v['new_fault_messages'], 'original failed observer status preserved')
            else:
                m.require(v['passed'] and v['complete'], 'actual completed qualification '+key)
            results[key] = v
        failed = results['counts-observer-failed']
        counts = results['counts-capture-review']
        timer = results['timer']
        m.require(counts['original_overall_status_unchanged'] and not timer['performance_eligible'] and
                  timer['model_normal_close'] and not timer['new_fault_messages'], 'exact limited diagnostic scope')
        m.require(failed['binary'] == timer['binary'] and failed['input'] == timer['input'] and
                  failed['argv'] == timer['argv'] and failed['effective_settings'] == timer['effective_settings'] and
                  failed['boot_id'] == timer['boot_id'], 'same binary/input/argv/effective settings/boot')
        e = dict(timer['environment'])
        m.require(e.pop('STRATA_PREFILL_TIMING') == '1' and e == failed['environment'], 'timer is only environment change')
        m.require(counts['transfer_ledger'] == timer['transfer_ledger'], 'exact full expert ledger equal in both actual captures')
        output = lambda x: [line for line in x['request_protocol'] if line.startswith(('T ','LP '))]
        m.require(output(failed) == output(timer), 'same actual selected outputs/logprobs')
        D = W/'bench/results/2026-10-11-xe-prefill-diagnostics32k'
        m.require(not D.exists(), 'new archive')
        D.mkdir()
        copies = {}
        def copy(src, name):
            dst = D/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            m.require(m.sha(src) == m.sha(dst), 'byte-identical archive')
            copies[str(dst.relative_to(W))] = dict(original_path=str(src), sha256=m.sha(src), bytes=src.stat().st_size)
        for key, root in roots.items():
            copy(B/root/'record.json', key+'/record.json')
        for name in ['xe-prefill-diagnostics-source-commit-v1.json','xe-prefill-diagnostics-combined-main-review-v1.json',
                     'xe-prefill-diagnostic-observer-main-review-v1.json',
                     'xe-prefill-diagnostic32k-controller-main-review-v1.json',
                     'xe-prefill-diagnostic32k-controller-main-review-v2.json',
                     'xe-prefill-native-transfer-layout-v1.json','xe-prefill-diagnostic-agent-snapshot-v1.json']:
            copy(B/name, name)
        report = B/'research-20261009/implementation-xe-prefill-diagnostic-observer-v1.txt'
        copy(report, report.name)
        for p in (B/'xe-prefill-diagnostic-observer-source-v1').iterdir():
            if p.is_file():
                copy(p, 'observer-source/'+p.name)
        for folder in ['xe-source-llvm-prefill-diagnostic32k-v1','xe-source-llvm-prefill-diagnostic32k-v2']:
            for p in (B/folder).iterdir():
                if p.suffix == '.py':
                    copy(p, 'recipes/'+folder+'/'+p.name)
        for name in ['xe-prefill-diagnostic-observer-cpu-v1.py','xe-prefill-diagnostics-source-publish-v1.py',
                     'xe-source-llvm-prefill-diagnostic32k-cpu-v1.py','xe-source-llvm-prefill-diagnostic32k-cpu-v2.py',
                     'xe-prefill-counts-closed-capture-review-v1.py','xe-prefill-diagnostics-integrate-v1.py',
                     'xe-prefill-diagnostics32k-boundary-v1.py']:
            copy(B/name, 'recipes/'+name)
        copy(B/'xe-e32-free-full-build-root-v14/commands/source-patch.stdout', 'production-prefill-diagnostics.patch')
        plans = []
        for key, root in roots.items():
            v = results[key]
            if key in ['counts-observer-failed', 'timer']:
                so = (B/root/'commands/model.stdout').read_text().splitlines()
                m.require(so == v['startup'] + v['request_protocol'], 'full completed native stdout retained')
                se = (B/root/'commands/model.stderr').read_text().splitlines()
                keep = [x for x in se if x.startswith(('strata generate:','strata prefill timing:','strata prefill timer overhead:',
                            'strata mtp:','strata verify:','strata serve:','strata:')) or
                        re.search(r'\b(?:WARN|WARNING|ERROR|FATAL)\b|GPU completion|terminate called',x)]
                # All384 expert rows/types/counters and total were independently checked and live in the structured capture-review/timer receipts.
                (D/key/'retained-stderr.txt').write_text('\n'.join(keep)+'\n')
            for p in (B/root/'commands').glob('*'):
                if p.suffix not in ['.stdout','.stderr']:
                    continue
                st = p.stat()
                plans.append(dict(path=str(p), bytes=st.st_size, allocated_bytes=st.st_blocks*512,
                    sha256=m.sha(p), retained_receipt=str((D/key/'record.json').relative_to(W)),
                    reason='Closed successful API/protocol/build/lint duplicate stream; useful exact flags/tests/mapped runtimes/owners/fault boundaries/individual status and full diagnostic row evidence retained. Resolved observer policy failure retained as FAILED.'))
        total = timer['transfer_ledger']['total']
        summary = dict(source=timer['candidate_revision'], boot=timer['boot_id'],
            input_sha256=timer['input']['sha256'], actual_batched_positions=32767, chunk_tokens=4096,
            requested_ring=8, actual_ring_source_minimum=16, effective_policy='full_nonresident_stream', issuer='inline',
            expert_h2d_logical_request_bytes=total['bytes'], expert_h2d_calls=total['calls'],
            per_chunk_submitted_bytes=counts['per_chunk_submitted_bytes'],
            transfer_scope='Expert blob successful H2D request extents only, checked terminal production drain before diagnostic declaration; excludes startup/decode/KV/activation/D2H/D2D and physical PCIe-exclusive service.',
            counts_original_overall_result=dict(passed=False, error=failed['error'], model_normal_close=True,
                native_request_timing=failed['diagnostic_request_timing'], new_faults=failed['new_fault_messages'],
                original_posthealth_not_reached=True, subsequent_complete_capture_cpu_validation=True),
            timer_actual_result=dict(passed=True, timing=timer['diagnostic_request_timing'],
                fresh_before_after_health=True, no_new_faults=True),
            timer_overhead=timer['timer_overhead'], phase_profile=timer['phase_profile'],
            same_full_transfer_ledger=True, same_selected16_outputs_and_lp=True,
            timer_minus_counts_prefill_ms=timer['validation']['prompt_ms']-failed['validation']['prompt_ms'],
            comparison_limit='One diagnostic process per setting; original counts overall FAILED preserved and excluded from qualified performance. No calibrated/exclusive attribution, optimal-setting, capability or full-context/RESTORE claim.',
            correction='Prior traffic-scope and quiet report claim that requested ring8 makes routed-only was wrong: ring_slots clamps to16 for T>=2048, so source and actual ledger both show full nonresident streaming. Historical performance/correctness outcomes remain unchanged.',
            performance_adopted=False, official_independent_settings_search_complete=False,
            next=['Measure admissible larger prefill chunks/VRAM lifetime choices to reduce full model traffic per token; keep decode decision separate',
                  'Use independent profiler/component service measurements before exclusive hardware bottleneck claims',
                  'Finish Xe evaluation then optimize and evaluate official pristine settings per tracked TODO',
                  'Actual full262144 and RESTORE remain required before adoption'])
        (D/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        overhead = timer['timer_overhead']['overhead']
        text = ('診断用コードはfree・contrib-llvmのrelease/host ASan+UBSanとも既存CPUテスト12件PASS/1件native ISA skip。'
            'actual baseline/current mandatory lintは各mode既存10件、新規0件。source commit %s。\n\n'
            '32768入力の32767位置を4096幅8チャンクで処理し、各48layer、合計384行を全モデルmetadataの型とblob extentで照合した。'
            '各layer/chunkは509個の非resident expertを転送し、1チャンク49,997,644,800 byte、合計399,981,158,400 byte（約400GB、195456コピー）だった。'
            'これは実際の成功したexpert blob H2D API要求のbyte数で、物理PCIe帯域や全GPU転送量ではない。\n\n'
            '以前のring8でrouted-onlyになるというsource解釈は誤りだった。ring_slotsは指定8を最低16へ補正するため、実際は全非resident streamingだった。'
            '最初のcountsは推論16token・QUITが正常完了しfaultなしだが、検証側の誤ったrouted期待により全体FAILED。'
            'FAILEDは保存し、実行後GPU health検査に到達していない範囲も明記した。complete stdout/stderrをCPUでsource由来の期待に照合して転送証拠を取り出した。'
            '検証期待だけを直した新controllerのtimer実行は推論・normal QUIT・全384行・正しいmapped runtime・新faultなし・fresh before/after GPU healthを含めPASS。\n\n'
            '両実captureの転送行・合計と16token/LP/top5は完全一致。timerのhost mark %.3f ms、fold %.3f ms。'
            'markのevent APIとbookkeeping、foldのtimestamp query等のhost wallで、blocking/deschedulingも含み、exclusiveなCPU計算やGPU serviceではない。'
            '既存stream_sync中の待ちを含まず、差を較正値として引かない。phase event区間もhost/copy idleを含む。'
            '個別timeはsummary.json。性能採用・最適設定・広い能力の劣化率・実256K/RESTORE完了は主張しない。\n\n'
            '次はchunk拡大とprefill中に保持しているdecode用VRAMの寿命を検討して、token当たりの約400GBの往復を減らす。'
            '本家もprefill/decodeを個別に最適化して比較する[TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を引き続き未完了として扱う。\n'
            ) % (timer['candidate_revision'], overhead['mark_ms'], overhead['fold_ms'])
        (D/'RESULTS_JA.md').write_text(text)
        previous = W/'bench/results/2026-10-11-xe-profile32k-quiet/report-registry-v124.json'
        m.require(json.loads(previous.read_text())['research_completed'] == 496, 'previous count')
        registry = dict(registry_version=125, research_completed=497, created_utc=m.utc(),
            registry_format='incremental; follow previous_committed_registry',
            previous_committed_registry=dict(path=str(previous.relative_to(W)), sha256=m.sha(previous), reports=496),
            new_reports=[report.name], reports=[dict(path=str(report), archive=str((D/report.name).relative_to(W)),
                sha256=m.sha(report), bytes=report.stat().st_size, model='gpt-6.1-sol', read_only=False,
                qualification='Full Main source/report/manifest review; actual CPU9 positive/61 rejection parser fixtures; actual complete model capture rows checked, first overall observer failure preserved; fresh timer run qualifies diagnostics. No performance adoption/global capability/256K/RESTORE.')],
            live_agent_snapshot=json.loads((B/'xe-prefill-diagnostic-agent-snapshot-v1.json').read_text()),
            next_research_scopes=['Measured copy traffic vs component roofline and prefill VRAM/chunk lifetime', 'Independent official parameter selection'],
            dependency='Actual selected Luna quota until Oct14 2026 12:33PM; no substitute')
        (D/'report-registry-v125.json').write_text(json.dumps(registry, indent=2)+'\n')
        (D/'archive-manifest.json').write_text(json.dumps(dict(byte_identical_copies=copies, original_statuses_preserved=True,
            raw_text_retention_review='No named whole-history question remains for these closed successful streams or resolved observer mismatch. All useful expert rows/counts and timer summaries retained; model/runtime fixtures remain current.'), indent=2)+'\n')
        m.require(sum(p.stat().st_size for p in D.rglob('*') if p.is_file()) < 8 << 20, 'compact archive under8MiB')
        run('secret-check', [str(XE/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(D)])
        run('diff-check', ['/usr/bin/git','diff','--check'])
        run('stage', ['/usr/bin/git','add','--',str(D.relative_to(W))])
        run('commit', ['/usr/bin/git','commit','-m','docs(bench): record actual prefill traffic and timer overhead'])
        r['evidence_revision'] = run('evidence-head', ['/usr/bin/git','rev-parse','HEAD'])
        manifest = D/'retention-v1.json'
        manifest.write_text(json.dumps(dict(retained_commit=r['evidence_revision'], deletions=plans,
            logical_bytes=sum(x['bytes'] for x in plans), allocated_file_bytes=sum(x['allocated_bytes'] for x in plans),
            does_not_measure_available_space=True, original_statuses_unchanged=True), indent=2)+'\n')
        run('manifest-secret-check', [str(XE/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(manifest)])
        for x in plans:
            p = Path(x['path'])
            m.require(m.sha(p) == x['sha256'] and p.stat().st_size == x['bytes'], 'unchanged closed raw file')
            p.unlink()
            r['deleted'].append(x['path'])
        m.require(all(m.sha(B/root/'record.json') == copies[str((D/key/'record.json').relative_to(W))]['sha256']
                      for key,root in roots.items()), 'original actual receipts unchanged')
        run('stage-retention', ['/usr/bin/git','add','--',str(manifest.relative_to(W))])
        run('commit-retention', ['/usr/bin/git','commit','-m','docs(bench): retire closed diagnostic and build streams'])
        r['committed_revision'] = run('final-head', ['/usr/bin/git','rev-parse','HEAD'])
        run('publish', ['/usr/bin/git','push','origin','HEAD'])
        m.require(not run('final-status', ['/usr/bin/git','status','--porcelain']), 'clean report tree')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','evidence_revision','committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
