import fcntl
import importlib.util
import json
import os
import re
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
XE = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O = B/'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B/'xe-chunk6144-ring96-boundary-root-v1'
    m.require(not out.exists(), 'immutable publication')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), gpu_executed=False, model_executed=False, deleted=[])
    def save():
        (out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
    owner = m.Owner(out/'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv, cwd=W):
        e,so,se = owner.run(label, argv, ENV, cwd, wall=60,cpu=120,
                            text_cap=4 << 20,total_cap=16 << 20,rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0, 'closed '+label)
        return so.read_text().strip()
    try:
        m.require(not run('initial-status',['/usr/bin/git','status','--porcelain']), 'clean report tree')
        m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'],XE) == '9c0b291d85adecb6c1dc489c3e9965b82226b799', 'unchanged committed source')
        m.require(not run('source-status',['/usr/bin/git','status','--porcelain'],XE), 'clean source')
        roots = {'ring96':'xe-source-llvm-ring96-diagnostic32k-root-v1/counts',
                 'ring96-cpu':'xe-source-llvm-ring96-diagnostic32k-cpu-root-v1',
                 'chunk6144-v1-preflight-failed':'xe-source-llvm-chunk6144-diagnostic32k-root-v1/counts',
                 'chunk6144-v2-observer-failed':'xe-source-llvm-chunk6144-diagnostic32k-root-v2/counts',
                 'chunk6144-v3':'xe-source-llvm-chunk6144-diagnostic32k-root-v3/counts',
                 'chunk6144-cpu-v1':'xe-source-llvm-chunk6144-diagnostic32k-cpu-root-v1',
                 'chunk6144-cpu-v2':'xe-source-llvm-chunk6144-diagnostic32k-cpu-root-v2',
                 'chunk6144-cpu-v3':'xe-source-llvm-chunk6144-diagnostic32k-cpu-root-v3',
                 'closed-capture-and-observer-cpu':'xe-chunk6144-capture-and-observer-cpu-root-v1'}
        results = {}
        for key,root in roots.items():
            v = json.loads((B/root/'record.json').read_text())
            m.require(not v['active'] and all(m.closed(c) for c in v['commands']), 'closed '+key)
            for c in v['commands']:
                for ident in c['owners']:
                    now = m.identity(ident['pid'])
                    m.require(not now or now['start_ticks'] != ident['start_ticks'], 'owned identity absent '+key)
            if key == 'chunk6144-v1-preflight-failed':
                m.require(not v['passed'] and not v['complete'] and not v['model_launch_attempted'] and v['error'] == "KeyError('validation')", 'preflight failure preserved')
            elif key == 'chunk6144-v2-observer-failed':
                m.require(not v['passed'] and not v['complete'] and v['model_normal_close'] and v['error'] == "ValueError('exact eight chunks')" and not v['new_fault_messages'], 'original observer failure preserved')
            else:
                m.require(v['passed'] and v['complete'], 'actual complete qualification '+key)
            results[key] = v
        ring,chunk,cpu = (results[k] for k in ('ring96','chunk6144-v3','closed-capture-and-observer-cpu'))
        m.require(cpu['transfer_ledger'] == chunk['transfer_ledger'] and cpu['selected16_ids_and_lp_top5_equal_quiet'], 'independent actual complete capture ledgers/outputs agree')
        m.require(ring['binary'] == chunk['binary'] and ring['input'] == chunk['input'] and ring['source_pins'] == chunk['source_pins'] and ring['boot_id'] == chunk['boot_id'], 'same source/binary/input/boot')
        m.require(not ring['new_fault_messages'] and not chunk['new_fault_messages'] and
                  ring['model_normal_close'] and chunk['model_normal_close'], 'closed fault-free model diagnostics')
        m.require(all(m.sha(XE/p) == h for p,h in chunk['source_pins'].items()), 'source pins current')
        D = W/'bench/results/2026-10-11-xe-chunk6144-and-ring96'
        m.require(not D.exists(), 'new archive')
        D.mkdir()
        copies = {}
        def copy(src, name):
            dst = D/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            m.require(m.sha(dst) == m.sha(src), 'exact archive bytes')
            copies[str(dst.relative_to(W))] = dict(original_path=str(src),bytes=src.stat().st_size,sha256=m.sha(src))
        for key,root in roots.items():
            copy(B/root/'record.json',key+'/record.json')
        for name in ['xe-ring96-diagnostic32k-main-review-v1.json',
                     'xe-chunk6144-diagnostic32k-main-review-v1.json','xe-chunk6144-diagnostic32k-main-review-v2.json','xe-chunk6144-diagnostic32k-main-review-v3.json',
                     'xe-prefill-route-branch-census-main-review-v1.json','xe-mmq-group-gather-byte-main-review-v1.json','xe-prefill-route-census-observer-main-review-v1.json',
                     'xe-chunk6144-agent-snapshot-v1.json']:
            copy(B/name,name)
        reports = ['implementation-xe-mmq-group-gather-byte-v1.txt','implementation-xe-prefill-route-branch-census-v1.txt','implementation-xe-prefill-route-census-observer-v1.txt']
        for name in reports:
            copy(B/'research-20261009'/name,name)
        artifact_folders = ['xe-mmq-group-gather-byte-source-v1','xe-prefill-route-branch-census-source-v1','xe-prefill-route-census-observer-source-v1']
        for folder in artifact_folders:
            for p in (B/folder).rglob('*'):
                if p.is_file():
                    copy(p,'agent-source/'+folder+'/'+str(p.relative_to(B/folder)))
        recipe_folders = ['xe-source-llvm-ring96-diagnostic32k-v1',
                          'xe-source-llvm-chunk6144-diagnostic32k-v1','xe-source-llvm-chunk6144-diagnostic32k-v2','xe-source-llvm-chunk6144-diagnostic32k-v3']
        for folder in recipe_folders:
            for p in (B/folder).glob('*.py'):
                copy(p,'recipes/'+folder+'/'+p.name)
        for name in ['xe-source-llvm-ring96-diagnostic32k-cpu-v1.py',
                     'xe-source-llvm-chunk6144-diagnostic32k-cpu-v1.py','xe-source-llvm-chunk6144-diagnostic32k-cpu-v2.py','xe-source-llvm-chunk6144-diagnostic32k-cpu-v3.py',
                     'xe-chunk6144-capture-and-observer-cpu-v1.py','xe-chunk6144-ring96-boundary-v1.py']:
            copy(B/name,'recipes/'+name)
        plans = []
        for key,root in roots.items():
            v = results[key]
            if v.get('model_normal_close'):
                so = (B/root/'commands/model.stdout').read_text().splitlines()
                m.require(so == v['startup'] + v['request_protocol'], 'full successful native stdout represented exactly')
                se = (B/root/'commands/model.stderr').read_text().splitlines()
                keep = [x for x in se if x.startswith(('strata generate:','strata mtp:','strata verify:','strata serve:','strata:')) or re.search(r'\b(?:WARN|WARNING|ERROR|FATAL)\b|GPU completion|terminate called',x)]
                (D/key/'retained-stderr.txt').write_text('\n'.join(keep)+'\n')
            for p in (B/root/'commands').glob('*'):
                if p.suffix in ('.stdout','.stderr'):
                    st = p.stat()
                    plans.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=m.sha(p),
                        retained_receipt=str((D/key/'record.json').relative_to(W)),
                        reason='Closed successful repeated API/protocol/journal/fixture stream. Exact commands, individual status, ownership, runtime, progress/output, complete H2D rows and useful warnings retained. Both resolved observer failures stay FAILED; their concrete errors are retained. No named full-history consumer.'))
        total = chunk['transfer_ledger']['total']
        offsets = [0,6144,12288,18432,24576,30720]
        summary = dict(source=chunk['candidate_revision'],boot=chunk['boot_id'],input_sha256=chunk['input']['sha256'],
            ring96_actual=dict(chunk=4096,actual_ring=96,timing=ring['diagnostic_request_timing'],transfer_total=ring['transfer_ledger']['total']),
            chunk6144_actual=dict(chunk=6144,actual_ring=16,timing=chunk['diagnostic_request_timing'],transfer_total=total,
                ready_free_mib=int(chunk['effective_settings']['vram_free_mib']),allocation_plan=chunk['chunk_plan'],
                per_chunk=[dict(offset=o,calls=sum(x['calls'] for x in chunk['transfer_ledger']['rows'] if x['offset']==o),bytes=sum(x['bytes'] for x in chunk['transfer_ledger']['rows'] if x['offset']==o)) for o in offsets]),
            original_failed_statuses_preserved=True, selected16_ids_and_lp_top5_equal_quiet=True,
            fresh_before_after_health=True,new_faults=[],performance_adopted=False,
            logical_request_reduction_fraction=1.0-total['bytes']/ring['transfer_ledger']['total']['bytes'],
            measurement_scope='Successful expert blob H2D request extents, not physical PCIe or exclusive DMA service. Diagnostic counters ON, TIMING/TRACE off. One passing process per setting; ring96 vs chunk6144 changes both ring and chunk. Original quiet3 was older source. No optimal-setting/adoption or calibrated bottleneck attribution.',
            cold_decode_scope='16 outputs from a new process, includes first graph captures; no warm/RESTORE measurement or repeated decode selection.',
            observer_actual_fixtures=cpu['observer_main_review']['fixtures'],
            pending_implementation=dict(census='Fully Main-reviewed source only; not applied/build/lint/runtime qualified',gather_probe='Fully Main-reviewed source only; no build/GPU byte-parity result',observer='Main CPU10 positive39 rejection fixtures PASS; actual new census capture unqualified'),
            open_requirements=['Integrate reviewed route census after this boundary, build and test both free/contrib-llvm with host sanitizers and actual baseline/current mandatory lints; use CPU-qualified observer on real32K capture',
                'Measure MMQ/fallback and unused-expert branch counts before prioritizing grouped-gather probe or selective streaming',
                'Quiet independently repeated6144/64K measurement with separate prefill and decode selection',
                'Prefill-only VRAM lifetime/recreate choices and broader dtype numerical capability evaluation',
                'Actual full262144/RESTORE, Xe pristine evaluation and official latest feature integration; official independent parameter tuning per committed TODO remains OPEN'])
        (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (D/'RESULTS_JA.md').write_text(
            '32K入力・同一9c0b291dソース/バイナリで、ring96/4096幅は215.56 tok/s、ring16/6144幅は266.35 tok/sだった。'
            '診断付きの個別試行であり、最適設定や性能採用の結論ではない。decodeはそれぞれ8.10/9.15 tok/sの初回16tokenで、独立した反復とRESTOREは未実施。\n\n'
            'expert blobの成功H2D要求量は399,981,158,400から295,956,300,800 byteへ26.0%減少した。'
            '6144幅では5つのfull-nonresident chunkと2047幅のrouted tailを、全48layer・288行と型/extentまで照合した。'
            '要求byte数は物理PCIe帯域やexclusive GPU serviceではない。リングだけを16から96へ広げた試行には速度改善が見えず、チャンク拡大と転送回数削減を次の候補とする。\n\n'
            '6144幅の追加initバッファ要求をソースで最大886.0625 MiBと見積もり、17allocに計68 MiBの計画上の丸め余裕を置いた。'
            'これはdriver page granularityの実測ではない。起動後の空きは実際に744 MiBで、512 MiBの開始条件を満たした。'
            '既存のdecode graph capture時には空きがさらに減る。8192幅へ無条件に拡大しない。\n\n'
            '最初のv1は出力比較の位置を誤りKeyErrorでGPU起動前にFAILED。v2では推論16tokenとQUITは正常だが、検証の固定8chunk条件によりFAILED。'
            '元のstatus/hashを保持し、正しい6chunk期待のCPU再検証で転送行と16token/LP/top5の完全一致を確認した。'
            '新しいv3の実測はnormal QUIT、正しいruntime、全転送行、基準出力、前後のfresh GPU health、新faultなしまでPASS。\n\n'
            'グループgatherは既定で有効で、整列した単体gatherも1kernelである。以前の複数copyという説明は非整列時のfallbackに限る。'
            '実際のMMQ/FP16選択と未使用expert転送量は未測定なので、そのcensusとbyte parity probeを別実装エージェントから受領し、全文とhashを確認した。'
            'census observerのCPU10正常・39拒否ケースはMainでPASS。census sourceとgather probeのbuild/GPUは未検証で、次にMainが順次実施する。\n\n'
            '本家は箱出し設定だけの測定ではないが、最適設定探索は未完了。[追跡TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を維持する。'
            '実256K、RESTORE、広い能力評価、最新本家の独立設定調整は未完了。\n')
        previous = W/'bench/results/2026-10-11-xe-prefill-diagnostics32k/report-registry-v125.json'
        m.require(json.loads(previous.read_text())['research_completed'] == 497, 'prior exact registry count')
        registry = dict(registry_version=126,research_completed=500,created_utc=m.utc(),registry_format='incremental; follow previous_committed_registry',
            previous_committed_registry=dict(path=str(previous.relative_to(W)),sha256=m.sha(previous),reports=497),
            new_reports=reports,reports=[dict(path=str(B/'research-20261009'/name),archive=str((D/name).relative_to(W)),sha256=m.sha(B/'research-20261009'/name),bytes=(B/'research-20261009'/name).stat().st_size,model='gpt-6.1-sol',read_only=False,qualification='Full Main source/report/manifest review; source-only probe and census runtime remain unqualified. Observer actual CPU10 positive39 refusal cases PASS.') for name in reports],
            live_agent_snapshot=json.loads((B/'xe-chunk6144-agent-snapshot-v1.json').read_text()),
            next_research_scopes=['Measured branch/counts and component service before selective transfer or MMQ changes','Official independent parameter selection'],
            dependency='Actual selected gpt-6-luna quota until Oct14 2026 12:33PM; no substitute. These were concrete implementation agents, not research substitutes.')
        (D/'report-registry-v126.json').write_text(json.dumps(registry,indent=2)+'\n')
        (D/'archive-manifest.json').write_text(json.dumps(dict(byte_identical_copies=copies,original_statuses_preserved=True,
            current_external_consumers={'census_source':'Main next integration/build/lint/runtime qualification','gather_probe':'Main production byte parity build when measured branch needs it','observer_source':'Main next real census capture validation'},
            raw_text_retention_review='Useful evidence extracted and verified. Resolved observer-only mismatches require errors and original receipts, not repeated full API traces. Current sources/helper/model fixtures are retained.'),indent=2)+'\n')
        m.require(sum(p.stat().st_size for p in D.rglob('*') if p.is_file()) < 8 << 20, 'compact archive8MiB')
        run('secret-check',[str(XE/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(D)])
        run('diff-check',['/usr/bin/git','diff','--check'])
        run('stage',['/usr/bin/git','add','--',str(D.relative_to(W))])
        run('commit',['/usr/bin/git','commit','-m','docs(bench): measure larger prefill chunks and review route census'])
        r['evidence_revision'] = run('evidence-head',['/usr/bin/git','rev-parse','HEAD'])
        manifest = D/'retention-v1.json'
        manifest.write_text(json.dumps(dict(retained_commit=r['evidence_revision'],deletions=plans,
            logical_bytes=sum(x['bytes'] for x in plans),allocated_file_bytes=sum(x['allocated_bytes'] for x in plans),
            does_not_measure_available_space=True,original_statuses_unchanged=True),indent=2)+'\n')
        run('retention-secret-check',[str(XE/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(manifest)])
        for x in plans:
            p = Path(x['path'])
            m.require(m.sha(p) == x['sha256'] and p.stat().st_size == x['bytes'], 'unchanged closed raw stream')
            p.unlink()
            r['deleted'].append(x['path'])
        m.require(all(m.sha(B/root/'record.json') == copies[str((D/key/'record.json').relative_to(W))]['sha256'] for key,root in roots.items()), 'original receipts byte-identical after retention')
        run('stage-retention',['/usr/bin/git','add','--',str(manifest.relative_to(W))])
        run('commit-retention',['/usr/bin/git','commit','-m','docs(bench): retire closed chunk measurement streams'])
        r['committed_revision'] = run('final-head',['/usr/bin/git','rev-parse','HEAD'])
        run('publish',['/usr/bin/git','push','origin','HEAD'])
        m.require(not run('final-status',['/usr/bin/git','status','--porcelain']), 'clean report tree')
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
