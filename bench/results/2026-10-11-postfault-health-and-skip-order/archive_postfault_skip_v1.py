"""Root-owned closed-run evidence archive; no GPU or active-run cleanup."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, re, shutil, subprocess

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R = B/'research-20261009'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A = W/'bench/results/2026-10-11-postfault-health-and-skip-order'
P = W/'bench/results/2026-10-11-parallel-round87'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v, msg):
    if not v: raise RuntimeError(msg)
def pin(p): return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

with (B/'owned-v0141-measurement.lock').open('a') as lk:
    fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not A.exists() and not P.exists(), 'new archive only')
    require(not subprocess.run(['git','status','--porcelain'],cwd=W,text=True,capture_output=True,check=True).stdout,'clean storage boundary')
    run_names=['postfault-status-health-root-r1','postfault-status-health-root-r2',
               'prefill-skipped-entry-order-cpu-root-r1','prefill-skip-order-linked-root-r1',
               'prefill-skip-order-linked-root-r2','prefill-single-gpu-census-cpu-root-r1',
               'xe-reset-guard-cpu-root-v1']
    records={name:json.loads((B/name/'record.json').read_text()) for name in run_names}
    for name,j in records.items():
        require(j['active'] is False, 'closed run '+name)
        # All old owners are from the prior boot. No unknown process is killed.
        for c in j.get('commands',[]):
            require(c.get('observation_complete') is True and c.get('session_empty') is True, 'closed owned command '+name)
    require(sha(B/'postfault-status-health-root-r2/record.json')=='7d83b26c2bf260004e83d1c3bab81fe6b73d67a419b763cea458a26d25937ea6','original failed receipt')
    require(records['postfault-status-health-root-r2']['passed'] is False,'failure preserved')
    A.mkdir();P.mkdir()
    copied=[]
    def copy(src,rel):
        src=Path(src);dest=A/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest);require(sha(src)==sha(dest),'exact archive copy')
        copied.append(dict(original=pin(src),archive=str(dest.relative_to(W))))
    for name in run_names: copy(B/name/'record.json',name+'/record.json')
    for name in ['postfault-status-health-v1','postfault-status-health-v2',
                 'prefill-skipped-entry-order-v1','prefill-single-gpu-census-v1']:
        for src in (B/name).rglob('*'):
            if src.is_file() and src.suffix in ('.cpp','.hpp','.py','.txt'):
                copy(src,name+'/'+str(src.relative_to(B/name)))
    for rel in ['classify_postfault_kernel_r2_v1.py','qualify_xe_reset_guard_v1.py',
                'postfault-status-health-root-r2/kernel-classification.json',
                'postfault-status-health-root-r1/commands/build.stderr',
                'postfault-status-health-root-r2/commands/integer-health.stdout',
                'postfault-status-health-root-r2/commands/integer-health.stderr',
                'prefill-skipped-entry-order-cpu-root-r1/extracted/extraction.json',
                'prefill-skipped-entry-order-cpu-root-r1/extracted/old_release.inc',
                'prefill-skipped-entry-order-cpu-root-r1/extracted/new_release.inc',
                'xe-reset-guard-cpu-root-v1/commands/unit-tests.stderr',
                'prefill-skip-order-linked-root-r2/commands/compile-SYCL-prefill.stderr']:
        copy(B/rel,rel)
    F=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011')
    for rel in ['sycl/tools/recover-xe.sh','sycl/tools/test_recover_xe.py']: copy(F/rel,'health-guard/'+Path(rel).name)
    copy(Path(__file__),'archive_postfault_skip_v1.py')

    old=json.loads((R/'report-registry-v109.json').read_text())
    new=[]
    files=[]
    for n in range(313,322):
        matches=list(R.glob(f'round{n}-*.txt'));require(len(matches)==1,'one frozen report')
        files+=matches
    files += [R/'implementation-prefill-skipped-entry-order-v1.txt',R/'implementation-prefill-single-gpu-census-v1.txt']
    for src in files:
        dest=P/src.name;shutil.copyfile(src,dest);require(sha(src)==sha(dest),'report copy')
        item=pin(src);item.update(archive_path=str(dest),model='gpt-6.1-sol' if src.name.startswith('implementation') else 'gpt-6-luna',
                                status='completed-source-handoff' if src.name.startswith('implementation') else 'completed-read-only',
                                root_full_report_reviewed=True,scope=src.stem)
        new.append(item)
    review={
        'census_predicate_correction':'R310/root earlier interpretation was inverted: parameter unsupported_multi_gpu false admits single-GPU census. Actual unchanged header is already opt-in eligible; clarity refactor not required/adopted. Preserve original report.',
        'R315_R318':'No independent crosschunk UAF established. Xe Runtime::free(pinned) drains queues; Stager-reset gap only all-pageable fallback, not established for V8 hostUSM arena. Ignored bool failures and conditional skipped-entry publication remain source holes, not proven CAT18 cause.',
        'R316':'896MiB decode lease is conditional arithmetic; request/server bind, Prefill reset/drain, allocation restore peak and graph reconstruction must be implemented/verified. No fit or speed claim.',
        'R317_R319':'199 relevant kernel entries include32 reset-log sequences while integer words exactly pass. Allocation ownership unresolved. R317 Codebrowser explanation is source hypothesis; installed vendor module metadata from R319 is passive and active-image/source equivalence unproven. Root could not fetch cited primary source URLs.',
        'R320_R321':'Plain first-distribution seam excludes actual native_pack. New native verifier investigation is pending. R321 benchmark selection is a proposal; 300 zero-regression cases is conditional binomial floor, not actual ability evidence. Historical timing fixture is not independent quality data.',
        'current_gpu':'New boot cfda0b6a-7c13-4fe2-bdf5-8f9202770840 observed after user reboot. No post-reboot GPU/model execution in this archive. User requires source countermeasures before faulting-code rerun.'}
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    old.update(registry_version=110,created_utc=now,previous_committed_registry='3bf7a81d2d6e8a10d8830e339c9630906d6500d6 / report-registry-v109.json',new_reports=new)
    old['reports']+=new;old['research_completed']=len(old['reports'])
    old['root_review_corrections_round87']=review
    old['live_agent_snapshot']=[{'agent':'/root/research_native_quality_seam_r322','model':'gpt-6-luna','scope':'native verifier first-distribution quality seam'},
                                {'agent':'/root/research_xe_submission_fail_closed_r323','model':'gpt-6-luna','scope':'independent fail-closed/thread failure audit'},
                                {'agent':'/root/implement_xe_prefill_failclosed_v1','model':'gpt-6.1-sol','scope':'source-only Xe derivative, root owns tests'}]
    dump(R/'report-registry-v110.json',old);shutil.copyfile(R/'report-registry-v110.json',P/'report-registry-v110.json')
    dump(P/'root-review.json',review)

    # Kernel duplicate is retired only after exact relevant-row reconciliation.
    raw=B/'postfault-status-health-root-r2/commands/kernel-after.stdout'
    rows=[json.loads(v) for v in raw.read_text().splitlines() if v.startswith('{')]
    relevant=[v for v in rows if '0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))]
    require(relevant==records['postfault-status-health-root-r2']['kernel_device_entries'],'all relevant199 rows preserved')
    require(len(relevant)==199,'recorded interval count')
    dump(A/'retention.json',dict(health_trace=dict(path='postfault-status-health-root-r2/commands/integer-health.stderr',
        bytes=104731,sha256='28ddaf0f58c93054496d48fca70b2770d3738d27914495d37e7c90ec03b981fe',budget_bytes=262144,
        owner='postfault-status-health-root-r2',consumer='R317/R323 status-success versus initialization/GuC-reset chronology',
        question='Which driver/context transition causes fresh integer pass concurrently with reset sequences?',
        next_review='after fail-closed derivative small-run qualification or chronology replacement'),
        candidate_binary_consumer='b8ada1df... pending root math/lifecycle qualification; not adopted',
        source_only_census_refactor='clarity only, not required for opt-in; not compiled/adopted'))
    (A/'RESULTS_JA.md').write_text('''新bootを確認したが、この区切りではGPUを実行していない。faulting pristine Xeの32K測定はFAILED/incompleteのまま。同じ設定の再実行を避け、同期とsubmit失敗処理を先に修正する。

旧bootの小テストは16,384整数一致・exit0だったが、同区間199件にGuC timeoutと32組のresetログがありhealth FAILED。これは独立hardware reset counterではない。新health分類は保存区間192/199を検出しCPU既存18テストPASS。修正はd1b8c105でcommit/push済み、旧helperをpinする測定controllerは変更していない。

Dのskip entry再利用前にissuer publicationを待つ変更は実際のlambdaを抽出したCPU112ケースとSYCL TU/linkがPASS。新binary b8ada1df04c4a19ce8347a11875b3bd6be759916fc54847604939434fd00ed17 はGPU/math/能力/物理262144未検証。現在の明示RING8はそもそも当該stream_all分岐を通らない。CAT18の原因と断定しない。

single-GPU censusの条件を読み違えていた。unsupported_multi_gpu=falseは許可であり、既存headerは環境変数1で既に有効。実体headerのCPU9ケースはPASSだがsynthetic trafficでありGPU production trafficではない。任意の名前明確化refactorは未採用。

最後の実測prefillは実入力65,536で403.5396 tok/s、prompt162.4029s。出力3 tokenで終了したためdecode性能は評価対象外。prefill1000判定は64Kでもよいというユーザー条件を採用し、decode70目標とは別々に評価する。
''')
    dump(A/'archive-manifest.json',copied)
    # Never serialize inherited environment; check actual sensitive values locally.
    sensitive=[v.encode() for k,v in os.environ.items() if re.search(r'token|secret|password|credential',k,re.I) and len(v)>=8]
    for path in list(A.rglob('*'))+list(P.rglob('*')):
        if path.is_file(): require(not any(v in path.read_bytes() for v in sensitive),'credential scan failed')
    deletion=[]
    for name in run_names:
        cmd=B/name/'commands'
        if not cmd.exists():continue
        for src in cmd.iterdir():
            if not src.is_file():continue
            reason=None
            if src.stat().st_size==0: reason='empty successful command stream; original command outcome/hash retained'
            elif name=='prefill-skip-order-linked-root-r2' and re.fullmatch(r'member-(before|after)-\d+\.stdout',src.name):
                reason='duplicate archive-member binary extraction; exact original member comparisons and hashes retained'
            elif src==raw: reason='all199 relevant kernel rows byte-content-equivalent to immutable original receipt; cumulative raw duplicate'
            elif name=='prefill-single-gpu-census-cpu-root-r1' and src.name in ['valid32.stderr','valid64.stderr','valid262.stderr','collector-bound.stderr']:
                case=src.stem;item=next(v for v in records[name]['cases'] if v['case']==case)
                require(sha(src)==item['raw_stderr_sha256'] and src.stat().st_size==item['raw_stderr_bytes'] and item['check']['passed'],'checked exact synthetic case')
                reason='synthetic repeated per-layer rows; individual checker result, receipt counts, raw hash and source preserved'
            if reason:deletion.append(dict(**pin(src),reason=reason,replacement=str((A/name/'record.json').relative_to(W))))
    dump(A/'deletion-manifest.json',dict(created_utc=now,files=deletion,total_logical_bytes=sum(v['bytes'] for v in deletion),original_statuses_unchanged=True))
    # Verify every retained copy before any deletion.
    for v in copied:require(sha(W/v['archive'])==v['original']['sha256'],'durable copy verification')
    for v in deletion:
        src=Path(v['path']);require(sha(src)==v['sha256'] and src.stat().st_size==v['bytes'],'deletion pin');src.unlink()
    subprocess.run(['git','diff','--check'],cwd=W,check=True)
    print(json.dumps(dict(archive=str(A),research=str(P),reports=len(new),removed_files=len(deletion),removed_logical_bytes=sum(v['bytes'] for v in deletion))))
