import datetime, fcntl, hashlib, json, os, re, shutil
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R = B / 'research-20261009'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A = W / 'bench/results/2026-10-11-xestrata-small-qualification'
P = W / 'bench/results/2026-10-11-parallel-round84'
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def identity(p): return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
def write(p, x): p.write_text(json.dumps(x, indent=2)+'\n')
def copy(p, q):
    q.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p,q)
    assert sha(p)==sha(q)
def absent(x):
    try:
        v=Path('/proc',str(x['pid']),'stat').read_text()
        assert int(v[v.rfind(')')+2:].split()[19]) != x['start_ticks'], 'owned PID still present'
    except FileNotFoundError: pass

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not A.exists() and not P.exists()
    A.mkdir(); P.mkdir()
    summary=[]; retired=[]; excerpts=[]
    names=['xestrata-device-first-diagnostic-root-v1']
    for role in ['GU','Down']:
        for case in ['1','127','128','129','257','sparse','empty']:
            names.append('xestrata-grouped-GU-1-first-diagnostic-root-v1' if (role,case)==('GU','1') else f'xestrata-grouped-{role}-{case}-diagnostic-root-v1')
    for name in names:
        out=B/name; rec=json.loads((out/'record.json').read_text())
        assert rec['passed'] and rec['complete'] and not rec['active'] and rec['boot_unchanged'] and not rec['new_fault_messages']
        for cmd in rec['commands']:
            assert cmd['session_empty'] and cmd['observation_complete'] and not cmd['survivors'] and cmd['direct_child_reaped'] and cmd['exit_code']==0
            for x in cmd['owners']: absent(x)
            for fn,pin in cmd['logs'].items():
                p=out/fn
                assert p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256']
                if p.suffix=='.stderr':
                    text=p.read_text(errors='replace')
                    assert not re.search(r'\berror\b|\bwarning\b|ZE_RESULT_ERROR_',text,re.I)
                    if p.stat().st_size:
                        retained=None
                        if len(excerpts)<1:
                            q=A/'first-api-excerpt.stderr'; q.write_bytes(p.read_bytes()[:32768]); retained=identity(q);excerpts.append(retained)
                        retired.append(dict(**identity(p),allocated_bytes=p.stat().st_blocks*512,reason='Successful API trace; complete numerical/provenance/health/ownership evidence retained; no unresolved consumer',replacement=str(A/name/'record.json'),validation_warning_error_matches=0,representative_excerpt=retained))
                elif p.suffix=='.stdout' and ('health' in fn or fn.startswith(('grouped-','device-first','embedding-state','kernel-before'))):
                    copy(p,A/name/fn)
        copy(out/'record.json',A/name/'record.json')
        summary.append(dict(name=name,original_receipt=identity(out/'record.json'),qualifier=rec.get('qualifier_results'),scope=rec['scope']))
    for name in ['xestrata-grouped-gemm-build-root-v1','xestrata-grouped-gemm-build-root-v2']:
        out=B/name; rec=json.loads((out/'record.json').read_text()); assert not rec['active']
        for cmd in rec['commands']:
            assert cmd['session_empty'] and not cmd['survivors'] and cmd['direct_child_reaped']
            for x in cmd['owners']: absent(x)
        copy(out/'record.json',A/name/'record.json')
        for p in out.glob('*.stderr'): copy(p,A/name/p.name)
        for p in out.glob('*.stdout'): copy(p,A/name/p.name)
    for p in [B/'xestrata-grouped-gemm-qualifier-v1/qualify_grouped.cpp',B/'xestrata-grouped-gemm-qualifier-v1/root_recipe.txt',B/'build_xestrata_grouped_qualifier_v1.py',B/'build_xestrata_grouped_qualifier_v2.py',B/'run_owned_xestrata_device_first_v1.py',B/'run_owned_xestrata_grouped_first_v1.py',B/'run_owned_xestrata_grouped_tail_v1.py',B/'run_owned_xestrata_grouped_matrix_v2.py',B/'xestrata-device-preparation-cpu-failure-v1.json',B/'xestrata-diagnostic-contract-root-v3/run_first_diagnostic_v4.py',B/'xestrata-diagnostic-contract-root-v3/v4-reserve-cpu-contract.json',Path(__file__)]: copy(p,A/'sources'/p.name)
    previous=R/'report-registry-v106.json'; registry=json.loads(previous.read_text())
    new=[]
    for n in range(293,299):
        hits=list(R.glob(f'round{n}-*.txt')); assert len(hits)==1
        p=hits[0]; copy(p,P/p.name)
        new.append(dict(**identity(p),archive_path=str(P/p.name),model='gpt-6-luna',agent='/root/research_bottleneck_evidence_audit_v203' if n in [293,295,298] else '/root/research_native_iq4nl_esimd_k640_v202',status='completed-read-only',root_full_report_reviewed=True,scope=p.stem))
    for p in [R/'implementation-xestrata-grouped-gemm-qualifier-v1.txt',R/'implementation-xestrata-clean64k-comparison-v1.txt',R/'provenance-incidents/round285-working-copy-restoration.json',R/'provenance-incidents/round285-unrequested-overwrite-20261011.txt']: copy(p,P/p.name)
    correction={
      'R294_R297':'Actual compiled generator is sycl/src/program/generate.cpp SHA8bb78a21..., cached object292bbf76...; root mirror extra8guard was not compiled. LOGPOS still does not independently qualify batched candidate prefill plus first forced completion token.',
      'R297_receipt_scope':'Link-only receipt says no runtime at build time. Later current81375binary32K runtime controls PASSED independently. Do not infer never executed from build receipt. Newbinary physical262144 lifecycle remains unqualified.',
      'R298_source_scope':'Xe caller source919c6961 verified; compared D in report is workspace6913f3ab, not actual linked service-only1cf7e911. Root actual service-only confirms explicit env8 gives STAGE8 and routed-only, whereas Xe clamps16 and stream_all. Do not identify alternate D file as linked source.',
      'R295_R298':'No valid-input grouped tail/bound/slot lifetime violation shown. Empirical independent FP64 bound is frozen ordinary-FP32 model assumption, not Intel numerical guarantee. Unchecked copy/event submit bool is a failure-path hole, not observed normal-path fault.',
      'R285_incident':'Unrequested private working-copy overwrite preserved; original committed report remained unchanged and was restored byte-exact. No canonical evidence reclassified.'}
    registry.update(registry_version=107,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),research_completed=registry['research_completed']+len(new),previous_committed_registry=identity(previous),new_reports=new)
    registry['reports']+=new; registry['root_review_corrections_round84']=correction
    registry['live_agent_snapshot']=[dict(agent='/root/research_native_iq4nl_esimd_k640_v202',model='gpt-6-luna',status='running-read-only-R299'),dict(agent='/root/research_bottleneck_evidence_audit_v203',model='gpt-6-luna',status='running-read-only-R300'),dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='running-source-only-first-diagnostic-v5')]
    registry['current_root_decisions']=dict(adopted=False,model_executed=False,small_gpu_qualified=True,checked_grouped_cells=sum(x['qualifier'][-1]['checked'] for x in summary if x['qualifier']),actual_prefill_goal_input=65536,full_physical_context=262144,next='Explicit-env/owned first raw32K model then independently qualified direct clean64K three pairs')
    write(R/'report-registry-v107.json',registry);copy(R/'report-registry-v107.json',P/'report-registry-v107.json')
    write(A/'summary.json',dict(passed=True,model_quality_qualified=False,performance_eligible=False,tests=summary,checked_cells=registry['current_root_decisions']['checked_grouped_cells'],root_corrections=correction))
    (A/'RESULTS_JA.md').write_text('XeStrata 39bdadccの未変更ソースで、Arc B570のgrouped XMX GEMMを14条件検証した。GU/Downそれぞれ1/127/128/129/257行、空expertを含む387行、全expert空を検証。独立FP64参照と3,951,360出力を比較し、全ケースPASS、非有限値・ガード破壊・新規GPU障害0。GU最大絶対誤差6.6121e-5、Down9.1661e-6。モデル品質、速度、262144セルの検証ではない。\n\n小さなallocation selftestと独立H2D/kernel/D2H healthもPASS。qualifierの最初のビルドは相反するFPオプションのWerrorで失敗し、未変更ソースを -fno-fast-math -ffp-contract=off で再ビルド、CPU契約24条件PASS。元の失敗receiptを保持した。\n\n成功APIログは構造化結果・入力/出力ハッシュ・所有プロセスの閉鎖・faultチェックを確認後に削除する。削除は元のreceiptを変更しない。次は未変更forkの32K診断、続いて実入力64Kの独立clean比較。1000tok/s目標は64K、decodeは別に3回。\n')
    (P/'REVIEW_JA.md').write_text('R293–R298を全文確認した。CPU IQ2_S GU NT1の限定dispatchが移植候補。grouped GEMMのproduction shape/boundsは整合し、ring8要求がXeでは16へclampされる差を記録する。LOGPOSはbatched prefillの独立品質検証には不十分。compiled generatorとroot mirrorを区別し、buildのみのreceiptから後続runtimeが未実行とは推論しない。補正はregistry107のroot_review_corrections_round84に記録、原報告は維持した。R299/R300は別scopeで継続。\n')
    # Block credential-value publication without disclosing values or variable names.
    secrets=[v.encode() for k,v in os.environ.items() if re.search(r'TOKEN|SECRET|PASSWORD|API_KEY',k,re.I) and len(v)>=16]
    for root in [A,P]:
        for p in root.rglob('*'):
            if p.is_file(): assert not any(v in p.read_bytes() for v in secrets), 'credential value in prospective archive'
    for entry in retired:
        p=Path(entry['path']); assert identity(p)['sha256']==entry['sha256']; p.unlink(); entry['removed']=True
    manifest=dict(complete=True,original_receipts_unchanged=True,removed=retired,removed_logical_bytes=sum(x['bytes'] for x in retired),removed_file_allocation_bytes=sum(x['allocated_bytes'] for x in retired),physical_space_gain_claimed=False,lock=str(B/'owned-v0141-measurement.lock'))
    write(A/'retention-manifest.json',manifest)
    print(json.dumps(dict(passed=True,checked_cells=registry['current_root_decisions']['checked_grouped_cells'],research_completed=registry['research_completed'],removed_files=len(retired),removed_logical_bytes=manifest['removed_logical_bytes'],archive=str(A))))
