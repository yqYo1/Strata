import datetime,fcntl,hashlib,json,os,re,shutil
from pathlib import Path
B=Path(__file__).parent;R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-11-clean64k-and-xestrata-admission';P=W/'bench/results/2026-10-11-parallel-round85'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def copy(p,q):
 p=Path(p);q.parent.mkdir(parents=True,exist_ok=True);assert p.stat().st_size<4<<20;shutil.copyfile(p,q);assert sha(p)==sha(q)
def absent(x):
 if not x:return
 try:
  v=Path('/proc',str(x['pid']),'stat').read_text();assert int(v[v.rfind(')')+2:].split()[19])!=x['start_ticks'],'owned PID still present'
 except FileNotFoundError:pass
with (B/'owned-v0141-measurement.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not A.exists()and not P.exists();A.mkdir();P.mkdir()
 retired=[];runs=[]
 names=['owned-clean64k-v3-baseline-r1','owned-xestrata39-first32k-diagnostic-v6-r1','owned-xestrata39-hostusm32k-diagnostic-v7-r1','owned-xestrata39-hostusm4k-code32k-diagnostic-v8-r1']
 for name in names:
  out=B/name;d=json.loads((out/'record.json').read_text());assert d['active']is False
  if name.startswith('owned-clean'):
   assert d['completed']and d['timing_valid']and d['health_after_passed']and d['boot_unchanged']and d['new_fault_messages']==[]
   assert d['model_result']==dict(exit_code=0,owned_closed=True)
   calls=d['commands']
  else:
   assert d['completed']is False and d['diagnostic_passed']is False and d['owned_gdb_closed']and d['auxiliary_ownership_closed']
   for key in ['inferior','debugger','actual_inferior']:absent(d.get(key))
   assert not d['cleanup']['inferior_survived']and not d['cleanup']['gdb_survived'];calls=d['health_calls']
  for c in calls:
   assert c['session_empty']and c['observation_complete']and not c.get('errors')
   for x in c.get('owners',[]):absent(x)
  copy(out/'record.json',A/name/'record.json')
  journal_files=list(out.rglob('*kernel*stdout'))
  journals=[]
  for p in journal_files:
   entries=[];cursor_lines=[]
   for line in p.read_text(errors='strict').splitlines():
    if line.startswith('{'):entries.append(json.loads(line))
    else:cursor_lines.append(line)
   relevant=[e for e in entries if 'xe 0000:05:00.0' in e.get('MESSAGE','')]
   journals.append(dict(original=pin(p),total_entries=len(entries),filter='MESSAGE contains exact xe 0000:05:00.0; retain complete entry',relevant=relevant,cursor_lines=cursor_lines))
  write(A/name/'kernel-evidence.json',journals)
  for p in out.rglob('*'):
   if not p.is_file()or p.name=='record.json':continue
   rel=p.relative_to(out)
   success_api=p.name=='health.stderr'or(re.fullmatch(r'health-(before|after)-\d+-health.stderr',p.name)is not None)
   duplicate_mi=p.name=='gdb-mi.stdout'and name.endswith(('v7-r1','v8-r1'))
   unneeded_reserve_stack=p.name=='failure.mi.txt'and name.endswith('v7-r1')
   if success_api:
    text=p.read_text(errors='strict');assert not re.search(r'\berror\b|\bwarning\b|ZE_RESULT_ERROR_',text,re.I)
   if success_api or duplicate_mi or unneeded_reserve_stack or p in journal_files:
    reason='successful repeated GPU-health API trace; exact health verdict/environment/source/closure retained'if success_api else 'duplicated debugger transcript; distinct failure snapshot retained for v8; v7 is preGEN reserve refusal'if duplicate_mi or unneeded_reserve_stack else 'kernel interval compacted to complete BDF-relevant entries and exact boundary/hash; unrelated firewall entries retired'
    retired.append(dict(**pin(p),allocated_bytes=p.stat().st_blocks*512,reason=reason,replacement=str(A/name/'record.json'if success_api else A/name/'kernel-evidence.json'if p in journal_files else A/name/'debugger/failure.mi.txt'if name.endswith('v8-r1')else A/name/'record.json')))
   else:copy(p,A/name/rel)
  runs.append(dict(name=name,original_receipt=pin(out/'record.json'),original_result_unchanged=True,completed=d['completed'],timing_valid=d.get('timing_valid',False),new_fault_messages=d['new_fault_messages'],scope=d.get('scope',d.get('source_scope'))))
 for n in ['xestrata-diagnostic-v5-cpu-root-v1','xestrata-diagnostic-v6-cpu-root-v2','xestrata-diagnostic-v7-cpu-root-v3','xestrata-diagnostic-v8-cpu-root-v4','xestrata-clean64k-owner-contract-root-v3','xestrata-clean64k-protocol-contract-root-v3','xestrata-clean64k-v4-protocol-root-v1','xestrata-clean64k-v4-owner-root-v1']:
  out=B/n;d=json.loads((out/'record.json').read_text());assert d['active']is False
  for p in out.rglob('*'):
   if p.is_file()and '__pycache__'not in str(p):copy(p,A/'cpu'/n/p.relative_to(out))
 for n in ['xestrata-clean-64k-comparison-v1','xestrata-clean-64k-comparison-v2','xestrata-clean-64k-comparison-v3','xestrata-clean-64k-comparison-v4']:
  for p in (B/n).iterdir():
   if p.is_file():copy(p,A/'sources'/n/p.name)
 for p in (B/'xestrata-diagnostic-contract-root-v3').iterdir():
  if p.is_file()and p.suffix in ['.py','.json']:copy(p,A/'sources'/'xestrata-diagnostic-contract-root-v3'/p.name)
 for n in ['prepare_xestrata_clean64k_v3.py','qualify_xestrata_clean64k_v3_cpu.py','prepare_xestrata_diagnostic_v8.py','qualify_xestrata_diagnostic_v5_cpu.py','qualify_xestrata_diagnostic_v6_cpu.py','qualify_xestrata_diagnostic_v7_cpu.py','qualify_xestrata_diagnostic_v8_cpu.py','qualify_clean64k_v4_root.py']:
  copy(B/n,A/'sources'/n)
 copy(Path(__file__),A/'sources'/Path(__file__).name)
 registry_path=R/'report-registry-v107.json';reg=json.loads(registry_path.read_text());new=[]
 for n in range(299,309):
  hits=list(R.glob(f'round{n}-*.txt'));assert len(hits)==1;p=hits[0];copy(p,P/p.name)
  new.append(dict(**pin(p),archive_path=str(P/p.name),model='gpt-6-luna',agent='/root/research_bottleneck_evidence_audit_v203'if n in [300,302,303,305,306]else '/root/research_native_iq4nl_esimd_k640_v202',status='completed-read-only',root_full_report_reviewed=True,scope=p.stem))
 for n in ['implementation-xestrata-clean64k-comparison-v2.txt','implementation-xestrata-first-diagnostic-env-owner-v5.txt','implementation-xestrata-clean64k-chunks-v4.txt']:
  copy(R/n,P/n)
 correction=dict(
  R300_terminal='Terminal prompt token is processed before DONE, after prompt_ms closes; denominator uses original engine n=65536 convention though batched positions=65535.',
  R301_runtime='Build-only unexecuted scope is time-of-build; later exact81375binary32K math controls passed. FP16 Dm alone yields zero currentshared-region saving under PLEmax; no quality/adoption conclusion.',
  R303_compiled='Compared aa58 workspace mirror is not current81375compiled generator8bb78/object292bb. Xe source triage stands; reject that D runtime attribution.',
  R304_runtime='Exact compute-runtime39395.14 public source was located in R271; source unavailable claim rejected. Installed-library equivalence remains unqualified; hostwrites-after-prepare are publicly allowed.',
  R306_placement='V7actual4204=>3923MiB;3944 was erroneous root setup wording. Do not force identical filledslots29vs128 when comparing unchanged whole engines. V7ready50 below512 has no GEN.',
  R305_versions='Previously reviewed private SHA567fafc603f34b04151cfbef895a078c9b49bdc8254d299c79fcf3683a8e31c9 was subsequently edited by reporter; latest74709a91c26de01460d89e23f9815a4b049180077a50932aa043912a15c56d18 fully reread. Reported intermediate aac9bb... unavailable. No byte-equivalence or original recovery claimed; freeze current archive. R309 was explicitly reassigned after oldR305 completion notification.',
  original_status='V5CPU original PASS was cleanup-only misclassification; separatefailure-classification marks overallFAIL exactGDBenv mismatch; preserve original. Baseline original eventledgerFAILED invalid101timestamps remains unchanged; separate math-only admission does not qualify event timing.',
  new_gpu_fault='V8actualREADY1503 andPP4096 then secondchunkSIGABRT, UR40, GPUaddressfault/CAT18/CCSreset. No fork performance/adoption. Throwstackdequant is not proven faultingkernel.'
 )
 reg.update(registry_version=108,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),research_completed=reg['research_completed']+len(new),previous_committed_registry=pin(registry_path),new_reports=new)
 reg['reports']+=new;reg['root_review_corrections_round85']=correction
 reg['live_agent_snapshot']=[dict(agent='/root/research_bottleneck_evidence_audit_v203',model='gpt-6-luna',status='running-read-only-R309-reassigned'),dict(agent='/root/research_native_iq4nl_esimd_k640_v202',model='gpt-6-luna',status='running-read-only-R310'),dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='running-source-only-taskfixture-v1')]
 baseline=json.loads((B/names[0]/'record.json').read_text())
 sample=baseline['sample'];summary=dict(runs=runs,hardware='Ryzen5 5600X /128GB RAM /ArcB57010GB; IntelLLVM2026.1.1, compute-runtime1.17.39395+14',actual_goal_input=65536,minimum_comparison_input=32768,prefill_goal_tps=1000,decode_goal_tps=70,baseline64K_sample=sample,prefill_performance_eligible=True,decode_performance_eligible=False,reason='3IDs honeststop is too short for decode comparison',adopted=False,full262144_newbinary_qualified=False,corrections=correction,additional_GPU_work='stopped after v8newfault; no reset/rebind/service changes; failure sourceaudit pending',next='restore documented task/chat fixture boundary; mainCPUchecks first; independent integerhealth before any knownbaselineGPU continuation; no rawXe faultpath retry untilqualified')
 write(A/'summary.json',summary);reg['current_root_decisions']=summary;write(R/'report-registry-v108.json',reg);copy(R/'report-registry-v108.json',P/'report-registry-v108.json')
 (A/'RESULTS_JA.md').write_text('prefill1000tok/sの評価入力は実65536トークン（約64K）を認める。短い固定費支配の入力は比較に使わず、32K以上。同じ入力ID・設定・測定区間で各armを比べる。モデル読み込みはprefillと分ける。decodeは別に十分な生成長で3回以上測り、個別値を残す。最大262144設定だけでは境界検証にならず、新候補採用前に物理262144セルを使い切る状態/restore検証が必要。\n\n現実装81375..の実65536入力・8Kchunk・freshprocess（再利用0）ではprefill162402.9ms、403.53959196541444tok/s。ロードからREADY44.595850970014秒は別枠。DONE定義の入力65536を分子とする。batched65535位置、最後のprompt tokenはprompt timer後に処理。生成3IDsでstop、decode813.4ms、MTP0/6。短いdecodeは速度比較から除外し、prefillのみ有効。現時点は1標本で1000目標未達、繰り返し比較未完了。新規障害0、正常exit0、全所有process閉鎖。\n\n未変更XeStrata39bdadcc/xe0.1.40.2.1（ghq取得、IntelLLVM2026.1.1でビルド）はデフォルトmmap-importでcache slot84 byte0不一致、通常exit1、GENなし。ソース対応hostUSMへ切り替えるとcache verifierは成功するが8KではREADYfree50MiBで512MiBguardがGENを拒否。4KではREADYfree1503MiB、PP4096完了後、第二chunkでGPU addressfault/CAT18/CCSresetとSIGABRT。後者はGPU障害で、単なるVRAM容量不足という推定をしない。throw位置のdequant kernelとfault発生kernelを同一と断定しない。失敗ログ/stackとfreshkernel entriesを保持。未変更forkの速度比較は未完了、移植なし、追加GPU測定停止。\n\nclean64K controllerv4はcommon4K/8K対応とprefill/decode適格性分離を追加。mainCPUprotocol/gate/actualfailure拒否とexactowner5casesPASS。モデル未実行。baseline元invalid-eventledgerFAILEDは不変、別math-only admissionを使いGPUeventtimingを合格扱いしない。次はtask/chat末尾を復元した64K fixtureのCPU確認とfaultsourceaudit。成功healthAPI/重複MI/無関係kernelnoiseはmanifest確認後退役。\n')
 (P/'REVIEW_JA.md').write_text('R299–R308を全文確認した。実64Kの基準を記録し、3IDstopのdecodeを除外。低精度の隣接モデル資料には方式次第で小さな能力劣化と大きな劣化の両方があり、単純BF16再帰castの採用根拠にはならない。Dm16のみは現PLEmax下でregion削減0、alias同時なら条件付き削減。R307のNT1CPU候補はproductionNT/type/owner分母未測定で保留。R305の後続private版は別hashとして記録し現版全文読了、過去版byte同一とは主張しない。root補正と各原報告のscopeをregistry108へ保存。R309faultaudit/R310capacityscale、Soltaskfixtureは並行継続、mainはCPU契約と証拠保存を実施。\n')
 secrets=[v.encode()for k,v in os.environ.items()if re.search(r'TOKEN|SECRET|PASSWORD|API_KEY',k,re.I)and len(v)>=16]
 for root in [A,P]:
  for p in root.rglob('*'):
   if p.is_file():assert not any(v in p.read_bytes()for v in secrets),'credential value in prospective archive'
 for e in retired:
  p=Path(e['path']);assert sha(p)==e['sha256']and p.stat().st_size==e['bytes'];p.unlink();e['removed']=True
 manifest=dict(complete=True,original_receipts_unchanged=True,removed=retired,removed_logical_bytes=sum(x['bytes']for x in retired),removed_file_allocation_bytes=sum(x['allocated_bytes']for x in retired),physical_space_gain_claimed=False,retained_failure_consumers=dict(default_mmap='R303/R304 plus smallarena qualifier origin stillunlocalized',v8='R309 allocation/lifetime/secondchunk fault localization; revisit afterminimalqualifiedreproducer'),lock=str(B/'owned-v0141-measurement.lock'))
 write(A/'retention-manifest.json',manifest)
 print(json.dumps(dict(passed=True,archive=str(A),reports=len(new),research_completed=reg['research_completed'],retired_files=len(retired),retired_logical_bytes=manifest['removed_logical_bytes'])))
