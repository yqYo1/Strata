import fcntl
import importlib.util
import json
import os
import re
from collections import Counter
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
XE=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-census32k-boundary-root-v1';m.require(not out.exists(),'immutable boundary');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),gpu_executed=False,model_executed=False,deleted=[])
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 def run(label,argv,cwd=W):
  e,so,se=owner.run(label,argv,ENV,cwd,wall=60,cpu=120,text_cap=4<<20,total_cap=16<<20,rss_cap=2<<30);m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label);return so.read_text().strip()
 try:
  m.require(not run('initial-status',['/usr/bin/git','status','--porcelain']),'clean report tree')
  publication=json.loads((B/'xe-prefill-route-census-source-commit-v2.json').read_text())
  m.require(publication['passed'] and publication['complete'] and not publication['active'],'qualified source publication')
  m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'],XE)==publication['committed_revision'],'current committed source')
  m.require(not run('source-status',['/usr/bin/git','status','--porcelain'],XE),'clean source')
  roots={}
  for mode in ['free','contrib-llvm']:
   for ver in [15,16]:roots[f'{mode}-release-v{ver}']=f'xe-e32-{mode}-full-build-root-v{ver}'
   for ver in [12,13]:roots[f'{mode}-host-asan-v{ver}']=f'xe-e32-{mode}-asan-full-build-root-v{ver}'
  roots.update({'source-integration':'xe-prefill-route-census-integrate-root-v1','lints-v10-failed':'xe-e32-lints-baseline-compare-root-v10','lints-v11':'xe-e32-lints-baseline-compare-root-v11','census-controller-cpu':'xe-source-llvm-census32k-cpu-root-v3','long-protocol-cpu':'xe-long-prefill-protocol-cpu-root-v1','source-publication':'xe-prefill-route-census-source-publish-root-v2','census32k':'xe-source-llvm-census32k-root-v3/counts'})
  results={}
  for case_name,root in roots.items():
   v=json.loads((B/root/'record.json').read_text());m.require(not v['active'] and all(m.closed(c) for c in v['commands']),'closed '+case_name)
   for c in v['commands']:
    for ident in c['owners']:
     now=m.identity(ident['pid']);m.require(not now or now['start_ticks']!=ident['start_ticks'],'owned identity absent '+case_name)
   if case_name=='lints-v10-failed':m.require(not v['passed'] and not v['complete'] and v['error']=="RuntimeError('no new diagnostic occurrences free')" and len(v['comparisons']['free']['new_findings'])==1,'original new lint finding retained')
   else:m.require(v['passed'] and v['complete'],'actual complete '+case_name)
   results[case_name]=v
  actual=results['census32k'];rows=actual['route_census']['rows'];totals=actual['route_census']['aggregates']
  m.require(actual['model_normal_close'] and not actual['new_fault_messages'] and actual['quiet_reference_comparison']['output_ids_equal'] and actual['quiet_reference_comparison']['selected_logprob_top5_equal'],'normal model with strict selected-output equality and fresh fault-free health')
  m.require(len(rows)==288 and actual['route_census']['capture_valid'],'complete actual route rows')
  m.require(all(m.sha(XE/p)==h for p,h in actual['source_pins'].items()),'source remains exact')
  D=W/'bench/results/2026-10-11-xe-prefill-route-census32k';m.require(not D.exists(),'new evidence archive');D.mkdir()
  copies={}
  def copy(src,name):
   dst=D/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());m.require(m.sha(dst)==m.sha(src),'archive exact bytes');copies[str(dst.relative_to(W))]=dict(original_path=str(src),bytes=src.stat().st_size,sha256=m.sha(src))
  for case_name,root in roots.items():copy(B/root/'record.json',case_name+'/record.json')
  for name in ['xe-prefill-route-census-applied-main-review-v1.json','xe-prefill-route-census-applied-main-review-v2.json','xe-census32k-controller-main-review-v1.json','xe-census32k-controller-main-review-v2.json','xe-census32k-controller-main-review-v3.json','xe-long-prefill-protocol-main-review-v1.json','xe-native-dequant-group-api-main-review-v1.json','xe-census32k-agent-snapshot-v1.json','xe-census32k-agent-snapshot-v2.json','xe-prefill-route-census-source-commit-v2.json']:
   copy(B/name,name)
  reports=['implementation-xe-long-prefill-protocol-v1.txt','implementation-xe-native-dequant-group-api-v1.txt']
  for name in reports:copy(B/'research-20261009'/name,name)
  for folder in ['xe-long-prefill-protocol-source-v1','xe-native-dequant-group-api-source-v1']:
   for p in (B/folder).rglob('*'):
    if p.is_file():copy(p,'agent-source/'+folder+'/'+str(p.relative_to(B/folder)))
  for folder in ['xe-e32-full-build-v15','xe-e32-full-build-v16','xe-e32-host-asan-build-v12','xe-e32-host-asan-build-v13','xe-e32-lints-baseline-compare-v10','xe-e32-lints-baseline-compare-v11','xe-source-llvm-census32k-v1','xe-source-llvm-census32k-v2','xe-source-llvm-census32k-v3']:
   for p in (B/folder).glob('*.py'):copy(p,'recipes/'+folder+'/'+p.name)
  for name in ['xe-prefill-route-census-integrate-v1.py','xe-prefill-route-census-source-publish-v1.py','xe-prefill-route-census-source-publish-v2.py','xe-source-llvm-census32k-cpu-v2.py','xe-source-llvm-census32k-cpu-v3.py','xe-long-prefill-protocol-cpu-v1.py','xe-census32k-boundary-v1.py']:
   copy(B/name,'recipes/'+name)
  for mode in ['free','contrib-llvm']:copy(B/f'xe-e32-{mode}-full-build-root-v16/source.patch',mode+'-release-v16/source.patch')
  model_so=(B/roots['census32k']/'commands/model.stdout').read_text().splitlines();m.require(model_so==actual['startup']+actual['request_protocol'],'full successful model stdout represented exactly')
  model_se=(B/roots['census32k']/'commands/model.stderr').read_text().splitlines()
  keep=[x for x in model_se if x.startswith(('strata generate:','strata mtp:','strata verify:','strata serve:','strata:')) or re.search(r'\b(?:WARN|WARNING|ERROR|FATAL)\b|GPU completion|terminate called',x)]
  (D/'census32k/retained-stderr.txt').write_text('\n'.join(keep)+'\n')
  # Persist useful non-model warnings and refused CLI evidence, before raw stream deletion.
  diagnostics={}
  for case_name,root in roots.items():
   for p in (B/root/'commands').glob('*'):
    if p.name.startswith('model.'):continue
    if p.suffix=='.stderr' and p.stat().st_size:
     text=p.read_text();m.require(len(text)<2<<20,'compact diagnostic budget');diagnostics[str(p)]=text
    elif p.suffix=='.stdout':
     lines=p.read_text().splitlines();windows=[]
     for i,line in enumerate(lines):
      if re.search(r'\b(?:warning|error):',line):windows.extend(lines[max(0,i-1):min(len(lines),i+4)])
     if windows:diagnostics[str(p)]='\n'.join(windows)+'\n'
  (D/'retained-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
  branch_counts=Counter(tuple(row[k] for k in ['mmq_enabled','plan_fallback','mmq_direct','group_gather']) for row in rows)
  transfers={(x['offset'],x['layer']):x for x in actual['transfer_ledger']['rows']}
  unused_full=[dict(offset=x['offset'],layer=x['layer'],zero_nonresident=x['zero_nonresident'],blob_bytes=transfers[x['offset'],x['layer']]['blob_bytes'],logical_unused_bytes=x['zero_nonresident']*transfers[x['offset'],x['layer']]['blob_bytes']) for x in rows if x['mode']=='full_nonresident_stream']
  m.require(all(transfers[x['offset'],x['layer']]['calls']==509 and x['active_nonresident']+x['zero_nonresident']==509 for x in rows if x['mode']=='full_nonresident_stream'),'exact all-nonresident producer extent and independent zero-route count')
  unused_total=sum(x['logical_unused_bytes'] for x in unused_full)
  unused=dict(scope='Full-nonresident stream copies whose same-layer/chunk actual route count is zero. Logical request extents; not physical traffic or avoidable exclusive service.',calls=sum(x['zero_nonresident'] for x in unused_full),bytes=unused_total,fraction_of_all_expert_h2d_bytes=unused_total/actual['transfer_ledger']['total']['bytes'],per_layer_chunk=unused_full)
  summary=dict(source_revision=actual['candidate_revision'],boot_id=actual['boot_id'],input_sha256=actual['input']['sha256'],input_tokens=32768,actual_batched_positions=32767,chunk=6144,rows=288,diagnostic_request_timing=actual['diagnostic_request_timing'],route_aggregates=totals,branch_layer_chunk_occurrences=[dict(mmq_enabled=k[0],plan_fallback=k[1],mmq_direct=k[2],group_gather=k[3],occurrences=v) for k,v in sorted(branch_counts.items())],transfer_total=actual['transfer_ledger']['total'],ready_free_mib=int(actual['effective_settings']['vram_free_mib']),strict_selected16_ids_lp_top5_equal=True,fresh_before_after_health=True,new_faults=[],original_failed_lint_v10_preserved=True,all_mode_release_host_asan_and_mandatory_lints_passed=True,general_protocol_cpu=results['long-protocol-cpu']['fixtures'],performance_adopted=False,measurement_scope='Actual host routing counts, successful wrapper returns and completed logical H2D extents. No exclusive GPU component service, physical bandwidth, attribution of elapsed time or optimal settings claim.',cold_decode_scope='Single independent process, first16 outputs including initial graph capture. No warm/RESTORE/variance selection claim.',pending_group_api='Main fully reviewed source-only candidate. Neither integrated, built, GPU byte-parity qualified nor adopted.',open_requirements=['Measure actual native dequant/group component service and submission overhead after byte parity','Quiet6144/64K independent prefill and decode measurements','Prefill-only VRAM lifetime/recreate and dtype quality','Actual full262144/RESTORE','Xe pristine evaluation then latest official required feature integration','Official independent parameter tuning and three independent decode validation runs per committed TODO'])
  summary['logical_unused_expert_h2d']=unused
  summary['next_priority']='Actual FP16 branch GU/Down normal wrapper calls273692; qualify grouped dequant bytes and component service before integration. MMQ gather probe deferred for this measured configuration. Unused full-stream extent alone is a smaller measured subset, without assumed elapsed-time gain.'
  (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  (D/'RESULTS_JA.md').write_text('32K・6144幅の診断実測で、処理分岐とexpert数を全48layer・6chunk・288行で確認した。計算・転送・同期処理には変更を加えず、消費側のhost集計だけを追加した。通常ビルドとhost ASan／UBSanをfree/contrib-llvmの両方で通し、実際の必須lintを変更前と比較した。初回lintの整数cast指摘1件は診断出力の加算前castに修正した。元のv10 FAILEDは保持し、修正後v11は両方追加指摘ゼロ。\n\n'+f"診断付きprefillは{actual['diagnostic_request_timing']['prefill_tps']:.2f} tok/s、初回16token decodeは{actual['diagnostic_request_timing']['cold_decode_tps']:.2f} tok/s。採用判断用の静かな反復測定ではない。MMQ/FP16等の実分岐と全counterは[summary.json](summary.json)。成功wrapperの回数と転送要求extentであり、exclusive GPU serviceや物理PCIe帯域の測定ではない。\n\n"+f"全288行でMMQ=0、fallback=1で、GU/Down展開の成功wrapper呼び出しは各{totals['native_dequant_gu_calls']:,}回だった。グループgather呼び出しは0回。未使用expertへのfull-stream転送要求は{unused['calls']:,}件・{unused['bytes']:,} byte（全expert転送要求量の{100*unused['fraction_of_all_expert_h2d_bytes']:.2f}%）である。まず実際に使われたFP16展開をまとめる候補をbyte単位で検証し、component serviceを測る。MMQのgather最適化はこの設定の実測経路には寄与しない。\n\n"+'16token ID・selected logprob・top5は従来の同入力基準と完全一致した。normal QUIT、実runtimeの照合、前後のfresh GPU health、新faultなしを確認した。256Kは設定値であり、実際の全262144tokenやRESTORE、広い能力評価は未完了。\n\n'+'別の実装エージェントから、最大16expertのGU/Down展開APIと、32K/64Kなどの可変chunk protocolをファイルで受領し全文・hashを確認した。前者は未適用・未build・未GPU検証である。後者はMainのCPU12正常／97拒否とtrace12正常／8拒否を通り、閉じた実6144入力captureも検証した。zero-output fixtureは明示された仮の契約であり、実256Kで0出力になることを証明しない。\n\n'+'本家単体の設定最適化は[追跡TODO](../2026-10-11-upstream-parameter-audit/TODO.md)で未完了を維持する。今後のprefill/decode採用は個別に決め、decodeは独立反復でばらつきを確認する。\n')
  previous=W/'bench/results/2026-10-11-xe-chunk6144-and-ring96/report-registry-v126.json';m.require(json.loads(previous.read_text())['research_completed']==500,'exact previous registry count')
  registry=dict(registry_version=127,research_completed=502,created_utc=m.utc(),registry_format='incremental; follow previous_committed_registry',previous_committed_registry=dict(path=str(previous.relative_to(W)),sha256=m.sha(previous),reports=500),new_reports=reports,reports=[dict(path=str(B/'research-20261009'/name),archive=str((D/name).relative_to(W)),sha256=m.sha(B/'research-20261009'/name),bytes=(B/'research-20261009'/name).stat().st_size,model='gpt-6.1-sol',read_only=False,qualification='Complete Main source/report/manifest review; grouped API remains unapplied and unqualified. General protocol actual Main CPU12/97 and trace12/8 plus completed6144 protocol PASS.') for name in reports],live_agent_snapshot=json.loads((B/'xe-census32k-agent-snapshot-v2.json').read_text()),dependency='Selected gpt-6-luna research quota until Oct14 2026 12:33PM; no substitute. Concrete implementation agents do not count as replacement research.',next_research_scopes=['Actual branch counts and component service','Official independent parameter tuning'])
  (D/'report-registry-v127.json').write_text(json.dumps(registry,indent=2)+'\n')
  plans=[]
  for case_name,root in roots.items():
   for p in (B/root/'commands').glob('*'):
    if p.suffix not in ('.stdout','.stderr'):continue
    st=p.stat();plans.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=m.sha(p),retained_receipt=str((D/case_name/'record.json').relative_to(W)),reason='Closed repeated build/lint/test/API/journal protocol. Exact commands/source/individual status/owners and actual rows, IDs/LP/progress/runtime/fault results retained; useful errors/warnings preserved. Resolved v10 lint remains FAILED. No named complete-history consumer.'))
  capture=B/roots['long-protocol-cpu']/'actual-protocol.txt';st=capture.stat();m.require(m.sha(capture)==results['long-protocol-cpu']['capture_sha256'],'closed duplicate actual protocol unchanged');plans.append(dict(path=str(capture),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=m.sha(capture),retained_receipt='bench/results/2026-10-11-xe-chunk6144-and-ring96/chunk6144-v3/record.json',reason='Exact duplicated completed request protocol is retained in original committed record; current protocol validation and explicit expectations retained separately.'))
  copy(B/roots['long-protocol-cpu']/'expectations.json','long-protocol-cpu/expectations.json')
  (D/'archive-manifest.json').write_text(json.dumps(dict(byte_identical_copies=copies,original_statuses_preserved=True,current_external_consumers={'group_api':'Main serial full-byte/guard parity and component qualification','general_protocol':'Main future explicit32K/64K/full-context controller','gather_probe':'GPU qualification deferred (MMQ inactive); active grouped-dequant probe implementation reads this existing source as a Runtime/drain convention guide. Review again when that concrete report returns.'},retention_review='All raw deletion plans only cover closed command streams and one duplicate request text. Active/source/model/tensor fixtures are not deleted; no raw full-history consumer remains.'),indent=2)+'\n')
  m.require(sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<12<<20,'compact archive12MiB')
  run('secret-check',[str(XE/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(D)])
  run('diff-check',['/usr/bin/git','diff','--check']);run('stage',['/usr/bin/git','add','--',str(D.relative_to(W))]);run('commit',['/usr/bin/git','commit','-m','docs(bench): qualify prefill route census and record actual branches']);r['evidence_revision']=run('evidence-head',['/usr/bin/git','rev-parse','HEAD'])
  manifest=D/'retention-v1.json';manifest.write_text(json.dumps(dict(retained_commit=r['evidence_revision'],deletions=plans,logical_bytes=sum(x['bytes'] for x in plans),allocated_file_bytes=sum(x['allocated_bytes'] for x in plans),does_not_measure_available_space=True,original_statuses_unchanged=True),indent=2)+'\n')
  run('retention-secret-check',[str(XE/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(manifest)])
  for x in plans:
   p=Path(x['path']);m.require(m.sha(p)==x['sha256'] and p.stat().st_size==x['bytes'],'unchanged closed stream');p.unlink();r['deleted'].append(x['path'])
  m.require(all(m.sha(B/root/'record.json')==copies[str((D/case_name/'record.json').relative_to(W))]['sha256'] for case_name,root in roots.items()),'original receipts unchanged after retention')
  run('stage-retention',['/usr/bin/git','add','--',str(manifest.relative_to(W))]);run('commit-retention',['/usr/bin/git','commit','-m','docs(bench): retire closed census qualification streams']);r['committed_revision']=run('final-head',['/usr/bin/git','rev-parse','HEAD']);run('publish',['/usr/bin/git','push','origin','HEAD']);m.require(not run('final-status',['/usr/bin/git','status','--porcelain']),'clean report tree');r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:
  r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','evidence_revision','committed_revision']}));raise SystemExit(0 if r['passed'] else 1)
