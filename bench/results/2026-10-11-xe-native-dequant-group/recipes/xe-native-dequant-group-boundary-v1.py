import ast,fcntl,importlib.util,json,os
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009');X=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');D=W/'bench/results/2026-10-11-xe-native-dequant-group';O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-native-dequant-group-boundary-root-v1';m.require(not out.exists(),'immutable boundary');out.mkdir();r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),gpu_executed=False,model_executed=False,deleted=[])
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 def run(label,args,cwd=W):
  e,so,se=owner.run(label,args,ENV,cwd,wall=60,cpu=120,text_cap=4<<20,total_cap=16<<20,rss_cap=2<<30);m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label);return so.read_text().strip()
 try:
  m.require(not run('report-status',['/usr/bin/git','status','--porcelain']),'clean report branch');m.require(not D.exists(),'new archive directory');D.mkdir()
  roots={'api-integration':'xe-native-dequant-group-api-integrate-root-v1','lints':'xe-e32-lints-baseline-compare-root-v12','probe-build-original-free':'xe-native-dequant-group-byte-build-root-v1-free','gpu-smoke':'xe-native-dequant-group-byte-gpu-root-v3-free-smoke','gpu-full':'xe-native-dequant-group-byte-gpu-root-v3-free-full','source-publication':'xe-native-dequant-group-source-publish-root-v1'}
  for mode in ['free','contrib-llvm']:
   roots[mode+'-release']='xe-e32-'+mode+'-full-build-root-v17';roots[mode+'-host-asan']='xe-e32-'+mode+'-asan-full-build-root-v14';roots[mode+'-probe-build']='xe-native-dequant-group-byte-build-root-v2-'+mode;roots[mode+'-component-build']='xe-native-dequant-group-component-build-root-v1-'+mode
  for phase in ['smoke-host','full-host','smoke-tags','full-tags']:roots['component-'+phase]='xe-native-dequant-group-component-gpu-root-v1-free-'+phase
  actual={};copies={};plans=[]
  def copy(src,name):
   dst=D/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());m.require(m.sha(dst)==m.sha(src),'exact archive');copies[str(dst.relative_to(W))]=dict(original_path=str(src),bytes=src.stat().st_size,sha256=m.sha(src));return dst
  for name,root in roots.items():
   p=B/root/'record.json';v=json.loads(p.read_text());m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'closed '+name)
   for c in v['commands']:
    for ident in c['owners']:
     now=m.identity(ident['pid']);m.require(not now or now['start_ticks']!=ident['start_ticks'],'closed owner '+name)
   actual[name]=v;copy(p,name+'/record.json')
   if v.get('byte_comparison_ledger'):copy(Path(v['byte_comparison_ledger']['path']),name+'/byte-comparison-ledger.txt')
   if (B/root/'source.patch').exists():copy(B/root/'source.patch',name+'/source.patch')
   for p in (B/root/'commands').glob('*'):
    if p.suffix not in ['.stdout','.stderr']:continue
    st=p.stat();plans.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=m.sha(p),retained_receipt=str((D/name/'record.json').relative_to(W)),reason='Closed successful full build/lint/test/API/drain/copy/journal traces. Exact commands/env/process/source/binary status, full case-identity byte ledger or all individual component samples, runtime/health/fault outcomes retained and verified. No unresolved complete-history consumer.'))
  pub=actual['source-publication'];m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'],X)==pub['committed_revision'] and not run('source-status',['/usr/bin/git','status','--porcelain'],X),'clean published source');m.require(all(m.sha(X/f)==h for f,h in pub['source_pins'].items()),'exact published source')
  for name in ['xe-native-dequant-group-api-main-review-v1.json','xe-native-dequant-group-api-applied-main-review-v1.json','xe-native-dequant-group-api-main-review-v2.json','xe-native-dequant-group-byte-probe-main-review-v1.json','xe-native-dequant-prefill-batch-main-review-v1.json','xe-native-dequant-group-component-main-review-v1.json','xe-native-dequant-group-byte-main-controllers-review-v1.json','xe-native-dequant-group-byte-main-controllers-review-v2.json','xe-native-dequant-group-byte-main-controllers-review-v3.json','xe-native-dequant-group-component-main-controllers-review-v1.json','xe-native-dequant-group-byte-qualification-plan-v1.json','xe-native-dequant-group-byte-mode-identity-v1.json','xe-native-dequant-group-final-mode-reuse-v1.json','xe-native-dequant-group-component-analysis-v1.json']:copy(B/name,name)
  recipe_paths=[B/'xe-native-dequant-group-api-integrate-v1.py',B/'xe-native-dequant-group-byte-build-v1.py',B/'xe-native-dequant-group-byte-build-v2.py',B/'xe-native-dequant-group-byte-gpu-v1.py',B/'xe-native-dequant-group-byte-gpu-v2.py',B/'xe-native-dequant-group-byte-gpu-v3.py',B/'xe-native-dequant-group-component-build-v1.py',B/'xe-native-dequant-group-component-gpu-v1.py',B/'xe-native-dequant-group-source-publish-v1.py',Path(__file__),B/'xe-e32-full-build-v17/run_root.py',B/'xe-e32-host-asan-build-v14/run_root.py',B/'xe-e32-lints-baseline-compare-v12/run_root.py']
  for p in recipe_paths:
   ast.parse(p.read_text(),filename=str(p));name=p.name if p.name!='run_root.py' else p.parent.name+'-run_root.py';copy(p,'recipes/'+name)
  for folder in ['xe-native-dequant-group-api-source-v1','xe-native-dequant-group-api-source-v2','xe-native-dequant-group-byte-probe-source-v1','xe-native-dequant-group-byte-probe-source-v2','xe-native-dequant-group-byte-probe-source-v3','xe-native-dequant-prefill-batch-source-v1','xe-native-dequant-group-component-source-v1','xe-native-dequant-group-component-source-v2']:
   for p in sorted((B/folder).rglob('*')):
    if p.is_file():copy(p,'sources/'+folder+'/'+str(p.relative_to(B/folder)))
  reports=['implementation-xe-native-dequant-group-api-v2.txt','implementation-xe-native-dequant-group-byte-probe-v1.txt','implementation-xe-native-dequant-prefill-batch-v1.txt','implementation-xe-native-dequant-group-component-v1.txt']
  entries=[]
  for name in reports:
   p=B/'research-20261009'/name;copy(p,name);entries.append(dict(path=str(p),archive=str((D/name).relative_to(W)),sha256=m.sha(p),bytes=p.stat().st_size,model='gpt-6.1-sol',read_only=False,qualification='Full Main source/report/manifest review completed. All original deliveries unexecuted source only; actual Main API full-byte/guards and component results recorded separately, caller integration remains unapplied. Original tag-guard defect recorded and corrected in Main v2.'))
  prev=W/'bench/results/2026-10-11-xe-prefill-route-census32k/report-registry-v127.json';registry=dict(registry_version=128,research_completed=506,created_utc=m.utc(),registry_format='incremental; follow previous_committed_registry',previous_committed_registry=dict(path=str(prev.relative_to(W)),sha256=m.sha(prev),reports=502),new_reports=reports,reports=entries,dependency='gpt-6-luna research quota until Oct14 2026 12:33PM; no alternative research model. Concrete implementation agents do not substitute for research.',next_research_scopes=['Actual component versus end-to-end phase attribution','Official independent parameter tuning'],live_agent_snapshot={'root_status':'running','implementation_agents':'All4 new concrete source-only reports returned and reviewed; byte/component tests serialized by Main; minimal APIv2 source only pending generated-code qualification','research_agents':'No running research agents due selected Luna quota','main_current':'Optional APIv1 all18 full-byte PASS but same-work component severe regression: REJECT production caller adoption; minimal pointer-lookup APIv2 source only pending optimized IR and fresh byte/service qualification'})
  (D/'report-registry-v128.json').write_text(json.dumps(registry,indent=2)+'\n')
  summary=dict(source_revision=pub['committed_revision'],prefill_caller_batch_applied=False,performance_adopted=False,model32k_newly_measured=False,all18_coverage=actual['gpu-full']['coverage'],all18_qualification_passed=True,all18_gpu_executed_mode='free',same_probe_bytes_built_both_modes=True,component_analysis=json.loads((B/'xe-native-dequant-group-component-analysis-v1.json').read_text()),official_parameter_search='Audit done, default-only false, independent official best-setting search and validation OPEN',next=['Reject v1 component adoption; inspect generated device code, qualify minimal constant-index APIv2 before considering caller batching','32K strict token/logprob/H2D validation before clean performance comparison','Continue independent official pristine tuning and updates after Xe qualification','Full physical262144/RESTORE and broad quality remain open'])
  (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  doc='''# Native expert FP16展開をまとめる候補の検証

本家設定のTODO監査は完了。過去測定は箱出しだけではないが、本家単体の最適設定探索・独立検証は未完了。[TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を継続する。

実装元はb3e1de06。追加した任意API以外の既存関数は元のbyteのまま。free・contrib-llvmのrelease/CPUテストとhost ASan/UBSan、実際のbaseline/current lintを通過。元36件と同じで新規0件。

18形式、GU/Down、null/明示Runtime stream、実640×2560と部分superblockを含む3×768を実行。n1..16、first、stride/capacity、共有/個別入力、inactive領域・全guard・全sourceを比較した。144区分18432ケース、3744拒否、初期化前34空呼び出し。前後health/最新kernel境界、所有processの正常終了・reap/session終了、実際のsourceLLVM runtime hashを確認。具体的な数・全case identity・fingerprintはgpu-fullのledger/record。通常のAPI戻りはcompletion/低level launch数の証明と扱わない。

両モードで実際にビルドしたstandalone検査バイナリはSHA256まで同一。同じsource/binary/runtime/bootの明示proofでfree側の完全GPU検証を共有した。contrib-llvm側の別GPU実行を行ったとは記録しない。モード固有のdense GEMM性能はこの証明の範囲外。

同一GU+Down作業のcomponentを7形式組合せ、n1/8/16で各arm6サンプル、ABBA順に測る。hostのみとproduction tagsの計測は別process・別sample。byte/guard/入力一致を各pairで確認し、全個別sampleと計算したばらつきを保持。完了したqueue区間は発行間隙も含み、exclusive kernel durationではない。具体値と限界はcomponent-analysis/各recordにある。

時間測定の元sourceは、通常のenable_profiling queue属性を誤って要求していた。実際のRuntime computeはin_orderのみで、以前実測確認したexperimental profiling tagは利用可能なため、このguardをMain v2で修正した。元source/reportはUNEXECUTEDのまま保存。新しいqueueや失敗後fallbackは追加していない。成功時比較は全byteのmemcmp、異常時は元のbyte走査。補助fingerprintはnative-endian uint64 word FNV形式で標準byte FNV-1aとは異なる。実際のequality判定・guard範囲・SHA256ファイルidentityは変えていない。

componentは性能面で不採用。n16では7形式組合せすべてでgroupの発行＋checked drain中央値が旧個別APIより29.799〜44.592倍遅かった。別processの完了queueタグ区間でも19.974〜44.904倍遅い。GU18/Down20の4反復ではhost発行が1.600007→0.159080msに短縮した一方、発行＋checked drainは4.419419→159.993150msに増えた。各条件の6個別sample、平均・中央値・標準偏差・範囲はanalysisに残した。host発行回数削減の効果をGPU側の増加が上回っており、現行callerへの採用を却下する。このcomponentからmodel全体の支配的割合は断定しない。

捕捉したpointer配列の動的添字がprivate trafficを増やした可能性は原因候補であり未証明。最小APIv2は定数添字の16caseを選ぶhelperだけを追加し、元のgeometry・admission・decoder・個別APIをbyteのまま保つ。Mainが完全patch/reportと依存hashを確認したが、このv2は未適用・未compile・未GPU実行。compilerが動的loadへ戻す可能性もあるため、生成device IR/backendを確認してから、新たなbyte/service検証へ進む。v1の測定結果や拒否判断は変更しない。

prefill caller候補はsource reviewだけで未適用。現行model処理は切り替えていない。追加任意APIのcommitは実験の保存であり性能採用ではない。改善したcomponentが確認できてからring lifetime・独立GU/Down fallback・G境界・32K厳密output/LP5/H2D、その後quiet速度比較を扱う。full262144/RESTORE・広い品質、本家の独立設定選別は引き続き未完了。

502件の過去registryと今回4件の完全source報告を接続し、v128の506件を保存した。quota中のLuna調査に別モデルを代替していない。実装agentのテストはMainだけが直列実行した。
'''
  (D/'RESULTS_JA.md').write_text(doc)
  # Retire only unconsumed build intermediates/obsolete original probe binary after exact receipts are committed.
  for name in ['probe-build-original-free','free-probe-build','contrib-llvm-probe-build','free-component-build','contrib-llvm-component-build']:
   root=B/roots[name]
   candidates=[root/'probe.o']+([Path(actual[name]['binary']['path'])] if name=='probe-build-original-free' else [])
   for p in candidates:
    if p.exists():
     st=p.stat();plans.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=m.sha(p),retained_receipt=str((D/name/'record.json').relative_to(W)),reason='Completed standalone intermediate or superseded never-GPU-executed original probe binary; current v3 probe/current component executables remain for exact active comparison consumers. Exact build/source/binary recipe/status retained.'))
  (D/'archive-manifest.json').write_text(json.dumps(dict(byte_identical_copies=copies,preserved_original_statuses=True,large_tensor_files_created=False,raw_complete_history_consumer=None),indent=2)+'\n')
  m.require(sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<32<<20,'compact archive32MiB including all required individual byte case ledger')
  run('archive-secret-check',[str(X/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(D)])
  run('diff-check',['/usr/bin/git','diff','--check']);run('stage-evidence',['/usr/bin/git','add','--',str(D.relative_to(W))]);run('commit-evidence',['/usr/bin/git','commit','-m','docs(bench): qualify native dequant groups and measure same-work service']);r['evidence_revision']=run('evidence-head',['/usr/bin/git','rev-parse','HEAD'])
  retention=D/'retention-v1.json';retention.write_text(json.dumps(dict(retained_commit=r['evidence_revision'],deletions=plans,logical_bytes=sum(x['bytes'] for x in plans),allocated_file_bytes=sum(x['allocated_bytes'] for x in plans),does_not_measure_available_space=True,original_statuses_unchanged=True,still_current_fixture_consumers=['Current v3 byte binaries/receipts/source; current component binaries/receipts/source; unapplied caller source; native model assets unchanged']),indent=2)+'\n')
  run('retention-secret-check',[str(X/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(retention)])
  for item in plans:
   p=Path(item['path']);m.require(m.sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'exact closed retired path');p.unlink();r['deleted'].append(item['path'])
  m.require(all(m.sha(B/root/'record.json')==m.sha(D/name/'record.json') for name,root in roots.items()),'all original receipts unchanged after cleanup')
  run('stage-retention',['/usr/bin/git','add','--',str(retention.relative_to(W))]);run('commit-retention',['/usr/bin/git','commit','-m','docs(bench): retire closed native group qualification traces']);r['committed_revision']=run('final-head',['/usr/bin/git','rev-parse','HEAD']);run('publish-own-fork',['/usr/bin/git','push','origin','HEAD']);m.require(not run('final-status',['/usr/bin/git','status','--porcelain']),'clean report tree');r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','evidence_revision','committed_revision']}));raise SystemExit(0 if r['passed'] else 1)
