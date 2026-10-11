"""Main: publish closed six-sample results and actual phase CPU/admission boundary."""
import base64, fcntl, gzip, hashlib, importlib.util, json, os, sys
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest()=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
sp=importlib.util.spec_from_file_location('quiet_results_publish_owner',O);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
H=B/'xe-compact-moe-phase-observer-controller-source-v1';sys.path.insert(0,str(H));import admission as a
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']
REPORTS={
'research-xe-sycl-prefill-retire-cache-growth-callpath-contract-round158-v1.txt':'3f3f2f3b13a127cea16cff451a4cb0edbed0595b639cc9ec0fe081de7b708be7',
'research-xe-native-seven-pair-half-store-device-evidence-coverage-round159-v1.txt':'6b66e76b63d3313bbae95ef875dfbca5aa0e2845cff08d0c6dc653ab5a2287b0',
'research-xe-prefill-early-raii-destruction-rebuild-minimal-seam-round160-v1.txt':'4dd9f6ea3236df8d598398139301557d2b6bfff6d95a0d2f6a611050987210b3',
'research-xe-gdn-bounded-row-production-kernel-policy-statecarry-contract-round160-v1.txt':'b04940820c27be1e82ed4ebe69c158da9c33b477c78c65c5a980f5f95908b7d0',
'research-xe-qsa-bounded-query-row-sharedmax-state-contract-round161-v1.txt':'770f3661a973c560d09744de4c4faebd4d7c24edc7439e32b44dee708608169c',
'research-upstream-xestrata-latest-release-server-protocol-delta-round161-v1.txt':'ae66dd4ab01e0e9d1281f5087bcf4d6304c92d77be1f393e39c3b483db9344c7',
'research-xe-row-scratch-source-provenance-arithmetic-verification-round162-v1.txt':'0accfeab06eccc360d9e08c84cbc0e672b3c356c01b5dda7a2c854c9ddafea45',
'research-xestrata-xe0142-source-safety-effective-controls-delta-round162-v1.txt':'73e101593f9ce79292ddd49bc52a5cbd4471c6b7670f5ea6edf7069e3b31f7f9',
'research-xe-fixed8192-cache-lending-larger-decode-cache-settings-contract-round163-v1.txt':'bfb846ced4d9d83a2e7a2db75020ada39d7eae8259fb55e71f0c00820b2d2e8e',
'research-xe-native-decode-cpu-avx2-multirow-service-dispatch-contract-round163-v1.txt':'26bfb965427cb83194dd2cec3ca9e30639dfc274bc5fc6f7156a69c347d09e67',
'research-xe-native-fixed8192-cache-loan-admission-byte-budget-round164-v1.txt':'5c6364da776f060a00aa4b7ebddec8d210458ae742975a49d3af6859192a4de2',
'research-xe-grouped-xmx-rowcount-bucketing-order-lifetime-contract-round164-v1.txt':'5afcd11376b42454d914e61420739c3a8c70b1833e230ad4e16d22c5de517454'}
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-compact-moe-quiet64k-results-phase-admission-publish-root-v1';dest=W/'bench/results/2026-10-12-xe-compact-moe-quiet64k-results'
 m.require(not out.exists() and not dest.exists(),'fresh immutable publication')
 out.mkdir();r=dict(active=True,complete=False,passed=False,commands=[],artifacts=[],started_utc=m.utc(),owner_sha256=m.sha(O),recipe_sha256=m.sha(__file__),gpu_executed=False,model_executed=False,original_results_modified=False)
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands
 def run(label,argv,cwd=W):
  c,so,se=owner.run(label,argv,ENV,cwd,wall=60,cpu=120,rss_cap=2<<30,text_cap=2<<20,total_cap=16<<20,file_cap=64<<20)
  m.require(a.strong_closed_command(c) and c['exit_code']==0,'normal owned '+label)
  return so.read_text().strip()
 def write(name,v):
  p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 def copy(p,name,wanted=None):
  h=m.sha(p);m.require(wanted is None or wanted==h,'exact original '+str(p));target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());m.require(m.sha(target)==h,'exact archive');e=dict(path=str(p),archive=str(target.relative_to(W)),sha256=h,bytes=p.stat().st_size);r['artifacts'].append(e);return e
 save()
 try:
  m.require(not run('status',['/usr/bin/git','status','--porcelain']),'clean report worktree')
  identity=run('identity',['/usr/bin/git','config','--show-origin','--get-regexp',r'^user\.(name|email)$']);m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()),'global identity')
  m.require(run('origin',['/usr/bin/git','remote','get-url','--push','origin'])=='git@github.com:yqYo1/Strata.git','own fork')
  m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'],S)==a.PUBLISHED_REVISION and not run('source-status',['/usr/bin/git','status','--porcelain'],S),'actual clean engine')
  m.require(not (B/'xe-compact-moe-phase-observer-root-v1').exists(),'phase GPU unlaunched at boundary')
  analysis_path=B/'xe-compact-moe-quiet64k-analysis-root-v1/result.json';analysis=json.loads(analysis_path.read_text())
  m.require(analysis['passed'] and analysis['complete'] and not analysis['active'] and analysis['all_six_exact16_ID_LP5'] and len(analysis['samples'])==6,'actual six closed analysis')
  for p,h in analysis['receipt_pins'].items():
   m.require(m.sha(Path(p))==h,'all quiet records stable');v=json.loads(Path(p).read_text());a.require_normal_model(v,Path('/proc/sys/kernel/random/boot_id').read_text().strip())
  for root in ('xe-compact-moe-phase-observer-controller-cpu-root-v1','xe-compact-moe-phase-observer-main-admission-root-v1'):
   v=json.loads((B/root/'record.json').read_text());a.require_closed_prerequisite(v);m.require(all(a.strong_closed_command(c) and c['exit_code']==0 for c in v['commands']),'actual phase CPU/converter closure')
  qpath=B/'xe-compact-moe-phase-observer-main-admission-root-v1/qualification.resolved.json';q=json.loads(qpath.read_text())
  m.require(all(m.sha(Path(p))==h for p,h in q['receipt_sha256'].items()),'actual phase proof dependencies stable')
  review=json.loads((B/'xe-compact-moe-phase-observer-controller-main-review-v1.json').read_text());m.require(review['main_reviewed'],'Main full source review')
  dest.mkdir()
  for sample in analysis['samples']:copy(Path(sample['record_path']),'quiet-samples/'+Path(sample['record_path']).parent.name+'/record.json',sample['record_sha256'])
  copy(analysis_path,'analysis/result.json');copy(B/'xe-compact-moe-quiet64k-analysis-v1.py','analysis/xe-compact-moe-quiet64k-analysis-v1.py',analysis['recipe_sha256'])
  for name,h in review['all_nineteen_file_pins'].items():copy(H/name,'phase-controller/'+name,h)
  for name in ('xe-compact-moe-phase-observer-controller-cpu-qualification-v1.py','xe-compact-moe-phase-observer-main-admission-v1.py','xe-compact-moe-phase-observer-controller-main-review-v1.json'):copy(B/name,name)
  for root,names in {'xe-compact-moe-phase-observer-controller-cpu-root-v1':['record.json','matrix.closed.json','three-cases.closed.json'], 'xe-compact-moe-phase-observer-main-admission-root-v1':['record.json','qualification.resolved.json','publication.bridge.resolved.json','phase-proof.resolved.json','phase-recipe.resolved.json']}.items():
   for name in names:copy(B/root/name,root+'/'+name)
  copy(B/'research-20261009/implementation-xe-compact-moe-phase-observer-controller-source-v1.txt','implementation-phase-report.txt','5b720fd094413fdabce82ddee5586114c3bd93f611ebbacb1b0ef7b52304e356')
  plain=out/'research-plaintext';plain.mkdir();entries=[]
  for name,h in REPORTS.items():
   p=B/'research-20261009'/name;m.require(m.sha(p)==h,'fully read original report');raw=p.read_bytes();(plain/name).write_bytes(raw);blob=gzip.compress(raw,mtime=0);m.require(gzip.decompress(blob)==raw,'lossless report');archive='research/'+name+'.lossless.json';write(archive,dict(encoding='gzip+base64',original_sha256=h,original_bytes=len(raw),compressed_sha256=hashlib.sha256(blob).hexdigest(),data=base64.b64encode(blob).decode()));e=dict(path=str(p),archive=str((dest/archive).relative_to(W)),sha256=h,bytes=len(raw),archive_sha256=m.sha(dest/archive),model='gpt-6-luna',main_full_report_review=True,implemented=False);entries.append(e);r['artifacts'].append(e)
  prior=W/'bench/results/2026-10-12-xe-compact-moe-quiet64k-controller/report-registry-v143.json';m.require(json.loads(prior.read_text())['research_completed']==604,'actual prior registry')
  write('report-registry-v144.json',dict(research_completed=604+len(entries),prior_registry_path=str(prior.relative_to(W)),prior_registry_sha256=m.sha(prior),new_reports=entries,source_revision=a.PUBLISHED_REVISION,measured=False,agent_snapshot_utc=m.utc(),assignments={'research_pass_reduction_scheduling_round135':dict(model='gpt-6-luna',round164='fixed8192 cache loan native startup byte budget',status='completed_and_full_Main_reviewed'),'research_routezero_fullstream_round153':dict(model='gpt-6-luna',round164='groupedXMX rowcount bucket order and lifetime',status='completed_and_full_Main_reviewed'),'implement_prefill_row_bounded_scratch':dict(model='gpt-6.1-sol',scope='phase controller source returned; Main owns CPU/admission and all GPU tests',status='completed')}))
  write('main-review-and-corrections.json',dict(original_reports_modified=False,actual_quiet_samples=6,quiet_all_pass=True,actual_phase_cpu_pass=True,actual_phase_admission_pass=True,phase_qualification_sha256=m.sha(qpath),phase_gpu_executed=False,default_adopted=False,compact_default='off',full262144_qualified=False,restore_qualified=False,broad_quality_qualified=False,corrections={'prefill_source_identity':dict(actual=m.sha(S/'src/prefill/prefill.cpp'),published_revision=a.PUBLISHED_REVISION,note='R158/R160/R161 malformed or mistyped identities are corrected separately. R162 independently confirms published source; reported6913... discrepancy is not reproducible and its cause is unknown.'),'R163_cpu_down_shape':'Logical Down is N2560 K640 (51200 quant blocks), not N640 K2560. Block product and stated921600/1126400B compressed totals are unchanged.','R159_half_store':'Historical252 device group-v-individual comparisons exist. Direct device constructor vs independentRNE remains a distinct missing proof; avoid repeating those252 cases.','R160_R161_shared_max':'BoundedGDN+QSA both required beforeMoE dominates; reducingDm alone does not lower current8192 shared maximum. Logical byte counts are not physical peak/runtime gains.','R163_loan':'Existing lend/refill is source-eligible alternative for phase cache sharing; current no-borrow measurements do not qualify any larger actual cache/loan/physical fit.'},latest_remote=dict(official_tag='v0.1.43',official_commit='7695859aa13afb248bdb4d36f9b2e95707db079d',xe_tag='xe0.1.42',xe_commit='da3d426079019daa6e01388d03b0d94ae67070b7',actual_latest_gpu_measured=False,official_independent_parameter_optimization_complete=False,notes='R161/R162 primary source update and safety audit only. Update integration and pristine/settings-tuned qualification remain separate TODOs.'),phase_observer=dict(cpu_pass=True,actual_timestamp_sanity_pending=True,exclusive_H2D_service=False,critical_path_complete=False)))
  rows=['| chunk | prefill mean t/s | sample SD | CV% | cold decode16 mean t/s | sample SD | CV% |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
  for chunk in ('6144','8192'):
   pp=analysis['summary'][chunk]['prefill'];tg=analysis['summary'][chunk]['cold_decode'];rows.append(f"| {chunk} | {pp['mean']:.6f} | {pp['sample_sd']:.6f} | {pp['cv_percent']:.6f} | {tg['mean']:.6f} | {tg['sample_sd']:.6f} | {tg['cv_percent']:.6f} |")
  write('round164-main-corrections.json',dict(original_reports_modified=False,prefill_source_sha256=m.sha(S/'src/prefill/prefill.cpp'),rowcount_grouping='Permuting members within the same fixedG16 expert set leaves max_rows and reserved grid unchanged. Reducing group maxima requires changing group membership/partition or grid representation; a within-window count sort alone establishes no gain. Compact capacity partition may be affected, but that needs a distinct calculation and proof.',source_identity='R164 scheduling supplied prefill digest is missing a character; actual published digest is recorded here.',cache_lending='R164 excludes native artifacts and gives symbolic B/P formulas. Source per-layer mode is a boolean flag; quota3 is a resulting capacity rather than a fixed CLI quota. Request768 is neither minimal proven fit nor measured. Exact native layout and actual startup free memory remain prerequisites.'))
  text='# Quiet64K six-sample results and phase admission\n\nSix fresh processes completed normally with exact16-ID/LP5 parity, same39e source/freeLLVM-v711 binary, same65536-token coding fixture and fixed262144 capacity. AB/BA/AB pairs compare requested6144 versus8192 compactON; all other model settings are fixed. Individual original receipts and times are archived.\n\n'+'\n'.join(rows)+'\n\nEach summary has three independent samples. Prefill and cold decode are judged separately; cold decode includes first-use work and only16 output tokens, so it does not establish warm/stationary throughput. Variability is reported, not eliminated. No default change, broad quality, physical262144 use or RESTORE qualification follows from these results.\n\nThe separate phase controller passed Main serial CPU fixtures and proof→recipe→front-door admission after all six quiet runs closed. Original pending templates remain byte-exact. The next sequence is N64 timestamp sanity thenN32768 diagnostics, fixed8192, with TIMING1. No new profiling queue/kernel is introduced. Point-marker spans are not exclusive H2D/kernel service or complete critical-path attribution. No phase GPU run exists at this boundary.\n\nTwelve new full-reviewed Luna source reports are preserved losslessly, with separate identity/shape/qualification corrections. R164 parallel research completed while Main measured. Latest officialv0.1.43 and Xe xe0.1.42 are identified, but neither latest pristine implementation nor independent official best-settings comparison has been measured. Updates remain pending after this boundary.\n'
  (dest/'README.md').write_text(text)
  copy(Path(__file__),Path(__file__).name);write('artifact-manifest.json',r['artifacts'])
  leak=str(S.parent/'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
  for label,where in [('original-report-secret-scan',plain),('archive-secret-scan',dest)]:
   scan=out/(label+'.json');run(label,[leak,'dir',str(where),'--no-banner','--redact','--report-format','json','--report-path',str(scan)]);m.require(json.loads(scan.read_text())==[],'secret-free '+label)
  run('stage',['/usr/bin/git','add','--',str(dest.relative_to(W))]);run('staged-check',['/usr/bin/git','diff','--cached','--check']);run('commit',['/usr/bin/git','commit','-m','docs(sycl): record six quiet64K samples and phase admission']);r['committed_revision']=run('head',['/usr/bin/git','rev-parse','HEAD']);run('push',['/usr/bin/git','push','origin','HEAD']);m.require(not run('final-status',['/usr/bin/git','status','--porcelain']),'clean published worktree');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ('passed','error','committed_revision')}));raise SystemExit(0 if r['passed'] else 1)
