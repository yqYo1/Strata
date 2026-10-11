"""Main: publish actual quiet64K CPU/admission before first GPU sample."""
import base64, fcntl, gzip, hashlib, importlib.util, json, os, sys
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest()=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
sp=importlib.util.spec_from_file_location('quiet_publish_owner',O);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
H=B/'xe-compact-moe-quiet64k-phase-controller-source-v1';sys.path.insert(0,str(H));import admission as a
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']
REPORTS={
'research-xe-grouped-xmx-smallm-32-64-tile-dispatch-safety-round156-v1.txt':'b0e68b404f2fc3d984a9d13a0248c351da3092e875e5627784d18e3fc838b9d2',
'research-xe-compact-down-intermediate-fp16-group-sink-order-quality-contract-round156-v1.txt':'b0261d77a4c16a846455039e9415ba1d002892030d8dc25dc0ed227240c5c699',
'research-xe-prefill-decode-owned-vram-rebuild-cache-authority-round157-v1.txt':'f1b0f0ba58c172de930c7e6e43399fc790136baa32899df39b4fb822ba2ed9c0',
'research-xe-native-iq-dequant-service-reuse-overlap-contract-round158-v1.txt':'871c6b613d3b8fd1d45e872cde73d7268c246c8089a9ee3b689a9ed96c4b15c2'}
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-compact-moe-quiet64k-controller-publish-root-v1';dest=W/'bench/results/2026-10-12-xe-compact-moe-quiet64k-controller'
 m.require(not out.exists() and not dest.exists(),'fresh publication')
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
  h=m.sha(p);m.require(wanted is None or wanted==h,'exact source '+str(p));target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());m.require(m.sha(target)==h,'exact archive');e=dict(path=str(p),archive=str(target.relative_to(W)),sha256=h,bytes=p.stat().st_size);r['artifacts'].append(e);return e
 save()
 try:
  m.require(not run('status',['/usr/bin/git','status','--porcelain']),'clean report worktree')
  identity=run('identity',['/usr/bin/git','config','--show-origin','--get-regexp',r'^user\.(name|email)$']);m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()),'global identity')
  m.require(run('origin',['/usr/bin/git','remote','get-url','--push','origin'])=='git@github.com:yqYo1/Strata.git','own fork')
  m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'],S)==a.PUBLISHED_REVISION and not run('source-status',['/usr/bin/git','status','--porcelain'],S),'exact clean source')
  m.require(not (B/'xe-compact-moe-quiet64k-root-v1').exists(),'no quiet GPU launch at boundary')
  for root in ('xe-compact-moe-quiet64k-controller-cpu-root-v1','xe-compact-moe-quiet64k-main-admission-root-v1'):
   v=json.loads((B/root/'record.json').read_text());a.require_closed_prerequisite(v);m.require(all(a.strong_closed_command(c) and c['exit_code']==0 for c in v['commands']),'closed CPU commands')
  q=B/'xe-compact-moe-quiet64k-main-admission-root-v1/qualification.resolved.json';m.require(m.sha(q)=='f67cee5cd040aa03076ef8dc4dd48f9d76c6670cba8753f3f5e53316e61c9c09','actual quiet admission')
  review=json.loads((B/'xe-compact-moe-quiet64k-controller-main-review-v1.json').read_text());m.require(review['main_reviewed'],'Main reviewed')
  dest.mkdir()
  for name,h in review['all_seventeen_file_pins'].items():copy(H/name,'controller/'+name,h)
  for name in ('xe-compact-moe-quiet64k-controller-cpu-qualification-v1.py','xe-compact-moe-quiet64k-main-admission-v1.py','xe-compact-moe-quiet64k-controller-main-review-v1.json'):copy(B/name,name)
  for root,names in {'xe-compact-moe-quiet64k-controller-cpu-root-v1':['record.json','matrix.closed.json','three-cases.closed.json','negative_cli.py'],'xe-compact-moe-quiet64k-main-admission-root-v1':['record.json','qualification.resolved.json','publication.bridge.resolved.json']}.items():
   for name in names:copy(B/root/name,root+'/'+name)
  copy(B/'research-20261009/implementation-xe-compact-moe-quiet64k-phase-controller-source-v1.txt','implementation-report.txt','63438d364038b8073d5099b9676f25a8517389f51366f350b5018a91358f950b')
  plain=out/'research-plaintext';plain.mkdir();entries=[]
  for name,h in REPORTS.items():
   p=B/'research-20261009'/name;m.require(m.sha(p)==h,'reviewed original report');raw=p.read_bytes();(plain/name).write_bytes(raw);blob=gzip.compress(raw,mtime=0);m.require(gzip.decompress(blob)==raw,'lossless report');archive='research/'+name+'.lossless.json';write(archive,dict(encoding='gzip+base64',original_sha256=h,original_bytes=len(raw),compressed_sha256=hashlib.sha256(blob).hexdigest(),data=base64.b64encode(blob).decode()));e=dict(path=str(p),archive=str((dest/archive).relative_to(W)),sha256=h,bytes=len(raw),archive_sha256=m.sha(dest/archive),model='gpt-6-luna',main_full_report_review=True,implemented=False);entries.append(e);r['artifacts'].append(e)
  prior=W/'bench/results/2026-10-12-xe-compact-moe-chunk8192-gpu/report-registry-v142.json';m.require(json.loads(prior.read_text())['research_completed']==600,'prior registry')
  write('report-registry-v143.json',dict(research_completed=604,prior_registry_path=str(prior.relative_to(W)),prior_registry_sha256=m.sha(prior),new_reports=entries,source_revision=a.PUBLISHED_REVISION,measured=False,agent_snapshot_utc=m.utc(),assignments={'research_pass_reduction_scheduling_round135':dict(model='gpt-6-luna',round158='SYCL-specific scratch retirement/cache growth eligibility',status='assigned_after_final_and_full_review'),'research_routezero_fullstream_round153':dict(model='gpt-6-luna',round159='seven-pair actual half-store historical qualification coverage',status='assigned_after_final_and_full_review'),'implement_prefill_row_bounded_scratch':dict(model='gpt-6.1-sol',scope='separate coarse-phase bounded launch controller SOURCE ONLY',status='assigned_after_final_and_full_review')}))
  write('main-review-and-corrections.json',dict(original_reports_modified=False,actual_cpu_pass=True,actual_admission_pass=True,qualification_sha256=m.sha(q),quiet_gpu_executed=False,compact_default='off',performance_eligible=False,full262144_qualified=False,restore_qualified=False,broad_quality_qualified=False,corrections={'R157_prefill_hash':dict(original='c7a427dc58fd5f03732b4a807274f2cc5d406dc2f6c39d64a620c0cd3518cc67',actual=m.sha(S/'src/prefill/prefill.cpp')),'R157_graph_and_cache': 'General graph/cache-growth statements do not establish SYCL compiled call-path eligibility; R158 follow-up examines these. Scratch retirement is source proposal, not an actual capacity/performance result.','R156_Dm': 'AtT8192/12288 GDN remains largest shared carve; halvingDm alone saves logical400/600MiB yet shared maximum stays unchanged. Group-local rank reordering changes addition order; FP16 output is numerical change even with FP32 accumulation. Proposed quality thresholds are evaluation proposals, not additional permission requirements.','R158_half_store': 'R252 inherited missing device proof is not proof later qualifications do not exist. R159 follows committed historical evidence before deciding new fixtures. Hypothetical256 expert expanded bytes are not actual512 model bank extent; 512 pairs are5033164800B,509nonresident pairs5003673600B per layer if all expanded.'},phase_observer=dict(source_parser_cpu_pass=True,launch_route_present=False,actual_timestamp_sanity_pending=True,exclusive_H2D_service=False,critical_path_complete=False),planned_measurement=dict(input_tokens=65536,fixed_context=262144,sequence=[[1,6144],[1,8192],[2,8192],[2,6144],[3,6144],[3,8192]],prefill_decode_evaluated_separately=True,strict16_ID_LP5=True,raw_per_process_budget_bytes=64<<20,campaign_wall_seconds=5400,automatic_retry=False)))
  (dest/'README.md').write_text("""# Quiet64K controller CPU/admission boundary

Actual Main CPU qualification and separate admission converter passed before any
quiet GPU launch. Thirteen code files and seventeen source files are pinned;
original six6144 and three current8192/64K diagnostic receipts remain unchanged.
Pending source templates are preserved; the separate resolved qualification is
the real launch input. Source39e keeps compact scratch off by default.

Next measure six independent quiet64K processes in AB/BA/AB order, A=6144 and
B=8192, fixed262144 context and compactON. Preserve each prefill/cold-decode
sample and exact16-ID/LP5 result; judge prefill and decode separately. Only nine
explicit diagnostic model environment keys are removed. Health remains diagnostic.
The campaign includes gaps in its finite5400s deadline and stops on failure without
retry. No quiet sample, default adoption, broad quality, full262144 or RESTORE
qualification exists at this boundary.

The optional coarse-phase parser passed CPU fixtures but has no launch route;
a separate source implementer is preparing that route. Point-marker spans must
not be presented as exclusive H2D service or complete critical-path attribution.

Four full-reviewed Luna R156/R157/R158 originals are archived losslessly, with
separate Main corrections. Two distinct Luna follow-ups continue while Main
owns all tests and measurements. Source-only proposals establish no speed/fit.
""")
  copy(Path(__file__),Path(__file__).name);write('artifact-manifest.json',r['artifacts'])
  leak=str(S.parent/'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
  for label,where in [('original-report-secret-scan',plain),('archive-secret-scan',dest)]:
   scan=out/(label+'.json');run(label,[leak,'dir',str(where),'--no-banner','--redact','--report-format','json','--report-path',str(scan)]);m.require(json.loads(scan.read_text())==[],'secret-free '+label)
  run('stage',['/usr/bin/git','add','--',str(dest.relative_to(W))]);run('staged-check',['/usr/bin/git','diff','--cached','--check']);run('commit',['/usr/bin/git','commit','-m','docs(sycl): qualify quiet64K balanced controller on CPU']);r['committed_revision']=run('head',['/usr/bin/git','rev-parse','HEAD']);run('push',['/usr/bin/git','push','origin','HEAD']);m.require(not run('final-status',['/usr/bin/git','status','--porcelain']),'clean published worktree');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ('passed','error','committed_revision')}));raise SystemExit(0 if r['passed'] else 1)
