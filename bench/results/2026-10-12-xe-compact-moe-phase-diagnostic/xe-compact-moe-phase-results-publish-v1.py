"""Main: close coarse phase evidence without misattributing the observer's overhead."""
import base64, fcntl, gzip, hashlib, importlib.util, json, os, sys
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest()=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
sp=importlib.util.spec_from_file_location('phase_publish_owner',O);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
H=B/'xe-compact-moe-phase-observer-controller-source-v1';sys.path.insert(0,str(H));import admission as a
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']
REPORTS={
'research-latest-official-xe-functional-merge-dependency-units-round165-v1.txt':'f2da0823cf1547788ad20c58a38487751c79e3e89d090f542ff6901d63be810e',
'research-latest-pristine-sycl-preload-selectedpath-safety-avoidability-round165-v1.txt':'8efcaf0a777176ab21ffbea4fb96a99a3a1702c2167645b95580cd21e762fd75',
'research-official-v0143-whole-request-timing-protocol-merge-contract-round166-v1.txt':'80e1892e048009f1b84ed658e5d16eafb99f2b20693e06c4843da34da753af11',
'research-current-unitrace-v711-health-only-instrumentation-contract-round166-v1.txt':'775a62aa1af166c126c4b082f5516eddda80b2fee5ff9eab6aa53fc32344b38f'}
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-compact-moe-phase-results-publish-root-v1';dest=W/'bench/results/2026-10-12-xe-compact-moe-phase-diagnostic'
 m.require(not out.exists() and not dest.exists(),'new immutable publication')
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
  h=m.sha(p);m.require(wanted is None or wanted==h,'original identity');target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());m.require(m.sha(target)==h,'byte-exact copy');r['artifacts'].append(dict(path=str(p),archive=str(target.relative_to(W)),sha256=h,bytes=p.stat().st_size))
 save()
 try:
  m.require(not run('status',['/usr/bin/git','status','--porcelain']),'clean report worktree')
  identity=run('identity',['/usr/bin/git','config','--show-origin','--get-regexp',r'^user\.(name|email)$']);m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()),'global identity')
  m.require(run('origin',['/usr/bin/git','remote','get-url','--push','origin'])=='git@github.com:yqYo1/Strata.git','own origin')
  m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'],S)==a.PUBLISHED_REVISION and not run('source-status',['/usr/bin/git','status','--porcelain'],S),'unchanged clean engine')
  qpath=B/'xe-compact-moe-phase-observer-main-admission-root-v1/qualification.resolved.json';m.require(m.sha(qpath)=='94f5b50b2b92c7e3e941567588af00cf6269e06c7d86de664c6d382be9631f0a','actual phase admission')
  q=json.loads(qpath.read_text());m.require(all(m.sha(Path(p))==h for p,h in q['receipt_sha256'].items()),'phase prerequisites stable')
  boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();phase={}
  for n in (64,32768):
   p=a.phase_root_path(B,n,8192)/'record.json';v=json.loads(p.read_text());a.require_normal_model(v,boot)
   m.require(v['qualification_sha256']==m.sha(qpath) and v['source_pins']==q['production_source_pins'] and v['binary']==q['free_release_binary'] and v['reference_comparison']['strict_continuation_passed'],'actual unchanged source and exact16LP5')
   m.require(v['phase_observer']['capture_valid'] and v['validation']['generated_tokens']==16 and not v['phase_observer']['performance_eligible'],'actual valid diagnostic')
   phase[n]=(p,v)
  dest.mkdir()
  for n,(p,v) in phase.items():copy(p,f'n{n}-t8192-g16-on/record.json')
  baseline_path=B/'xe-compact-moe-chunk8192-root-v1/n32768-t8192-g16-on/record.json';baseline=json.loads(baseline_path.read_text());a.require_normal_model(baseline,boot)
  target=phase[32768][1];obs=target['phase_observer'];tim=target['timer_overhead']['overhead']
  m.require(q['three_case_receipts']['n32768-t8192']['path']==str(baseline_path) and q['three_case_receipts']['n32768-t8192']['sha256']==m.sha(baseline_path),'actual same-chunk reference')
  write('analysis.json',dict(passed=True,complete=True,active=False,source_revision=a.PUBLISHED_REVISION,phase_qualification_sha256=m.sha(qpath),boot_id=boot,
   records={str(n):dict(path=str(p),sha256=m.sha(p),validation=v['validation'],raw_phase_lines=v['phase_raw_report_lines'],observer=v['phase_observer'],transfer_total=v['transfer_ledger']['total']) for n,(p,v) in phase.items()},
   reference_diagnostic=dict(path=str(baseline_path),sha256=m.sha(baseline_path),prompt_ms=baseline['validation']['prompt_ms'],generated_tokens=16),
   n32768=dict(prompt_ms=target['validation']['prompt_ms'],single_diagnostic_elapsed_change_percent=100*(target['validation']['prompt_ms']/baseline['validation']['prompt_ms']-1),mark_cpu_api_ms=tim['mark_ms'],fold_cpu_api_ms=tim['fold_ms'],marker_calls=tim['marker_calls'],observed_span_ranking=sorted(obs['observed_phases'].items(),key=lambda p:p[1]['ms'],reverse=True)),
   performance_eligible=False,observer_total_overhead_calibrated=False,exclusive_kernel_service=False,exclusive_H2D_service=False,critical_path_complete=False,default_adopted=False,full262144_qualified=False,restore_qualified=False,broad_quality_qualified=False,
   decision='Current200451-marker diagnostic substantially changes request timing. Host-grouping/dequant named spans are observations in instrumented execution, not exclusive useful host/GPU service or quiet bottleneck shares. Do not subtract timerCPU overhead to infer uninstrumented service. Next qualify bounded profiler/sparser evidence before kernel prioritization.',
   limits=['Single same-chunk previous diagnostic comparison is not balanced total observer calibration.','Queued marker command_end intervals include host/copy waits and marker work.','Positive hardware timestamp access for these reports does not establish independent support/coverage for all kernels or other graph paths.']))
  plain=out/'research-plaintext';plain.mkdir();entries=[]
  for name,h in REPORTS.items():
   p=B/'research-20261009'/name;m.require(m.sha(p)==h,'fully reviewed original report');raw=p.read_bytes();(plain/name).write_bytes(raw);blob=gzip.compress(raw,mtime=0);m.require(gzip.decompress(blob)==raw,'lossless original');archive='research/'+name+'.lossless.json';write(archive,dict(encoding='gzip+base64',original_sha256=h,original_bytes=len(raw),compressed_sha256=hashlib.sha256(blob).hexdigest(),data=base64.b64encode(blob).decode()));e=dict(path=str(p),archive=str((dest/archive).relative_to(W)),sha256=h,bytes=len(raw),archive_sha256=m.sha(dest/archive),model='gpt-6-luna',main_full_report_review=True,implemented=False);entries.append(e);r['artifacts'].append(e)
  prior=W/'bench/results/2026-10-12-xe-compact-moe-quiet64k-results/report-registry-v144.json';m.require(json.loads(prior.read_text())['research_completed']==616,'prior registry')
  write('report-registry-v145.json',dict(research_completed=616+len(entries),prior_registry_path=str(prior.relative_to(W)),prior_registry_sha256=m.sha(prior),new_reports=entries,source_revision=a.PUBLISHED_REVISION,measured=False,agent_snapshot_utc=m.utc(),assignments={'research_pass_reduction_scheduling_round135':dict(model='gpt-6-luna',round166='whole-request timing protocol dependency',status='completed_and_full_Main_reviewed'),'research_routezero_fullstream_round153':dict(model='gpt-6-luna',round166='currentunitrace tinyhealth instrumentation contract',status='completed_and_full_Main_reviewed'),'implement_prefill_row_bounded_scratch':dict(model='gpt-6.1-sol',scope='boundedunitrace tinyhealth controller SOURCE ONLY',status='assigned_after_final_and_full_review')}))
  refs=run('official-tag-identity',['/usr/bin/git','ls-remote','https://github.com/Niko1221/Strata.git','refs/tags/v0.1.43','refs/tags/v0.1.43^{}'])
  m.require(dict(line.split()[::-1] for line in refs.splitlines())=={'refs/tags/v0.1.43':'16374b3e6dfe452a163c3a961ae5faa3b4aec84b','refs/tags/v0.1.43^{}':'7695859aa13afb248bdb4d36f9b2e95707db079d'},'actual official release tag proof')
  write('main-review-and-corrections.json',dict(original_reports_modified=False,official_tag_ref_stdout=refs,official_release='https://github.com/Niko1221/Strata/releases/tag/v0.1.43',corrections={'R165_identity':'Report silently spells assignedofficialcommit suffix0790 rather than actual079d; lookup of that different commit cannot establish actual release unavailable. Main tag proof here and release page verify assigned079d. No original report rewritten.','R165_current_paths':'Current39e already has >=32K and64K narrow exact16LP5 gates. Historical startup errors are a different preload-on source/setting question; neither currentpasses nor oldfailure qualify latest untouchedpristine.','R165_feature_scope':'User requires functional/server updates. logit_bias needs a complete engine-backed API unit rather than indefinite server-only deferral. NaN recovery can retain failfast diagnostics with explicit guard0 while separately testing recovery semantics; no adoption implied.','R143_profiler_availability':'InstalledB/unitrace-build-system-cc/build/unitrace and historicalclosedreceipts exist outside/opt. R143 search absence does not prove profiler unavailable. Historicalsuccesses are a differentboot/compiler/binary, not currentqualification.'},latest_pristine_gpu_measured=False,official_independent_parameter_optimization_complete=False,default_adopted=False))
  write('unitrace-r166-main-corrections.json',dict(original_report_modified=False,health_protocol='Current qualified exact lines are STATUS/QUEUE/H2D/KERNEL/D2H/PASS:1 round16384 exactwords, not the proposed literal PASS BDF. Read the actual currenthealth source/closedreceipt before defining acceptance.',environment='Current diagnostic logger/validation policy should be retained for first checks; report proposal to remove logs is not a requirement or tested qualification. No inherited unrelatedLD_PRELOAD is allowed. Health-only tracing does not qualify modelgraph coverage.',resource_bounds='PTI has no overall trace byte limit; externally owned finite file/text/time/aggregate caps and complete parsing are mandatory for the selected consumer.'))
  (dest/'README.md').write_text('# Coarse phase diagnostic closure\n\nBoth N64 and N32768 fixed8192 compactON diagnostics completed normally with exact16-ID/LP5 output and fault/boot/health gates. Source39e and the qualified freeLLVM-v711 binary remained unchanged. Each original receipt retains every phase line and transfer/route count.\n\nAt32768 input, prompt took146757.0ms; the earlier same-chunk diagnostic took94070.8ms. This single comparison is not a balanced calibration. The observer made200451 marker calls; reportedCPU mark API time was30852.558ms and fold API time176.250ms. Host-grouping50866ms and dequant55518ms are adjacent compute-marker spans that include waiting and observer effects. They are not exclusive useful service or quiet bottleneck shares. No arithmetic subtraction repairs the attribution. A bounded profiler/sparser measurement is required before assigning a kernel bottleneck.\n\nNo default, performance, broad quality, full physical262144 or RESTORE qualification follows. Successful raw diagnostic logs may be retired only after this compact evidence is verified; current source/compiler/binaries/protocol fixtures and unresolved startup failure evidence remain required.\n\nFour full-read research originals are archived losslessly with separate Main corrections. Actual official release tag7695859...079d is verified; an agent report looked up a different0790 spelling. Functional updates, including engine-backed API features, remain mandatory integration work. The official optimized-settings sweep and latest pristine long-input measurements remain uncompleted.\n')
  copy(Path(__file__),Path(__file__).name);write('artifact-manifest.json',r['artifacts'])
  leak=str(S.parent/'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
  for label,where in [('original-report-secret-scan',plain),('archive-secret-scan',dest)]:
   scan=out/(label+'.json');run(label,[leak,'dir',str(where),'--no-banner','--redact','--report-format','json','--report-path',str(scan)]);m.require(json.loads(scan.read_text())==[],'secret-free '+label)
  run('stage',['/usr/bin/git','add','--',str(dest.relative_to(W))]);run('staged-check',['/usr/bin/git','diff','--cached','--check']);run('commit',['/usr/bin/git','commit','-m','docs(sycl): close coarse phase diagnostics with observer limits']);r['committed_revision']=run('head',['/usr/bin/git','rev-parse','HEAD']);run('push',['/usr/bin/git','push','origin','HEAD']);m.require(not run('final-status',['/usr/bin/git','status','--porcelain']),'clean published worktree');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ('passed','error','committed_revision')}));raise SystemExit(0 if r['passed'] else 1)
