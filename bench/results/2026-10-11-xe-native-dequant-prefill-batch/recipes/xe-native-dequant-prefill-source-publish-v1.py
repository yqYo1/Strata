import fcntl,importlib.util,json,os
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','owner')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-native-dequant-prefill-source-publish-root-v1';m.require(not out.exists(),'immutable publication');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),base_revision='b72640fff05098f1762cb3bf9c427a5f8d7db31c',adopted=False,production_callers_changed=True, model_qualified=False, performance_eligible=False,gpu_executed=False)
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 def run(label,args):
  e,so,se=owner.run(label,args,ENV,W,wall=60,cpu=120,text_cap=4<<20,total_cap=16<<20,rss_cap=2<<30);m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label);return so.read_text().strip()
 try:
  paths={'lint':B/'xe-e32-lints-baseline-compare-root-v16/record.json','observer':B/'xe-native-dequant-batch-census-observer-cpu-root-v2/record.json'}
  for mode in ['free','contrib-llvm']:
   paths[mode+'-release']=B/('xe-e32-'+mode+'-full-build-root-v21/record.json');paths[mode+'-host-asan']=B/('xe-e32-'+mode+'-asan-full-build-root-v18/record.json')
  r['qualification_pins']={}
  for name,p in paths.items():
   v=json.loads(p.read_text());m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'closed '+name);m.require(all(m.sha(Path(f) if Path(f).is_absolute() else W/f)==h for f,h in v['source_pins'].items()),'exact qualified source '+name);r['qualification_pins'][name]=dict(path=str(p),sha256=m.sha(p))
  observer=json.loads(paths['observer'].read_text());m.require(observer['fixtures']['positive_cases']==3 and observer['fixtures']['negative_cases']==47 and len(observer['linkage'])==2,'actual complete observer and retained caller references')
  review_path=B/'xe-native-dequant-prefill-batch-applied-main-review-v4.json';review=json.loads(review_path.read_text());m.require(review['main_reviewed'] and review['applied'],'reviewed opt-in native prefill caller');r['source_pins']=review['source_pins'];r['source_review_sha256']=m.sha(review_path)
  m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'])==r['base_revision'],'exact committed base')
  m.require(run('changed-files',['/usr/bin/git','diff','--name-only','HEAD']).splitlines()==['src/prefill/prefill.cpp'],'only reviewed opt-in prefill caller')
  m.require(not run('untracked',['/usr/bin/git','ls-files','--others','--exclude-standard']),'no unreviewed source')
  run('diff-check',['/usr/bin/git','diff','--check']);run('stage',['/usr/bin/git','add','--','src/prefill/prefill.cpp']);run('commit',['/usr/bin/git','commit','-m','experiment(xe): batch native prefill dequant with checked slot release']);r['committed_revision']=run('committed-head',['/usr/bin/git','rev-parse','HEAD'])
  run('publish-own-fork',['/usr/bin/git','push','git@github.com:yqYo1/Strata.git','HEAD:refs/heads/fix/xestrata-e32-host-handoff-20261011']);m.require(not run('final-status',['/usr/bin/git','status','--porcelain']),'clean source tree');r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','committed_revision']}));raise SystemExit(0 if r['passed'] else 1)
