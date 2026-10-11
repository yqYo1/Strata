import fcntl,importlib.util,json
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','owner pin')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=B/'xe-prefill-diagnostics-integrate-root-v1';m.require(not out.exists(),'immutable integration');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),source_base='ad41ad97b3a918727f89aace09b9a424b077bc8c',build_executed=False,gpu_executed=False,model_executed=False)
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 def run(label,args):
  e,so,se=owner.run(label,args,ENV,W,wall=60,cpu=60,text_cap=4<<20,total_cap=16<<20,rss_cap=2<<30);m.require(m.closed(e) and e['exit_code']==0,'closed '+label);return so.read_text().strip()
 try:
  m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'])==r['source_base'] and not run('source-status',['/usr/bin/git','status','--porcelain']),'clean committed base')
  boundary=B/'xe-profile32k-quiet3-boundary-root-v2/record.json';p=json.loads(boundary.read_text());m.require(p['passed'] and p['complete'] and not p['active'] and all(m.closed(c) for c in p['commands']),'previous measurement/retention boundary closed');r['previous_boundary_sha256']=m.sha(boundary)
  for i in range(1,4):
   q=B/'xe-source-llvm-quiet32k-root-v2'/str(i)/'record.json';v=json.loads(q.read_text());m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'quiet baseline closed')
   for c in v['commands']:
    for ident in c['owners']:
     now=m.identity(ident['pid']);m.require(not now or now['start_ticks']!=ident['start_ticks'],'owned baseline identities absent')
  r['reviews']=[]
  for review_name,patch_name in [('xe-profiling-overhead-accounting-main-review-v1.json','xe-profiling-overhead-accounting-source-v1/profiling-overhead-accounting.patch'),('xe-prefill-transfer-counts-main-review-v1.json','xe-prefill-transfer-counts-source-v1/prefill-transfer-counts.patch')]:
   q=B/review_name;v=json.loads(q.read_text());m.require(v['main_reviewed'] and v['base_revision']==r['source_base'] and all(m.sha(Path(path))==h for path,h in v['artifact_pins'].items()),'fully reviewed exact delivered artifact')
   patch=B/patch_name;run('check-'+review_name,['/usr/bin/git','apply','--check',str(patch)]);run('apply-'+review_name,['/usr/bin/git','apply',str(patch)]);r['reviews'].append(dict(path=str(q),sha256=m.sha(q),patch=str(patch),patch_sha256=m.sha(patch)))
  m.require(run('changed-files',['/usr/bin/git','diff','--name-only','HEAD']).splitlines()==['src/prefill/prefill.cpp'] and not run('untracked',['/usr/bin/git','ls-files','--others','--exclude-standard']),'only reviewed source changed')
  run('diff-check',['/usr/bin/git','diff','--check']);patch=run('combined-diff',['/usr/bin/git','diff','HEAD']);(out/'source.patch').write_text(patch+'\n');r['source_patch_sha256']=m.sha(out/'source.patch');r['source_pins']={'src/prefill/prefill.cpp':m.sha(W/'src/prefill/prefill.cpp')}
  q=B/'xe-prefill-diagnostics-combined-main-review-v1.json';m.require(not q.exists(),'new combined review');v=dict(main_reviewed=True,base_revision=r['source_base'],source_pins=r['source_pins'],applied=True,compile_build_lint_test_gpu_qualified=False,adopted=False,only_changed_file='src/prefill/prefill.cpp',component_reviews=r['reviews'],combined_patch=dict(path=str(out/'source.patch'),sha256=r['source_patch_sha256']),review='Full original diffs reviewed independently; exact patches applied without manual conflict resolution. Combined diagnostic-only changes retain default enablement guards, existing copy/event/sync/ownership boundaries and model arithmetic; no performance/quality claim before actual qualification.');q.write_text(json.dumps(v,indent=2)+'\n');r['combined_review_sha256']=m.sha(q);r['complete']=True
 except BaseException as error:r['error']=repr(error)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','source_pins']}));raise SystemExit(0 if r['passed'] else 1)
