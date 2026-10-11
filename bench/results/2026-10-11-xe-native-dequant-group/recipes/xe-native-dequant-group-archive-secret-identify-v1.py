import fcntl,importlib.util,json
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');X=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');D=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009/bench/results/2026-10-11-xe-native-dequant-group');O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('o',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','owner')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=B/'xe-native-dequant-group-archive-secret-identify-root-v1';m.require(not out.exists(),'new root');out.mkdir();r=dict(active=True,complete=False,passed=False,commands=[],gpu_executed=False);
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 try:
  report=out/'redacted-findings.json';e,so,se=owner.run('redacted-findings',[str(X/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--report-format','json','--report-path',str(report),'--source',str(D)],dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8'),D,wall=60,cpu=120,text_cap=4<<20,total_cap=8<<20,rss_cap=2<<30);m.require(m.closed(e) and e['exit_code'] in [0,1],'normal scanner');data=json.loads(report.read_text());r['findings']=[{k:v for k,v in x.items() if k in ['RuleID','Description','File','StartLine','EndLine','Fingerprint']} for x in data];r['redacted_report_sha256']=m.sha(report);r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','findings']}));raise SystemExit(0 if r['passed'] else 1)
