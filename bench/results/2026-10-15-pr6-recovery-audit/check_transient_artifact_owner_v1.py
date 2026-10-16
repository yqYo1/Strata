#!/usr/bin/env python3
"""CPU-only observation-race and retained supervision checks; no GPU runtime."""
from pathlib import Path
import fcntl,hashlib,importlib.util,json,sys,tempfile
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
helper=A/'direct_owner_transient_artifacts_v1.py'
spec=importlib.util.spec_from_file_location('transient_owner_check',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Item:
 def __init__(self,error=None):self.error=error;self.calls=0
 def stat(self):
  self.calls+=1
  if self.error:raise self.error
  class S:st_mode=0o100600;st_size=123
  return S()
class Root:
 def __init__(self,items):self.items=items
 def rglob(self,_):return iter(self.items)
def main():
 root=B/'main-transient-artifact-owner-check-v1';dst=A/'transient-artifact-owner-check-v1.json';assert not root.exists() and not dst.exists()
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
  rec=dict(schema='strata-transient-artifact-owner-check-v1',passed=False,complete=False,GPU_executed=False,helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),checks=[],commands=[])
  # Deterministic deletion between enumeration and metadata: missing contributes
  # zero, surviving regular file contributes exact bytes with only one stat.
  gone=Item(FileNotFoundError(2,'gone'));live=Item();assert m.artifact_bytes(Root([gone,live]))==123 and gone.calls==live.calls==1
  rec['checks'].append('deterministic disappeared entry skipped; present file exact bytes/one stat')
  try:m.artifact_bytes(Root([Item(PermissionError(13,'denied'))]));raise AssertionError('permission hidden')
  except PermissionError:rec['checks'].append('permission failure propagates')
  owner=m.Owner(root/'owned');env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
  script="from pathlib import Path;import sys,time;p=Path(sys.argv[1]);p.mkdir();\nfor i in range(1200):\n q=p/str(i);q.write_bytes(b'x'*128);q.unlink();time.sleep(.001)\nprint('transient_files=1200 status=PASS',flush=True)"
  c,so,se=owner.run('transient-writer',['/usr/bin/python3','-B','-c',script,str(root/'transient')],env,P,wall=15,cpu=10,rss_cap=256<<20,text_cap=1<<20,file_cap=1<<20,total_cap=2<<20)
  rec['commands']=owner.commands;assert m.closed(c) and c['exit_code']==0 and 'status=PASS' in so.read_text()
  rec['checks'].append('actual owned transient writer normal exit, anchored observation/reap/empty session, no cleanup signals')
  over="from pathlib import Path;import sys,time;Path(sys.argv[1]).write_bytes(b'x'*65536);time.sleep(10)"
  c,so,se=owner.run('artifact-cap-negative',['/usr/bin/python3','-B','-c',over,str(root/'oversize')],env,P,wall=15,cpu=10,rss_cap=256<<20,text_cap=1<<20,file_cap=1<<20,total_cap=32768)
  assert 'artifact cap' in c.get('error','') and c['direct_child_reaped'] and c['session_empty'] and not c['survivors'] and c['cleanup'] and not m.closed(c)
  rec['checks'].append('artifact cap still fails and pinned-pidfd cleanup reaps/empties owned session; negative result not normal PASS')
  for c in owner.commands:
   for ident in c['owners']+[c['direct_identity']]:
    now=m.identity(ident['pid']);assert now is None or now['start_ticks']!=ident['start_ticks']
  rec.update(passed=True,complete=True,all_owned_identities_absent=True,ownership_closed=True)
  dst.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({'passed':True,'GPU_executed':False,'checks':rec['checks']}))
if __name__=='__main__':main()
