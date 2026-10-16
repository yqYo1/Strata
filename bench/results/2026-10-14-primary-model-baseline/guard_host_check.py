import fcntl, importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path
p=Path('bench/results/2026-10-14-primary-model-baseline')
sp=importlib.util.spec_from_file_location('runner',p/'run_server_job_v2.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
b=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
with (b/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with tempfile.TemporaryDirectory(prefix='strata-stop-guard-host-') as tmp:
  root=Path(tmp);(root/'owned').mkdir()
  (root/'plan.json').write_text(json.dumps({'expected_long_prompt_tokens':61055}))
  code="import sys; line=sys.stdin.buffer.readline(); sys.stdout.buffer.write(line); sys.stdout.buffer.flush()"
  child=subprocess.Popen([sys.executable,'-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
  try:
   v=m.identity(child.pid); native={k:v[k] for k in ['pid','ppid','sid','start_ticks']}
   v=m.identity(os.getpid()); wrapper={k:v[k] for k in ['pid','ppid','sid','start_ticks']}
   pipe=os.stat('/proc/'+str(child.pid)+'/fd/0')
   entry={'native':native,'wrapper':wrapper,'engine':str(Path('/proc',str(child.pid),'exe').resolve()),'stdin':{'device':pipe.st_dev,'inode':pipe.st_ino}}
   (root/'engine-start.json').write_text(json.dumps(entry))
   (root/'owned/engine.stderr').write_text('strata trace: prompt start 28 -1\nstrata trace: prompt chunk 0 of 28\n')
   rejected=False
   try:m.stop_single_long(root,{})
   except AssertionError:rejected=True
   assert rejected and child.poll() is None
   (root/'owned/engine.stderr').write_text('strata trace: prompt start 61054 -1\nstrata trace: prompt chunk 0 of 61054\n')
   entry['native']['start_ticks']+=1;(root/'engine-start.json').write_text(json.dumps(entry))
   stale=False
   try:m.stop_single_long(root,{})
   except AssertionError:stale=True
   assert stale and child.poll() is None
   entry['native']['start_ticks']-=1;(root/'engine-start.json').write_text(json.dumps(entry))
   action={};m.stop_single_long(root,action)
   assert action['sent'] and child.stdout.read()==b'STOP\n' and child.wait(timeout=5)==0
   result={'passed':True,'scope':'host-only FIFO and ownership guard; no model/GPU proof','rejected_wrong_prompt':rejected,'rejected_stale_identity':stale,'exact_pipe_command':'STOP\\n','stub_exit_code':0}
   (p/'guard-host-check.json').write_text(json.dumps(result,indent=2)+'\n')
   print(json.dumps(result))
  finally:
   if child.poll() is None:child.kill();child.wait()
