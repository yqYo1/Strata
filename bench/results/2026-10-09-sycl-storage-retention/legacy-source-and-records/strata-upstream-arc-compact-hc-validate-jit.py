import json,os,subprocess,sys,shutil,hashlib
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
refpath=Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json');ref=json.loads(refpath.read_text())
env=dict(os.environ,**ref['env']);env['SYCL_CACHE_PERSISTENT']='0';env['STRATA_PREFILL_COMPACT']='0'
env['STRATA_PLE_GGUF']=str(Path.home()/'.local/share/strata-sycl/models/qwen3.8-flash-next-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf')
for k in list(env):
 if k.startswith('STRATA_GDN_'):env.pop(k)
with open('/tmp/strata-upstream-arc-compact-hc-jit-ctest.log','w') as log:
 subprocess.run(['ctest','--test-dir',str(root/'build-sycl-upstream-jit'),'--output-on-failure','--timeout','180','-j','1'],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
exe=Path('/tmp/strata-upstream-arc-compact-hc-jit');shutil.copy2(root/'build-sycl-upstream-jit/strata',exe)
print('JIT 30 tests passed; snapshot',hashlib.sha256(exe.read_bytes()).hexdigest(),flush=True)
for label,compact in [('baseline',0),('compact-hc',2)]:
 args=[sys.executable,str(root/'sycl/tools/prefill_resident_profile.py'),'--reference',str(refpath),'--exe',str(exe),'--output','/tmp/strata-upstream-arc-compact-hc-resident-initial/'+label,'--repeats','1','--warmup','--mode','profile','--cwd',str(root)]
 for k,v in dict(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_COMPACT=str(compact),STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1').items():args+=['--set-env',k+'='+v]
 subprocess.run(args,env=env,cwd=root,check=True,timeout=900)
 print(label+' full 4k/8k heads match original',flush=True)
