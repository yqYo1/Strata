import json,os,subprocess,sys
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
ref=Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json')
env=dict(os.environ,**json.loads(ref.read_text())['env'])
for label,topk,reverse in [('attention',0,False),('attention-topk',1,True)]:
 out=Path('/tmp/strata-upstream-arc-topk-resident-compare')/label
 args=[sys.executable,str(root/'sycl/tools/prefill_resident_profile.py'),'--reference',str(ref),'--exe','/tmp/strata-upstream-arc-topk-resident-jit','--output',str(out),'--warmup','--repeats','3','--mode','profile','--cwd',str(root),'--set-env','STRATA_PREFILL_ATTN_BATCH=128','--set-env','STRATA_PREFILL_ATTN_LAYOUT=3','--set-env',f'STRATA_PREFILL_TOPK_TUNED={topk}']
 if reverse:args+=['--reverse-order']
 print(label,'start',flush=True)
 subprocess.run(args,cwd=root,env=env,check=True,timeout=600)
 print(label,'done',flush=True)
