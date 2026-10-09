import json,os,subprocess,sys
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
refpath=Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json');ref=json.loads(refpath.read_text())
env=dict(os.environ,**ref['env'])
for key in ['STRATA_PREFILL_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_PRELOAD_PLE','STRATA_SERVE_NO_MTP','STRATA_DUMP_FIRST_LOGITS']:env.pop(key,None)
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='3',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0')
records=[]
for label,tuned in [('original',0),('tuned',1)]:
 dest=Path('/tmp/strata-upstream-arc-gdn-long-serve')/(label+'.json')
 subprocess.run([sys.executable,str(root/'bench/results/2026-10-05-sycl-prefill-scaling/gdn/serve_long_check.py'),'--engine','/tmp/strata-upstream-arc-gdn-tuned-jit','--reference',str(refpath),'--out',str(dest)],cwd=root,env=dict(env,STRATA_GDN_KEYHEAD_TUNED=str(tuned)),check=True,timeout=600)
 records.append(json.loads(dest.read_text()))
 print(label+' long MTP/checkpoint regression passed',flush=True)
assert records[0]['requests'][0]['logits_sha256']==records[1]['requests'][0]['logits_sha256']
for a,b in zip(records[0]['requests'],records[1]['requests']):assert a['ids']==b['ids'] and a['logprobs']==b['logprobs']
print('PASS original/tuned long first head identical; all four requests IDs and logprobs equal',flush=True)
