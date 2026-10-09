import json, os, subprocess, sys
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
ref=json.loads(Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json').read_text())
a=ref['runs'][0]['args']
expected={r['tokens']:(r['logits_sha256'],r['logits_count'],r['output_ids']) for r in ref['runs']}
env=dict(os.environ,**ref['env'])
for key in ['STRATA_PREFILL_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_PRELOAD_PLE','STRATA_SERVE_NO_MTP','STRATA_DUMP_FIRST_LOGITS']:
 env.pop(key,None)
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1')
print('normal-MTP serve regression start',flush=True)
subprocess.run([sys.executable,str(root/'bench/results/2026-10-05-sycl-upstream-arc/serve_check.py'),'--engine','/tmp/strata-upstream-arc-attn-order-jit','--out','/tmp/strata-upstream-arc-attn-order-normal-serve.json'],cwd=root,env=env,check=True,timeout=600)
common=[sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe','/tmp/strata-upstream-arc-attn-order-jit','--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],'--expert-profile',a[a.index('--expert-profile')+1],'--tokens-file',ref['fixture'],'--lengths','4096,8087','--max-context','8192','--chunk','8192','--expert-cache','128','--mode','wall','--repeats','1','--cwd',str(root)]
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1')
for pair in range(1,4):
 for label,layout in ([('layout3',3),('layout4',4)] if pair%2 else [('layout4',4),('layout3',3)]):
  dest=Path('/tmp/strata-upstream-arc-attn-order-cli-confirm')/f'{label}-p{pair}'
  run_env=dict(env,STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT=str(layout),STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1')
  print(f'pair={pair} label={label} start',flush=True)
  subprocess.run(common+(['--reverse-order'] if pair%2==0 else [])+['--label',f'{label}-p{pair}','--output',str(dest)],cwd=root,env=run_env,check=True,timeout=600)
  result=json.loads((dest/'run.json').read_text())
  for r in result['runs']:
   assert (r['logits_sha256'],r['logits_count'],r['output_ids'])==expected[r['tokens']],r['name']
   assert r['logits_finite'] and r['chunks']==1 and not any(k in r for k in ('transfer','phases','preload'))
  print(f'pair={pair} label={label} all-heads-bit-identical',flush=True)
