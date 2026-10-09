import json,os,subprocess,sys,shutil,hashlib
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
reference=Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json');ref=json.loads(reference.read_text())
env=dict(os.environ,**ref['env']);env['SYCL_CACHE_PERSISTENT']='0'
env['STRATA_PLE_GGUF']=str(Path.home()/'.local/share/strata-sycl/models/qwen3.8-flash-next-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf')
with open('/tmp/strata-upstream-arc-selection-aot-ctest.log','w') as log:
 subprocess.run(['ctest','--test-dir',str(root/'build-sycl-upstream-aot'),'--output-on-failure','--timeout','180','-j','1'],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
exe=Path('/tmp/strata-upstream-arc-selection-aot');shutil.copy2(root/'build-sycl-upstream-aot/strata',exe)
print('AOT 28 tests passed; snapshot',hashlib.sha256(exe.read_bytes()).hexdigest(),flush=True)
a=ref['runs'][0]['args']
args=[sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe',str(exe),'--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],'--expert-profile',a[a.index('--expert-profile')+1],'--tokens-file',ref['fixture'],'--lengths','4096,8087','--chunk','8192','--max-context','8192','--expert-cache','128','--mode','wall','--repeats','1','--label','aot-attention-topk','--output','/tmp/strata-upstream-arc-selection-aot-cli','--cwd',str(root)]
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='3',STRATA_PREFILL_TOPK_TUNED='1')
subprocess.run(args,env=env,cwd=root,check=True,timeout=600)
expected={r['tokens']:(r['logits_sha256'],r['logits_count'],r['output_ids']) for r in ref['runs']}
j=json.load(open('/tmp/strata-upstream-arc-selection-aot-cli/run.json'))
for r in j['runs']:
 assert (r['logits_sha256'],r['logits_count'],r['output_ids'])==expected[r['tokens']],r['name']
 assert r['chunks']==1 and r['logits_finite'] and not any(k in r for k in ('phases','transfer','preload'))
print('AOT full 4k and 8k first heads match original',flush=True)
subprocess.run([sys.executable,str(root/'bench/results/2026-10-05-sycl-upstream-arc/serve_check.py'),'--engine',str(exe),'--out','/tmp/strata-upstream-arc-selection-aot-serve.json'],env=env,cwd=root,check=True,timeout=600)
print('AOT normal MTP serve passed',flush=True)
