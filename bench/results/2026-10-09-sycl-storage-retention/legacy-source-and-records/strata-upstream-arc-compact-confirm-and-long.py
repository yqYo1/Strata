import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe=Path('/tmp/strata-upstream-arc-compact-diag-jit')
refpath=Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json');ref=json.loads(refpath.read_text());a=ref['runs'][0]['args']
expected={r['tokens']:(r['logits_sha256'],r['logits_count'],r['output_ids']) for r in ref['runs']}
env=dict(os.environ,**ref['env'])
for k in ['STRATA_PREFILL_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_PRELOAD_PLE','STRATA_SERVE_NO_MTP','STRATA_DUMP_FIRST_LOGITS']:env.pop(k,None)
env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',STRATA_PREFILL_COMPACT='1')
out=Path('/tmp/strata-upstream-arc-compact-confirm');out.mkdir(exist_ok=True)
for script,name,reference in [('bench/results/2026-10-05-sycl-upstream-arc/serve_check.py','normal-mtp',False),('bench/results/2026-10-05-sycl-prefill-scaling/gdn/serve_long_check.py','long-mtp',True)]:
 dest=out/(name+'.json');args=[sys.executable,str(root/script),'--engine',str(exe),'--out',str(dest)]
 if reference:args+=['--reference',str(refpath)]
 print(name+' compact start',flush=True);subprocess.run(args,cwd=root,env=env,check=True,timeout=600)
 previous=Path('/tmp/strata-upstream-arc-attn-order-normal-serve.json') if not reference else Path('/tmp/strata-upstream-arc-attn-order-long-serve/layout4.json')
 old=json.loads(previous.read_text());new=json.loads(dest.read_text())
 for r,s in zip(old['requests'],new['requests']):
  assert r['ids']==s['ids'] and r['logprobs']==s['logprobs'],s['name']
  if 'logits_sha256' in r:assert r['logits_sha256']==s['logits_sha256'] and s['logits_finite']
 assert len(old['requests'])==len(new['requests'])==4
 print(name+' all four IDs/logprobs and available complete heads match original',flush=True)
common=[sys.executable,str(root/'sycl/tools/prefill_profile.py'),'--exe',str(exe),'--pack',a[a.index('--pack')+1],'--native',a[a.index('--native')+1],'--expert-profile',a[a.index('--expert-profile')+1],'--expert-cache','128','--cwd',str(root),'--repeats','1']
for pair in range(1,4):
 for label,compact in ([('baseline',0),('compact',1)] if pair%2 else [('compact',1),('baseline',0)]):
  dest=out/f'{label}-p{pair}'
  args=common+['--tokens-file',ref['fixture'],'--lengths','4096,8087','--max-context','8192','--chunk','8192','--mode','wall','--label',f'{label}-p{pair}','--output',str(dest)]
  if pair%2==0:args+=['--reverse-order']
  print(f'pair={pair} {label} normal CLI start',flush=True);subprocess.run(args,cwd=root,env=dict(env,STRATA_PREFILL_COMPACT=str(compact)),check=True,timeout=600)
  for r in json.loads((dest/'run.json').read_text())['runs']:
   assert (r['logits_sha256'],r['logits_count'],r['output_ids'])==expected[r['tokens']] and r['logits_finite'] and r['chunks']==1
   assert not any(k in r for k in ['transfer','phases','preload'])
  print(f'pair={pair} {label} full heads bit-identical',flush=True)
 if pair==1:
  prior={}
  for label,compact in [('baseline',0),('compact',1)]:
   dest=out/f'long-{label}-chunk4096'
   args=common+['--tokens-file','/tmp/strata-upstream-arc-long-context-fixture/coding-context-64k-tokens.txt','--lengths','16384,32768,65536','--max-context','65538','--chunk','4096','--multiple-chunks','--mode','transfer','--timeout','900','--label',f'long-{label}','--output',str(dest)]
   print(label+' long 16K/32K/64K same 4096 chunk start',flush=True);subprocess.run(args,cwd=root,env=dict(env,STRATA_PREFILL_COMPACT=str(compact)),check=True,timeout=2100)
   for r in json.loads((dest/'run.json').read_text())['runs']:
    assert r['logits_finite'] and r['logits_count']==248320 and r['reported_tokens']==r['tokens']
    assert r['chunks']==r['tokens']//4096 and r['transfer']['chunks']==r['chunks']
    head=(r['logits_sha256'],r['logits_count'],r['output_ids'])
    if compact:assert head==prior[r['tokens']],(r['tokens'],head,prior[r['tokens']])
    else:prior[r['tokens']]=head
   print(label+' all long full heads finite and chunk counts correct'+('; original heads bit-identical' if compact else ''),flush=True)
print('PASS compact model/MTP checks, three normal CLI pairs and long 16K/32K/64K same-chunk parity',flush=True)
