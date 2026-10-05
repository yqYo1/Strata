import array,hashlib,json,math,os,re,subprocess,time
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe=Path('/tmp/strata-upstream-arc-cache-policy6-jit')
out=Path('/tmp/strata-residual-control-check');out.mkdir(exist_ok=True)
ref=json.loads(Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json').read_text());a=ref['runs'][0]['args']
env=dict(os.environ,**ref['env'])
for k in list(env):
 if k.startswith('STRATA_'):env.pop(k)
env.update(STRATA_IO_THREADS='16',STRATA_PREFILL_RING='8',STRATA_PREFILL_FIRST='0',STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',SYCL_CACHE_PERSISTENT='1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def value(k):return a[a.index(k)+1]
report=dict(binary_sha256=sha(exe),runs=[],completed=False)
def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
 for name,n,c,gpu,first,inplace in [('control-257',257,128,257,0,0)]:
  assert subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()=='ActiveState=inactive'
  dest=out/name;dest.mkdir(exist_ok=True)
  baseline=Path(f'/tmp/strata-upstream-arc-layer-major-gpu-initial/baseline-{n}-c{c}')
  runenv=dict(env,STRATA_PREFILL_FIRST=str(first),STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu),STRATA_PREFILL_LAYER_MAJOR_R_INPLACE=str(inplace),STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',STRATA_PREFILL_CACHE_RESTORE='ram',STRATA_PREFILL_CACHE_VERIFY='1',STRATA_DUMP_FIRST_LOGITS=str(dest/'head.bin'),STRATA_PREFILL_DUMP_STATE=str(dest/'state.bin'),STRATA_PREFILL_DUMP_R=str(dest/'residual.bin'),STRATA_PREFILL_DUMP_R_ALL='1',STRATA_PREFILL_TRANSFER_TIMING='1',STRATA_DBG_NAN='1')
  args=[str(exe),'--pack',value('--pack'),'--native',value('--native'),'--tokens-file',str(baseline/'tokens.txt'),'--max-new','1','--spec','2','--max-context','8194','--prefill',str(c),'--no-prefill-borrow','--expert-cache','128','--expert-profile',value('--expert-profile'),'--expert-cache-per-layer','--pool-workers','5','--pcie-frac','0','--adapt-swaps','0','--ple-io','direct','--suffix-draft','0','--greedy','--stats']
  print(name+' start',flush=True);start=time.monotonic()
  with (dest/'engine.log').open('w') as log:rc=subprocess.run(args,cwd=root,env=runenv,stdout=log,stderr=subprocess.STDOUT).returncode
  record=dict(name=name,tokens=n,chunk=c,gpu_tokens=gpu,first=first,inplace=inplace,args=args,env={k:v for k,v in runenv.items() if k.startswith(('STRATA_','ONEAPI_','SYCL_')) or k=='LD_LIBRARY_PATH'},exit_code=rc,wall_seconds=time.monotonic()-start)
  report['runs'].append(record);save();assert rc==0,name
  v=array.array('f');v.frombytes((dest/'head.bin').read_bytes());assert len(v)==248320 and all(map(math.isfinite,v))
  text=(dest/'engine.log').read_text();assert 'non-finite (max' in text and not re.search(r' [1-9][0-9]* non-finite',text)
  assert (dest/'residual.bin').stat().st_size==n*(8+10240*4)
  for key in ['head.bin','state.bin','residual.bin']:
   record[key]=dict(bytes=(dest/key).stat().st_size,sha256=sha(dest/key))
   comparison=(out/'short-first-control') if first else baseline
   assert sha(dest/key)==sha(comparison/key),(name,key)
  transfers=json.loads(re.search(r'^strata prefill transfer: (.*)$',text,re.M)[1])
  if inplace:assert transfers['device_residual_bytes']==0,(name,transfers)
  assert transfers['activation_bytes']==2*47*(n-transfers['residual_gpu_tokens'])*10240*4
  record.update(all_head_residual_state_bytes_equal=True,transfer=transfers);save();print(name+' PASS',flush=True)
 report['completed']=True;save()
except BaseException as e:report['terminal_error']=repr(e);save();raise
