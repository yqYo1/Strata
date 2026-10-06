import array,hashlib,json,math,os,re,struct,subprocess,time
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
recovery=Path('/home/yayoi/.local/state/strata-sycl/residual-inplace-recovery')
exe=recovery/'strata-upstream-arc-residual-inplace2-jit'
out=Path('/home/yayoi/.local/state/strata-sycl/gpu-stall-diagnosis/post-reboot-20261006/residual-proof');out.mkdir(exist_ok=True)
ref=json.loads((recovery/'reference.json').read_text());a=ref['runs'][0]['args']
env=dict(os.environ,**ref['env'])
for k in list(env):
 if k.startswith('STRATA_'):env.pop(k)
env.update(STRATA_IO_THREADS='16',STRATA_PREFILL_RING='8',STRATA_PREFILL_FIRST='0',STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',SYCL_CACHE_PERSISTENT=os.environ.get('SYCL_CACHE_PERSISTENT','0'))
def run_diagnostic(args,runenv,name,log):
 stage=Path('/home/yayoi/.local/state/strata-sycl/gpu-stall-diagnosis/post-reboot-20261006')
 command=['python3',str(stage/'run-stage.py'),'--runtime','custom','--no-persistent-cache','--timeout','600']
 for key,value in runenv.items():
  if key.startswith('STRATA_'):command+=['--env',key+'='+value]
 command+=['residual-'+name,'--','/usr/bin/gdb','--batch','--return-child-result','-iex','set debuginfod enabled off','-x',str(stage/'first-fault.gdb'),'--args']+args
 parent=subprocess.run(command,cwd=root,stdout=subprocess.DEVNULL,stderr=log)
 prefix=stage/('residual-'+name)
 record=json.loads(prefix.with_suffix('.json').read_text())
 log.write(prefix.with_suffix('.stdout').read_text()+prefix.with_suffix('.stderr').read_text())
 if parent.returncode:raise RuntimeError('Diagnostic runner failed: '+name)
 # Never advance to another GPU test when its predecessor remains alive.
 assert not record.get('still_live',False),record
 return record['exit_code']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def value(k):return a[a.index(k)+1]
def finite_residual_rows(path,n):
 # Each record is an int64 position followed by 10240 FP32 values. Inspect
 # every value without treating the integer header as a floating-point value.
 with path.open('rb') as f:
  for pos in range(n):
   assert struct.unpack('<q',f.read(8))[0]==pos,(path,pos)
   row=array.array('f');row.frombytes(f.read(10240*4))
   assert len(row)==10240 and all(map(math.isfinite,row)),(path,pos)
  assert f.read(1)==b'',path
report=dict(binary_sha256=sha(exe),runs=[],completed=False)
def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
 for name,n,c,gpu,first,inplace in [('all-257',257,128,257,0,1),('mixed-257',257,128,128,0,1),('host-257',257,128,0,0,1),('single-4096',4096,4096,4096,0,1),('all-8087',8087,4096,8087,0,1),('mixed-8087',8087,4096,4096,0,1),('short-first-control',257,128,257,32,0),('short-first-all',257,128,257,32,1),('short-first-mixed',257,128,160,32,1)]:
  assert subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()=='ActiveState=inactive'
  dest=out/name;dest.mkdir(exist_ok=True)
  baseline=(recovery/f'baseline-{n}-c{c}')
  runenv=dict(env,STRATA_PREFILL_FIRST=str(first),STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu),STRATA_PREFILL_LAYER_MAJOR_R_INPLACE=str(inplace),STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',STRATA_PREFILL_CACHE_RESTORE='ram',STRATA_PREFILL_CACHE_VERIFY='1',STRATA_DUMP_FIRST_LOGITS=str(dest/'head.bin'),STRATA_PREFILL_DUMP_STATE=str(dest/'state.bin'),STRATA_PREFILL_DUMP_R=str(dest/'residual.bin'),STRATA_PREFILL_DUMP_R_ALL='1',STRATA_PREFILL_TRANSFER_TIMING='1')
  args=[str(exe),'--pack',value('--pack'),'--native',value('--native'),'--tokens-file',str(baseline/'tokens.txt'),'--max-new','1','--spec','2','--max-context','8194','--prefill',str(c),'--no-prefill-borrow','--expert-cache','128','--expert-profile',value('--expert-profile'),'--expert-cache-per-layer','--pool-workers','5','--pcie-frac','0','--adapt-swaps','0','--ple-io','direct','--suffix-draft','0','--greedy','--stats']
  print(name+' start',flush=True);start=time.monotonic()
  with (dest/'engine.log').open('w') as log:rc=run_diagnostic(args,runenv,name,log)
  record=dict(name=name,tokens=n,chunk=c,gpu_tokens=gpu,first=first,inplace=inplace,args=args,env={k:v for k,v in runenv.items() if k.startswith(('STRATA_','ONEAPI_','SYCL_')) or k=='LD_LIBRARY_PATH'},exit_code=rc,wall_seconds=time.monotonic()-start)
  report['runs'].append(record);save();assert rc==0,name
  v=array.array('f');v.frombytes((dest/'head.bin').read_bytes());assert len(v)==248320 and all(map(math.isfinite,v))
  text=(dest/'engine.log').read_text();assert f'compact-hc layout for {c} tokens' in text,name
  assert (dest/'residual.bin').stat().st_size==n*(8+10240*4)
  finite_residual_rows(dest/'residual.bin',n)
  for key in ['head.bin','state.bin','residual.bin']:
   record[key]=dict(bytes=(dest/key).stat().st_size,sha256=sha(dest/key))
   comparison=(out/'short-first-control') if first else baseline
   assert sha(dest/key)==sha(comparison/key),(name,key)
  transfers=json.loads(re.search(r'^strata prefill transfer: (.*)$',text,re.M)[1])
  if inplace:assert transfers['device_residual_bytes']==0,(name,transfers)
  assert transfers['activation_bytes']==2*47*(n-transfers['residual_gpu_tokens'])*10240*4
  record.update(all_head_residual_state_bytes_equal=True,head_and_all_residual_values_finite=True,transfer=transfers);save();print(name+' PASS',flush=True)
 report['completed']=True;save()
except BaseException as e:report['terminal_error']=repr(e);save();raise
