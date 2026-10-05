"""Run after GPU recovery. Fill the configured context, with normal MTP in serve.

Finite jobs end naturally. Protocol timeouts queue QUIT and wait; they never kill
an active GPU process. The full tests can take substantially longer than 64K.
"""
import argparse,array,hashlib,json,math,os,re,selectors,subprocess,time
from pathlib import Path
from observe_memory import MemoryObserver
p=argparse.ArgumentParser()
p.add_argument('--context',type=int,default=262144)
p.add_argument('--serve-chunk',type=int,default=128)
p.add_argument('--gpu-rows',type=int,default=0)
p.add_argument('--recovery',type=Path,default=Path.home()/'.local/state/strata-sycl/residual-inplace-recovery')
p.add_argument('--stage',choices=['boundary','cli','serve'],required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[4];recovery=a.recovery
exe=recovery/'strata-upstream-arc-residual-inplace2-jit'
ref=json.loads((recovery/'reference.json').read_text());args0=ref['runs'][0]['args']
def value(k):return args0[args0.index(k)+1]
env=dict(os.environ,**ref['env'])
for k in list(env):
 if k.startswith('STRATA_'):env.pop(k)
env.update(STRATA_IO_THREADS='16',STRATA_PREFILL_RING='8',STRATA_PREFILL_FIRST='0',STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(a.gpu_rows),STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',STRATA_PREFILL_CACHE_RESTORE='ram',SYCL_CACHE_PERSISTENT='1')
source=list(map(int,(recovery/'coding-context-256k-tokens.txt').read_text().split()))
assert len(source)>=a.context and a.context>=64
out=recovery/f'{a.stage}-{a.context}';out.mkdir(exist_ok=True)
report=dict(stage=a.stage,context=a.context,completed=False,runs=[],binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),env={k:v for k,v in env.items() if k.startswith(('STRATA_','ONEAPI_','SYCL_')) or k=='LD_LIBRARY_PATH'})
def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
def finite_head(path):
 v=array.array('f');v.frombytes(path.read_bytes());assert len(v)==248320 and all(map(math.isfinite,v));return hashlib.sha256(path.read_bytes()).hexdigest()
common=[str(exe),'--pack',value('--pack'),'--native',value('--native'),'--max-context',str(a.context),'--kv','int8','--spec','4','--suffix-draft','0','--no-prefill-borrow','--expert-cache','128','--expert-profile',value('--expert-profile'),'--expert-cache-per-layer','--pool-workers','5','--pcie-frac','0','--adapt-swaps','0','--ple-io','direct','--greedy']
assert subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()=='ActiveState=inactive'
save()
try:
 if a.stage in ('boundary','cli'):
  cases=[('full-no-output-room',a.context,1,False),('one-too-many',a.context-2,3,False)] if a.stage=='boundary' else [('fills-context',a.context-2,2,True)]
  for name,n,new,ok in cases:
   fixture=out/(name+'.tokens.txt');fixture.write_text(' '.join(map(str,source[:n]))+'\n')
   head=out/(name+'.head.bin');head.unlink(missing_ok=True);runenv=dict(env,STRATA_DUMP_FIRST_LOGITS=str(head))
   # 1K fits the old 256K capacity probe; normal MTP is tested separately.
   args=common+['--tokens-file',str(fixture),'--max-new',str(new),'--prefill','1024','--stats']
   start=time.monotonic()
   with (out/(name+'.log')).open('w') as log:
    child=subprocess.Popen(args,cwd=root,env=runenv,stdout=log,stderr=subprocess.STDOUT)
    observer=MemoryObserver(child.pid,out/(name+'.memory.jsonl'))
    try:rc=child.wait()
    finally:memory=observer.finish()
   text=(out/(name+'.log')).read_text();r=dict(name=name,args=args,input_tokens=n,max_new=new,exit_code=rc,wall_seconds=time.monotonic()-start,memory=memory);report['runs'].append(r);save()
   if ok:
    assert rc==0,(name,rc);m=re.search(r'^output\s*:\s*(.*)$',text,re.M);ids=list(map(int,m[1].split()));assert len(ids)==new
    assert f'prefill {n-1} tokens' in text;r.update(ids=ids,logits_sha256=finite_head(head),logical_length=n+len(ids));assert r['logical_length']==a.context
   else:assert rc==2 and 'must fit the prompt and generation' in text
   save()
 else:
  args=common+['--serve','--mtp',str(Path.home()/'.local/share/strata-sycl/mtp/rt'),'--spec-split','--prefill',str(a.serve_chunk),'--prompt-cache','0']
  head=out/'first-head.bin';runenv=dict(env,STRATA_DUMP_FIRST_LOGITS=str(head),STRATA_TRACE='1')
  report['args']=args;save()
  with (out/'engine.log').open('w') as log:
   child=subprocess.Popen(args,cwd=root,env=runenv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
   observer=MemoryObserver(child.pid,out/'engine.memory.jsonl')
   sel=selectors.DefaultSelector();sel.register(child.stdout,selectors.EVENT_READ);pending=bytearray()
   def line(timeout=7200):
    end=time.monotonic()+timeout
    while True:
     if b'\n' in pending:
      b,_,tail=pending.partition(b'\n');pending[:]=tail;return b.decode().strip()
     remaining=end-time.monotonic()
     if remaining<=0 or not sel.select(remaining):raise TimeoutError('engine protocol timeout; no forced kill')
     b=os.read(child.stdout.fileno(),65536)
     if not b:raise RuntimeError('engine exited before reply')
     pending.extend(b)
   try:
    startup=[]
    while True:
     s=line();startup.append(s)
     if s.startswith('ERR'):raise RuntimeError(s)
     if s.startswith('READY '):break
    report['startup']=startup;save()
    # A large request fills the window. Small boundary runs also exercise T=1,
    # T=2 tails and request reuse with the same normal MTP graphs.
    cases=[('fills-context',a.context-4,4,True),('no-room',a.context,1,False),('one-too-many',a.context-2,3,False),('works-after-refusal',37,2,True)]
    for name,n,new,ok in cases:
     trace_start=log.tell()
     if ok:head.unlink(missing_ok=True)
     ids=source[:n];request=f'GEN {new} logprobs=5 '+','.join(map(str,ids))+'\n';child.stdin.write(request.encode());child.stdin.flush()
     lines=[];output=[];lp=[];start=time.monotonic()
     while True:
      s=line();lines.append(s)
      if s.startswith('T '):output.append(int(s.split()[1]))
      if s.startswith('LP '):
       assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in s.split()[1:]);lp.append(s)
      if s.startswith(('DONE ','ERR ')):break
     r=dict(name=name,input_tokens=n,max_new=new,ids=output,protocol=lines,wall_seconds=time.monotonic()-start);report['runs'].append(r);save()
     with (out/'engine.log').open('rb') as trace:
      trace.seek(trace_start);text=trace.read(log.tell()-trace_start).decode()
     windows=[(int(pos),int(count)) for pos,count in re.findall(r'strata trace: window (-?\d+) (-?\d+)',text)]
     r['verify_windows']=windows
     assert all(pos>=0 and count>0 and pos+count<=a.context for pos,count in windows),windows
     if ok:
      assert lines[-1].startswith('DONE ') and len(output)==new and len(lp)==new,lines[-10:]
      r['logits_sha256']=finite_head(head)
      if name=='fills-context':
       assert n+len(output)==a.context;r['logical_length']=a.context
       # S=4 with confidence clipping disabled reaches the final allocated cell,
       # even when the last speculative token is rejected rather than emitted.
       assert windows and max(pos+count for pos,count in windows)==a.context,windows
       r['last_executed_kv_cell']=a.context-1
     else:assert lines[-1].startswith('ERR prompt') and not output
     save()
    child.stdin.write(b'QUIT\n');child.stdin.flush();rc=child.wait();assert rc==0;report['exit_code']=rc
   finally:
    if child.poll() is None:
     try:child.stdin.write(b'QUIT\n');child.stdin.flush()
     except BrokenPipeError:pass
     child.wait()
    report['memory']=observer.finish();save()
 report['completed']=True;save()
except BaseException as e:report['terminal_error']=repr(e);save();raise
