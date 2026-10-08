"""Run after GPU recovery. Fill the configured context, with normal MTP in serve.

Finite jobs end naturally. Explicit job deadlines bound protocol I/O and cleanup;
failed cleanup records a surviving PID. Full tests can take longer than 64K.
"""
import argparse,array,hashlib,json,math,os,re,selectors,signal,subprocess,time
from pathlib import Path
from observe_memory import MemoryObserver
p=argparse.ArgumentParser()
p.add_argument('--context',type=int,default=262144)
p.add_argument('--serve-chunk',type=int,default=128)
p.add_argument('--gpu-rows',type=int,default=0)
p.add_argument('--ple-io',choices=['direct','ram'],default='direct',help='PLE table policy; RAM reports use a separate directory')
p.add_argument('--release-draft',action='store_true',help='Lease immutable MTP decode weights during layer-major prefill')
p.add_argument('--verify-draft',action='store_true',help='Compare every leased weight byte before and after restoration')
p.add_argument('--recovery',type=Path,default=Path.home()/'.local/state/strata-sycl/residual-inplace-recovery')
p.add_argument('--executable',type=Path,help='Frozen alternate engine; its reports use a separate directory')
p.add_argument('--environment-file',type=Path,help='Complete execution environment as JSON; replaces the historical reference environment')
p.add_argument('--job-timeout',type=float,default=7200,help='Maximum engine lifetime, seconds')
p.add_argument('--protocol-timeout',type=float,default=7200,help='Maximum time for each protocol reply, seconds')
p.add_argument('--shutdown-timeout',type=float,default=30,help='Grace for QUIT before bounded TERM/KILL cleanup, seconds')
p.add_argument('--stage',choices=['boundary','cli','serve'],required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[4];recovery=a.recovery
exe=a.executable.resolve() if a.executable else recovery/'strata-upstream-arc-residual-inplace2-jit'
ref=json.loads((recovery/'reference.json').read_text());args0=ref['runs'][0]['args']
def value(k):return args0[args0.index(k)+1]
env=json.loads(a.environment_file.read_text()) if a.environment_file else dict(os.environ,**ref['env'])
assert isinstance(env,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in env.items())
assert a.job_timeout>0 and a.protocol_timeout>0 and a.shutdown_timeout>0
for k in list(env):
 if k.startswith('STRATA_'):env.pop(k)
env.update(STRATA_IO_THREADS='16',STRATA_PREFILL_RING='8',STRATA_PREFILL_FIRST='0',STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='4',STRATA_PREFILL_TOPK_TUNED='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='1',STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(a.gpu_rows),STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',STRATA_PREFILL_CACHE_RESTORE='ram',SYCL_CACHE_PERSISTENT=os.environ.get('SYCL_CACHE_PERSISTENT','0'))
if a.verify_draft and not a.release_draft:p.error('--verify-draft requires --release-draft')
if a.release_draft:env['STRATA_PREFILL_RELEASE_DRAFT']='1'
if a.verify_draft:env['STRATA_PREFILL_DRAFT_VERIFY']='1'
source=list(map(int,(recovery/'coding-context-256k-tokens.txt').read_text().split()))
assert len(source)>=a.context and a.context>=64
# The raw coding prefix is an unfinished user message and greedily predicts
# im_end. Use the same verified nine-token assistant suffix as serve_long_check;
# reserve its space inside the requested length. Keep ordinary EOS handling.
assistant_suffix=[248046,198,248045,74455,198,248068,198,248069,271]
def prompt_ids(n):
 ids=source[:n-len(assistant_suffix)]+assistant_suffix
 assert len(ids)==n
 return ids
variant=('-draft-lease' if a.release_draft else '')+('-verified' if a.verify_draft else '')+('-ple-ram' if a.ple_io=='ram' else '')
out=recovery/(f'{a.stage}-{a.context}'+('-'+exe.stem if a.executable else '')+variant);out.mkdir(exist_ok=True)
report=dict(stage=a.stage,context=a.context,ple_io=a.ple_io,completed=False,runs=[],binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),env={k:v for k,v in env.items() if k.startswith(('STRATA_','ONEAPI_','SYCL_','UR_','ZE_')) or k=='LD_LIBRARY_PATH'})
report['assistant_suffix']=assistant_suffix
report['source_fixture_sha256']=hashlib.sha256((recovery/'coding-context-256k-tokens.txt').read_bytes()).hexdigest()
report['environment_file_sha256']=hashlib.sha256(a.environment_file.read_bytes()).hexdigest() if a.environment_file else None
report['timeouts']={'job_seconds':a.job_timeout,'protocol_seconds':a.protocol_timeout,'shutdown_seconds':a.shutdown_timeout}
report['processes']=[]
def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
def start_child(args,**kw):
 child=subprocess.Popen(args,start_new_session=True,**kw)
 stat=Path(f'/proc/{child.pid}/stat').read_text()
 proc=dict(pid=child.pid,start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),started_monotonic=time.monotonic(),exit_code=None,still_alive=True)
 report['processes'].append(proc);save()
 return child,proc
def drain(sel,child,raw):
 if sel is not None and sel.select(0):
  data=os.read(child.stdout.fileno(),65536)
  if data:raw.write(data);raw.flush()
  else:sel.unregister(child.stdout)
def cleanup(child,proc,sel=None,raw=None):
 actions=[]
 if child.poll() is None and child.stdin is not None:
  try:os.write(child.stdin.fileno(),b'QUIT\n');actions.append('QUIT')
  except (BrokenPipeError,BlockingIOError):pass
 def wait_until(seconds):
  end=time.monotonic()+seconds
  while child.poll() is None and time.monotonic()<end:
   drain(sel,child,raw);time.sleep(0.02)
 wait_until(a.shutdown_timeout)
 if child.poll() is None:
  for sig in (signal.SIGTERM,signal.SIGKILL):
   try:os.killpg(child.pid,sig);actions.append(sig.name)
   except ProcessLookupError:break
   wait_until(1)
 drain(sel,child,raw)
 proc.update(exit_code=child.poll(),still_alive=child.poll() is None,cleanup_actions=actions)
 if proc['still_alive']:
  stat=Path(f'/proc/{child.pid}/stat').read_text();proc['start_ticks']=int(stat[stat.rfind(')')+2:].split()[19])
  (recovery/'stalled-writer.json').write_text(json.dumps(proc,indent=2)+'\n')
 save()
 if proc['still_alive']:raise RuntimeError(f'engine PID {child.pid} survived SIGKILL; do not start another GPU job')
 if actions and actions!=['QUIT']:raise RuntimeError(f'engine required forced cleanup: {actions}')
 return child.returncode
def finite_head(path):
 v=array.array('f');v.frombytes(path.read_bytes());assert len(v)==248320 and all(map(math.isfinite,v));return hashlib.sha256(path.read_bytes()).hexdigest()
def ple_startup(text):
 if a.ple_io!='ram':return {}
 m=re.search(r'strata generate: PLE table (locked in RAM|loaded \(not locked\)) \(--ple-io ram\) in ([\d.]+) s',text)
 assert m,'RAM PLE startup record missing'
 seconds=float(m[2]);assert math.isfinite(seconds) and seconds>=0
 return dict(ple_table_locked=m[1]=='locked in RAM',ple_table_startup_seconds=seconds)
common=[str(exe),'--pack',value('--pack'),'--native',value('--native'),'--max-context',str(a.context),'--kv','int8','--spec','4','--spec-min-p','0','--suffix-draft','0','--no-prefill-borrow','--expert-cache','128','--expert-profile',value('--expert-profile'),'--expert-cache-per-layer','--pool-workers','5','--pcie-frac','0','--adapt-swaps','0','--ple-io',a.ple_io,'--greedy']
assert subprocess.check_output(['systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],text=True).strip()=='ActiveState=inactive'
stalled=recovery/'stalled-writer.json'
if stalled.exists():
 previous=json.loads(stalled.read_text());path=Path(f"/proc/{previous['pid']}/stat")
 if path.exists():
  stat=path.read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
  assert ticks!=previous['start_ticks'],'previous GPU process is still alive; no new GPU job'
save()
try:
 if a.stage in ('boundary','cli'):
  cases=[('full-no-output-room',a.context,1,False),('one-too-many',a.context-2,3,False)] if a.stage=='boundary' else [('fills-context',a.context-2,2,True)]
  for name,n,new,ok in cases:
   fixture=out/(name+'.tokens.txt');fixture.write_text(' '.join(map(str,prompt_ids(n)))+'\n')
   head=out/(name+'.head.bin');head.unlink(missing_ok=True);runenv=dict(env,STRATA_DUMP_FIRST_LOGITS=str(head))
   # 1K fits the old 256K capacity probe; normal MTP is tested separately.
   args=common+['--tokens-file',str(fixture),'--max-new',str(new),'--prefill','1024','--stats']
   start=time.monotonic()
   with (out/(name+'.log')).open('w') as log:
    child,proc=start_child(args,cwd=root,env=runenv,stdout=log,stderr=subprocess.STDOUT)
    observer=MemoryObserver(child.pid,out/(name+'.memory.jsonl'))
    try:
     try:rc=child.wait(timeout=a.job_timeout)
     except BaseException as e:report['processing_error']=repr(e);save();raise
     finally:cleanup(child,proc)
    finally:memory=observer.finish()
   text=(out/(name+'.log')).read_text();r=dict(name=name,args=args,input_tokens=n,max_new=new,exit_code=rc,wall_seconds=time.monotonic()-start,memory=memory,fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest());report['runs'].append(r);save()
   if ok:
    r.update(ple_startup(text));save()
    assert rc==0,(name,rc);m=re.search(r'^output\s*:\s*(.*)$',text,re.M);ids=list(map(int,m[1].split()));assert len(ids)==new
    assert f'prefill {n-1} tokens' in text;r.update(ids=ids,logits_sha256=finite_head(head),logical_length=n+len(ids));assert r['logical_length']==a.context
   else:assert rc==2 and 'must fit the prompt and generation' in text
   save()
 else:
  args=common+['--serve','--mtp',str(Path.home()/'.local/share/strata-sycl/mtp/rt'),'--spec-split','--prefill',str(a.serve_chunk),'--prompt-cache','0']
  head=out/'first-head.bin';runenv=dict(env,STRATA_DUMP_FIRST_LOGITS=str(head),STRATA_TRACE='1')
  report['args']=args;save()
  with (out/'engine.log').open('w') as log,(out/'protocol.stdout.raw').open('wb') as raw:
   child,proc=start_child(args,cwd=root,env=runenv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
   os.set_blocking(child.stdin.fileno(),False)
   observer=MemoryObserver(child.pid,out/'engine.memory.jsonl')
   sel=selectors.DefaultSelector();sel.register(child.stdout,selectors.EVENT_READ);pending=bytearray()
   def line():
    end=min(time.monotonic()+a.protocol_timeout,proc['started_monotonic']+a.job_timeout)
    while True:
     if b'\n' in pending:
      b,_,tail=pending.partition(b'\n');pending[:]=tail;return b.decode().strip()
     remaining=end-time.monotonic()
     if remaining<=0 or not sel.select(remaining):raise TimeoutError('engine protocol/job deadline exceeded')
     b=os.read(child.stdout.fileno(),65536)
     if not b:raise RuntimeError('engine exited before reply')
     raw.write(b);raw.flush()
     pending.extend(b)
   def send(data):
    end=min(time.monotonic()+a.protocol_timeout,proc['started_monotonic']+a.job_timeout)
    with selectors.DefaultSelector() as writable:
     writable.register(child.stdin,selectors.EVENT_WRITE)
     view=memoryview(data)
     while view:
      remaining=end-time.monotonic()
      if remaining<=0 or not writable.select(remaining):raise TimeoutError('engine request-write deadline exceeded')
      try:n=os.write(child.stdin.fileno(),view)
      except BlockingIOError:continue
      view=view[n:]
   try:
    startup=[]
    while True:
     s=line();startup.append(s)
     if s.startswith('ERR'):raise RuntimeError(s)
     if s.startswith('READY '):break
    report['startup']=startup
    report.update(ple_startup((out/'engine.log').read_text()));save()
    # Four output tokens can finish without shortening the normal T=4 window.
    # Two force the second verify window to T=2, regardless of draft acceptance.
    cases=[('fills-context',a.context-4,4,True),('fills-context-tail-2',a.context-2,2,True),('no-room',a.context,1,False),('one-too-many',a.context-2,3,False),('works-after-refusal',37,2,True)]
    for name,n,new,ok in cases:
     trace_start=log.tell()
     if ok:head.unlink(missing_ok=True)
     ids=prompt_ids(n);request=f'GEN {new} logprobs=5 '+','.join(map(str,ids))+'\n';report['active_request']=name;save();send(request.encode())
     lines=[];output=[];lp=[];start=time.monotonic()
     while True:
      s=line();lines.append(s)
      if s.startswith('T '):output.append(int(s.split()[1]))
      if s.startswith('LP '):
       assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in s.split()[1:]);lp.append(s)
      if s.startswith(('DONE ','ERR ')):break
     r=dict(name=name,input_tokens=n,max_new=new,ids=output,protocol=lines,wall_seconds=time.monotonic()-start,request_sha256=hashlib.sha256(request.encode()).hexdigest());report['runs'].append(r);save()
     with (out/'engine.log').open('rb') as trace:
      trace.seek(trace_start);text=trace.read(log.tell()-trace_start).decode()
     if a.release_draft:
      releases=re.findall(r'strata mtp decode release: (\d+) physical bytes,.*?total ([\d.]+) ms, wait ([\d.]+) ms, unmap ([\d.]+) ms, verify ([\d.]+) ms, verified=(\d+)',text)
      restores=re.findall(r'strata mtp decode restore: (\d+) payload bytes from RAM, same addresses; total ([\d.]+) ms, map ([\d.]+) ms, copy ([\d.]+) ms, verify ([\d.]+) ms, verified=(\d+)',text)
      r['mtp_decode_lease']={
       'releases':[dict(physical_bytes=int(v[0]),total_ms=float(v[1]),wait_ms=float(v[2]),unmap_ms=float(v[3]),verify_ms=float(v[4]),verified=bool(int(v[5]))) for v in releases],
       'restores':[dict(payload_bytes=int(v[0]),total_ms=float(v[1]),map_ms=float(v[2]),copy_ms=float(v[3]),verify_ms=float(v[4]),verified=bool(int(v[5]))) for v in restores]}
      assert len(releases)==len(restores),(name,releases,restores)
      if name=='fills-context' and n-1>a.serve_chunk:assert releases,'layer-major prompt did not lease MTP decode weights'
      if a.verify_draft:assert all(int(v[5])==1 for v in releases+restores)
     windows=[(int(pos),int(count)) for pos,count in re.findall(r'strata trace: window (-?\d+) (-?\d+)',text)]
     r['verify_windows']=windows
     assert all(pos>=0 and count>0 and pos+count<=a.context for pos,count in windows),windows
     if ok:
      assert lines[-1].startswith('DONE ') and len(output)==new and len(lp)==new,lines[-10:]
      done=lines[-1].split()
      assert len(done)>=6 and int(done[1])==new and int(done[2])==n and done[5]=='length',done
      assert all(math.isfinite(float(x)) and float(x)>=0 for x in done[3:5]),done
      r['finish']=done[5]
      r['logits_sha256']=finite_head(head)
      if name.startswith('fills-context'):
       assert n+len(output)==a.context;r['logical_length']=a.context
       # S=4 with confidence clipping disabled reaches the final allocated cell,
       # even when the last speculative token is rejected rather than emitted.
       assert windows and max(pos+count for pos,count in windows)==a.context,windows
       r['last_executed_kv_cell']=a.context-1
       if name=='fills-context-tail-2':
        assert windows==[(a.context-3,1),(a.context-2,2)],windows
        r['clipped_verify_tail']=2
     else:
      assert lines[-1].startswith('ERR prompt') and not output
      assert not windows and not any(s.startswith(('REUSED ','LP ')) for s in lines),lines
     save()
   except BaseException as e:report['processing_error']=repr(e);save();raise
   finally:
    try:report['exit_code']=cleanup(child,proc,sel,raw)
    finally:
     report['memory']=observer.finish();sel.close();save()
   assert report['exit_code']==0
   report['active_request']=None
 report['completed']=True;save()
except BaseException as e:report['terminal_error']=repr(e);save();raise
