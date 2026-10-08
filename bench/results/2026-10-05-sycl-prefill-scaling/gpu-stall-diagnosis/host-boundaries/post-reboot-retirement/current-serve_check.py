"""Check repeated requests and snapshot restoration on the real IQ3_S model."""
import argparse, array, hashlib, json, math, os, re, selectors, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--engine',required=True,type=Path)
p.add_argument('--out',required=True,type=Path)
p.add_argument('--model-root',type=Path,default=Path.home()/'.local/share/strata-sycl')
p.add_argument('--force-prefill',action='store_true')
p.add_argument('--dump-first-head',action='store_true',help='Save and validate all first-window head values for each request')
p.add_argument('--uncached',action='store_true',help='Disable prompt and conversation reuse for a cold-state comparison')
p.add_argument('--protocol-timeout',type=float,default=180)
a=p.parse_args();base=a.model_root;exe=a.engine.resolve()
args=['--pack',str(base/'packs/qwen3.8-flash-next-iq3_s'),'--native',str(base/'models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'),'--serve','--max-context','128','--spec','4','--spec-min-p','0','--mtp',str(base/'mtp/rt'),'--suffix-draft','0','--prefill','32','--no-prefill-borrow','--expert-cache','600','--expert-profile','data/expert-profile.bin','--expert-cache-per-layer','--pool-workers','5','--pcie-frac','0.55','--spec-split','--adapt-swaps','0','--conversation-cache-mib','256','--conversation-cache-slots','4','--greedy']
if a.force_prefill:args+=['--short-read','0']
if a.uncached:
 args[args.index('--conversation-cache-mib')+1]='0'
 args+=['--prompt-cache','0']
sha=hashlib.sha256(exe.read_bytes()).hexdigest()
stderr=a.out.with_suffix('.log');start=time.monotonic();records=[]
startup=[];lines=[];current_request=None;failure=None;cleanup=[]
with stderr.open('w') as log, a.out.with_suffix('.stdout.raw').open('wb',buffering=0) as raw_log, a.out.with_suffix('.events.jsonl').open('w',buffering=1) as events:
 child_env=dict(os.environ)
 head_path=a.out.with_suffix('.head.bin')
 if a.dump_first_head:child_env['STRATA_DUMP_FIRST_LOGITS']=str(head_path)
 child=subprocess.Popen([str(exe)]+args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=child_env)
 sel=selectors.DefaultSelector();sel.register(child.stdout,selectors.EVENT_READ);pending=bytearray()
 def event(kind,**fields):
  events.write(json.dumps(dict(kind=kind,time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),pid=child.pid,**fields))+'\n')
 def progress(**fields):
  a.out.with_suffix('.partial.json').write_text(json.dumps(dict(binary_sha256=sha,args=args,pid=child.pid,startup=startup,requests=records,active_request=current_request,active_protocol=lines,error=failure,exit_code=child.poll(),cleanup=cleanup,**fields),indent=2)+'\n')
 def received(data):
  raw_log.write(data);event('stdout',bytes=len(data),raw_offset=raw_log.tell())
 def drain_to_exit(timeout):
  end=time.monotonic()+timeout
  while child.poll() is None and time.monotonic()<end:
   if not sel.get_map():
    time.sleep(min(0.05,max(0,end-time.monotonic())));continue
   if sel.select(min(0.05,max(0,end-time.monotonic()))):
    data=os.read(child.stdout.fileno(),65536)
    if data:received(data)
    else:sel.unregister(child.stdout)
  # Preserve any tail already written when poll first observes process exit.
  if child.poll() is not None:
   while sel.get_map() and sel.select(0):
    data=os.read(child.stdout.fileno(),65536)
    if data:received(data)
    else:sel.unregister(child.stdout)
  return child.poll()
 event('spawn');progress()
 def line(timeout=None):
  if timeout is None:timeout=a.protocol_timeout
  end=time.monotonic()+timeout
  while True:
   if b'\n' in pending:
    raw,_,tail=pending.partition(b'\n');pending[:]=tail;return raw.decode().strip()
   wait=end-time.monotonic()
   if wait<=0:raise TimeoutError('engine protocol timeout')
   if not sel.select(wait):raise TimeoutError('engine protocol timeout')
   data=os.read(child.stdout.fileno(),65536)
   if not data:
    sel.unregister(child.stdout)
    drain_to_exit(2)
    event('protocol-eof',exit_code=child.poll())
    raise RuntimeError(f'engine exited before protocol response: pid={child.pid} exit={child.poll()}')
   received(data)
   pending.extend(data)
 try:
  while True:
   s=line();startup.append(s)
   if s.startswith('ERR'):raise RuntimeError(s)
   if s.startswith('READY '):break
  progress()
  prompt_ids=list(map(int,Path(__file__).with_name('writing-tokens.txt').read_text().strip().split(',')))
  original=','.join(map(str,prompt_ids));alternate=prompt_ids.copy();alternate[3]+=1
  other=','.join(map(str,alternate))
  for name,prompt in [('first',original),('repeat',original),('other',other),('restored',original)]:
   current_request=name
   if a.dump_first_head:head_path.unlink(missing_ok=True)
   request=f'GEN 4 logprobs=5 {prompt}\n';child.stdin.write(request.encode());child.stdin.flush();event('request',name=name)
   lines=[];ids=[];lp=[]
   while True:
    s=line();lines.append(s)
    if s.startswith('ERR'):raise RuntimeError(s)
    if s.startswith('T '):ids.append(int(s.split()[1]))
    if s.startswith('LP '):
     values=s.split()[1:]
     if not all(math.isfinite(float(v.rsplit(':',1)[-1])) for v in values):
      raise RuntimeError('nonfinite logprob: '+s)
     lp.append(s)
    if s.startswith('DONE '):break
   if len(ids)!=4 or len(lp)!=4:raise RuntimeError('incomplete generated output/logprobs')
   record={'name':name,'request':request.strip(),'ids':ids,'logprobs':lp,'protocol':lines}
   if a.dump_first_head:
    data=head_path.read_bytes();values=array.array('f');values.frombytes(data)
    assert len(values)==248320 and all(map(math.isfinite,values)),'missing/nonfinite full head'
    saved=a.out.with_name(a.out.stem+'-'+name+'.head.bin');saved.write_bytes(data)
    record['first_head']={'file':str(saved),'bytes':len(data),'floats':len(values),'sha256':hashlib.sha256(data).hexdigest()}
   records.append(record)
   progress()
   print(name,ids,flush=True)
  for i in [1,3]:
   assert any(x.startswith('RESUME ') and (int(x.split()[1])==0 if a.uncached else int(x.split()[1])>0) for x in records[i]['protocol']),'unexpected checkpoint reuse'
   assert records[i]['ids']==records[0]['ids'],'repeated output differs'
   assert records[i]['logprobs']==records[0]['logprobs'],'repeated head logprobs differ'
   if a.dump_first_head:
    assert Path(records[i]['first_head']['file']).read_bytes()==Path(records[0]['first_head']['file']).read_bytes(),'repeated full first head differs'
  child.stdin.write(b'QUIT\n');child.stdin.flush();event('quit')
  if drain_to_exit(30) is None:raise TimeoutError('engine shutdown timeout')
  if child.returncode:raise RuntimeError('engine exit '+str(child.returncode))
  result={'binary_sha256':sha,'args':args,'startup':startup,'requests':records,'exit_code':child.returncode,'wall_seconds':time.monotonic()-start,'same_output_and_logprobs_on_repeat_and_restore':True,'full_first_head_compared':a.dump_first_head}
  a.out.write_text(json.dumps(result,indent=2)+'\n')
  print('PASS repeated generation and checkpoint restoration',flush=True)
 except BaseException as e:
  failure=f'{type(e).__name__}: {e}';event('failure',error=failure,exit_code=child.poll());raise
 finally:
  before_cleanup=child.poll()
  if before_cleanup is None:
   cleanup.append('QUIT');event('cleanup-quit')
   try:child.stdin.write(b'QUIT\n');child.stdin.flush()
   except (BrokenPipeError,OSError):pass
   if drain_to_exit(15) is None:
    cleanup.append('SIGTERM');event('cleanup-term');child.terminate()
    if drain_to_exit(1) is None:
     cleanup.append('SIGKILL');event('cleanup-kill');child.kill();drain_to_exit(1)
  event('exit',exit_code=child.poll(),still_alive=child.poll() is None)
  progress(exit_before_cleanup=before_cleanup,still_alive=child.poll() is None)
  sel.close();child.stdin.close();child.stdout.close()
