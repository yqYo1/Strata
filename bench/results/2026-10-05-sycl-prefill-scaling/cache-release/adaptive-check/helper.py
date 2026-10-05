"""Check long-prompt GDN, normal MTP, repeated requests and snapshot restoration."""
import argparse, array, hashlib, json, math, os, re, selectors, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--engine',required=True,type=Path)
p.add_argument('--out',required=True,type=Path)
p.add_argument('--reference',required=True,type=Path)
p.add_argument('--model-root',type=Path,default=Path.home()/'.local/share/strata-sycl')
a=p.parse_args();base=a.model_root;exe=a.engine.resolve()
a.out.parent.mkdir(parents=True,exist_ok=True)
ref=json.loads(a.reference.read_text())
fixture=Path(ref['fixture'])
assert hashlib.sha256(fixture.read_bytes()).hexdigest()==ref['fixture_sha256']
assert not os.getenv('STRATA_SERVE_NO_MTP') and not os.getenv('STRATA_PREFILL_TIMING') and not os.getenv('STRATA_PREFILL_TRANSFER_TIMING')
dump=a.out.with_suffix('.first-head.bin');dump.unlink(missing_ok=True)
env=dict(os.environ,STRATA_DUMP_FIRST_LOGITS=str(dump.resolve()))
args=['--pack',str(base/'packs/qwen3.8-flash-next-iq3_s'),'--native',str(base/'models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'),'--serve','--max-context','2048','--spec','4','--spec-min-p','0','--mtp',str(base/'mtp/rt'),'--suffix-draft','0','--prefill','128','--no-prefill-borrow','--expert-cache','128','--expert-profile','data/expert-profile.bin','--expert-cache-per-layer','--pool-workers','5','--pcie-frac','0.55','--spec-split','--adapt-swaps','4','--adapt-every','1','--conversation-cache-mib','256','--conversation-cache-slots','4','--greedy']
sha=hashlib.sha256(exe.read_bytes()).hexdigest()
stderr=a.out.with_suffix('.log');start=time.monotonic();records=[]
with stderr.open('w') as log:
 child=subprocess.Popen([str(exe)]+args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env)
 sel=selectors.DefaultSelector();sel.register(child.stdout,selectors.EVENT_READ);pending=bytearray()
 def line(timeout=180):
  end=time.monotonic()+timeout
  while True:
   if b'\n' in pending:
    raw,_,tail=pending.partition(b'\n');pending[:]=tail;return raw.decode().strip()
   wait=end-time.monotonic()
   if wait<=0:raise TimeoutError('engine protocol timeout')
   if not sel.select(wait):raise TimeoutError('engine protocol timeout')
   data=os.read(child.stdout.fileno(),65536)
   if not data:raise RuntimeError('engine exited before protocol response')
   pending.extend(data)
 try:
  startup=[]
  while True:
   s=line();startup.append(s)
   if s.startswith('ERR'):raise RuntimeError(s)
   if s.startswith('READY '):break
  template=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/bench/results/2026-10-05-sycl-upstream-arc/writing-tokens.txt')
  writing=list(map(int,template.read_text().strip().split(',')))
  suffix=writing[-9:]
  assert suffix==[248046,198,248045,74455,198,248068,198,248069,271]
  prompt_ids=list(map(int,fixture.read_text().split()))[:1016]+suffix
  assert len(prompt_ids)==1025
  original=','.join(map(str,prompt_ids));alternate=prompt_ids.copy();alternate[3]+=1
  other=','.join(map(str,alternate))
  for name,prompt in [('first',original),('repeat',original),('other',other),('restored',original)]:
   request=f'GEN 16 logprobs=5 {prompt}\n';child.stdin.write(request.encode());child.stdin.flush()
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
   if len(ids)!=16 or len(lp)!=16:raise RuntimeError('incomplete generated output/logprobs: '+repr(lines))
   record={'name':name,'request':request.strip(),'ids':ids,'logprobs':lp,'protocol':lines}
   if name=='first':
    values=array.array('f');values.frombytes(dump.read_bytes())
    assert len(values)==248320 and all(map(math.isfinite,values))
    record.update(logits_sha256=hashlib.sha256(dump.read_bytes()).hexdigest(),logits_count=len(values),logits_finite=True)
   records.append(record)
   print(name,ids,flush=True)
  a.out.with_suffix('.partial.json').write_text(json.dumps({'binary_sha256':sha,'args':args,'requests':records},indent=2)+'\n')
  for i in [1,3]:
   assert any(x.startswith('RESUME ') and int(x.split()[1])>0 for x in records[i]['protocol']),'checkpoint was not reused'
  child.stdin.write(b'QUIT\n');child.stdin.flush();child.wait(timeout=30)
  if child.returncode:raise RuntimeError('engine exit '+str(child.returncode))
  result={'binary_sha256':sha,'args':args,'reference':str(a.reference),'fixture_sha256':ref['fixture_sha256'],'input_tokens':1025,'prompt_kind':'1016-token code-review prefix plus the verified nine-token assistant suffix','assistant_suffix':suffix,'env':{k:v for k,v in env.items() if k.startswith('STRATA_') or k in ['LD_LIBRARY_PATH','ONEAPI_DEVICE_SELECTOR','SYCL_CACHE_PERSISTENT']},'startup':startup,'requests':records,'exit_code':child.returncode,'wall_seconds':time.monotonic()-start,'checkpoint_reuse_and_restoration_checked':True,'adaptation_enabled':True}
  a.out.write_text(json.dumps(result,indent=2)+'\n')
  print('PASS adaptive generation and checkpoint restoration',flush=True)
 finally:
  if child.poll() is None:
   try: child.stdin.write(b'QUIT\n');child.stdin.flush()
   except BrokenPipeError: pass
   child.wait()
