import os,json,subprocess,time,hashlib,sys,re
from pathlib import Path
label,chunk=sys.argv[1],sys.argv[2]
root=Path.home()/'.local/share/strata-sycl';cfg=json.loads((root/'serve-config-iq3_s.json').read_text())
cfg['exe']=os.environ['PERF_EXE']
a=cfg['args'].copy();a.remove('--serve');del a[a.index('--tokens'):a.index('--tokens')+2]
for k,v in [('--max-context','16384' if label=='8k' else '8192'),('--prefill',chunk),('--expert-cache','512'),('--adapt-swaps','0')]:a[a.index(k)+1]=v
a+=['--tokens-file',f'/tmp/strata-sycl-goal-prefill-scale-{label}-tokens.txt','--check-logits','--stats']
env=dict(cfg['env'],**os.environ);env.pop('STRATA_PREFILL_TIMING',None)
prefix=Path(f'/tmp/strata-sycl-goal-signspread-{os.environ["PACK_TRIAL"]}')
result=dict(label=label,requested_chunk=int(chunk),executable=cfg['exe'],binary_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),flags=a,environment={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','NEO_','ONEAPI_'))})
start=time.monotonic()
with prefix.with_suffix('.log').open('w') as f:
 r=subprocess.run([cfg['exe']]+a,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
result.update(exit_code=r.returncode,total_seconds=time.monotonic()-start)
s=prefix.with_suffix('.log').read_text();m=re.search(r'^prefill\s+(\d+) tokens in ([\d.]+) ms',s,re.M)
if m:result.update(prefetched_tokens=int(m[1]),prefill_ms=float(m[2]),prefill_tok_s=int(m[1])*1000/float(m[2]))
m=re.search(r'^output  : (.*)',s,re.M)
if m:result['output_ids']=list(map(int,m[1].split()));result['first_token_is_blue']=result['output_ids'][0]==11855
result['log']=s
prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['flags','environment','log']},flush=True)
sys.exit(r.returncode)
