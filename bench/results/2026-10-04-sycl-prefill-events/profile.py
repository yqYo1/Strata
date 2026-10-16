import os,json,subprocess,time,hashlib,sys,re
from pathlib import Path
label,trial=sys.argv[1:3];assert label in ('short','4k')
cfg=json.loads((Path.home()/'.local/share/strata-sycl/serve-config-iq3_s.json').read_text())
a=cfg['args'].copy();a.remove('--serve');del a[a.index('--tokens'):a.index('--tokens')+2]
for k,v in [('--max-context','2048' if label=='short' else '8192'),('--prefill','1024' if label=='short' else '4096'),('--expert-cache','1649' if label=='short' else '512'),('--adapt-swaps','0'),('--pool-workers','5')]:a[a.index(k)+1]=v
fixture='/tmp/strata-sycl-functional-context-v4-tokens.txt' if label=='short' else '/tmp/strata-sycl-goal-prefill-scale-4k-tokens.txt'
a+=['--tokens-file',fixture,'--check-logits','--stats']
env=dict(cfg['env'],**os.environ);env.pop('STRATA_PREFILL_TIMING',None)
exe=os.environ.get('PERF_EXE',cfg['exe']);prefix=Path(f'/tmp/strata-sycl-goal-prefill-event-{label}-{trial}')
result=dict(label=label,executable=exe,binary_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),flags=a,environment={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','NEO_','ONEAPI_'))})
start=time.monotonic()
with prefix.with_suffix('.log').open('w') as f:r=subprocess.run([exe]+a,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
result.update(exit_code=r.returncode,total_seconds=time.monotonic()-start)
s=prefix.with_suffix('.log').read_text();m=re.search(r'^prefill\s+(\d+) tokens in ([\d.]+) ms',s,re.M)
if m:result.update(prefetched_tokens=int(m[1]),prefill_ms=float(m[2]),prefill_tok_s=int(m[1])*1000/float(m[2]))
m=re.search(r'^output  : (.*)',s,re.M)
if m:result['output_ids']=list(map(int,m[1].split()))
result['log']=s;prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k not in ('flags','environment','log')},flush=True)
if r.returncode:sys.exit(r.returncode)
if result.get('output_ids')!=[11855,248046,198,248044,248045,846,198,248046]:raise RuntimeError('output IDs differ')
