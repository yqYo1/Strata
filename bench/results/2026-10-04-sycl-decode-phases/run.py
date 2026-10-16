import json,os,subprocess,hashlib,time,re
from pathlib import Path
cfg=json.loads((Path.home()/'.local/share/strata-sycl/serve-config-iq3_s.json').read_text())
reference=json.loads(Path('/tmp/strata-sycl-goal-iq-looped-serve-old.json').read_text())
for i,(label,fixture) in enumerate([('writing','strata-sycl-writing-tokens.txt'),('coding','strata-sycl-profile-coding-tokens.txt')]):
 args=cfg['args'].copy();args.remove('--serve');del args[args.index('--tokens'):args.index('--tokens')+2]
 for key,value in [('--max-new','128'),('--expert-cache','2137')]:args[args.index(key)+1]=value
 args+=['--tokens-file','bench/results/2026-10-03-sycl-mtp-floor/'+fixture,'--check-logits','--stats']
 env=dict(cfg['env'],**os.environ,STRATA_SYCL_MMQ_XMX_PACK='0')
 prefix=Path('/tmp/strata-sycl-goal-decode-phases-'+label);start=time.monotonic()
 with prefix.with_suffix('.log').open('w') as log:r=subprocess.run([cfg['exe']]+args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 s=prefix.with_suffix('.log').read_text();match=re.search(r'^output  : (.*)',s,re.M);ids=list(map(int,match[1].split())) if match else []
 result=dict(executable=cfg['exe'],binary_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),flags=args,environment={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','NEO_','ONEAPI_'))},exit_code=r.returncode,total_seconds=time.monotonic()-start,output_ids=ids,log=s)
 prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
 print(label,r.returncode,'tokens',len(ids),flush=True)
 print('\n'.join(x for x in s.splitlines() if x.startswith(('decode ','prefill ','speculation','window sizes','accepted per round','verify window','pool multi','dispatch','verify GPU','MTP ','mtp ','draft '))),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
