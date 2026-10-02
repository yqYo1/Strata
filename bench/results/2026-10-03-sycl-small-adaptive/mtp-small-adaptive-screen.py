import json,os,subprocess,re
from pathlib import Path
base=json.load(open('bench/results/2026-10-03-sycl-verify-overlap/run.json'))['runs'][0]['flags'].copy()
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu',STRATA_SYCL_NATIVE_RECORDING='0');env.pop('STRATA_TRACE',None);env.pop('STRATA_SYCL_VERIFY_LOGITS',None);env.pop('STRATA_SYCL_VERIFY_OVERLAP',None)
runs=[]
for prompt,tokens in [('writing','/tmp/strata-sycl-writing-tokens.txt'),('coding','/tmp/strata-sycl-profile-coding-tokens.txt')]:
 for swaps in ['0','64','128']:
  args=base.copy()
  for flag,value in [('--tokens-file',tokens),('--expert-cache','auto'),('--pool-workers','4'),('--adapt-swaps',swaps)]:args[args.index(flag)+1]=value
  log=f'/tmp/strata-sycl-mtp-small-adaptive-{prompt}-{swaps}.log'
  with open(log,'w') as f:r=subprocess.run(['./build-sycl/strata']+args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
  text=Path(log).read_text();result=dict(prompt=prompt,maximum_swaps=int(swaps),flags=args,exit_code=r.returncode,log=log)
  if r.returncode==0:
   m=re.search(r'decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s',text);result.update(generated_tokens=int(m[1]),decode_ms=float(m[2]),decode_tok_s=float(m[3]),output_ids=list(map(int,re.search(r'^output\s*:\s*(.*)$',text,re.M)[1].split())))
   c=re.search(r'R4 expert-cache hits\s+(\d+) of (\d+)',text);result.update(cache_hits=int(c[1]),cache_lookups=int(c[2]))
   spec=re.search(r'speculation\s+(\d+) rounds of \d+, drafts accepted (\d+) of (\d+)',text);result.update(rounds=int(spec[1]),drafts_accepted=int(spec[2]),drafts_attempted=int(spec[3]))
   slots=re.search(r'expert cache (\d+) slots',text);result['expert_slots']=int(slots[1])
   timing=re.search(r'verify window\s+wait for rings ([\d.]+)\s+pool ([\d.]+)',text);result.update(gpu_wait_ms_per_round=float(timing[1]),cpu_pool_ms_per_round=float(timing[2]))
   a=re.search(r'adaptive tier\s+(\d+) experts swapped[^\n]*?([\d.]+) ms/round',text)
   if a:result.update(swapped_experts=int(a[1]),adapt_host_ms_per_round=float(a[2]))
  else:result['failure_tail']=text[-1800:]
  runs.append(result);Path('/tmp/strata-sycl-mtp-small-adaptive-screen.json').write_text(json.dumps(runs,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['flags','output_ids']},flush=True)
