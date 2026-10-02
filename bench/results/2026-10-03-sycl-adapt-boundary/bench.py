import json,os,subprocess,re,statistics
from pathlib import Path
base=json.load(open('bench/results/2026-10-03-sycl-verify-overlap/run.json'))['runs'][0]['flags'].copy();base[base.index('--expert-cache')+1]='auto'
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu');env.pop('STRATA_TRACE',None);env.pop('STRATA_SYCL_VERIFY_LOGITS',None);env.pop('STRATA_SYCL_VERIFY_OVERLAP',None)
runs=[];known={}
for prompt,tokens in [('writing','/tmp/strata-sycl-writing-tokens.txt'),('coding','/tmp/strata-sycl-profile-coding-tokens.txt')]:
 for trial in range(1,4):
  for variant,sync_mode in [('before','0'),('after','1')]:
   swaps='64'
   args=base.copy();args[args.index('--tokens-file')+1]=tokens;args[args.index('--adapt-swaps')+1]=swaps
   log=f'/tmp/strata-sycl-adapt-boundary-{prompt}-{variant}-{trial}-128.log'
   with open(log,'w') as f:subprocess.run(['./build-sycl/strata']+args,env=dict(env,STRATA_SYCL_ADAPT_SYNC=sync_mode),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
   text=Path(log).read_text();result=dict(prompt=prompt,variant=variant,trial=trial,maximum_swaps=int(swaps),adapt_every=4,sync_mode=sync_mode,flags=args,log=log)
   m=re.search(r'decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s',text);result.update(generated_tokens=int(m[1]),decode_ms=float(m[2]),decode_tok_s=float(m[3]),output_ids=list(map(int,re.search(r'^output\s*:\s*(.*)$',text,re.M)[1].split())));assert len(result['output_ids'])==128
   c=re.search(r'R4 expert-cache hits\s+(\d+) of (\d+)',text);result.update(cache_hits=int(c[1]),cache_lookups=int(c[2]))
   spec=re.search(r'speculation\s+(\d+) rounds of \d+, drafts accepted (\d+) of (\d+)',text);result.update(rounds=int(spec[1]),drafts_accepted=int(spec[2]),drafts_attempted=int(spec[3]))
   slots=re.search(r'expert cache (\d+) slots',text);result['expert_slots']=int(slots[1])
   timing=re.search(r'verify window\s+wait for rings ([\d.]+)\s+pool ([\d.]+)',text);result.update(gpu_wait_ms_per_round=float(timing[1]),cpu_pool_ms_per_round=float(timing[2]))
   a=re.search(r'adaptive tier\s+(\d+) experts swapped[^\n]*?([\d.]+) ms/round',text)
   if a:result.update(swapped_experts=int(a[1]),adapt_host_ms_per_round=float(a[2]))
   identity=tuple(result[k] if k!='output_ids' else tuple(result[k]) for k in ['output_ids','cache_hits','cache_lookups','rounds','drafts_accepted','drafts_attempted','expert_slots'])
   key=(prompt,variant)
   if key not in known:known[key]=identity
   if variant=='after':assert identity==known[key],f'{key} routing or output varied across repeated identical settings'
   runs.append(result);Path('/tmp/strata-sycl-adapt-boundary-macro.json').write_text(json.dumps(dict(runs=runs),indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['flags','output_ids']},flush=True)
print('medians',{p:{v:statistics.median(r['decode_tok_s'] for r in runs if r['prompt']==p and r['variant']==v) for v in ['before','after']} for p in ['writing','coding']},flush=True)
