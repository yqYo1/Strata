import json,os,subprocess,re,statistics
from pathlib import Path
assert json.load(open('/tmp/strata-sycl-native-recording-check.json'))['cold32']['comparison']['bitwise_equal']
p=json.load(open('bench/results/2026-10-03-sycl-q4k-esimd/run.json'));base=p['model_flags']+['--tokens-file','/tmp/strata-sycl-writing-tokens.txt']+p['common_flags']
env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='level_zero:gpu');env.pop('STRATA_TRACE',None);env.pop('STRATA_SYCL_VERIFY_LOGITS',None);env.pop('STRATA_SYCL_ARENA_THP',None)
runs=[];ref=None;counts=None
for trial in range(1,4):
 for variant,mode in [('before','0'),('after','1')]:
  log=f'/tmp/strata-sycl-native-recording-{variant}-{trial}-ordinary128.log'
  with open(log,'w') as f:subprocess.run(['./build-sycl/strata']+base,env=dict(env,STRATA_SYCL_NATIVE_RECORDING=mode),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
  text=Path(log).read_text();ids=list(map(int,re.search(r'^output\s*:\s*(.*)$',text,re.M)[1].split()));assert len(ids)==128
  if ref is None:ref=ids
  assert ids==ref
  result=dict(variant=variant,trial=trial,flags=base,log=log,output_ids=ids,environment=dict(STRATA_SYCL_NATIVE_RECORDING=mode))
  m=re.search(r'decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s',text);result.update(generated_tokens=int(m[1]),decode_ms=float(m[2]),decode_tok_s=float(m[3]))
  m=re.search(r'R4 expert-cache hits\s+(\d+) of (\d+)',text);result.update(cache_hits=int(m[1]),cache_lookups=int(m[2]))
  if counts is None:counts=m.groups()
  assert m.groups()==counts
  for name,pat in [('host_after_ring_ms',r'host after ring\s+([\d.]+) ms/token'),('cpu_drain_ms',r'CPU expert pool\s+([\d.]+) ms/token'),('vram_free_mib',r'INFO[^\n]*?vram_free_mib=(\d+)')]:
   m=re.search(pat,text);result[name]=float(m[1]) if m else None
  runs.append(result);Path('/tmp/strata-sycl-native-recording-macro.json').write_text(json.dumps(dict(runs=runs),indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['flags','log','output_ids','environment']},flush=True)
print('medians',{v:statistics.median(r['decode_tok_s'] for r in runs if r['variant']==v) for v in ['before','after']},flush=True)
