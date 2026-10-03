import os,json,subprocess,re,statistics
from pathlib import Path
base='/tmp/strata-sycl-verifier-native-'
rows=[]
for trial in range(1,4):
 for mode in [0,1]:
  if trial>1:
   subprocess.run(['python3','/tmp/strata-sycl-verifier-native-serve.py'],env=dict(os.environ,STRATA_SYCL_VERIFY_NATIVE_CAPTURE=str(mode),STRATA_BENCH_TRIAL=str(trial)),check=True,timeout=360)
  prefix=base+str(mode)+(f'-trial{trial}' if trial!=1 else '')
  row=json.load(open(prefix+'.json'));assert row['exit_code']==0 and row['cancel_recovery_known_eight_ids']
  row.update(trial=trial,mode=mode,prefix=prefix,capture=re.findall(r'capture (\d+) tokens ([\d.]+) ms, free VRAM ([\d.]+) MiB',Path(prefix+'.stderr.log').read_text()))
  rows.append(row)
  for previous in rows:
   assert len(row['runs'])==len(previous['runs'])==6
   for current,old in zip(row['runs'],previous['runs']):
    for field in ['prompt','requested_tokens','output_ids','cache_hits','cache_lookups','reused_prompt_tokens','cancelled']:
     assert current[field]==old[field],(trial,mode,field)
  Path(base+'pairs.json').write_text(json.dumps(dict(runs=rows,all_ids_cache_reuse_cancel_counts_equal=True),indent=2)+'\n')
for i in range(4):
 print(i,rows[0]['runs'][i]['prompt'],{mode:statistics.median(row['runs'][i]['decode_tok_s'] for row in rows if row['mode']==mode) for mode in [0,1]},flush=True)
