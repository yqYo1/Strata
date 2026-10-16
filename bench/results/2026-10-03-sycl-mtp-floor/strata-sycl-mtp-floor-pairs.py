import os,json,subprocess,statistics
from pathlib import Path
rows=[]
for trial in range(1,4):
 for floor in ['0.5','0.9']:
  if trial>1:
   subprocess.run(['python3','/tmp/strata-sycl-mtp-floor-serve.py'],env=dict(os.environ,STRATA_BENCH_FLOOR=floor,STRATA_BENCH_TRIAL=str(trial),STRATA_DECODE_TIMING='1'),check=True,timeout=360)
  row=json.load(open(f'/tmp/strata-sycl-mtp-floor-{floor}-trial{trial}.json'));row.update(floor=floor,trial=trial);assert row['exit_code']==0 and row['cancel_recovery_known_eight_ids']
  for previous in rows:
   if previous['floor']!=floor:continue
   for old,new in zip(previous['runs'],row['runs']):
    for k in ['output_ids','cache_hits','cache_lookups','reused_prompt_tokens','cancelled']:assert old[k]==new[k],(floor,trial,k)
  rows.append(row);Path('/tmp/strata-sycl-mtp-floor-pairs.json').write_text(json.dumps(dict(runs=rows,same_config_repeat_ids_cache_reuse_cancel_counts_equal=True),indent=2)+'\n')
for i in range(4):
 print(i,{floor:statistics.median(row['runs'][i]['decode_tok_s'] for row in rows if row['floor']==floor) for floor in ['0.5','0.9']},flush=True)
