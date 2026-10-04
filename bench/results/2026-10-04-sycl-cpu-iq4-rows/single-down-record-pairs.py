from pathlib import Path
import json,hashlib,statistics,shutil
p=Path('bench/results/2026-10-04-sycl-cpu-iq4-rows/run.json');d=json.loads(p.read_text())
d['engine_pair_order']=['old1','new1','new2','old2','old3','new3']
d['engine_pairs']={}
for tag in d['engine_pair_order']:
 raw=Path('/tmp/strata-sycl-goal-single-down-pairs-'+tag+'.json');v=json.loads(raw.read_text())
 assert v['exit_code']==0 and v['comparison_ids_equal'] and v['cancel_recovery_first_eight_ids']
 assert v['environment']['STRATA_IQ_SINGLE_EXACT']=='1'
 for r in v['runs']:
  ids=r.pop('output_ids');r['output_count']=len(ids);r['output_ids_sha256']=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
 v['source_file']=raw.name;v['source_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest();d['engine_pairs'][tag]=v
for limit,key in [(3,'all_three_pairs')]:
 d[key]={}
 for idx,name in enumerate(['writing_first','writing_repeat','coding_first','coding_repeat','total_four_requests']):
  values={tag:[] for tag in ['old','new']}
  for tag in values:
   for trial in range(1,limit+1):
    runs=d['engine_pairs'][tag+str(trial)]['runs']
    values[tag].append(runs[idx]['decode_ms'] if idx<4 else sum(r['decode_ms'] for r in runs[:4]))
  med={tag:statistics.median(v) for tag,v in values.items()}
  d[key][name]=dict(samples_ms=values,median_ms=med,median_tok_s={tag:(128000 if idx<4 else 512000)/v for tag,v in med.items()},rate_gain_percent=100*(med['old']/med['new']-1))
d['validation']['engine_pairs']='all6processes passed equal output ids and prefill/decode cancellation with recovery'
d['scope']='Both binaries use STRATA_IQ_SINGLE_EXACT1, so existing IQ3_XXS/IQ3_S/IQ2_S gate/up tuning remains active. Only the new binary adds two-row IQ4_NL down kernels. Same persistent128-token prompts and localctx512/2137cache/workers5/spec4 p0.9/XMXexact8/packing0 setup.'
p.write_text(json.dumps(d,indent=2)+'\n')
archive=Path.home()/'.local/state/strata-sycl/measurement-archive/2026-10-04-speed-goal'
for f in Path('/tmp').glob('strata-sycl-goal-single-down-*'):
 if f.is_file():shutil.copy2(f,archive/f.name)
print(json.dumps(d['all_three_pairs'],indent=2))
