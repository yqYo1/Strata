from pathlib import Path
import json,re,hashlib,statistics,csv,sys
base=Path(__file__).parent
rows=[]
for p in sorted(base.glob('owned-v01402-code32k-*/record.json')):
 d=json.loads(p.read_text())
 if d['active']:continue
 startup=' '.join(d.get('startup',[]))
 item={'case':p.parent.name,'mode':d.get('mode'),'phase':d.get('phase'),'healthy':d['healthy'],'error':d.get('error'),'record_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'binary_sha256':d['binary_sha256'],'prompt_sha256':d.get('prompt_fixture',{}).get('sha256')}
 for k in ['expert_slots','expert_cache_mib','vram_free_mib']:
  m=re.search(r'\b'+k+r'=(\d+)',startup);item[k]=int(m[1]) if m else None
 if d['requests']:
  item.update(d['requests'][0]['measurement']);item['output_ids_sha256']=hashlib.sha256(json.dumps(d['requests'][0]['ids']).encode()).hexdigest();item['finite_logprobs']=d['requests'][0].get('finite_logprobs');item['finish_reason']=d['requests'][0].get('finish_reason')
  assert item['prompt_tokens']>=32768
 item['decode_comparison_eligible']=item.get('generated_tokens',0)>=64 and item['phase']=='clean' and item['healthy']
 item['performance_comparison_eligible']=item['healthy'] and item['phase']=='clean'
 rows.append(item)
result={'minimum_prompt_tokens':32768,'rows':rows,'clean_means':{}}
for mode in sorted(set(r['mode'] for r in rows)):
 values=[r for r in rows if r['mode']==mode and r['performance_comparison_eligible']]
 if values:result['clean_means'][mode]={'repetitions':len(values),**{k:statistics.mean(r[k] for r in values) for k in ['prompt_ms','decode_ms','prefill_tok_s','decode_tok_s']}}
result['within_configuration_output_consistency']={}
for mode in sorted(set(r['mode'] for r in rows)):
 records=[json.loads(p.read_text()) for p in sorted(base.glob('owned-v01402-code32k-*/record.json')) if json.loads(p.read_text()).get('mode')==mode and json.loads(p.read_text()).get('healthy')]
 result['within_configuration_output_consistency'][mode]={'completed_runs':len(records),'ids_equal':all(d['requests'][0]['ids']==records[0]['requests'][0]['ids'] for d in records),'protocol_logprobs_equal':all(d['requests'][0]['logprobs']==records[0]['requests'][0]['logprobs'] for d in records)}
result['interpretation']='Clean means are observed elapsed times. Accept a tuning reference only when repeated and diagnostic controls reproduce IDs and logprobs; process healthy describes lifecycle and GPU health, not parity.'
for r in result['rows']:
 consistency=result['within_configuration_output_consistency'][r['mode']]
 stable=consistency['completed_runs']>=3 and consistency['ids_equal'] and consistency['protocol_logprobs_equal']
 r['performance_comparison_eligible']=r['performance_comparison_eligible'] and stable
 r['decode_comparison_eligible']=r['decode_comparison_eligible'] and stable
result['accepted_clean_means']={mode:value for mode,value in result['clean_means'].items() if result['within_configuration_output_consistency'][mode]['ids_equal'] and result['within_configuration_output_consistency'][mode]['protocol_logprobs_equal']}
p=base/'v01402-code32k-summary.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
