"""Report repeated decode timing with process-block uncertainty, independently of prefill."""
from pathlib import Path
import hashlib
import itertools
import json
import math
import statistics

b=Path(__file__).parent
seq_path=b/'native-copy-decode-repeat-v0141-comparison-sequence-v1/record.json'
seq=json.loads(seq_path.read_text());assert seq['passed'] and not seq['active']
assert len(seq['steps'])==18
def digest(path):
    return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
groups={mode:[] for mode in ['baseline','nativeoff','nativeon']}
infos=[]
for step in seq['steps']:
    p=Path(step['receipt']);assert digest(p)==step['receipt_sha256']
    r=json.loads(p.read_text())
    assert r['healthy'] and r['math_gate_passed'] and r['phase_specific_decode_sequence_completed'] and not r['active']
    info_lines=[value for value in r['startup'] if value.startswith('INFO ')]
    assert len(info_lines)==1 and r['startup'][-1].split()[1]=='262144'
    info=dict(value.split('=',1) for value in info_lines[0].split()[1:])
    for key,value in {'context':'262144','kv':'int8','kv_resident':'32768','expert_slots':'128','expert_cache_mib':'325','spec':'4','pool_workers':'5','pcie_frac':'0.00','vram_free_mib':'1565','engine':'0.1.41'}.items():
        assert info[key]==value,(step['mode'],step['block'],key,info[key])
    infos.append(info)
    samples=[x for x in r['requests'] if x['timing_eligible']]
    assert len(samples)==12 and all(x['phase_kind']=='decode-only' and not x['warmup_decode'] for x in samples)
    assert all(x['resume_tokens']==[32831,32831] and x['mtp_counts']==[46,51] for x in samples)
    assert all(x['measurement']['generated_tokens']==64 and x['measurement']['prompt_tokens']==32832 for x in samples)
    assert all(x['math_gate_passed'] and all(x['qualified_reference_output_comparison'].values()) for x in samples)
    ms=[x['measurement']['decode_ms'] for x in samples]
    assert all(math.isfinite(x) and x>0 for x in ms)
    prime=r['requests'][0]['measurement'];warm=r['requests'][1]['measurement']
    groups[step['mode']].append({'block':step['block'],'order_position':step['position'],'receipt':str(p),'receipt_sha256':step['receipt_sha256'],
        'decode_ms':ms,'mean_decode_ms':statistics.mean(ms),'median_decode_ms':statistics.median(ms),
        'within_process_decode_ms_sd':statistics.stdev(ms),'within_process_cv_percent':100*statistics.stdev(ms)/statistics.mean(ms),
        'decode_tok_s':64000/statistics.mean(ms),'prefill_prime_ms_excluded':prime['prompt_ms'],
        'fresh32k_first_decode_ms_separate':prime['decode_ms'],'first_decode_after_restore_ms_excluded':warm['decode_ms']})
assert all(info==infos[0] for info in infos)
summary={}
for mode,processes in groups.items():
    assert sorted(x['block'] for x in processes)==list(range(1,7))
    values=[v for x in processes for v in x['decode_ms']]
    pm=[x['mean_decode_ms'] for x in processes]
    summary[mode]={'independent_processes':6,'measured_requests':len(values),'generated_tokens':64*len(values),
        'total_decode_ms':sum(values),'mean_decode_ms':statistics.mean(values),'decode_tok_s':64000/statistics.mean(values),
        'all_request_decode_ms_sd':statistics.stdev(values),'process_mean_decode_ms_sd':statistics.stdev(pm),
        'process_mean_cv_percent':100*statistics.stdev(pm)/statistics.mean(pm),
        'minimum_decode_ms':min(values),'maximum_decode_ms':max(values),'processes':processes}

# Six independent paired process blocks: Student t, df=5. Repeated requests
# improve each block estimate but are not treated as 72 independent blocks.
critical_t=2.570581835636314
comparisons={}
for candidate,reference in [('nativeoff','baseline'),('nativeon','nativeoff'),('nativeon','baseline')]:
    cp={x['block']:x for x in groups[candidate]};rp={x['block']:x for x in groups[reference]}
    logs=[math.log(cp[k]['mean_decode_ms']/rp[k]['mean_decode_ms']) for k in range(1,7)]
    mean=statistics.mean(logs);se=statistics.stdev(logs)/math.sqrt(6)
    lo,hi=mean-critical_t*se,mean+critical_t*se
    speed_change=100*(math.exp(-mean)-1)
    speed_interval=[100*(math.exp(-hi)-1),100*(math.exp(-lo)-1)]
    # Exact sign-flip check supplements, rather than replaces, the small-n t CI.
    permutations=[abs(statistics.mean(s*v for s,v in zip(signs,logs))) for signs in itertools.product([-1,1],repeat=6)]
    p=sum(v>=abs(mean)-1e-14 for v in permutations)/len(permutations)
    comparisons[f'{candidate}_vs_{reference}']={'paired_independent_blocks':6,'paired_block_log_latency_ratios':logs,
        'geometric_speed_change_percent':speed_change,'speed_change_95_percent_ci':speed_interval,
        'paired_log_latency_mean':mean,'paired_log_latency_standard_error':se,'student_t_degrees_of_freedom':5,
        'exact_sign_flip_two_sided_p':p,'confidence_interval_includes_zero':speed_interval[0]<=0<=speed_interval[1],
        'interpretation':'Measured decode regression' if speed_interval[1]<0 else ('Measured decode improvement' if speed_interval[0]>0 else 'Direction unresolved; does not establish equivalence'),
        'block_speed_changes_percent':[100*(math.exp(-v)-1) for v in logs]}
fresh_summary={}
for mode,processes in groups.items():
    values=[x['fresh32k_first_decode_ms_separate'] for x in processes]
    fresh_summary[mode]={'independent_processes':6,'requests':6,'generated_tokens':384,
        'mean_decode_ms':statistics.mean(values),'decode_tok_s':64000/statistics.mean(values),
        'decode_ms_sd':statistics.stdev(values),'values_ms':values}
fresh_comparisons={}
for candidate,reference in [('nativeoff','baseline'),('nativeon','nativeoff'),('nativeon','baseline')]:
    cp={x['block']:x for x in groups[candidate]};rp={x['block']:x for x in groups[reference]}
    logs=[math.log(cp[k]['fresh32k_first_decode_ms_separate']/rp[k]['fresh32k_first_decode_ms_separate']) for k in range(1,7)]
    mean=statistics.mean(logs);se=statistics.stdev(logs)/math.sqrt(6)
    interval=[100*(math.exp(-(mean+critical_t*se))-1),100*(math.exp(-(mean-critical_t*se))-1)]
    fresh_comparisons[f'{candidate}_vs_{reference}']={'paired_independent_blocks':6,
        'geometric_speed_change_percent':100*(math.exp(-mean)-1),'speed_change_95_percent_ci':interval,
        'paired_block_log_latency_ratios':logs,'confidence_interval_includes_zero':interval[0]<=0<=interval[1],
        'scope':'Secondary: first64 output tokens immediately after a fresh32768-token prefill. Exactly the qualified A output/logprobs/41-of66 MTP counts. Kept separate from restored continuation64 with46-of51 MTP counts; never pooled.'}
result={'passed':True,'active':False,'sequence_receipt_sha256':digest(seq_path),'summarizer_sha256':digest(__file__),
    'scope':'Decode only, fixed restored32831-token state plus one input token, greedy64 output/LP5/MTP4. Prefill priming and restore times excluded. Six independent order-balanced process blocks and12 measured repeats per process, one warm decode excluded. Same qualified output, logprobs and46/51 MTP counts in every measured request.',
    'summary':summary,'comparisons':comparisons,'actual_info_geometry':infos[0],'all18_actual_info_geometry_equal':True,
    'secondary_fresh32k_first_decode_summary':fresh_summary,'secondary_fresh32k_first_decode_comparisons':fresh_comparisons,
    'limitations':['Six independent process blocks; 72 repeats per mode are not 72 independent process samples.',
        'One qualified32K coding continuation; this does not cover every prompt or longer decode.',
        'Same numerical state restored each time; expert cache and runtime remain live within each process.',
        'The raw reused-request receipts inherit a generic prefill_tok_s calculation over total input tokens. That number is not a prefill rate because32831 tokens are reused; it is excluded from all summaries and adoption decisions.',
        'Intervals use a paired log-latency Student t model with five degrees of freedom; sign-flip p assumes exchangeable signs under the null.',
        'No claim that repetition eliminates physical variance or that an interval including zero proves equivalence.'],
    'phase_adoption_policy':'Prefill and decode assessed separately. No decode change accepted merely because prefill improves.'}
out=b/'native-copy-decode-repeat-v0141-summary-v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'summary':{k:{kk:vv for kk,vv in v.items() if kk!='processes'} for k,v in summary.items()},'comparisons':comparisons},indent=2))
