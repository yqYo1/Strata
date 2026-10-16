#!/usr/bin/env python3
"""Offline arithmetic from retained independent CPU prepare and paired-service samples."""
import hashlib,json,math,statistics
from pathlib import Path
D=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    initial=json.loads((D/'cpu-predecode-probe-v1.json').read_text())
    repeat=json.loads((D/'cpu-predecode-repeat-v1.json').read_text())
    summary=json.loads((D/'cpu-predecode-repeat-summary-v1.json').read_text())
    assert all(x['passed'] for x in [initial,repeat,summary])
    processes={'initial':initial['timings']['e64-nt1']['rows']}
    processes.update({x['repetition']:x['rows'] for x in repeat['repetitions']})
    out=[]
    for c in summary['comparisons']:
        assert c['control'] in ['ggml','existing_exact']
        rows=processes[c['repetition']];ty=c['type']
        fill=[r for r in rows if r['kind']=='prepare_sample' and r['type']==ty]
        assert len(fill)==2 and all(r['experts']==64 and not r['migration_observed'] for r in fill)
        prep={r['phase']:r['thread_cpu_ns']/r['experts']/1000 for r in fill}
        deltas=[]
        for block in c['paired_blocks']:
            v=block['thread_cpu_ns_per_output_row'];deltas.append((v['control']-v['candidate'])*640/1000)
        assert len(deltas)==12
        saved=statistics.median(deltas);assert saved>0
        out.append(dict(process=c['repetition'],type=ty,control=c['control'],nt=1,
            saved_thread_cpu_us_per_expert_call=deltas,median_saved_thread_cpu_us=saved,prepare_thread_cpu_us=prep,
            cold_calls_to_amortize=prep['cold_first_touch_fill']/saved,
            cold_calls_ceiling=math.ceil(prep['cold_first_touch_fill']/saved),
            refill_calls_to_amortize=prep['overwrite_refill']/saved,
            refill_calls_ceiling=math.ceil(prep['overwrite_refill']/saved)))
    result=dict(passed=True,GPU_executed=False,adopted=False,model_performance_qualified=False,
        source_receipts={n:sha(D/n) for n in ['cpu-predecode-probe-v1.json','cpu-predecode-repeat-v1.json','cpu-predecode-repeat-summary-v1.json']},rows=out,
        actual_model_setting='STRATA_IQ_SINGLE_EXACT=0 in the queued clean comparison; ggml is its packed baseline. Existing_exact is a separate already-available helper comparison, not the currently adopted setting.',
        scope='Single-thread CPU service arithmetic only, not concurrent model wall or a guaranteed hit-rate threshold. Fill includes destination touch but excludes allocation syscall. Calls include miss served after prepare. Request-cold simulation and actual persistent/prefaulted pool lifetimes differ; nt2..4 speed unmeasured.')
    (D/'cpu-predecode-amortization-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    for control in ['ggml','existing_exact']:
        for ty in [18,22,21]:
            samples=[r for r in out if r['control']==control and r['type']==ty]
            print(control,ty,'cold',min(r['cold_calls_to_amortize'] for r in samples),max(r['cold_calls_to_amortize'] for r in samples),'refill',min(r['refill_calls_to_amortize'] for r in samples),max(r['refill_calls_to_amortize'] for r in samples))
if __name__=='__main__':main()
