#!/usr/bin/env python3
"""Three fresh clean official-bench samples per window, never concurrent GPU jobs."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
A=Path(__file__).resolve().parent
ORDER=((2,0),(8,0),(8,1),(2,1),(2,2),(8,2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dst=A/'moe-accum-model-clean-cohort-v1.json';assert not dst.exists()
    runner=A/'run_moe_accum_model_clean_v3.py'
    gates=[A/f'moe-accum-model-profile-window{w}-tokens65536-v2.json' for w in (2,8)]
    for f in gates:
        x=json.loads(f.read_text());assert x['passed'] and x['complete'] and x['ownership_closed'] and not x['active'] and not x['fresh_faults'] and not x['driver_dump_exists']
    r={'schema':'strata-moe-accum-clean-cohort-v1','passed':False,'complete':False,'active_sample':None,
       'order':ORDER,'runner_sha256':sha(runner),'recipe_sha256':sha(__file__),'gates':{str(f):sha(f) for f in gates},
       'completed':[],'performance_adopted':False,'general_quality_qualified':False,'physical262144_RESTORE_qualified':False,
       'scope':'Official bench.sh64K-minimum, thinkingON24-token probe, same binary,128K resident context, allGPU residuals; window2 vs window8. Profiles disabled and STRATA_TRACE/UR layers absent. Each sample has its own ownership/health receipt. No sustained decode/general-quality/context adoption.',
       'budget':{'samples':6,'raw_text_cap_per_sample':64<<20,'aggregate_raw_text_cap':6*(64<<20)}}
    def save():dst.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for w,n in ORDER:
            assert sha(runner)==r['runner_sha256'] and sha(__file__)==r['recipe_sha256']
            assert all(sha(f)==h for f,h in r['gates'].items())
            r['active_sample']={'window':w,'repetition':n};save();print(json.dumps({'start':w,'repetition':n}),flush=True)
            command=[sys.executable,str(runner),str(w),'--prefill-tokens','65536','--repeat',str(n)]
            code=subprocess.run(command).returncode
            f=A/f'moe-accum-model-profile-window{w}-tokens65536-clean-r{n}-v3.json'
            item={'window':w,'repetition':n,'command':command,'controller_exit_code':code,'path':str(f),'sha256':sha(f) if f.exists() else None}
            if f.exists():
                sample=json.loads(f.read_text());item.update(passed=sample['passed'],complete=sample['complete'],ownership_closed=sample['ownership_closed'])
                requests=sample.get('benchmark_result',{}).get('results',{}).get('prefill_only',{}).get('requests',[])
                if requests:item['request']=requests[0]
            r['completed'].append(item);r['active_sample']=None;save()
            assert code==0 and item['passed'] and item['complete'] and item['ownership_closed'],item
            print(json.dumps({'finished':w,'repetition':n,'prompt_tps':item['request']['prompt_tps']}),flush=True)
        samples=[json.loads(Path(i['path']).read_text()) for i in r['completed']]
        binary=samples[0]['binary'];requests=[x['benchmark_result']['results']['prefill_only']['requests'][0] for x in samples]
        assert all(x['binary']==binary for x in samples)
        assert len({x['request_sha256'] for x in requests})==len({x['actual_prompt_tokens'] for x in requests})==1
        assert all(x['actual_prompt_tokens']>=65536 and x['actual_reused_tokens']==0 and x['eligible'] and x['stream_complete'] and x['thinking_requested'] and x['reasoning_observed'] for x in requests)
        r['summaries']={}
        for w in (2,8):
            values=[i['request']['prompt_tps'] for i in r['completed'] if i['window']==w]
            r['summaries'][str(w)]={'n':len(values),'samples':values,'median':statistics.median(values),'mean':statistics.mean(values),'stdev':statistics.stdev(values),'minimum':min(values),'maximum':max(values)}
        old,new=r['summaries']['2'],r['summaries']['8']
        r.update(passed=True,complete=True,binary=binary,median_ratio=new['median']/old['median'],all_candidate_samples_above600=new['minimum']>600,
                 all_candidate_samples_above_baseline_max=new['minimum']>old['maximum'])
    except BaseException as error:r['error']=repr(error)
    finally:r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print(json.dumps({k:r.get(k) for k in ('passed','complete','error','summaries','all_candidate_samples_above600')}),flush=True)
    return 0 if r['passed'] else 1
if __name__=='__main__':sys.exit(main())
