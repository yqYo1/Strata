#!/usr/bin/env python3
"""Offline recurrence and measured-service scenarios; never claims model wall benefit."""
import hashlib,json
from pathlib import Path
D=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rp=D/'cpu-gu-reuse-long-v1.json';cp=D/'cpu-predecode-amortization-v1.json'
    r,c=[json.loads(p.read_text()) for p in [rp,cp]]
    assert r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults'] and c['passed']
    req=r['requests'][0];obs=req['recurrence'];assert req['reference_ids_equal'] and obs['header']['complete']==1
    out=[]
    for lru in obs['lrus']:
        if lru['policy']!=1:continue
        buckets=[x for x in obs['lru_format_nt'] if x['policy']==1 and x['budget']==lru['budget'] and x['nt']==1]
        for control in ['ggml','existing_exact']:
            for process in ['initial',0,1,2]:
                entries=[]
                for b in buckets:
                    ty=[18,22,21][b['group']];cost=next(x for x in c['rows'] if x['type']==ty and x['process']==process and x['control']==control)
                    uses=b['hits']+b['misses'];benefit=uses*cost['median_saved_thread_cpu_us']
                    refill=b['misses']*cost['prepare_thread_cpu_us']['overwrite_refill']
                    cold=b['misses']*cost['prepare_thread_cpu_us']['cold_first_touch_fill']
                    entries.append(dict(type=ty,hits=b['hits'],misses=b['misses'],uses=uses,uses_per_fill=uses/b['misses'],
                        measured_service_saved_us=benefit,all_misses_overwrite_prepare_us=refill,
                        all_misses_cold_prepare_us=cold,net_overwrite_service_us=benefit-refill,net_cold_service_us=benefit-cold))
                out.append(dict(budget=lru['budget'],control=control,process=process,formats=entries,
                    net_overwrite_thread_cpu_seconds=sum(x['net_overwrite_service_us'] for x in entries)/1e6,
                    net_cold_thread_cpu_seconds=sum(x['net_cold_service_us'] for x in entries)/1e6))
    result=dict(passed=True,adopted=False,GPU_executed=False,model_performance_qualified=False,
        source_receipts={rp.name:sha(rp),cp.name:sha(cp)},actual_done=req['done'],lrus=obs['lrus'],scenarios=out,
        decision='Do not implement a synchronous whole-GU request-cold nt1 cache at these budgets from the current evidence. Even all-misses-overwrite scenarios consume more CPU service than they save in this128output workload. Keep packed and prioritize clean Direct benchmark. This does not exclude a different persistent/selective/fused/overlapped design.',
        scope='Ordered request-cold simulation plus single-thread synthetic measured cost arithmetic. No real expanded cache allocation, live RAM feasibility, concurrent model wall, or guaranteed lower/upper bound. Every eligible miss is assumed prepared then served by expanded kernel. No nt2..4 speed extrapolation.')
    assert all(x['net_overwrite_thread_cpu_seconds']<0 for x in out)
    (D/'cpu-gu-reuse-summary-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    for budget in [268435456,1073741824,4294967296,8589934592]:
        s=[x['net_overwrite_thread_cpu_seconds'] for x in out if x['control']=='ggml' and x['budget']==budget]
        print(budget//(1<<20),'MiB, net overwrite thread service seconds',min(s),max(s))
if __name__=='__main__':main()
