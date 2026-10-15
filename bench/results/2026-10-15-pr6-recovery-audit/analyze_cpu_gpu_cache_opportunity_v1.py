#!/usr/bin/env python3
"""Offline conditional route-coverage ceiling, never GPU capacity or speed evidence."""
import base64,hashlib,json,re
from pathlib import Path
D=Path(__file__).parent

def main():
    p=D/'cpu-gu-reuse-long-v1.json';r=json.loads(p.read_text());assert r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults']
    req=r['requests'][0];obs=req['recurrence'];raw=base64.b64decode(obs['expert_counts_base64'],validate=True)
    values=[];value=shift=0
    for b in raw:
        assert shift<64;value|=(b&127)<<shift
        if b&128:shift+=7
        else:values.append(value);value=shift=0
    assert not shift and len(values)==98304
    sizes={}
    for line in req['ledger']['lines']:
        if line.startswith('format l='):
            f=dict(re.findall(r'(\w+)=(\d+)',line));layer=int(f['l']);assert layer not in sizes;sizes[layer]=int(f['bytes'])
    assert len(sizes)==48
    entries=[]
    for l in range(48):
        for e in range(512):
            counts=values[(l*512+e)*4:(l*512+e+1)*4];jobs=sum(counts);rows=sum((n+1)*v for n,v in enumerate(counts))
            if jobs:entries.append(dict(layer=l,expert=e,jobs=jobs,rows=rows,blob_bytes=sizes[l]))
    assert len(entries)==obs['header']['first_touches_unique']
    assert sum(x['jobs'] for x in entries)==obs['header']['jobs'] and sum(x['rows'] for x in entries)==obs['header']['tokens']
    key=lambda x:(-x['rows'],x['layer'],x['expert'])
    entries.sort(key=key);out=[]
    for n in [128,256,512]:
        for policy in ['global_row_rank','append_legacy_quotas','append_balanced_quotas']:
            if policy=='global_row_rank':selected=entries[:n]
            else:
                selected=[]
                for layer in range(48):
                    count=n//48+((n%48 if layer==47 else 0) if policy=='append_legacy_quotas' else int(layer<n%48))
                    eligible=[x for x in entries if x['layer']==layer];assert len(eligible)>=count;selected+=eligible[:count]
            assert len(selected)==n
            rows=sum(x['rows'] for x in selected);jobs=sum(x['jobs'] for x in selected)
            out.append(dict(additional_slots=n,hypothetical_total_slots=128+n,policy=policy,CPU_rows=rows,CPU_jobs=jobs,
                CPU_row_fraction=rows/obs['header']['tokens'],CPU_job_fraction=jobs/obs['header']['jobs'],
                source_blob_bytes=sum(x['blob_bytes'] for x in selected),selected_pairs_sha256=hashlib.sha256(json.dumps([(x['layer'],x['expert']) for x in selected]).encode()).hexdigest()))
    result=dict(passed=True,GPU_executed=False,adopted=False,model_performance_qualified=False,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=out,
        ranking='Descending observed CPU expert-token rows, then ascending layer/expert IDs; no global optimal byte-knapsack claim.',
        scope='Hindsight fixed-route CPU-owner coverage only. Existing GPU-owner counts/IDs are missing; actual resident-set disjointness and transferable eligibility must be checked before admission. No prediction of future routing/output/GPU kernel time/H2D time/live VRAM or wall speed. Appended quota model preserves old residents conceptually; recomputing total-size partition can shrink a layer and is a different experiment.')
    (D/'cpu-gpu-cache-opportunity-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    for x in out:print(x['additional_slots'],x['policy'],x['CPU_rows'],x['CPU_jobs'],x['source_blob_bytes'])
if __name__=='__main__':main()
