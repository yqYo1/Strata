from pathlib import Path
from collections import defaultdict, Counter
import csv,json,sys,statistics
label=sys.argv[1]
prefix=Path(f'/tmp/strata-sycl-goal-prefill-device-full-{label}')
raw=list(csv.DictReader(prefix.with_suffix('.csv').open()))
rows=[r for r in raw if int(r['end_ns'])>=int(r['start_ns'])>0]
assert rows

def union_ns(entries):
    ranges=sorted((int(x['start_ns']),int(x['end_ns'])) for x in entries)
    total=0;begin=end=0
    for a,b in ranges:
        if a>end:total+=end-begin;begin,end=a,b
        else:end=max(end,b)
    return total+end-begin

by=defaultdict(list)
for r in rows:by[(r['kind'],int(r['type']),int(r['cols']))].append(r)
groups=[]
for key,rs in by.items():
    times=[(int(x['end_ns'])-int(x['start_ns']))/1e6 for x in rs]
    groups.append(dict(kind=key[0],type_or_source_line=key[1],cols=key[2],count=len(rs),
                       bytes=sum(int(x['bytes']) for x in rs),device_ms=sum(times),
                       union_ms=union_ns(rs)/1e6,median_ms=statistics.median(times),
                       cpu_submit_ms=sum(float(x['submit_us']) for x in rs)/1000 if key[0] in ('copy','xmx') else None))
groups.sort(key=lambda g:g['union_ms'],reverse=True)
shape=[]
for line in Path(str(prefix)+'.csv.shapes.tsv').read_text().splitlines():
    v=list(map(int,line.split('\t')));T,l,j0,ngx,nr,maxr,gt,dt=v[:8];counts=v[8:]
    assert len(counts)==ngx and sum(counts)==nr and max(counts)==maxr
    shape.append(dict(tokens=T,layer=l,first_expert_index=j0,experts=ngx,routed_rows=nr,max_rows=maxr,
                      gu_type=gt,down_type=dt,counts=counts,uniform_tiles8=ngx*((maxr+7)//8),
                      compact_tiles8=sum((x+7)//8 for x in counts)))
span=(max(int(x['end_ns']) for x in rows)-min(int(x['start_ns']) for x in rows))/1e6
counts=Counter((int(r['start_ns']),int(r['end_ns'])) for r in rows)
unique=sum((b-a)/1e6 for (a,b) in counts)
measured=union_ns(rows)/1e6
out=dict(label=label,raw_events=len(raw),valid_events=len(rows),invalid_intervals=len(raw)-len(rows),
         duplicate_intervals=sum(n-1 for n in counts.values()),event_span_ms=span,
         raw_duration_sum_ms=sum(g['device_ms'] for g in groups),unique_duration_sum_ms=unique,
         measured_union_ms=measured,overlapping_unique_interval_ms=unique-measured,
         unmeasured_span_ms=span-measured,groups=groups,shape_groups=len(shape),
         counterfactual_uniform_tiles8=sum(s['uniform_tiles8'] for s in shape),
         actual_compact_tiles8=sum(s['compact_tiles8'] for s in shape),
         warnings=['Diagnostic overhead is included in wall time.',
                   'Common finish events have no CPU submission measurement.',
                   'A oneMKL completion event may cover only its final internal command.',
                   'Unmeasured span includes uncaptured commands and host gaps, not necessarily idle time.'],shapes=shape)
Path(str(prefix)+'-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k not in ('groups','shapes')})
for g in groups[:25]:print(g)
