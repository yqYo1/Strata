from pathlib import Path
from collections import defaultdict,Counter
import csv,json,sys,statistics
label=sys.argv[1];prefix=Path(f'/tmp/strata-sycl-goal-prefill-device-trace-{label}')
rows=list(csv.DictReader(prefix.with_suffix('.csv').open()))
assert rows and all(int(r['end_ns'])>=int(r['start_ns'])>0 for r in rows)
def union_ns(entries):
 ranges=sorted((int(x['start_ns']),int(x['end_ns'])) for x in entries);total=0;begin=end=0
 for a,b in ranges:
  if a>end:total+=end-begin;begin,end=a,b
  else:end=max(end,b)
 return total+end-begin
by=defaultdict(list)
for r in rows:by[(r['kind'],int(r['type']),int(r['cols']))].append(r)
groups=[]
for key,rs in sorted(by.items()):
 times=[(int(x['end_ns'])-int(x['start_ns']))/1e6 for x in rs];nbytes=sum(int(x['bytes']) for x in rs)
 groups.append(dict(kind=key[0],type=key[1],cols=key[2],count=len(rs),bytes=nbytes,device_ms=sum(times),union_ms=union_ns(rs)/1e6,median_ms=statistics.median(times),cpu_submit_ms=sum(float(x['submit_us']) for x in rs)/1000,max_rows_histogram=dict(Counter(int(x['max_rows']) for x in rs))))
shape=[]
for line in Path(str(prefix)+'.csv.shapes.tsv').read_text().splitlines():
 v=list(map(int,line.split('\t')));T,l,j0,ngx,nr,maxr,gt,dt=v[:8];counts=v[8:];assert len(counts)==ngx and sum(counts)==nr and max(counts)==maxr
 shape.append(dict(tokens=T,layer=l,first_expert_index=j0,experts=ngx,routed_rows=nr,max_rows=maxr,gu_type=gt,down_type=dt,counts=counts,allocated_tiles8=ngx*((maxr+7)//8),valid_tiles8=sum((x+7)//8 for x in counts)))
alloc=sum(s['allocated_tiles8'] for s in shape);valid=sum(s['valid_tiles8'] for s in shape)
span=(max(int(x['end_ns']) for x in rows)-min(int(x['start_ns']) for x in rows))/1e6
out=dict(label=label,events=len(rows),event_span_ms=span,measured_union_ms=union_ns(rows)/1e6,groups=groups,shape_groups=len(shape),allocated_expert_tiles8=alloc,valid_expert_tiles8=valid,empty_expert_tile_fraction=1-valid/alloc,shapes=shape)
Path(str(prefix)+'-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k not in ('groups','shapes')})
for r in groups:print({k:v for k,v in r.items() if k!='max_rows_histogram'})
