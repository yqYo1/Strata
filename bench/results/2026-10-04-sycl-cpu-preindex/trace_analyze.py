from pathlib import Path
from collections import Counter,OrderedDict,defaultdict
import json
p=Path('/tmp/strata-sycl-goal-index-reuse.tsv')
rows=[]
for line in p.read_text().splitlines():
 b,ty,n,ff,row,up,ptr,nt,layer,expert=line.split('\t');rows.append(dict(batch=int(b),type=int(ty),n=int(n),ff=int(ff),ptr=ptr,nt=int(nt),key=(int(layer),int(expert))))
eligible=[r for r in rows if r['nt']==1 and r['type'] in (21,22) and r['n']==2560]
counts=Counter(r['key'] for r in eligible);types={r['key']:r['type'] for r in eligible};sizes={r['key']:2*r['ff']*(r['n']//256)*(166 if r['type']==21 else 106) for r in eligible}
ptrkeys=defaultdict(set)
for r in rows:ptrkeys[r['ptr']].add(r['key'])
results=[]
for mib in (512,1024,2048,4096,8192,16384):
 for admit in (1,2,4,8):
  seen=Counter();cache=OrderedDict();used=peak=hits=packs=evictions=0;hittypes=Counter();packtypes=Counter()
  for r in eligible:
   key=r['key'];seen[key]+=1
   if key in cache:cache.move_to_end(key);hits+=1;hittypes[r['type']]+=1
   elif seen[key]>=admit:
    while used+sizes[key]>mib*1024**2 and cache:
     old,sz=cache.popitem(last=False);used-=sz;evictions+=1
    cache[key]=sizes[key];used+=sizes[key];peak=max(peak,used);packs+=1;packtypes[r['type']]+=1
  # Estimates only: measured six-core E64 component savings and serial warm packing, excluding allocation.
  saved=sum(hittypes[t]*v for t,v in ((21,(4103.22-3235.865)/64),(22,(2460.44-2126.74)/64)))
  pack_us=packtypes[21]*16657.8/64+packtypes[22]*7799.18/64
  results.append(dict(mib=mib,admission_use=admit,hits=hits,packs=packs,evictions=evictions,hit_fraction=hits/len(eligible),peak_bytes=peak,hit_types=dict(hittypes),pack_types=dict(packtypes),estimated_saved_kernel_us=saved,estimated_serial_pack_us=pack_us))
out=dict(trace_rows=len(rows),eligible_rows=len(eligible),unique_eligible=len(counts),unique_expanded_bytes=sum(sizes.values()),type_uses=dict(Counter(r['type'] for r in eligible)),reuse_histogram=dict(Counter(counts.values())),pointer_addresses=len(ptrkeys),pointer_addresses_with_multiple_logical_experts=sum(len(x)>1 for x in ptrkeys.values()),simulations=results)
Path('/tmp/strata-sycl-goal-index-trace-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('simulations','reuse_histogram')},indent=2))
for r in results:print(r)
