import json
from protocol import validate, EXPECTED

def fixture():
 out=[]
 for name in ['fp16','int8']:
  out+=['FORMAT_BEGIN format=%s slots=8192 blocks=8256 nq=2,3,4,2,3,4 resolves=6'%name,'FORMAT_ALLOCATED format=%s usm_bytes=10000000 live_buffer_bytes_upper=12000000 limit=67108864'%name]
  for i,misses in enumerate([36,8,8,0,4,8]):
   nq=2+i%3
   out+=['RESOLVE_BEGIN format=%s pass=%d nq=%d expected_misses=%d'%(name,i+1,nq,misses),'RESOLVE_DONE format=%s pass=%d nq=%d misses=%d map_entries=16448 pool_compares=%d oracle_bytes=%d selected_cells=%d'%(name,i+1,nq,misses,16384 if name=='fp16' else 32768,16777216 if name=='fp16' else 8650752,nq*512)]
  out+=['FORMAT_PASS format='+name+' '+' '.join(k+'='+str(v) for k,v in EXPECTED[name].items())]
 for name in ['fp16','int8']:out+=['FINAL_COUNTS format='+name+' '+' '.join(k+'='+str(v) for k,v in EXPECTED[name].items())]
 out+=['PASS formats=2 resets=2 resolves=12 uploads=38 readbacks=72 direct_copies=110 direct_copy_bytes=179147776 drains=16 released=30 map_entries=197376 selected_cells=18432 pool_compares=294912 oracle_bytes=152567808 lookups=4608 missed_pages=128 production_copy_runs=384 production_copy_bytes=198656']
 return '\n'.join(out)+'\n'
so=fixture();assert len(validate(so,'')['resolve_results'])==12
cases=[(so.replace('format=int8','format=fp16',1),''),(so.replace('misses=36 map_entries','misses=35 map_entries',1),''),(so.replace('limit=67108864','limit=12000000',1),''),(so.replace('released=13','released=12',1),''),(so.replace('oracle_bytes=16777216','oracle_bytes=0',1),''),(so.replace('resolves=12 uploads=38','resolves=11 uploads=38'),''),(so,'UR_RESULT_ERROR_DEVICE_LOST'),(so.replace('RESOLVE_DONE','RESOLVE_MISSING',1),'')]
for n,(a,b) in enumerate(cases):
 try:validate(a,b)
 except ValueError:pass
 else:raise RuntimeError('negative accepted '+str(n))
print(json.dumps(dict(passed=True,positive_cases=1,negative_cases=8,gpu_executed=False,scope='parser only, fabricated fixtures')))
