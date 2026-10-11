import json
from protocol import validate

def fixture(mode):
 header='PROBE mode='+mode+' STRATA_NO_BLAS='+('1' if mode=='free' else '0')+' samples=8 device_bytes=16777216 host_usm_bytes=33554432 per_sample_transfer_bytes=33554432 queue=Runtime.compute(in_order) wall_limit=wall*1.25+10ms scope=queue_timeline_not_exclusive_kernel_service'
 cs=['unrecorded pair','completed non-timing pair'];es=['event_elapsed_ms: an event was not recorded','event_elapsed_ms: an event was not created for timing']
 out=[header]+['CONTROL name='+a+' result=0 sentinel_unchanged=1 error='+b for a,b in zip(cs,es)]
 err=['API operation='+a+' result=0 error='+b for a,b in zip(cs,es)]
 for i in range(8):
  reverse='event_elapsed_ms: end < start (start=%d, end=%d)'%(100000000+i*2000000+1000000,100000000+i*2000000)
  out+=['SAMPLE_BEGIN index=%d planned_copies=2 planned_transfer_bytes=33554432'%i,'SAMPLE index=%d copies=2 transfer_bytes=33554432 records=2 compute_waits=1 elapsed_queries=3 api_calls=8 checked_words=4194304 normal_result=1 elapsed_ms=1 normal_error=none host_submit_to_completion_wall_ms=2 reversed_result=0 sentinel_unchanged=1 raw_endpoints_available=1 raw_start_ns=%d raw_end_ns=%d reversed_error=%s same_result=1 same_ms=0 same_error=none plausible=1'%(i,100000000+i*2000000,100000000+i*2000000+1000000,reverse)]
  err+=['API operation=sample reversed event_elapsed_ms result=0 error='+reverse]
 out+=['RESULT status=PASS samples=8 planned_samples=8 positive_reversals=8 api_calls=75 records=18 copies=16 compute_waits=9 cleanup_waits=1 elapsed_queries=26 checked_words=33554432']
 return '\n'.join(out)+'\n','\n'.join(err)+'\n'
for mode in ['free','contrib-llvm']:
 so,se=fixture(mode);assert validate(so,se,mode)['sample_count']==8
 cases=[(so.replace('index=1 copies=2','index=0 copies=2'),se),(so.replace('sentinel_unchanged=1','sentinel_unchanged=0',1),se),(so.replace('elapsed_ms=1','elapsed_ms=nan',1),se),(so.replace('samples=8 planned_samples=8','samples=7 planned_samples=8'),se),(so,se+'API operation=unknown result=0 error=unexpected\n'),(so.replace('raw_end_ns=101000000','raw_end_ns=100000000',1),se),(so.replace('elapsed_ms=1','elapsed_ms=0.5',1),se),(so,se+'UR_RESULT_ERROR_DEVICE_LOST\n')]
 for n,(a,b) in enumerate(cases):
  try:validate(a,b,mode)
  except ValueError:pass
  else:raise RuntimeError('negative accepted '+mode+' '+str(n))
print(json.dumps(dict(passed=True,positive_cases=2,negative_cases=16,gpu_executed=False,scope='parser supervision only, fabricated fixtures')) )
