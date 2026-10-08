"""Stream host-only API events; distinguish elapsed unions from concurrent sums."""
from pathlib import Path
import collections,datetime,hashlib,json,time

base=Path(__file__).parent
out=base/'host-profile32k-v01402-analysis';out.mkdir(mode=0o700)
sequence=json.loads((base/'host-profile32k-v01402-sequence/record.json').read_text());assert sequence['passed'] and not sequence['active']
profile_path=base/'owned-event-ack-no-root-host-profile-v01402-code32k-state-r1/record.json'
control_path=base/'owned-event-ack-no-root-host-profile-v01402-code32k-control-r1/record.json'
profile=json.loads(profile_path.read_text());control=json.loads(control_path.read_text())
assert profile['healthy'] and control['healthy'] and not profile['active'] and not control['active']
trace=next(profile_path.parent.glob('strata*.json'))
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Quiet host-only unitrace32K native API trace. All state/head/IDs/logprobs match a fresh unprofiled32K control and completed baseline. No device timestamps or GPU metrics; these inclusive host calls overlap CPU/GPU work and cannot alone establish PCIe bandwidth or GPU/CPU bottlenecks. State/head capture excludes throughput claims.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def union(intervals):
 total=0;begin=end=None
 for a,z in sorted(intervals):
  if begin is None:begin,end=a,z
  elif a<=end:end=max(end,z)
  else:total+=end-begin;begin,end=a,z
 return (total+(end-begin if begin is not None else 0))/1e6
save()
try:
 timer=base/'pti-gpu-profiler-source/tools/unitrace/src/unitimer.h'
 source=timer.read_text()
 assert 'GetHostBootTimestamp()' in source and 'clock_gettime(CLOCK_MONOTONIC_RAW' in source
 assert 'return epoch_start_time_ + systime;' in source and 'GetEpochTimeInUs' in source
 # This collector builds its epoch offset from BOOTTIME but records API time
 # with MONOTONIC_RAW. Correct the observed boot/raw offset before using the
 # external protocol's wall clock. It is measured after the job, so keep phase
 # boundaries approximate and report sensitivity to +/-10ms correction.
 samples=[]
 for _ in range(50):
  before=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
  boot=time.clock_gettime_ns(time.CLOCK_BOOTTIME)
  after=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
  samples.append({'bracket_ns':after-before,'boot_minus_raw_ns':boot-(before+after)/2})
 best=min(samples,key=lambda value:value['bracket_ns']);shift_us=best['boot_minus_raw_ns']/1000
 record['clock_correction']={'measured_after_profile_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'add_to_trace_ts_us':shift_us,'samples':samples,'unitimer_sha256':digest(timer),'inference':'Native API timestamp is raw time plus a boot-derived epoch offset; adding BOOTTIME-MONOTONIC_RAW aligns it approximately with observed protocol wall times. No per-request calibration was captured, so phase attribution is approximate; test +/-10ms sensitivity rather than treating phase boundaries as exact.'}
 req=profile['requests'][0];other=control['requests'][0]
 assert req['first_head']['sha256']==other['first_head']['sha256'] and req['ids']==other['ids'] and req['logprobs']==other['logprobs']
 assert req['prefill_state']['parts']==other['prefill_state']['parts']
 bounds={'whole_request':(req['start_epoch_us'],req['end_epoch_us'])}
 pp=[item['epoch_us'] for item in req['protocol_timestamps'] if item['text'].startswith('PP ')]
 generated=[item['epoch_us'] for item in req['protocol_timestamps'] if item['text'].startswith('T ')]
 if pp:bounds['request_to_last_prefill_progress']=(req['start_epoch_us'],pp[-1])
 if generated:bounds['first_token_to_done']=(generated[0],req['end_epoch_us'])
 record['profile_measurement']=req['measurement'];record['control_measurement']=other['measurement']
 record['profile_to_control_elapsed_ratio']={key:req['measurement'][key]/other['measurement'][key] for key in ['prompt_ms','decode_ms','wall_seconds']}
 # Retain only scalars needed for analysis, rather than loading a multi-million
 # event document into JSON objects. Original bytes stay private and hashed.
 calls=[];counts=collections.Counter();metadata=0;gpu=0
 with trace.open() as stream:
  for number,line in enumerate(stream,1):
   text=line.strip().lstrip(',').strip().rstrip(',')
   if not text.startswith('{"'):continue
   event=json.loads(text)
   if event.get('cat')=='gpu_op':gpu+=1
   if event.get('ph')!='X' or event.get('cat')!='cpu_op':metadata+=1;continue
   assert event['dur']>=0 and event['pid']==profile['inferior']['pid']
   calls.append((float(event['ts']),float(event['dur']),int(event['tid']),event['name']))
   counts[event['name']]+=1
 assert calls and gpu==0
 record['trace']={'file':str(trace),'bytes':trace.stat().st_size,'sha256':digest(trace),'native_host_calls':len(calls),'gpu_events':gpu,'other_events':metadata,'recorded_target_pid':profile['inferior']['pid']}
 record['whole_process_api_counts']=counts.most_common(20)
 def summarize(a,z,shift):
  api_seconds=collections.Counter();api_count=collections.Counter();thread_intervals=collections.defaultdict(list);thread_apis=collections.defaultdict(collections.Counter)
  waits=[];queries=[];appends=[]
  for ts,duration,tid,name in calls:
   start=ts+shift;end=start+duration
   if end<=a or start>=z:continue
   begin=max(a,start);finish=min(z,end);seconds=(finish-begin)/1e6
   api_count[name]+=1;api_seconds[name]+=seconds
   thread_intervals[tid].append((begin,finish));thread_apis[tid][name]+=1
   if name in ['zeEventHostSynchronize','zeCommandListHostSynchronize','zeCommandQueueSynchronize']:waits.append((begin,finish))
   elif name=='zeEventQueryStatus':queries.append((begin,finish))
   elif name.startswith('zeCommandListAppend'):appends.append((begin,finish))
  return {'observed_wall_envelope_seconds':(z-a)/1e6,'native_wait_union_seconds':union(waits),'native_status_query_union_seconds':union(queries),'native_append_union_seconds':union(appends),'top_inclusive_api_time_sums':[{'name':name,'calls':api_count[name],'inclusive_seconds_clipped':seconds} for name,seconds in api_seconds.most_common(15)],'threads':[{'tid':tid,'is_main':tid==profile['inferior']['pid'],'native_api_union_seconds':union(intervals),'top_call_counts':thread_apis[tid].most_common(5)} for tid,intervals in sorted(thread_intervals.items(),key=lambda pair:len(pair[1]),reverse=True)]}
 record['phases']={name:summarize(a,z,shift_us) for name,(a,z) in bounds.items()}
 a,z=bounds['whole_request']
 record['clock_sensitivity']={str(delta):{key:value for key,value in summarize(a,z,shift_us+delta).items() if key not in ['threads','top_inclusive_api_time_sums']} for delta in [-10000,10000]}
 record['notes']=['No GPU event instrumentation or hardware counters; the host trace does not measure transfer execution duration/bytes or kernel execution time.','Host API time sums run concurrently and may be nested; do not add them or subtract them from elapsed time to estimate other work.','Native status querying is event polling, not a measured CPU utilization percentage.','Protocol timestamps are externally observed and include transport/poll delay; prefill progress and first-token boundaries are approximate.','Unitimer boot/raw clock offset correction is inferred from the exact pinned source and measured after the job; phase attribution remains approximate.','Both profile and control retain state/head capture and one process each; no clean speed gain or full256K claim.','The profiler source, option derivation, actual target environment and all full state/head parity gates were checked before interpreting the trace.']
 record['controller_sha256']=digest(Path(__file__));record['passed']=True
except BaseException as error:record['error']=repr(error);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({'passed':record['passed'],'trace':record['trace'],'profile_to_control_elapsed_ratio':record['profile_to_control_elapsed_ratio'],'whole_request':record['phases']['whole_request']},indent=2))
