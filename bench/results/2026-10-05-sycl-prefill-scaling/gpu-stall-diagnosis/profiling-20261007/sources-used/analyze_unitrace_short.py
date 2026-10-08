"""Summarize unions of GPU intervals; do not add nested host API durations."""
from pathlib import Path
import collections
import hashlib
import json
import re

b=Path(__file__).parent
p=b/'unitrace-model-short-profile'
trace=next(p.glob('strata-*.json'))
profile=json.loads((p/'record.json').read_text())
clean=json.loads((b/'unitrace-model-short-clean/record.json').read_text())
assert profile['passed'] and clean['passed']
data=json.loads(trace.read_bytes());events=data['traceEvents']
gpu=[e for e in events if e.get('cat')=='gpu_op' and e.get('ph')=='X']
host=[e for e in events if e.get('cat')=='cpu_op' and e.get('ph')=='X']
copy_pattern=re.compile(r'^zeCommandListAppendMemoryCopy(?:Region)?\(([^)]+)\)\[(\d+)\]')

def union(intervals):
    total=0;start=end=None
    for a,z in sorted(intervals):
        if start is None:start,end=a,z
        elif a<=end:end=max(end,z)
        else:total+=end-start;start,end=a,z
    return (total+(end-start if start is not None else 0))/1e6

def summarize(a,z):
    selected=[e for e in gpu if a<=e['ts']<z]
    clips=lambda rows:[(max(a,e['ts']),min(z,e['ts']+e['dur'])) for e in rows if e['ts']+e['dur']>a and e['ts']<z]
    copies=[];kernels=[];fills=[];directions={};kernel_time=collections.Counter();kernel_calls=collections.Counter()
    for e in selected:
        match=copy_pattern.match(e['name'])
        if match:
            direction,size=match[1],int(match[2]);copies.append(e)
            row=directions.setdefault(direction,dict(bytes=0,calls=0,sum_seconds=0))
            row['bytes']+=size;row['calls']+=1;row['sum_seconds']+=e['dur']/1e6
        elif e['name'].startswith('zeCommandListAppendMemoryFill'):fills.append(e)
        else:
            kernels.append(e);kernel_time[e['name']]+=e['dur']/1e6;kernel_calls[e['name']]+=1
    for row in directions.values():row['active_time_gb_per_second']=row['bytes']/1e9/row['sum_seconds'] if row['sum_seconds'] else None
    api_time=collections.Counter();api_calls=collections.Counter();waits=[]
    for e in host:
        if not a<=e['ts']<z or not e['name'].startswith(('ze','zex')):continue
        api_time[e['name']]+=e['dur']/1e6;api_calls[e['name']]+=1
        if e['name'] in ['zeEventHostSynchronize','zeCommandListHostSynchronize','zeCommandQueueSynchronize']:waits.append(e)
    return dict(wall_envelope_seconds=(z-a)/1e6,gpu_event_count=len(selected),
                gpu_busy_union_seconds=union(clips(gpu)),copy_busy_union_seconds=union(clips(copies)),
                kernel_busy_union_seconds=union(clips(kernels)),fill_busy_union_seconds=union(clips(fills)),
                host_wait_union_seconds=union(clips(waits)),copy_directions=directions,
                top_level_zero_inclusive_api_times=[dict(name=n,calls=api_calls[n],sum_seconds=t) for n,t in api_time.most_common(12)],
                top_kernel_duration_sums=[dict(name=n,calls=kernel_calls[n],sum_seconds=t) for n,t in kernel_time.most_common(12)])

record=dict(scope='Protocol-received timestamp envelopes of phase-synchronized context-128/chunk-32/layer-major-2 normal-MTP profile; not evidence for long-prompt transfer dominance',
            trace_path=str(trace),trace_bytes=trace.stat().st_size,
            trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            event_counts={label:sum(1 for e in events if (e.get('cat')==label if label in ['gpu_op','cpu_op'] else e.get('cat','').startswith('Flow_'))) for label in ['gpu_op','cpu_op','flow']},
            requests=[],notes=[
                'Explicit copy bytes omit PCIe traffic caused by GPU kernels reading mapped host memory.',
                'Copy bandwidth uses summed copy durations, not end-to-end elapsed time or an isolated bus benchmark.',
                'GPU busy is interval union across traced commands, not hardware engine/EU utilization.',
                'Host API durations are inclusive and overlap GPU work and nested UR/SYCL calls; do not add them.',
                'Protocol timestamps are observed by an external poller and may lag engine boundaries.',
                'Existing STRATA_PREFILL_SYNC=1 remains on in both runs; neither is an unsynchronized throughput baseline.',
                'Single controls are preliminary; trace parsing overlapped part of clean model startup/first request.',
                'Startup profiling is paused; no hardware performance counters collected.'
            ])
for req, control in zip(profile['requests'],clean['requests']):
    assert req['name']==control['name'] and req['head_sha256']==control['head_sha256']
    a,z=req['start_epoch_us'],req['end_epoch_us']
    pp=[x['epoch_us'] for x in req['protocol'] if x['text'].startswith('PP ')]
    tokens=[x['epoch_us'] for x in req['protocol'] if x['text'].startswith('T ')]
    row=dict(name=req['name'],profile_done=req['done'],clean_done=control['done'],
             exact_head_and_tokens_and_logprobs=all(req['equality'].values()) and all(control['equality'].values()),
             whole_request=summarize(a,z))
    if pp:row['request_to_last_prefill_progress']=summarize(a,pp[-1])
    if tokens:row['first_token_to_done']=summarize(tokens[0],z)
    record['requests'].append(row)
(b/'unitrace-short-analysis.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['requests']},indent=2))
for row in record['requests']:
    s=row['whole_request'];print(row['name'],{k:s[k] for k in ['wall_envelope_seconds','gpu_event_count','gpu_busy_union_seconds','copy_busy_union_seconds','kernel_busy_union_seconds','host_wait_union_seconds','copy_directions']})
