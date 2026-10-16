#!/usr/bin/env python3
"""Reconcile closed profiler samples and derive scoped hardware-resource evidence.
Repository LICENSE; offline only, no native/GPU work, cleanup or adoption.
"""
from collections import Counter
import datetime,fcntl,hashlib,json
from pathlib import Path
import re,sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def main():
 target=A/'hardware-resource-profile-v1.json';assert not target.exists()
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  p=A/'own-vllm-resource-timeline-model-v4.json'
  assert sha(p)=='a6e8d8c6f16482b0db8a29012017659746f0f877a7c25c41f4616df5be5bbee2'
  v=json.loads(p.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] and v['fresh_faults']==[] and v['driver_dump_exists'] is False
  sys.path.insert(0,str(A/'own-vllm-resource-timeline-v3'));import controller as c;import timeline as t
  absence=c.absence(v['commands']);coll=v['collection'];rt=coll['resource_timeline'];q=rt['source_identified_QSA'];sample=rt['individual_samples'];assert sha(sample['path'])==sample['sha256']
  table=json.loads(Path(sample['path']).read_text());assert table['columns']==['raw_line','name_index','thread','device','append_ns','submit_ns','start_ns','end_ns']
  names=table['names'];events=[];counts=Counter();bytes_by_direction=Counter();service=Counter()
  for row in table['rows']:
   assert len(row)==8 and all(type(x)is int and x>=0 for x in row)
   line,idx,thread,device,a,s,b,e=row;assert 0<a<=s<=b<=e and idx<len(names)
   name=names[idx];m=t.COPY.fullmatch(name)
   kind='copy' if m else ('compute' if t.SHAPE.search(name) else 'other');counts[kind]+=1
   ev=dict(line=line,name=name,thread=thread,device=device,append=a,submit=s,start=b,end=e,kind=kind);events.append(ev)
   if m:bytes_by_direction[m[1]+'2'+m[2]]+=int(m[3]);service[m[1]+'2'+m[2]]+=e-b
  assert dict(counts)==rt['counts'] and dict(bytes_by_direction)==rt['requested_copy_payload_bytes_by_allocation_type']
  checks=t.reconcile(events,coll);assert checks==rt['reconciliation']
  windows={}
  for label,window in [('whole_native_command_envelope',rt['whole_native_command_envelope']),('long_attention_envelope',q['attention_envelope'])]:
   a,b=window['window_ns'];computed=t.measure(events,[a,b],window['label']);assert computed==window
   payload,elapsed,rows=Counter(),Counter(),Counter();crossing=[]
   for ev in events:
    m=t.COPY.fullmatch(ev['name'])
    if not m or ev['end']<=a or ev['start']>=b:continue
    if ev['start']<a or ev['end']>b:crossing.append(ev['line']);continue
    direction=m[1]+'2'+m[2];payload[direction]+=int(m[3]);elapsed[direction]+=ev['end']-ev['start'];rows[direction]+=1
   windows[label]=dict(measurement=window,percent_of_window={k:100*window[k]/window['duration_ns'] for k in ['compute_union_ns','copy_union_ns','compute_copy_intersection_ns','no_observed_command_ns']},fully_contained_copy=dict(counts=rows,requested_payload_bytes=payload,command_service_ns=elapsed,requested_bytes_per_service_GBps={k:payload[k]/elapsed[k] for k in payload}),boundary_crossing_copy_rows_not_prorated=crossing)
  cpu={}
  for label,key in [('wait','cpu_wait_lines'),('host','cpu_host_lines')]:
   line,=v['model_evidence'][key];cpu[label]={k:(float(val) if '.' in val else int(val)) for k,val in re.findall(r'([a-z_][a-z_0-9]*)=([0-9]+(?:\.[0-9]+)?)',line)}
  result=dict(schema='strata-hardware-resource-profile-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis_source=pin(__file__),invocation=sys.orig_argv,GPU_executed=False,
   capture=pin(p),source_commit=v['source_commit'],binary_sha256=v['binary_sha256'],collector_library_sha256=v['collector_library_sha256'],native_environment=v['native_environment'],native_identity=v['actual_native']['native'],
   captured_case=dict(input_tokens=39262,batched_tokens=39261,chunks=[8192,8192,8192,8192,6493],thinking=True,output_tokens=24,residual='BF16 GPU all rows',backend='vllm',prompt_softmax_subgroup=0,QSA_native16=0),
   all_owned_IDs_absent=absence,faults=[],driver_dump_exists=False,request_journal_verified=v['request_journal_verification'],
   sample_pin=sample,counts=counts,per_name_count_sum_min_max_reconciliation=checks,windows=windows,
   long_QSA_service_sum_ns=q['long_QSA_service_sum_ns'],long_QSA_samples=q['samples'][12:],full_prefill_boundaries_proved=False,
   cpu=cpu,cpu_original_lines=v['model_evidence']['cpu_wait_lines']+v['model_evidence']['cpu_host_lines'],coarse_original_lines=v['model_evidence']['coarse_profile_lines'][-1:],
   requested_copy_payload_bytes=bytes_by_direction,copy_command_service_ns=service,copy_requested_bytes_per_service_GBps={k:bytes_by_direction[k]/service[k] for k in service},
   resource_verdicts=dict(
    PCIe='Host/device copy commands run near prior matched-stack bulk-copy throughput; in the long attention envelope ALL copy union is9.11% and overlap about8microseconds. Transfer matters but this capture does not support PCIe-only dominance. This is an envelope, not full-prefill critical path or physical wire counters.',
    GPU_compute='Compute-command coverage77.31% of long attention envelope, QSA60launch service33.963s. This identifies device compute commands as priority but DOES NOT determine ALU/XMX saturation versus VRAM/cache/SLM/barrier/dependency limits inside kernels.',
    CPU='Main thread98.420CPU s includes69.696CPU s in explicit GPU waits and28.723CPU s outside waits. Host grouping0.106CPU s; GEMM submission123192calls5.643CPU s; upload submission5.753CPU s; source copy4.846CPU s; unclassified12.329CPU s. Polling alone is not CPU latency evidence; launch/packing and gap causality remain candidates.',
    RAM='Host source-copy CPU time measured, physical RAM traffic/bandwidth not measured in this run. Historical36.87GB/s six-core streaming read is operation/stack scoped reference, not current utilization or random-access capacity.',
    VRAM='Whole-command D2D requested payload76.948GB; payload/service about163GB/s (logical read+write about326GB/s). Copy throughput is not kernel DRAM traffic. No per-kernel cache/VRAM bandwidth or occupancy collected.',
    scheduling='Copies and compute almost entirely serialize in the observed safe route. Overlap or fewer host launches may remove exposed time, but9%copy coverage cannot alone establish a600/1000tok/s path. Gaps12.893s of long attention envelope are uncovered commands, not proved hardware idle or CPU causality.'),
   next_experiments=['Actual non-root Level Zero metric-group enumeration, then qualify counters only if available; no policy changes inferred.','Use supported per-kernel counters or controlled component variations to distinguish VRAM/cache, XMX/ALU, SLM/barrier and launch gaps.','Keep native16 QSA alternatives and restored PR6 source behind numerical/lifetime/fault gates; no unsafe old route execution.'],
   no_performance_adoption=True,clean_speed_reference=dict(actual_input_tokens=39262,
    baseline_source='6354f0b4ff3731466795410e11ca9ef785c9d326',baseline_binary='85643f3926bb9d212945659cb7ddb264e101ca9b2d5fa8f19430e945140ad191',
    baseline_samples_tok_s=[402.6,403.0,403.4],baseline_median_tok_s=403.0,
    current_source='63326f98c36656cf3a3e1adcc54c3c25a02aa6ee',current_accepted_OFF_clean_samples_tok_s=[403.4],
    scope='Prior BF16/vLLM clean reference is a different compiled source. Current source OFF clean cohort remains incomplete after administrative source-pin failure; no adoption. Instrumented capture is not a clean speed comparison.'),
   limitations=['No fullprefill clock bridge','No direct physical bus/RAM/VRAM traffic','No PMU occupancy/cache/XMX/stall counters','No causal identification of command gaps','No new600milestone','No general-quality/full262144/RESTORE qualification'],
   retention=dict(owner='Main',canonical_samples_consumer='Main next GPU-internal counters/component comparison and host-submission-gap diagnosis',sample_byte_budget=64<<20,review_deadline_utc='2026-10-17T03:00:00Z',raw_retirement='Main may retire successful raw timeline/protocol after these exact sample/table/hash checks under measurement lock; preserve original receipts and manifest.'))
  assert sample['bytes']<=result['retention']['sample_byte_budget']
  with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  print(json.dumps(dict(receipt=str(target),sha256=sha(target),events=len(events),verified_names=len(checks['rows']),normal_native_exit=True)))
if __name__=='__main__':main()
