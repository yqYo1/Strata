"""Main: preserve all six quiet samples and assess prefill/decode separately."""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H=B/'xe-compact-moe-quiet64k-phase-controller-source-v1'
Q=B/'xe-compact-moe-quiet64k-main-admission-root-v1/qualification.resolved.json'
SEQUENCE=((1,6144),(1,8192),(2,8192),(2,6144),(3,6144),(3,8192))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise RuntimeError(message)
def stats(values):
 require(len(values)==3 and all(math.isfinite(x) and x>0 for x in values),'three finite positive samples')
 return dict(samples=values,mean=statistics.mean(values),median=statistics.median(values),sample_sd=statistics.stdev(values),cv_percent=100*statistics.stdev(values)/statistics.mean(values),minimum=min(values),maximum=max(values))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-compact-moe-quiet64k-analysis-root-v1'
 require(not out.exists(),'new immutable analysis')
 require(sha(Q)=='f67cee5cd040aa03076ef8dc4dd48f9d76c6670cba8753f3f5e53316e61c9c09','actual quiet qualification')
 require(sha(H/'run_root.py')=='04f425d650bc5a43320b1d228d95015a54cede0d20400adfdc6c110db80b1900','actual controller')
 q=json.loads(Q.read_text());pins={};samples=[];by_case={}
 require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='cfda0b6a-7c13-4fe2-bdf5-8f9202770840','same boot')
 for sample,chunk in SEQUENCE:
  p=B/f'xe-compact-moe-quiet64k-root-v1/sample-{sample:03d}-n65536-t{chunk}-g16-on/record.json'
  v=json.loads(p.read_text());pins[str(p)]=sha(p)
  require(v['passed'] is True and v['complete'] is True and v['active'] is False,'actual closed PASS')
  require(v['model_normal_close'] and v['model_close']['normal'] and not v['model_close']['forced'] and v['model_close']['exit_code']==0,'normal model exit')
  require(v['kernel_gate_passed'] and v['boot_unchanged'],'fault and boot checks')
  require(v['input_tokens']==65536 and v['requested_chunk']==chunk and v['sample_index']==sample and v['compact']=='on' and v['group']==16,'actual settings')
  require(v['controller_sha256']==sha(H/'run_root.py') and v['qualification_sha256']==sha(Q),'actual source qualification')
  require(v['source_pins']==q['production_source_pins'] and v['binary']==q['free_release_binary'],'same source binary')
  require(v['reference_comparison']['strict_continuation_passed'] and v['validation']['generated_tokens']==16,'same-chunk exact16 and LP5')
  require(v['effective_settings']['context']=='262144' and v['effective_settings']['kv']=='int8','fixed full capacity settings')
  for observer in ('STRATA_PREFILL_TIMING','STRATA_TRACE','STRATA_PREFILL_TRANSFER_COUNTS','ZE_ENABLE_VALIDATION_LAYER','ZE_ENABLE_PARAMETER_VALIDATION','ZEL_ENABLE_LOADER_LOGGING','ZEL_LOADER_LOG_CONSOLE','ZEL_LOADER_LOGGING_LEVEL','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):
   require(observer not in v['environment'],'quiet model environment')
  for c in [*v['commands'],*v['guard_commands']]:
   require(c.get('session_anchored') is True and c.get('exit_code')==0 and not c.get('errors') and not c.get('error') and not c.get('cleanup'),'normally closed owned commands')
  valid=v['validation'];pp=valid['prompt_ms'];tg=valid['decode_ms'];require(pp>0 and tg>0,'positive actual times')
  item=dict(sample=sample,chunk=chunk,pair_order=v['pair_order'],record_path=str(p),record_sha256=pins[str(p)],prompt_ms=pp,cold_decode_ms=tg,prefill_tokens_per_second=65536*1000/pp,cold_decode_tokens_per_second=16*1000/tg,generated_tokens=16)
  samples.append(item);by_case[(sample,chunk)]=v
 # Protocol endpoint identity and LP5 including every token checked across all six.
 anchor=by_case[SEQUENCE[0]]
 def exact_lines(v):return [s for s in v['request_protocol'] if s.startswith(('T ','LP '))]
 require(len(exact_lines(anchor))==32 and all(exact_lines(v)==exact_lines(anchor) for v in by_case.values()),'all six exact16 IDs/LP5')
 require(all(v['input']==anchor['input'] and v['model_inputs']==anchor['model_inputs'] and v['environment']==anchor['environment'] for v in by_case.values()),'same input/model/environment')
 summary={str(chunk):dict(prefill=stats([s['prefill_tokens_per_second'] for s in samples if s['chunk']==chunk]),cold_decode=stats([s['cold_decode_tokens_per_second'] for s in samples if s['chunk']==chunk])) for chunk in (6144,8192)}
 paired=[]
 for index in (1,2,3):
  a=next(x for x in samples if x['sample']==index and x['chunk']==6144);b=next(x for x in samples if x['sample']==index and x['chunk']==8192)
  paired.append(dict(sample=index,order=a['pair_order'],prefill_throughput_change_percent=100*(b['prefill_tokens_per_second']/a['prefill_tokens_per_second']-1),prefill_time_change_percent=100*(b['prompt_ms']/a['prompt_ms']-1),cold_decode_throughput_change_percent=100*(b['cold_decode_tokens_per_second']/a['cold_decode_tokens_per_second']-1),cold_decode_time_change_percent=100*(b['cold_decode_ms']/a['cold_decode_ms']-1)))
 # No automatic default change or universal quality/performance extrapolation.
 result=dict(passed=True,complete=True,active=False,recipe_sha256=sha(__file__),receipt_pins=pins,qualification_sha256=sha(Q),source_revision='39e395a103af36323262cdc0dbaea27cba81f3e4',measurement_order=[list(x) for x in SEQUENCE],samples=samples,summary=summary,paired_changes=paired,all_six_exact16_ID_LP5=True,default_adopted=False,compact_default='off',prefill_decode_separate=True,broad_quality_qualified=False,full262144_qualified=False,restore_qualified=False,limitations=['Three fresh balanced pairs estimate variability; variation is not eliminated.','Cold decode includes first-use work and only sixteen tokens; this does not establish warm/stationary decode throughput.','Health checks use diagnostic logging outside measured model. Quiet model timing excludes explicit trace/validation/timer keys; instrumentation total overhead is not calibrated.','Same64K synthetic coding-review fixture and fixed262144 capacity; neither full physical context nor broad model capability is validated.','Only requested chunk width differs; full memory peak and universal hardware ceiling remain unmeasured.'])
 require(all(sha(Path(p))==h for p,h in pins.items()),'original receipts unchanged during analysis')
 out.mkdir();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(passed=True,summary=summary,paired_changes=paired)))
