#!/usr/bin/env python3
"""Full cross-process byte comparison and resource sensitivity summary, no GPU."""
import fcntl,hashlib,importlib.util,json,statistics,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2];B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 root=B/'main-qsa-resource-analysis-v1';target=A/'own-qsa-resource-comparison-v1.json';assert not root.exists() and not target.exists()
 first_path=A/'own-qsa-resource-first-v1.json';clean_path=A/'own-qsa-resource-clean-v1.json';smoke_path=A/'own-qsa-resource-smoke-v1.json'
 first=json.loads(first_path.read_text());clean=json.loads(clean_path.read_text());smoke=json.loads(smoke_path.read_text())
 for j in [first,clean,smoke]:assert j['passed'] and j['complete'] and j['ownership_closed'] and not j['fresh_faults'] and not j['driver_dump_entries']
 frames=[c['frame'] for j in [first,clean] for c in j['cases']]
 for v in frames:assert sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
 ownerpath=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py';assert sha(ownerpath)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 spec=importlib.util.spec_from_file_location('qsa_resource_analysis_owner',ownerpath);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 comparator=A/'own-qsa-resource-control-v1/compare_frames.py'
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
  rec=dict(schema='strata-qsa-resource-comparison-v1',passed=False,complete=False,active=False,GPU_executed=False,commands=[],
   references={str(p):sha(p) for p in [smoke_path,first_path,clean_path,comparator]},binary=clean['binary'],
   environment=clean['native_environment'],scope=clean['scope'],arms={},frames=frames,
   independent_processes_per_arm=3,timed_repetitions_per_process=10,not_model_performance=True)
  owner=None
  def persist():
   rec['active']=bool(owner and owner.active is not None);(root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);rec['commands']=owner.commands
  try:
   c,out,err=owner.run('compare-all-frames',['/usr/bin/python3',str(comparator),*[v['path'] for v in frames]],dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1'),P,wall=60,cpu=120,rss_cap=512<<20,text_cap=8<<20,file_cap=64<<20,total_cap=64<<20)
   assert m.closed(c) and c['exit_code']==0;comparison=json.loads(out.read_text());assert comparison['status']=='PASS' and len(comparison['frames'])==10;rec['full_frame_comparison']=comparison
   metrics=['whole_wall_ms','enqueue_wall_ms','wait_wall_ms','whole_thread_cpu_ms','enqueue_thread_cpu_ms','wait_thread_cpu_ms']
   for arm in ['A','B','C']:
    rows=[c for c in clean['cases'] if c['spec']['name'].startswith(arm)];assert len(rows)==3
    values={key:[statistics.median(s[key] for s in row['samples']) for row in rows] for key in metrics}
    rec['arms'][arm]=dict(reuse=rows[0]['spec']['reuse'],grain=rows[0]['spec']['grain'],process_medians=values,median_of_process_medians={key:statistics.median(v) for key,v in values.items()},samples=[row['samples'] for row in rows])
   am=rec['arms']['A']['median_of_process_medians'];bm=rec['arms']['B']['median_of_process_medians'];cm=rec['arms']['C']['median_of_process_medians']
   rec['sensitivity']=dict(replicated_over_shared=bm['whole_wall_ms']/am['whole_wall_ms'],replicated_change_percent=100*(bm['whole_wall_ms']/am['whole_wall_ms']-1),grain8_over_whole=cm['whole_wall_ms']/am['whole_wall_ms'],grain8_extra_whole_wall_ms=cm['whole_wall_ms']-am['whole_wall_ms'],grain8_extra_enqueue_wall_ms=cm['enqueue_wall_ms']-am['enqueue_wall_ms'],grain8_extra_wait_wall_ms=cm['wait_wall_ms']-am['wait_wall_ms'])
   rec['source_proven_extents']=dict(nq=128,KV='K8V8',Q_O='FP32',cap=2051,head_dim=256,query_heads=24,KV_heads=2,shared_selected_payload_bytes=2165856,replicated_selected_payload_bytes=277229568,all_device_allocations_bytes=294681024,workgroups_total=256,whole_calls=1,whole_workgroups_per_call=256,grain8_calls=16,grain8_workgroups_per_call=16)
   rec['decisions']=['Large indexed-address working-set expansion128x gives only~2%timeincrease in this fixed synthetic kernel; does not directly measure physicalVRAM/cache traffic or exclude all memory bottlenecks.',
    'Same256 workgroups split16calls is~2.63x slower. Extra hostenqueuewall is a small fraction of extrawholewall; increasedtime mostly wait. Device parallelism/grid/command-boundary effects are a priority over host math or PCIe for this component.',
    'Compute-versus-SLM/barrier/register/cache activity remains unresolved; no arithmetic saturation/occupancy utilization claim.',
    'Current model already uses muchlargerQSA batches, so this is not a demonstrated model speedup or proof that further biggerQSA batches help.',
    'Keep production kernels/precision/layout unchanged; this is diagnostic source only.']
   rec.update(passed=True,complete=True,all_full_frames_equal=True,all_finite=True,new_GPU_faults_in_native_receipts=0,model_quality_qualified=False,physical262144_qualified=False,RESTORE_qualified=False,performance_adopted=False)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   rec['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands);persist();target.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(target),passed=rec['passed'],complete=rec['complete'],sensitivity=rec.get('sensitivity'),error=rec.get('error'))))
  return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
