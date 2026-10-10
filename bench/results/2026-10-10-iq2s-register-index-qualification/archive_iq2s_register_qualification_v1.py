from pathlib import Path
import datetime,fcntl,hashlib,json,subprocess,shutil
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010')
A=W/'bench/results/2026-10-10-iq2s-register-index-qualification'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='22064c0b26232b0964afd70e7d9fe47535b39224'
  assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=W,text=True).strip()
  out=B/'iq2s-register-index-cpu-validation-v2';r=json.loads((out/'record.json').read_text());assert r['complete'] and r['passed'] and not r['active'] and not r['cleanup'] and not r['survivors'] and r['exit_pin_gate_passed']
  failed=B/'iq2s-register-index-cpu-validation-v1';f=json.loads((failed/'record.json').read_text());assert not f['active'] and not f['passed'] and not f['complete'] and not f['survivors']
  assert sha(out/'record.json')=='be9fbe4246e0e420584e5baf98240bbbd4170efa19e299aadb2dcb1cefb6e548'
  assert sha(failed/'record.json')=='60cdb36ed8979a3744a83b27f46a0133bcc1191822eae31d43372ac24c225365'
  for name,h in r['source_pins'].items():assert sha(W/'sycl/tools/native-iq2s-nt1-index-spread'/name)==h
  binary=Path(r['variants']['release']['binary']);assert sha(binary)==r['variants']['release']['binary_sha256']
  assert not A.exists();A.mkdir()
  capture={'binary':str(binary),'binary_sha256':sha(binary),'scope':'offline complete callers only; no executable/model/GPU run','commands':[]}
  for label,symbol in [('rows','isolated_iq2s::rows()'),('main','main')]:
   argv=['/usr/bin/objdump','-d','-C','--disassemble='+symbol,str(binary)]
   v=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
   assert v.returncode==0 and not v.stderr and 1000<len(v.stdout)<512<<10
   p=A/(label+'.asm');p.write_bytes(v.stdout);capture['commands'].append({'argv':argv,'exit_code':v.returncode,'stdout_sha256':sha(p),'stdout_bytes':p.stat().st_size,'stderr_bytes':0})
  assert sha(binary)==capture['binary_sha256']
  (A/'additional-linked-callers-capture.json').write_text(json.dumps(capture,indent=2)+'\n')
  shutil.copyfile(out/'record.json',A/'passed-v2-record.json');shutil.copyfile(failed/'record.json',A/'failed-v1-record.json')
  for p in [B/'run_iq2s_register_index_cpu_v1.py',B/'run_iq2s_register_index_cpu_v2.py',Path(__file__)]:shutil.copyfile(p,A/p.name)
  raw={}
  for directory in [failed,out]:
   raw[directory.name]={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(directory.iterdir()) if p.is_file()}
  (A/'raw-file-identities.json').write_text(json.dumps(raw,indent=2)+'\n')
  names=['compiler.stdout','release-configure.stdout','asan-ubsan-configure.stdout','release-target-commands.stdout','asan-ubsan-target-commands.stdout','release-imports.stdout','asan-ubsan-imports.stdout','release-fixtures.stdout','asan-ubsan-fixtures.stdout','release-compile_commands.json','asan-ubsan-compile_commands.json','release-CMakeCache.txt','asan-ubsan-CMakeCache.txt']
  for name in names:shutil.copyfile(out/name,A/name)
  for label,name in [('control','control.asm'),('candidate','candidate.asm'),('trait','trait.asm'),('reference-quantizer','reference-quantizer.asm')]:shutil.copyfile(out/('linked-'+label+'.stdout'),A/name)
  symbols=(out/'linked-symbols.stdout').read_text().splitlines();selected=[x for x in symbols if any(s in x for s in ['isolated_iq2s::direct_control','isolated_iq2s::index_candidate','isolated_iq2s::rows','ggml_vec_dot_iq2_s_q8_K','quantize_row_q8_K',' main'])]
  (A/'selected-symbols.txt').write_text('\n'.join(selected)+'\n')
  excerpts=[]
  for directory in [failed,out]:
   for p in sorted(directory.iterdir()):
    if p.name.endswith('-build.stdout') or p.name.endswith('-build.stderr'):
     lines=p.read_text(errors='replace').splitlines()
     excerpts.append({'source':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'line_count':len(lines),'warning_line_count':sum('warning:' in x for x in lines),'head':lines[:4],'tail':lines[-5:]})
  (A/'build-output-excerpts.json').write_text(json.dumps(excerpts,indent=2)+'\n')
  decision={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':r['source_head'],'passed_receipt_sha256':sha(out/'record.json'),'failed_controller_receipt_preserved_sha256':sha(failed/'record.json'),'original_failure':'Exited /proc entry raised ProcessLookupError during sanitizer build; controller interrupted two known owned processes. No survivors. Release fixture passed but whole v1 gate remains failed/incomplete.','fix':'Fresh v2 tolerates only missing/exited process entries; no fixture or correctness threshold change.','correctness':'Fresh release and IntelLLVM ASan/UBSan,101lines each,96profiles/1536GU pairs/9216dots, all96profile rows identical to earlier qualified source. Mixed4194304active checks/1048576vectors/16alignments, uniform262144/all1024indices; MXCSR controls unchanged.','root_full_new_control_candidate_instructions_reviewed':True,'codegen':{'old_candidate_frame_bytes':184,'new_candidate_frame_bytes':32,'direct_control_frame_bytes':40,'new_candidate_four_64value_inner_iterations':True,'old_materialized_index_roundtrips_absent':True,'old_extra_YMM_accumulator_spills_absent':True,'eight_literal_PEXTRW_per_inner_iteration':True,'intentional_32byte_sign_scratch_retained':True,'new_two_HiSpread_loads_per_inner_iteration':True,'scalar_grid_reads_and_sign_scale_madd_accum_blockFMA_hsum_postscale_preserved':True,'hot_helpers_or_gathers':False,'speed_measured':False},'mechanism_gate':'Root accepts advancing register-only candidate to actual-weight/measurement qualification; independent R103 emitted-code review pending separately.','model_opened':False,'live_activations':False,'GPU_executed':False,'performance_adopted':False,'production_default_changed':False,'remaining_gates':['actual-weight frozen train/holdout cohort and all GU row bits/nonfinite counts','live quantizer/path/worker FP domain','repeated real workload >=32768 prefill and distinct repeated decode phases','physical262144 context lifecycle before production adoption']}
  (A/'root-qualification-and-codegen-review.json').write_text(json.dumps(decision,indent=2)+'\n')
  (A/'README.md').write_text('The register-return IQ2_S candidate passed fresh Release and IntelLLVM ASan/UBSan checks:96 profiles,1536 Gate/Up pairs and9216 dot calls per run, with every profile row identical to the previous source. Uniform/mixed/alignment checks and MXCSR controls also passed. Both processes exited normally and no owned cleanup was needed.\n\nThe exact linked candidate now uses fixed PEXTRW extracts and the four-iteration inner loop. Its stack frame is32 bytes (intentional sign scratch), compared with184 bytes in the earlier candidate and40 bytes in direct control. The old index-array store/reload and extra accumulator spill traffic are absent. The new dependent2KiB HiSpread table loads remain. This is emitted-code evidence, not a speed measurement. Actual weights, live inputs, model>=32K phase timings and full physical context remain unqualified, and production defaults are unchanged.\n\nThe first new-source gate was interrupted by a process-exit race in the supervisor during sanitizer compilation. Its original failed/incomplete receipt and known owned cleanup are preserved. Only the process observer changed for the successful fresh second run; fixtures and numerical comparisons did not change. Successful verbose compiler logs are represented by hashes, recipe/flags and bounded excerpts.\n')
  ids={str(p.relative_to(A)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(A.iterdir()) if p.is_file()}
  (A/'archive-file-identities.json').write_text(json.dumps(ids,indent=2)+'\n')
  print(json.dumps({'archive':str(A),'files':len(ids),'bytes':sum(x['bytes'] for x in ids.values()),'passed':True,'timed':False}))
if __name__=='__main__':main()
