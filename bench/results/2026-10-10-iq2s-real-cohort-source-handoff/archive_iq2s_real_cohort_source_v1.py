from pathlib import Path
import fcntl,hashlib,json,shutil,subprocess,datetime
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');R=B/'research-20261009';W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010');A=W/'bench/results/2026-10-10-iq2s-real-cohort-source-handoff'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={'sycl/tools/native-service-calibration/CMakeLists.txt':'b00f32668d59bea9d7e0380393046f44d1ad1616577f9c8611934f9d4d54a203','sycl/tools/native-service-calibration/native_service_calibration.cpp':'fda9a4972ef08d0350d5857ceedd8392fa2857a6906dc432bc8621342f469354','sycl/tools/native-service-calibration/README.md':'4c05f432e0b86180532ef1e0954b448d2425de2842e30d563517a3e78f5ed8d0','sycl/tools/native-service-calibration/iq2s_index_dot.hpp':'40497188e7079ad87c2f831b7556d5f1ce51d45b4582aec03583d5ee4f178720','sycl/tools/native-service-calibration/iq2s_index_dot.cpp':'bb20df83d7f76429f7c65dd4f03868a2cc8e161680092e49aa31cf2e6cf93bfd'}
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p,h in files.items():assert sha(W/p)==h,p
 source=W/'sycl/tools/native-iq2s-nt1-index-spread/iq2s_index_spread.cpp';assert sha(source)=='c51f580bc72cb303e7fe30b1cfb6f7c09ad82332ffd60d441af0eac2dbe77b89'
 a=source.read_text();b=(W/'sycl/tools/native-service-calibration/iq2s_index_dot.cpp').read_text();left=a.index('/*\nMIT License');right=a.index('\n// Independent',a.index('float index_candidate')) if '\n// Independent' in a[a.index('float index_candidate'):] else -1
 # Compare full copied region ending at the balanced candidate closing brace.
 start=a.index('__attribute__((noinline)) float index_candidate');opening=a.index('{',start);depth=0;end=None
 for i in range(opening,len(a)):
  if a[i]=='{':depth+=1
  if a[i]=='}':
   depth-=1
   if not depth:end=i+1;break
 assert end and a[left:end] in b,'exact helper/table/dot copy'
 assert not A.exists();A.mkdir(parents=True)
 p=R/'implementation-iq2s-real-cohort-correctness-v1.txt';assert sha(p)=='b68dd429906afad4df4551a23df6928d31cdfd7ac07fd4d1af02cc583efd1550';shutil.copyfile(p,A/p.name)
 review={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_commit':'89c2d37d529475c07416392f9c2ec6aaa266d2b1','source_only':True,'root_reviewed_complete_diff_and_new_files':True,'root_verified_exact_original_helper_table_and_two_dot_copy':True,'source_pins':files,'default_CMake_option':'OFF','scope':'384 frozen actual expert IDs, three IQ2S Gate/Up NT1 dot arms using production quantized synthetic layer inputs; no pool/timing, all IEEE bits compared','root_findings':'Production/default modes preserved in source; mode rejects before payload on OFF or unsupported cells, reuses bounded reader, branch precedes all full FFN/pool/schedule/timing work. Source review is not compile/runtime proof. Actual role content SHA1152 closure, emitted custom quantizer/order, Release/sanitizer/negative/no-GPU/default gates remain pending.','adopted':False,'actual_weights_read_in_this_handoff':False,'tests_run':False,'performance_or_model_lifecycle_qualified':False}
 (A/'root-source-review.json').write_text(json.dumps(review,indent=2)+'\n')
 (A/'REPORT.md').write_text('''# IQ2S actual-cohort comparison source handoff

Separate gpt-6.1-sol supplied an OFF-by-default correctness-only mode for the existing bounded actual-cohort fixture. Root fully reviewed all five changed/new files and verified the helper/table/direct/register dot region is byte-identical to the previously qualified standalone register-return source.

This compares the trait baseline, direct control and index candidate on384 frozen expert IDs,640 Gate/Up row pairs each, identical production Q8_K prepared inputs and the existing scalar finish. All IEEE bits enter equality with separate nonfinite classes. Reader/format/extents/frozen ordering are reused; no new parser, model selection, tensor capture, pool run or timing is introduced. Decode Gate/Up width is2560; Down20 width640 is not an index target.

This commit records source only. Release/sanitizer compilation and actual-weight correctness, source/1152-role-content closure, linked instruction/custom quantizer FP audit, OFF/negative/no-GPU/default-path validation are pending under the root-owned lock. No performance, live activation,32K model/full262144 lifecycle or adoption follows from this source review. Original implementation report remains source-only and unmodified.
''')
 shutil.copyfile(__file__,A/Path(__file__).name);p=A/'archive-file-identities.json';p.write_text(json.dumps({str(f.relative_to(A)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(A.rglob('*')) if f.is_file() and f!=p},indent=2)+'\n')
 subprocess.run(['git','-C',str(W),'diff','--check'],check=True)
 subprocess.run(['git','-C',str(W),'add',*files,str(A.relative_to(W))],check=True)
 subprocess.run(['git','-C',str(W),'commit','-m','test(sycl): add actual-cohort IQ2S three-arm correctness mode'],check=True)
 print(subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip())
