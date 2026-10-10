"""Root additional ASAN+UBSAN and fresh default-OFF CPU admissions."""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, importlib.util, json, time, types

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010')
RELEASE=B/'run_iq2s_actual_cohort_correctness_v1.py'
ADMISSION=B/'iq2s-actual-cohort-correctness-v1/record.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('variant',choices=('asan','default'));variant=ap.parse_args().variant
 with (B/'owned-v0141-measurement.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert sha(ADMISSION)=='a5e2ac74cda487aeb5e6ca4053041cb1a5f3ea8f06951b8083d51b47e622fcea'
  admission=json.loads(ADMISSION.read_text());assert admission['passed'] and admission['complete'] and not admission['active'] and not admission['cleanup'] and not admission['survivors']
  assert sha(RELEASE)==admission['controller_sha256']
  spec=importlib.util.spec_from_file_location('root_release_admission',RELEASE);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
  base=json.loads(r.BUILD.read_text());r.check_build(base)
  receipt=B/('iq2s-actual-cohort-'+variant+'-build-v1/record.json');build=json.loads(receipt.read_text());build_pin=sha(receipt)
  assert build['passed'] and build['complete'] and not build['active'] and not build['cleanup'] and not build['survivors']
  assert build['base_build_receipt_sha256']==sha(r.BUILD)
  assert not build['gpu_work_submitted'] and not build['model_opened'] and not build['inference_run']
  assert sha(build['binary']['path'])==build['binary']['sha256']
  assert sha(build['compile_commands']['path'])==build['compile_commands']['sha256']
  helper_text=r.HELPER.read_text();assert sha(r.HELPER)==r.PINS[r.HELPER]
  # ASAN reserves shadow virtual address space; retain bounded polled physical RSS.
  if variant=='asan':
   old='(resource.RLIMIT_AS, (96 << 30, 96 << 30))';new='(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))'
   assert helper_text.count(old)==1;helper_text=helper_text.replace(old,new)
   old="assert rss <= 1536 << 20, 'owned RSS limit'";new="assert rss <= 3072 << 20, 'owned RSS limit'"
   assert helper_text.count(old)==1;helper_text=helper_text.replace(old,new)
  h=types.ModuleType('root_variant_child');h.__file__=str(r.HELPER);exec(compile(helper_text,str(r.HELPER),'exec'),h.__dict__);h.W=W
  out=B/('iq2s-actual-cohort-'+variant+'-correctness-v1');out.mkdir(mode=0o700)
  rec=dict(active=True,complete=False,passed=False,variant=variant,controller_sha256=sha(__file__),release_admission_sha256=sha(ADMISSION),
           release_controller_sha256=sha(RELEASE),helper_original_sha256=r.PINS[r.HELPER],helper_effective_sha256=hashlib.sha256(helper_text.encode()).hexdigest(),
           helper_changes=('ASAN child virtual shadow reservation admitted; polled group RSS 3GiB' if variant=='asan' else 'none'),
           build_receipt_sha256=build_pin,binary=build['binary'],commands=[],cleanup=[],survivors=[],gpu_work_submitted=False,inference_run=False,
           performance_eligible=False,adopted=False,full_lifecycle_passed=False,scope='Standalone actual-weight CPU numerical/default admission, no model timing',
           limits=dict(AS_each_bytes=('unlimited_ASAN_shadow' if variant=='asan' else 96<<30),RSS_group_poll_bytes=(3072 if variant=='asan' else 1536)<<20,
                       CPU_soft_seconds=900,CPU_hard_seconds=901,FSIZE_each_bytes=4<<20,NOFILE=256,CORE=0,text_budget_bytes=8<<20),
           started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
  start=time.monotonic();h.save(rec,out)
  try:
   prior=json.loads(r.PRIOR.read_text());extents=prior['selected_extent_sha256'];manifest=json.loads((r.OWNER/'manifest.json').read_text())
   models=[h.identity(x['path']) for x in prior['model_identity']];assert models==prior['model_identity'];rec['model_identity']=models
   rec['selected_extents_before']=r.extent_hashes(extents)
   env=dict(r.ENV)
   if variant=='asan':env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
   def argv(mode,tasks=0,batch=1,missing=False):
    return [build['binary']['path'],str(out/'missing-pack') if missing else str(r.PACK),str(out/'missing-primary') if missing else str(r.PRIMARY),str(r.OWNER/'cohort.tsv'),'22','20','1',str(tasks),str(batch),'2026101001',mode]
   if variant=='default':
    h.run_child(rec,out,'disabled-before-pack-open',argv('--iq2s-index-correctness-only',missing=True),env,expect_error='IQ2S index mode disabled at build time',wall=20)
    rows,stderr=h.run_child(rec,out,'default-original-correctness',argv('--correctness-only',tasks=6,batch=6),env,wall=120)
    h.validate_rows(rows,manifest,6,6,True);rec['default_legacy_complete_pool_and_reference_parity']=True
   else:
    rows,stderr=h.run_child(rec,out,'asan-untraced-correctness',argv('--iq2s-index-correctness-only'),env,wall=120)
    rec['numerical_admission']=r.validate(rows,manifest,extents)
    assert not stderr.strip(),'sanitizer diagnostic stderr'
    release_rows=list(h.csv.reader((B/'iq2s-actual-cohort-correctness-v1/correctness-observed/stdout.csv').read_text().splitlines()))
    tags=('ID','EXTENT','INDEX_ID','INDEX_SPLIT','INDEX_COMPLETE','RESULT')
    assert [x for x in rows if x and x[0] in tags]==[x for x in release_rows if x and x[0] in tags],'release/sanitizer numerical records differ'
    rec['release_sanitizer_records_identical']=True
   rec['selected_extents_after']=r.extent_hashes(extents);assert rec['selected_extents_after']==rec['selected_extents_before']
   assert all(h.identity(x['path'])==x for x in models)
   r.check_build(base);assert sha(receipt)==build_pin and sha(build['binary']['path'])==build['binary']['sha256']
   assert sha(__file__)==rec['controller_sha256'];rec.update(complete=True,passed=True)
  except BaseException as e:rec['error']=type(e).__name__+': '+str(e)
  finally:
   try:
    r.check_build(base);assert sha(receipt)==build_pin and sha(__file__)==rec['controller_sha256']
    assert sha(build['binary']['path'])==build['binary']['sha256']
    if 'model_identity' in rec:assert all(h.identity(x['path'])==x for x in rec['model_identity'])
    rec['exit_source_pin_gate_passed']=True
   except BaseException as e:rec['passed']=False;rec['exit_source_error']=type(e).__name__+': '+str(e)
   rec.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());h.save(rec,out)
  print(json.dumps(dict(record=str(out/'record.json'),sha256=sha(out/'record.json'),passed=rec['passed'],error=rec.get('error'),elapsed_seconds=rec['elapsed_seconds'])))
  return 0 if rec['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
