"""Root serial numerical admission and replicated caller-dot screen."""
from pathlib import Path
import argparse, csv, datetime, fcntl, hashlib, importlib.util, json, math, re, statistics, time, types
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-actual-cohort-timing-20261010')
RELEASE=B/'run_iq2s_actual_cohort_correctness_v1.py'
ADMISSION=B/'iq2s-actual-cohort-correctness-v1/record.json'
SNAP=B/'research-source-snapshots/iq2s-actual-timing-source-v2/manifest.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def orderhash(order):
 h=14695981039346656037
 for n in order:
  for b in n.to_bytes(4,'little'):h=((h^b)*1099511628211)&((1<<64)-1)
 return str(h)
ENV_KEYS={'STRATA_FORCE_ISA','STRATA_FORCE_AVX2','STRATA_NO_AVXVNNI','STRATA_NO_Q8K_AVX2','STRATA_IQ_MT_MIN','STRATA_IQ3S_MT1','STRATA_IQ_PREFETCH','STRATA_NO_IQ512','STRATA_NO_IQ256','STRATA_NO_IQ4NL','STRATA_KQ256','STRATA_IQ256_GATHER','STRATA_Q2_BITPLANE','STRATA_POOL_SPIN_US','STRATA_HOST_CORE','STRATA_NATIVE_DISPATCH_HISTOGRAM'}
def validate_timing(rows,seed):
 assert rows[-2:]==[['INDEX_TIMING_COMPLETE','216','36','18','3','2','2','3'],['RESULT','iq2s_index_timing_complete','not_model_performance_or_adoption']]
 hot=[x for x in rows if x[:1]==['INDEX_HOT_ID']];assert len(hot)==16 and all(len(x)==5 for x in hot)
 ids={int(x[2]):x[3:5]for x in rows if x[:1]==['ID']}
 assert all(x[3:5]==ids[int(x[2])]for x in hot)
 host=next(x[2]for x in rows if x[:2]==['META','index_host_cpu'])
 assert [tuple(map(int,x[1:3]))for x in hot]==[(co,co*192+i)for co in (0,1)for i in range(0,192,24)]
 ws=[x for x in rows if x[:1]==['INDEX_WORKING_SET']];assert len(ws)==4
 expected={}
 for co in (0,1):
  for stratum in ('stream192','hot8'):
   order=list(range(co*192,(co+1)*192,1 if stratum=='stream192' else 24));n=len(order)
   expected[(co,stratum)]=(n,n*640,n*1280,n*1049600,orderhash(order))
   assert ['INDEX_WORKING_SET',str(co),stratum,str(n),str(n*1049600),orderhash(order)]in ws
 samples=[x for x in rows if x[:1]==['INDEX_TIMING']];assert len(samples)==252 and all(len(x)==15 for x in samples)
 seen=set();outputs={};times={};positions={}
 for x in samples:
  co,st,arm,roundno,warm,pos=int(x[1]),x[2],x[3],int(x[4]),int(x[5]),int(x[6])
  assert co in (0,1)and st in ('stream192','hot8')and arm in ('trait','direct','register')and -3<=roundno<18
  assert warm==int(roundno<0)and pos in (0,1,2)
  assert arm==('trait','direct','register')[(pos+roundno+3+seed%3)%3]
  key=(co,st,roundno,arm);assert key not in seen;seen.add(key)
  assert tuple(map(int,x[7:11]))==expected[(co,st)][:4]and x[11]==expected[(co,st)][4]
  outputs.setdefault((co,st),x[12]);assert outputs[(co,st)]==x[12]
  t=float(x[13]);assert math.isfinite(t)and t>0
  # Field14 is caller CPU; CSV has15 columns including tag, checked below.
  assert x[14]==host
  times[key]=t;positions.setdefault((co,st,arm),[]).append((roundno,pos,t))
 summaries={}
 for co in (0,1):
  for st in ('stream192','hot8'):
   v={}
   for a,b in [('register','trait'),('direct','trait'),('register','direct')]:
    ratios=[times[(co,st,i,a)]/times[(co,st,i,b)]for i in range(18)]
    v[a+'/'+b]=dict(median=statistics.median(ratios),minimum=min(ratios),maximum=max(ratios),individual_paired_ratios=ratios)
   v['arm_median_ms']={a:statistics.median(times[(co,st,i,a)]for i in range(18))for a in ('trait','direct','register')}
   v['arm_position_median_ms']={a:{str(p):statistics.median(t for rr,pp,t in positions[(co,st,a)]if rr>=0 and pp==p)for p in range(3)}for a in ('trait','direct','register')}
   summaries[f'{co}/{st}']=v
 return dict(recorded_samples=216,warmups=36,output_hashes={f'{co}/{st}':v for(co,st),v in outputs.items()},process_cell_summaries=summaries,scope='Paired same-round caller-dot screen; process is replication unit, no confidence interval/model/adoption claim')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('correctness','asan','timing'));ap.add_argument('--repeat',type=int,default=1);a=ap.parse_args();assert a.repeat in (1,2,3)
 with (B/'owned-v0141-measurement.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert sha(ADMISSION)=='a5e2ac74cda487aeb5e6ca4053041cb1a5f3ea8f06951b8083d51b47e622fcea'
  priorad=json.loads(ADMISSION.read_text());assert sha(RELEASE)==priorad['controller_sha256']
  spec=importlib.util.spec_from_file_location('root_prior_admission',RELEASE);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
  base=json.loads(r.BUILD.read_text());r.check_build(base)
  assert sha(SNAP)=='bf44394cdbe744514015b756c9560702560e1778ba62868b63732100c5be4c90'
  buildfile=B/('iq2s-timing-'+('asan'if a.mode=='asan'else'release')+'-build-v2/record.json');build=json.loads(buildfile.read_text());bp=sha(buildfile)
  assert build['passed']and build['complete']and not build['active']and not build['cleanup']and not build['survivors']
  def pins():
   r.check_build(base);assert sha(buildfile)==bp and sha(build['binary']['path'])==build['binary']['sha256']
   for rel,x in json.loads(SNAP.read_text()).items():assert sha(W/rel)==x['sha256']
   for rel,x in base['production_source_pins'].items():assert sha(W/rel)==x
  pins()
  text=r.HELPER.read_text();assert sha(r.HELPER)==r.PINS[r.HELPER]
  if a.mode=='asan':
   text=text.replace('(resource.RLIMIT_AS, (96 << 30, 96 << 30))','(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))').replace("assert rss <= 1536 << 20, 'owned RSS limit'","assert rss <= 3072 << 20, 'owned RSS limit'")
  h=types.ModuleType('root_timing_child');h.__file__=str(r.HELPER);exec(compile(text,str(r.HELPER),'exec'),h.__dict__);h.W=W
  out=B/f'iq2s-timing-{a.mode}-r{a.repeat}-v1';out.mkdir(mode=0o700);seed=2026101000+a.repeat
  rec=dict(active=True,complete=False,passed=False,mode=a.mode,seed=seed,commands=[],cleanup=[],survivors=[],controller_sha256=sha(__file__),build_receipt_sha256=bp,binary=build['binary'],source_snapshot_sha256=sha(SNAP),helper_sha256=r.PINS[r.HELPER],helper_effective_sha256=hashlib.sha256(text.encode()).hexdigest(),gpu_work_submitted=False,inference_run=False,performance_eligible=False,adopted=False,full_lifecycle_passed=False,scope='Selected actual weights with synthetic finite inputs; CPU caller-dot comparison only',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),limits=dict(AS_each=('ASAN_shadow_unlimited'if a.mode=='asan'else 96<<30),RSS_group_poll_bytes=(3072 if a.mode=='asan'else 1536)<<20,wall_child_seconds=180,CPU_soft_seconds=900,CPU_hard_seconds=901,FSIZE_each_bytes=4<<20,text_budget_bytes=8<<20,NOFILE=256,CORE=0))
  h.save(rec,out);start=time.monotonic()
  try:
   original=json.loads(r.PRIOR.read_text());extents=original['selected_extent_sha256'];manifest=json.loads((r.OWNER/'manifest.json').read_text());models=[h.identity(x['path'])for x in original['model_identity']];assert models==original['model_identity'];rec['model_identity']=models
   rec['selected_extents_before']=r.extent_hashes(extents)
   env=dict(r.ENV)
   if a.mode=='asan':env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
   mode='--iq2s-index-correctness-only'if a.mode=='correctness'else'--iq2s-index-timing-only'
   cmd=[build['binary']['path'],str(r.PACK),str(r.PRIMARY),str(r.OWNER/'cohort.tsv'),'22','20','1','0','1',str(seed),mode]
   if a.mode=='correctness':
    trace=out/'syscalls.txt';cmd=['/usr/bin/strace','-f','-yy','-s','512','-e','trace=open,openat,openat2,close,close_range,mmap,ioctl,execve','-o',str(trace)]+cmd;env['LD_DEBUG']='libs'
   rows,stderr=h.run_child(rec,out,'owned-sample',cmd,env,wall=180)
   envrows={x[1]:x[2]for x in rows if x[:1]==['ENV']};assert set(envrows)==ENV_KEYS and all(x=='<unset>'for x in envrows.values())
   endpoint=next(i for i,x in enumerate(rows)if x==['RESULT','iq2s_index_correctness_only_pass','no_timing_no_adoption'])
   rec['numerical_admission']=r.validate(rows[:endpoint+1],manifest,extents)
   if a.mode=='correctness':
    t=trace.read_text();assert trace.stat().st_size<=512<<10 and not re.search(r'/dev/(dri|nvidia|kfd)|lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)',t,re.I)
    rec['runtime_audit']=dict(trace_sha256=sha(trace),trace_bytes=trace.stat().st_size,scope='pre-exec to exit scoped no GPU/runtime observation')
   else:
    assert not stderr.strip();rec['timing_validation']=validate_timing(rows,seed)
    rec['timing_validation']['duration_interpretation']='sanitizer correctness only'if a.mode=='asan'else'replicated screen, not model performance'
   rec['selected_extents_after']=r.extent_hashes(extents);assert rec['selected_extents_before']==rec['selected_extents_after']
   assert all(h.identity(x['path'])==x for x in models);pins();assert sha(__file__)==rec['controller_sha256'];rec.update(complete=True,passed=True)
  except BaseException as e:rec['error']=type(e).__name__+': '+str(e)
  finally:
   try:
    pins();assert sha(__file__)==rec['controller_sha256']
    if 'model_identity'in rec:assert all(h.identity(x['path'])==x for x in rec['model_identity'])
    rec['exit_source_pin_gate_passed']=True
   except BaseException as e:rec['passed']=False;rec['exit_source_error']=type(e).__name__+': '+str(e)
   rec.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());h.save(rec,out)
  print(json.dumps(dict(record=str(out/'record.json'),sha256=sha(out/'record.json'),passed=rec['passed'],error=rec.get('error'),elapsed_seconds=rec['elapsed_seconds'])))
  return 0 if rec['passed']else 1
if __name__=='__main__':raise SystemExit(main())
