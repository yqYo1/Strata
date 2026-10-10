"""Root-only finite native CPU admission/timing cell. No GPU work."""
from pathlib import Path
from collections import Counter
import csv,fcntl,hashlib,json,math,os,subprocess,sys,types,traceback
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-service-capacity-20261010')
HEAD='dba6bea90192f907cbeecdb75c29aa0e7d1fa449'
STAGE=sys.argv[1];NT=int(sys.argv[2]);MODE=sys.argv[3];REPEAT=int(sys.argv[4])
assert STAGE in ('qualify','measure') and NT in (1,2,3,4) and MODE in ('streaming','hot') and REPEAT in (1,2,3)
assert STAGE!='qualify' or (MODE=='streaming' and REPEAT==1)
OUT=B/f'native-capacity-v2-{STAGE}-nt{NT}-{MODE}-r{REPEAT}-controller2'
BUILD=B/'native-capacity-v2-build/record.json'
TSV=B/'native-service-calibration-cohort-22-20-nt1-v2/cohort.tsv'
MANIFEST=TSV.with_name('manifest.json')
PACK=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
PRIMARY=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
def ident(p):
 with Path(p).open('rb') as f:return dict(bytes=Path(p).stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def file_stat(p):
 s=p.stat();return dict(bytes=s.st_size,dev=s.st_dev,ino=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def git(*args):return subprocess.check_output(['git',*args],cwd=W,text=True).strip()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert git('rev-parse','HEAD')==HEAD and not git('status','--porcelain')
 build=json.loads(BUILD.read_text());assert build['passed'] and build['complete'] and not build['active']
 assert build['source_commit']==HEAD and ident(build['source']['path'])=={k:build['source'][k] for k in ('bytes','sha256')}
 binary=Path(build['binary']['path']);assert ident(binary)=={k:build['binary'][k] for k in ('bytes','sha256')}
 qualifiers={}
 if STAGE=='measure':
  q=B/f'native-capacity-v2-qualify-nt{NT}-streaming-r1-controller2/record.json';qr=json.loads(q.read_text());assert qr['passed'] and qr['complete'] and not qr['active'] and qr['binary']==build['binary']
  qualifiers[str(q)]=ident(q)
 gate_source=W/'bench/results/2026-10-10-native-service-capacity-source/root-admission.json'
 declared=json.loads(gate_source.read_text())['predeclared_calibration_gates']
 assert declared['GU']==dict(maxabs=1e-4,nrms=1e-5) and declared['Down']==declared['GU'] and declared['full_chain']==dict(maxabs=1e-3,nrms=1e-4)
 manifest=json.loads(MANIFEST.read_text());assert ident(TSV)['sha256']==manifest['cohort_sha256']=='49ab6545beda7a1d16cc02da07981b80d4fd36fc4fa9385f56039771fb4d21f1'
 parent=B/'run_gdn_gate_factor_probe_v2.py';assert ident(parent)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 source=parent.read_text().split('\ndef parse_probe(',1)[0]
 needle='(16 << 30, 16 << 30)';assert source.count(needle)==1;source=source.replace(needle,'(96 << 30, 96 << 30)')
 module=types.ModuleType('native_capacity_cell_owner');exec(compile(source,str(parent),'exec'),module.__dict__)
 assert not OUT.exists();OUT.mkdir(mode=0o700);module.W=OUT;owner=module.Owner(OUT)
 env=dict(PATH='/opt/intel/oneapi/compiler/2026.1/bin:/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
 seed=2026101001+REPEAT-1
 args=[str(binary),str(PACK),str(PRIMARY),str(TSV),'22','20',str(NT),'0','1',str(seed)]
 if STAGE=='qualify':args+=['--correctness-only']
 if MODE=='hot':args+=['--hot-eight']
 if NT>1:
  for prefix,key in [('gu','GU'),('down','Down'),('chain','full_chain')]:
   for field in ('maxabs','nrms'):args+=['--'+prefix+'-'+field,str(declared[key][field])]
 r=dict(active=True,complete=False,passed=False,stage=STAGE,NT=NT,mode=MODE,repeat=REPEAT,source_commit=HEAD,
  binary=build['binary'],source=build['source'],build_receipt=dict(path=str(BUILD),**ident(BUILD)),
  controller=ident(__file__),parent_owner=ident(parent),owner_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
  predeclared_gates=dict(path=str(gate_source),**ident(gate_source)),qualifiers=qualifiers,
  fixture=dict(TSV=dict(path=str(TSV),**ident(TSV)),manifest=dict(path=str(MANIFEST),**ident(MANIFEST)),primary=dict(path=str(PRIMARY),**file_stat(PRIMARY)),
    native_experts=dict(path=str(PACK/'native_experts.txt'),**ident(PACK/'native_experts.txt'))),
  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),environment=env,argv=args,commands=owner.commands,
  cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_policy={str(p):p.read_text().strip() for name in ['scaling_governor','scaling_min_freq','scaling_max_freq','energy_performance_preference'] for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/'+name)},
  started_utc=module.utc(),gpu_work_submitted=False,model_inference=False,adopted=False,
  scope='Actual384 selected weight payloads; synthetic prequantized activations, direct single pinned host plus unchanged5-worker pool. Intended hot8/cohort or192 host-resident streaming jobs, no proven cache/physicalDRAM roof.',
  limits=dict(wall_seconds=240,AS_each_bytes=96<<30,RSS_session_bytes=2<<30,CPU_each_seconds=[120,121],text_bytes=8<<20))
 def save():
  p=OUT/'record.json.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(OUT/'record.json')
 owner.persist=save
 try:
  command,stdout,stderr=owner.run('native-cell',args,env,wall=240,text_cap=8<<20,file_cap=32<<20);save()
  rows=list(csv.reader(stdout.read_text().splitlines()));r['output_rows']=rows;r['stderr']=dict(path=str(stderr),**ident(stderr));save()
  assert module.completed(command) and command['exit_code']==0,command
  assert rows[-1]==(['RESULT','correctness_only_pass','no_calibration_rounds'] if STAGE=='qualify' else ['RESULT','correctness_pass','statistical_holdout_adoption_and_full_lifecycle_not_qualified'])
  refs=[x for x in rows if x[0]=='REFERENCE_TOKEN'];expected={(i,t,s) for i in range(384) for t in range(NT) for s in ['GU_SwiGLU','Down_actual_FF','full_chain_independent_FF']}
  keys=[(int(x[1]),int(x[5]),x[6]) for x in refs];assert len(keys)==len(set(keys))==len(expected) and set(keys)==expected
  for x in refs:
   assert int(x[4])==NT and int(x[13])==(640 if x[6]=='GU_SwiGLU' else 2560)
   assert all(math.isfinite(float(v)) and float(v)>=0 for v in x[7:13])
  count=[x for x in rows if x[0]=='REFERENCE_COUNTS'];assert count==[['REFERENCE_COUNTS',str(384*NT),str(384*NT*(640+5120)),str(384*NT*3)]]
  ids=[x for x in rows if x[0]=='ID'];assert len(ids)==384 and len({(int(x[2]),int(x[3])) for x in ids})==384
  working=[x for x in rows if x[0]=='ACTIVE_LOGICAL_WORKING_SET'];jobs=8 if MODE=='hot' else 192
  assert len(working)==2 and {int(x[1]) for x in working}=={0,1}
  for x in working:assert list(map(int,x[2:]))==[jobs,jobs*1049600,jobs*921600,jobs*1971200]
  hot=[x for x in rows if x[0]=='HOT_ID'];assert len(hot)==(16 if MODE=='hot' else 0)
  if hot:
   assert {(int(x[1]),int(x[2])) for x in hot}=={(c,c*192+rank) for c in range(2) for rank in range(0,192,24)}
  numeric=[x for x in rows if x[0] in ['ROUND','WARMUP'] and x[1].isdecimal()]
  if STAGE=='qualify':assert not numeric and not [x for x in rows if x[0]=='ROUND_COUNTS']
  else:
   assert [x for x in rows if x[0]=='ROUND_COUNTS']==[['ROUND_COUNTS','30','250']]
   expected_round={(c,a,n) for c in (0,1) for a in ('GU','FFquant','Down','direct_complete','pool') for n in range(-3,25)}
   keys=[(int(x[1]),x[2],int(x[3])) for x in numeric];assert len(keys)==len(set(keys))==280 and set(keys)==expected_round
   for x in numeric:
    assert len(x)==18 and list(map(int,x[4:11]))==[22,20,NT,0,1,jobs,jobs]
    assert all(math.isfinite(float(v)) and float(v)>=0 for v in x[12:16]) and float(x[12])>0
    assert x[0]==('WARMUP' if int(x[3])<0 else 'ROUND')
  assert ident(binary)=={k:build['binary'][k] for k in ('bytes','sha256')}
  assert file_stat(PRIMARY)=={k:r['fixture']['primary'][k] for k in ('bytes','dev','ino','mtime_ns','ctime_ns')}
  assert ident(TSV)=={k:r['fixture']['TSV'][k] for k in ('bytes','sha256')}
  assert git('rev-parse','HEAD')==HEAD and not git('status','--porcelain')
  r.update(passed=True,complete=True,reference_stage_vectors=len(refs),numeric_rows=len(numeric))
 except BaseException as error:r['error']=type(error).__name__+': '+str(error);r['controller_traceback']=traceback.format_exc()
 finally:
  r.update(active=owner.active is not None,finished_utc=module.utc());r['passed']=r['passed'] and not r['active'] and all(module.completed(c) for c in owner.commands);save()
  print(json.dumps({k:r.get(k) for k in ['stage','NT','mode','repeat','passed','complete','active','error']}),flush=True)
 if not r['passed']:sys.exit(1)
