"""Root-only host fixture rebuild against byte-pinned unchanged CPU objects."""
from pathlib import Path
import fcntl,hashlib,json,shlex,subprocess,sys,types
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-service-capacity-20261010')
T=W.parent/'diag-sycl-native-role-plan-v0141-20261010';D=T/'build-native-service-calibration-v1'
HEAD='dba6bea90192f907cbeecdb75c29aa0e7d1fa449';OUT=B/'native-capacity-v2-build'
def ident(p):
 with Path(p).open('rb') as f:return dict(bytes=Path(p).stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
 assert git(T,'rev-parse','HEAD')=='af12591ca840deb4649b703cbb7469fbc2417fc9' and not git(T,'status','--porcelain')
 prior=B/'native-service-calibration-cpu-build-v4/record.json'
 assert ident(prior)['sha256']=='dd4d7776f6a426ae0a159e76281d2c6ed00476e083f887de62d91a2506491d89'
 old=json.loads(prior.read_text());assert old['passed'] and old['complete'] and not old['active'] and not old['cleanup'] and not old['survivors']
 for rel,pin in old['production_source_pins'].items():
  assert ident(T/rel)['sha256']==pin and ident(W/rel)['sha256']==pin
 for filename,pin in old['pins'].items():assert ident(filename)['sha256']==pin
 parent=B/'run_gdn_gate_factor_probe_v2.py';assert ident(parent)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 owner_source=parent.read_text().split('\ndef parse_probe(',1)[0]
 module=types.ModuleType('native_capacity_build_owner');exec(compile(owner_source,str(parent),'exec'),module.__dict__)
 assert not OUT.exists();OUT.mkdir(mode=0o700);module.W=OUT;owner=module.Owner(OUT)
 rel='sycl/tools/native-service-calibration/native_service_calibration.cpp';source=W/rel;pin=ident(source)
 assert pin['sha256']=='f062f24ea57443c4cd59d09d2b56d498165ea497254ae9fa8998375b062b2ac3'
 cc=json.loads((D/'compile_commands.json').read_text());entries=[e for e in cc if e['file']==str(T/rel)];assert len(entries)==1
 original_compile=shlex.split(entries[0]['command']);args=[s.replace(str(T),str(W)) for s in original_compile];args[args.index('-o')+1]=str(OUT/'native_service_calibration.cpp.o')
 raw=subprocess.check_output(['ninja','-C',str(D),'-t','commands','native_service_calibration'],text=True).splitlines()[-1]
 assert raw.startswith(': && ') and raw.endswith(' && :')
 original_link=shlex.split(raw[5:-5]);link=list(original_link);link[link.index('-o')+1]=str(OUT/'native_service_calibration')
 cached={}
 for i,arg in enumerate(link):
  if arg=='CMakeFiles/native_service_calibration.dir/native_service_calibration.cpp.o':link[i]=str(OUT/'native_service_calibration.cpp.o')
  elif arg.endswith(('.o','.a')) and not arg.startswith('-'):
   p=D/arg;link[i]=str(p);cached[str(p)]=ident(p)
 r=dict(active=True,complete=False,passed=False,source_commit=HEAD,source=dict(path=str(source),**pin),controller=ident(__file__),
  parent_owner=ident(parent),prior_build_receipt=dict(path=str(prior),**ident(prior)),cached_base_commit='af12591ca840deb4649b703cbb7469fbc2417fc9',
  cached_objects=cached,unchanged_production_sources=old['production_source_pins'],
  original_compile=original_compile,changed_compile=args,original_link=original_link,changed_link=link,
  environment=old['environment'],commands=owner.commands,gpu_work_submitted=False,model_inference=False,adopted=False,started_utc=module.utc())
 def save():
  p=OUT/'record.json.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(OUT/'record.json')
 owner.persist=save
 def run(label,cmd,wall=120):
  e,so,se=owner.run(label,cmd,old['environment'],wall=wall,text_cap=8<<20,file_cap=32<<20);save()
  assert module.completed(e) and e['exit_code']==0,(label,e);return so
 try:
  run('compile-fixture',args)
  run('link-host-fixture',link)
  binary=OUT/'native_service_calibration'
  elf=run('readelf-host-closure',['/usr/bin/readelf','-d',str(binary)])
  assert not any(name in elf.read_text().lower() for name in ['libsycl','libur_adapter','libze_loader','libcudart','libamdhip'])
  assert ident(source)==pin and {p:ident(p) for p in cached}==cached
  assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
  r.update(passed=True,complete=True,binary=dict(path=str(binary),**ident(binary)))
 except BaseException as error:r['error']=type(error).__name__+': '+str(error)
 finally:
  r.update(active=owner.active is not None,finished_utc=module.utc());r['passed']=r['passed'] and not r['active'] and all(module.completed(e) for e in owner.commands);save()
  print(json.dumps({k:r.get(k) for k in ['passed','complete','active','error','binary']}),flush=True)
 if not r['passed']:sys.exit(1)
