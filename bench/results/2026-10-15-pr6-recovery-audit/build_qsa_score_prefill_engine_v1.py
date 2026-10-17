#!/usr/bin/env python3
"""Fresh full engine and actual dedicated-prefill scorer fixture, Main-owned."""
import datetime,fcntl,hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEV=Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 root=B/'main-qsa-score-prefill-engine-build-v1';out=A/'qsa-score-prefill-engine-build-v1.json';assert not root.exists() and not out.exists()
 owner_path=A/'direct_owner_transient_artifacts_v1.py';assert pin(owner_path)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
 owned=module('qsa_score_engine_owner',owner_path)
 package=A/'own-vllm-xe2-qualification-v1';meta_path=package/'manifest.json'
 assert pin(meta_path)['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
 meta=json.loads(meta_path.read_text());helper=package/'main_controller.py';assert pin(helper)['sha256']==meta['package_files']['main_controller.py']['sha256'];guard=module('qsa_score_engine_guard',helper)
 gates=[]
 for name in ['qsa-score-querytile-decision-v1.json','qsa-score-querytile-cohort-v1.json','moe-accum-model-profile-window10-tokens65536-warm4-v5.json']:
  path=A/name;v=json.loads(path.read_text());assert v['passed'] and v['complete'] and not v.get('active')
  if 'commands' in v:
   assert v['ownership_closed'] and v['fresh_faults']==[] and not v.get('driver_dump_exists')
   for c in v['commands']:
    assert owned.closed(c) and c['exit_code']==0
    for old in c['owners']+[c['direct_identity']]:
     now=owned.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
  gates.append(path)
 previous_path=A/'moe-accum-engine-build-v2.json';previous=json.loads(previous_path.read_text());assert previous['passed'] and previous['complete'] and previous['ownership_closed']
 source_path=A/'qsa-score-prefill-source-v1.json';source=json.loads(source_path.read_text())
 for name,h in source['source_pins'].items():assert pin(P/name)['sha256']==h
 subprocess.run(['git','diff','--exit-code','HEAD','--','src','include','tests','cmake','CMakeLists.txt'],check=True,cwd=P)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=P).strip()
 names=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','--','src','include','tests','cmake','CMakeLists.txt'],text=True,cwd=P).splitlines()
 frozen={str(P/name):pin(P/name) for name in names}
 oldroot=Path(previous['configure_argv'][previous['configure_argv'].index('-S')+1])
 for name,h in previous['source_pins'].items():
  path=Path(name)
  if any(path.is_relative_to(x) for x in [P,B,oldroot]):continue
  if '/MistVVK/XeStrata/' in name:assert '/.tools/intel-llvm-v711/install/' in name
  v=pin(path);assert v['sha256']==h;frozen[name]=v
 for path in [Path(__file__),owner_path,meta_path,helper,source_path,previous_path,*gates]:frozen[str(path)]=pin(path)
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==previous['boot_id']
 with (B/'owned-v0141-measurement.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.no_model_owners(meta['known_model_files']);guard.driver_guard()
  root.mkdir(mode=0o700);(root/'tmp').mkdir()
  env={**previous['environment'],'TMPDIR':str(root/'tmp'),'XDG_CACHE_HOME':str(root/'cache')}
  rec=dict(schema='strata-qsa-score-prefill-engine-build-v1',passed=False,complete=False,active=False,GPU_executed=False,source_commit=commit,boot_id=boot,environment=env,commands=[],frozen_files=frozen,source_pins={p:v['sha256'] for p,v in frozen.items()},
   provenance=source['provenance'],production_selected=False,performance_adopted=False,model_qualified=False,physical262144_qualified=False,RESTORE_qualified=False,
   scope='Actual dedicated-prefill opt-in query8 FP32 scorer, original public scorer retained; full Release engine plus actual-entry oracle/guard fixture. No prior binary overwritten.')
  owner=None
  def save():
   rec['active']=bool(owner and owner.active is not None);rec['ownership_closed']=bool(owner and owner.active is None and all(owned.closed(c) for c in owner.commands));out.write_text(json.dumps(rec,indent=2)+'\n')
  owner=owned.Owner(root/'owned',save);rec['commands']=owner.commands
  def run(name,argv,wall=60,cpu=120):
   c,so,se=owner.run(name,argv,env,P,wall=wall,cpu=cpu,rss_cap=16<<30,text_cap=64<<20,file_cap=1<<30,total_cap=4<<30)
   assert owned.closed(c) and c['exit_code']==0,(name,c.get('exit_code'),str(se));return so.read_text()
  cursor=None;save()
  try:
   raw=run('journal-before',['/usr/bin/journalctl','-k','-b','-n','0','--show-cursor','--no-pager']);cursor=re.search(r'-- cursor: (\S+)',raw)[1];rec['journal_before_cursor']=cursor
   configure=list(previous['configure_argv']);configure[configure.index('-S')+1]=str(P);configure[configure.index('-B')+1]=str(root/'build');rec['configure_argv']=configure
   run('configure',configure,wall=300,cpu=600)
   targets=['strata','sycl_qsa_score_prefill_test'];run('compile-link',['/usr/bin/cmake','--build',str(root/'build'),'--target',*targets,'--parallel','2','--verbose'],wall=2400,cpu=18000)
   rec['products']={t:pin(root/'build'/t) for t in targets};rec['binary']=rec['products']['strata']
   entries=json.loads((root/'build/compile_commands.json').read_text());selected=['src/kernels/sycl/qsa_select.cpp','src/prefill/prefill.cpp','src/prefill/sycl/native16.cpp','src/program/generate.cpp','tests/sycl/qsa_score_prefill.cpp']
   rec['actual_compile_entries']=[x for x in entries if x['file'] in [str(P/n) for n in selected]];assert {x['file'] for x in rec['actual_compile_entries']}=={str(P/n) for n in selected}
   for entry in rec['actual_compile_entries']:
    if entry['file'].endswith(('qsa_select.cpp','qsa_score_prefill.cpp')):assert '-ffp-contract=off' in entry['command']
   rec['compile_database']=pin(root/'build/compile_commands.json')
   for t in targets:rec.setdefault('dynamic',{})[t]=run('readelf-'+t,['/usr/bin/readelf','-d',str(root/'build'/t)])
   assert all(pin(p)==v for p,v in frozen.items());rec.update(passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   try:
    assert owner.active is None
    if cursor:
     raw=run('journal-final',['/usr/bin/journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(x) for x in raw.splitlines() if x.strip()];rec['fresh_faults']=[x for x in rows if FAULT.search(str(x.get('MESSAGE','')))]
    rec['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot;rec['driver_dump_entries']=[str(x) for x in DEV.glob('devcoredump*')]
    assert cursor and rec['fresh_faults']==[] and rec['boot_unchanged'] and not rec['driver_dump_entries'];guard.driver_guard()
   except BaseException as e:rec.update(closure_error=repr(e),passed=False,complete=False)
   rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
  print(json.dumps(dict(receipt=str(out),passed=rec['passed'],complete=rec['complete'],error=rec.get('error'))),flush=True)
  return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
