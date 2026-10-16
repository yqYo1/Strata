#!/usr/bin/env python3
"""Fresh full engine build for opt-in BF16-prefix dispatch; Main serial owner."""
import fcntl,hashlib,importlib.util,json,os,re,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
BOOT='cfda0b6a-7c13-4fe2-bdf5-8f9202770840'
DEV=Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def main():
 root=B/'main-bf16-prefix-tilepack-engine-build-v1';dst=A/'own-qsa-bf16-prefix-tilepack-engine-build-v2.json'
 assert root.exists() and not dst.exists() and not (root/'owned-v2').exists()
 priorpath=A/'own-vllm-engine-build-v2.json';prior=json.loads(priorpath.read_text());assert prior['passed'] and prior['complete'] and prior['ownership_closed']
 compiler=Path(prior['configure_argv'][prior['configure_argv'].index('-DCMAKE_CXX_COMPILER='+prior['configure_argv'][6].split('=',1)[-1])].split('=',1)[1]) if False else Path(next(x.split('=',1)[1] for x in prior['configure_argv'] if x.startswith('-DCMAKE_CXX_COMPILER=')))
 assert pin(compiler)['sha256']=='1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
 ownerpath=A/'direct_owner_transient_artifacts_v1.py';assert pin(ownerpath)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
 spec=importlib.util.spec_from_file_location('bf16_tilepack_engine_owner',ownerpath);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 subprocess.run(['git','diff','--exit-code','HEAD','--','src','include','cmake','CMakeLists.txt','tests'],cwd=P,check=True,stdout=subprocess.DEVNULL)
 names=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','--','src','include','cmake','CMakeLists.txt','tests'],cwd=P,text=True).splitlines()
 frozen={str(P/n):pin(P/n) for n in names}
 # Reuse independently pinned external inputs, never mutate any previous build.
 for group in ['pins']:
  for name,v in prior[group].items():
   if Path(name).is_relative_to(P):continue
   assert pin(name)['sha256']==v['sha256'],name
   frozen[name]=pin(name)
 for p in [Path(__file__),ownerpath,priorpath,compiler]:frozen[str(p)]=pin(p)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  interrupted_path=A/'own-qsa-bf16-prefix-tilepack-engine-build-v1.json'
  interrupted=json.loads(interrupted_path.read_text())
  assert not interrupted['passed'] and not interrupted['active'] and not interrupted['fresh_faults']
  check_path=A/'transient-artifact-owner-check-v1.json'
  check=json.loads(check_path.read_text());assert check['passed'] and check['complete'] and check['GPU_executed'] is False
  assert check['helper_sha256']==pin(ownerpath)['sha256']
  for command in interrupted['commands']:
   assert command['direct_child_reaped'] and command['session_empty'] and not command.get('survivors')
   for old_id in command['owners']+[command['direct_identity']]:
    now=m.identity(old_id['pid']);assert now is None or now['start_ticks']!=old_id['start_ticks']
  for p in [interrupted_path,check_path,A/'qsa-bf16-prefix-tilepack-build-interruption-audit-v1.json']:
   frozen[str(p)]=pin(p)
  rec=dict(schema='strata-bf16-prefix-tilepack-engine-build-v2',resumed_failed_build=pin(interrupted_path),resume_scope='Original v1 remains FAIL. After explicit absent-owner audit and new observer CPU gates, reuse only partial compile outputs in its private build directory; unchanged source/toolchain, fresh command supervision/logs/receipt.',owner_source=pin(ownerpath),boot_id=BOOT,passed=False,complete=False,active=False,GPU_executed=False,offline_AOT_may_query_device=True,commands=[],frozen_files=frozen,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),performance_adopted=False,model_qualified=False,physical262144_model_qualified=False,RESTORE_qualified=False)
  owner=None
  def persist():
   rec['active']=bool(owner and owner.active is not None);(root/'record-v2.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned-v2',persist);rec['commands']=owner.commands
  env=dict(prior['environment']);env.update(XDG_CACHE_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'));rec['environment']=env
  def verify():
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
   assert DEV.joinpath('driver').resolve().name=='xe' and not list(DEV.glob('devcoredump*'))
   for p,v in frozen.items():assert pin(p)==v,p
  def run(label,argv,wall=300,cpu=1800):
   verify();c,so,se=owner.run(label,argv,env,P,wall=wall,cpu=cpu,rss_cap=32<<30,text_cap=64<<20,file_cap=2<<30,total_cap=8<<30)
   assert m.closed(c),(label,'ownership');rec.setdefault('output',{})[label]=dict(stdout=so.read_text(),stderr=se.read_text());assert c['exit_code']==0,(label,c['exit_code']);verify();return so.read_text()
  cursor=None
  try:
   raw=run('journal-before',['/usr/bin/journalctl','-k','-b','-n','0','--show-cursor','--no-pager']);cursor=re.search(r'-- cursor: (\S+)',raw)[1];rec['journal_before_cursor']=cursor
   options=list(prior['configure_argv']);options[options.index('-B')+1]=str(root/'build');rec['configure_argv']=options
   run('configure',options)
   targets=['strata','sycl_qsa_bf16_prefix_test','sycl_qsa_bf16_prefix_optimized_test','sycl_qsa_bf16_prefix_tilepack_test','sycl_qsa_bf16_late_flags_test','sycl_native16_qsa_test','prefill_layer_major_plan_test']
   run('compile-link',['/usr/bin/cmake','--build',str(root/'build'),'--target',*targets,'--parallel','2','--verbose'],wall=2400,cpu=18000)
   run('host-plan',[str(root/'build/prefill_layer_major_plan_test')])
   rec['products']={name:pin(root/'build'/name) for name in targets}
   rec['binary']=rec['products']['strata'];rec['actual_compile_entries']=json.loads((root/'build/compile_commands.json').read_text())
   raw=run('readelf',['/usr/bin/readelf','-d',str(root/'build/strata')]);rec['dynamic']=raw
   rec.update(passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   try:
    assert owner.active is None
    if cursor:
     raw=run('journal-final',['/usr/bin/journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(x) for x in raw.splitlines() if x.strip()];rec['fresh_faults']=[x for x in rows if FAULT.search(x.get('MESSAGE',''))]
     assert not rec['fresh_faults']
    rec['ownership_closed']=all(m.closed(c) for c in owner.commands); rec['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT; rec['driver_dump_entries']=[str(p) for p in DEV.glob('devcoredump*')]; verify(); assert rec['ownership_closed'] and rec['boot_unchanged'] and not rec['driver_dump_entries']
   except BaseException as e:rec.update(closure_error=repr(e),passed=False,complete=False)
   persist();dst.write_bytes((root/'record-v2.json').read_bytes())
  print(json.dumps(dict(receipt=str(dst),passed=rec['passed'],complete=rec['complete'],GPU_executed=False,error=rec.get('error'),closure_error=rec.get('closure_error'))))
  return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
