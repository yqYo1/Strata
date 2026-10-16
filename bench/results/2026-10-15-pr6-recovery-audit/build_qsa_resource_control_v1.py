#!/usr/bin/env python3
"""Build the independently reviewed fixture against immutable current-source archives."""
import fcntl,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2];B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def main():
 root=B/'main-qsa-resource-build-v1';target=A/'own-qsa-resource-build-v1.json';assert not root.exists() and not target.exists()
 priorpath=A/'own-prompt-softmax-build-v3.json';prior=json.loads(priorpath.read_text());assert prior['passed'] and prior['complete'] and prior['ownership_closed']
 assert prior['source_pins']['src/kernels/sycl/prompt_attention.cpp']==pin(P/'src/kernels/sycl/prompt_attention.cpp')['sha256']=='5fa7927e611b8cb746e1d11bd70a8e2aae3c66a48b02a93eb55e86eb9bf0076f'
 ownerpath=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py';assert pin(ownerpath)['sha256']=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 spec=importlib.util.spec_from_file_location('qsa_resource_build_owner',ownerpath);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 libraries=[Path(prior['build_path'])/n for n in ['libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a']]
 frozen={}
 for name,h in prior['source_pins'].items():
  if name.startswith(('include/','src/')):
   v=pin(P/name);assert v['sha256']==h,name;frozen[v['realpath']]=v
 for p in [*libraries,P/'tests/sycl/qsa_resource_control.cpp',priorpath,ownerpath,L/'bin/clang++',Path(__file__)]:
  v=pin(p);frozen[v['realpath']]=v
 assert pin(L/'bin/clang++')['sha256']=='1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700);(root/'tmp').mkdir();(root/'cache').mkdir()
  record=dict(schema='strata-qsa-resource-control-build-v1',passed=False,complete=False,active=False,commands=[],GPU_executed=False,
   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),frozen_files=frozen,
   inherited_closed_build=pin(priorpath),archive_origin='Current63326 source5fa qualifiedbuild outputs, read-only; archives newly pinned here. No historical or excluded project code.',
   source=pin(P/'tests/sycl/qsa_resource_control.cpp'),limits='Compile/link qualification only; no runtime, model or performance result.')
  owner=None
  def persist():
   record['active']=bool(owner and owner.active is not None);(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
  env=dict(PATH=str(L/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(L/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib',TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
  def verify():
   for path,v in frozen.items():assert pin(path)==v,path
  try:
   verify();exe=root/'sycl_qsa_resource_control'
   argv=[str(L/'bin/clang++'),'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1','-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror',str(P/'tests/sycl/qsa_resource_control.cpp'),*[str(p) for p in libraries],'-o',str(exe)]
   c,out,err=owner.run('compile-link',argv,env,P,wall=300,cpu=600,rss_cap=8<<30,text_cap=64<<20,file_cap=128<<20,total_cap=1<<30)
   record.update(exit_code=c['exit_code'],stdout=out.read_text(),stderr=err.read_text());assert m.closed(c) and c['exit_code']==0,(c['exit_code'],out.read_text()[-4000:],err.read_text()[-4000:]);verify()
   record.update(binary=pin(exe),passed=True,complete=True)
  except BaseException as e:
   import traceback
   record.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   record['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands);persist();target.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(target),passed=record['passed'],exit_code=record.get('exit_code'),GPU_executed=False,error=(record.get('error') or '')[:400])))
  return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
