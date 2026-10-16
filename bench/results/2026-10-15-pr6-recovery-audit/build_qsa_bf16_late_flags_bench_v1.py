#!/usr/bin/env python3
"""Link matched QSA timing fixture; no GPU execution or engine modification."""
import fcntl, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 helper=A/'build_qsa_bf16_operand_v1.py';h=module('qsa_bf16_build_helpers',helper);pin=h.pin;B=h.B;L=h.L
 root=B/'main-qsa-bf16-late-flags-bench-build-v1';dst=A/'own-qsa-bf16-late-flags-bench-build-v1.json'
 assert not root.exists() and not dst.exists()
 buildpath=A/'own-qsa-bf16-late-flags-build-v1.json';build=json.loads(buildpath.read_text())
 assert build['passed'] and build['complete'] and build['ownership_closed']
 frozen=dict(build['frozen_files']);obj=build['object'];frozen[obj['path']]=obj
 oldpath=A/'own-qsa-resource-build-v4.json';old=json.loads(oldpath.read_text());assert old['passed'] and old['complete'] and old['ownership_closed']
 libraries=[]
 for name in ['libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a']:
  matches=[v for v in old['frozen_files'].values() if Path(v['path']).name==name];assert len(matches)==1
  v=matches[0];frozen[v['path']]=v;libraries.append(v['path'])
 for path in [buildpath,oldpath,P/'tests/sycl/qsa_bf16_bench.cpp',Path(__file__),P/'include/strata/kernels/qsa_bf16_attention.hpp']:
  v=pin(path);frozen[v['path']]=v
 ownerpath=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
 assert pin(ownerpath)['sha256']=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 m=module('qsa_bf16_fixture_build_owner',ownerpath)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700);(root/'tmp').mkdir();(root/'cache').mkdir()
  rec=dict(schema='strata-qsa-bf16-bench-build-v1',passed=False,complete=False,active=False,GPU_executed=False,commands=[],frozen_files=frozen,
   object=obj,production_selected=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
   scope='Unchanged timing fixture redirects bf16 arm to actual late-flags entry by explicit compile-time identifier substitution. All inputs/allocations/host-oracle/timing code identical to original source. No GPU execution.')
  owner=None
  def persist():
   rec['active']=bool(owner and owner.active is not None);(root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);rec['commands']=owner.commands
  def verify():
   for path,v in frozen.items():assert pin(v['path'])==v,path
  try:
   verify();exe=root/'sycl_qsa_bf16_late_flags_bench'
   env=dict(PATH=str(L/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(L/'lib'),TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
   argv=[str(L/'bin/clang++'),'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1','-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror','-Dqsa_bf16_attention_batch=qsa_bf16_attention_late_flags_batch',str(P/'tests/sycl/qsa_bf16_bench.cpp'),obj['path'],*libraries,'-o',str(exe)]
   c,out,err=owner.run('compile-link',argv,env,P,wall=300,cpu=600,rss_cap=8<<30,text_cap=64<<20,file_cap=128<<20,total_cap=1<<30)
   rec.update(exit_code=c['exit_code'],stdout=out.read_text(),stderr=err.read_text());assert m.closed(c) and c['exit_code']==0
   verify();rec['binary']=pin(exe)
   c,out,err=owner.run('readelf',['/usr/bin/readelf','-d',str(exe)],env,P,wall=30,cpu=30,rss_cap=256<<20,text_cap=1<<20,file_cap=1<<20,total_cap=1<<30)
   rec.update(dynamic=out.read_text(),readelf_stderr=err.read_text());assert m.closed(c) and c['exit_code']==0
   rec.update(passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   rec['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands);persist();dst.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(dst),passed=rec['passed'],exit_code=rec.get('exit_code'),GPU_executed=False,error=rec.get('error'))))
  return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
