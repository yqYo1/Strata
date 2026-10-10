import fcntl,importlib.util,json,time
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
WX=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
DEST=WX/'.tools/intel-llvm-v711';LLVM=DEST/'install'
SRC=Path('/home/yayoi/ghq/github.com/intel/llvm/.worktree/toolchain-sycl-v711-b570-20261011')
OWNER=B/'xe-llvm-v711-source-v5/direct_owner.py'
spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
BASE=dict(PATH=str(LLVM/'bin')+':'+str(DEST/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib'))
def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  proof=B/'xe-llvm-v711-source-root-v5/record.json';toolchain=json.loads(proof.read_text())
  m.require(toolchain['passed'] and not toolchain['active'] and toolchain['complete'],'closed compiler')
  out=B/'xe-llvm-v711-host-sanitizer-runtime-root-v1';m.require(not out.exists(),'immutable output');out.mkdir()
  r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(OWNER),compiler_receipt_sha256=m.sha(proof),model_executed=False,gpu_executed=False,system_packages_installed=False,scope='Source-matched native x86_64 host compiler-rt ASan/UBSan runtimes missing from deploy-sycl-toolchain; prerequisite to host sanitizer tests',total_wall_budget_seconds=2400,raw_log_budget_bytes=64<<20,build_jobs=6,references=['https://clang.llvm.org/docs/AddressSanitizer.html','https://compiler-rt.llvm.org/'],source_commit='504366f4b82ff00bbac7b0c956635eed8bc599d8')
  def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
  owner=m.Owner(out/'commands',save);r['commands']=owner.commands;start=time.monotonic();save()
  def cmd(label,args,wall=60):
   m.require(time.monotonic()-start<2400,'whole runtime build deadline')
   e,so,se=owner.run(label,args,BASE,HERE,wall=min(wall,2400-(time.monotonic()-start)),cpu=int(wall)*6,text_cap=32<<20,total_cap=64<<20,rss_cap=24<<30,file_cap=1<<30)
   m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label)
   return so.read_text()
  try:
   for item in toolchain['installed_files'].values():m.require(m.sha(Path(item['path']))==item['sha256'],'installed compiler identity')
   m.require(cmd('source-head',['/usr/bin/git','-C',str(SRC),'rev-parse','HEAD']).strip()==r['source_commit'],'compiler-rt source matches installed compiler')
   resource=Path(cmd('resource-dir',[str(LLVM/'bin/clang++'),'--print-resource-dir']).strip());m.require(resource==LLVM/'lib/clang/22','exact private compiler resource directory')
   r['resource_dir']=str(resource)
   r['source_pins']={f:m.sha(SRC/f) for f in ['compiler-rt/CMakeLists.txt','compiler-rt/cmake/base-config-ix.cmake','compiler-rt/cmake/Modules/CompilerRTUtils.cmake']}
   build=DEST/'compiler-rt-build';m.require(not build.exists(),'new host runtime build')
   options=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER='+str(LLVM/'bin/clang'),'-DCMAKE_CXX_COMPILER='+str(LLVM/'bin/clang++'),'-DCMAKE_ASM_COMPILER='+str(LLVM/'bin/clang'),'-DLLVM_CONFIG_PATH='+str(LLVM/'bin/llvm-config'),'-DLLVM_DIR='+str(LLVM/'lib/cmake/llvm'),'-DCMAKE_INSTALL_PREFIX='+str(LLVM),'-DCOMPILER_RT_INSTALL_PATH=lib/clang/22','-DCOMPILER_RT_INSTALL_LIBRARY_DIR=lib/clang/22/lib/linux','-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF','-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON','-DCMAKE_C_COMPILER_TARGET=x86_64-unknown-linux-gnu','-DCMAKE_CXX_COMPILER_TARGET=x86_64-unknown-linux-gnu','-DCOMPILER_RT_BUILD_BUILTINS=OFF','-DCOMPILER_RT_BUILD_SANITIZERS=ON','-DCOMPILER_RT_SANITIZERS_TO_BUILD=asan;ubsan_minimal','-DCOMPILER_RT_INCLUDE_TESTS=OFF','-DCOMPILER_RT_BUILD_XRAY=OFF','-DCOMPILER_RT_BUILD_LIBFUZZER=OFF','-DCOMPILER_RT_BUILD_PROFILE=OFF','-DCOMPILER_RT_BUILD_CTX_PROFILE=OFF','-DCOMPILER_RT_BUILD_MEMPROF=OFF','-DCOMPILER_RT_BUILD_ORC=OFF','-DCOMPILER_RT_BUILD_GWP_ASAN=OFF']
   cmd('configure',['/usr/bin/cmake','-S',str(SRC/'compiler-rt'),'-B',str(build),'-G','Ninja',*options],wall=300)
   r['cache_sha256']=m.sha(build/'CMakeCache.txt');save()
   cmd('compile-install',['/usr/bin/nice','-n','10','/usr/bin/cmake','--build',str(build),'--parallel','6','--target','install'],wall=1800)
   r['installed_runtime_files']=[]
   for p in sorted((resource/'lib/linux').glob('libclang_rt.*')):
    r['installed_runtime_files'].append(dict(path=str(p),sha256=m.sha(p),bytes=p.stat().st_size))
   m.require(any('libclang_rt.asan-x86_64.a' in x['path'] for x in r['installed_runtime_files']),'host asan library installed')
   m.require(any('libclang_rt.ubsan_standalone-x86_64.a' in x['path'] for x in r['installed_runtime_files']),'host ubsan library installed')
   m.require(all(m.sha(Path(x['path']))==x['sha256'] for x in toolchain['installed_files'].values()),'original compiler/SYCL installation unchanged')
   r['complete']=True
  except BaseException as exc:r['error']=repr(exc)
  finally:
   r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
  print(json.dumps({k:r.get(k) for k in ['passed','error','resource_dir','installed_runtime_files']}))
  return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
