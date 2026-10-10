"""Main-owned pristine XeStrata configure/build, no device or model execution."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,types
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/eval-b570-20261010')
OUT=B/'xestrata-pristine-icpx-build-root-v1'
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
GGML=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
BUILD=W/'build/eval-b570-icpx-v1'
def ident(p):
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='39bdadcc9e2b89b1e3c8be7bb2a603b04fa0e197'
 assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=GGML,text=True).strip()=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
 assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 assert not OUT.exists() and not BUILD.exists();OUT.mkdir(mode=0o700)
 parent=PARENT.read_text().split('\ndef parse_probe(',1)[0]
 # Explicit finite build-specific budgets. GPU/model supervision is unchanged.
 assert '(120, 121)' in parent and 'rss <= 2 << 30' in parent
 parent=parent.replace('(120, 121)','(600, 601)').replace('rss <= 2 << 30','rss <= 24 << 30')
 copied=OUT/'build_owner.py';copied.write_text(parent)
 m=types.ModuleType('root_pristine_xestrata_build');exec(compile(parent,str(copied),'exec'),m.__dict__);m.W=W;owner=m.Owner(OUT)
 env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR','LANG','LC_CTYPE'}}
 env.update(LC_ALL='C',LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib',MKLROOT='/opt/intel/oneapi/mkl/2026.1')
 cmake=['/usr/bin/cmake','-S',str(W),'-B',str(BUILD),'-G','Ninja','-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx','-DCMAKE_C_COMPILER=/usr/bin/cc','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DSTRATA_LICENSE=contrib-icpx','-DSTRATA_ENABLE_XE=ON','-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_GGML_DIR='+str(GGML),'-DSTRATA_BUILD_TESTS=OFF','-DSTRATA_CUDA_ARCHS=','-DSTRATA_HIP_ARCHS=','-DMKL_ROOT=/opt/intel/oneapi/mkl/2026.1']
 r=dict(active=True,complete=False,passed=False,mode='contrib-icpx',fork_pristine=True,source_commit='39bdadcc9e2b89b1e3c8be7bb2a603b04fa0e197',root=str(W),build=str(BUILD),gpu_executed=False,model_executed=False,controller=ident(Path(__file__)),parent_owner=ident(PARENT),build_owner=ident(copied),build_budgets=dict(per_child_AS_bytes=16<<30,per_child_CPU_seconds=600,owned_session_RSS_bytes=24<<30,file_bytes=512<<20,combined_text_bytes=64<<20,configure_wall_seconds=600,build_wall_seconds=1800,jobs=3),environment=env,commands=owner.commands,ggml_commit='3cf03257f219afbe7334045ff7c6a06ac68c627d',ggml_pristine=not subprocess.check_output(['git','status','--porcelain'],cwd=GGML,text=True).strip(),source={name:ident(W/name) for name in ['CMakeLists.txt','AGENTS.md','src/core/device.cpp','src/core/pinned.cpp','src/core/graph.cpp','src/core/expert_source.cpp','src/program/generate.cpp','src/prefill/gemm.cpp','src/prefill/prefill.cpp','src/kernels/cpu/native_expert.cpp','src/kernels/cpu/iq_avx2.cpp']})
 def save():
  temp=OUT/'record.json.tmp';temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(OUT/'record.json')
 owner.persist=save
 try:
  e,_,_=owner.run('configure',cmake,env,wall=600,text_cap=64<<20,file_cap=512<<20);assert m.completed(e) and e['exit_code']==0,'configure did not complete normally'
  e,_,_=owner.run('build',['/usr/bin/cmake','--build',str(BUILD),'--target','strata','strata-device','-j3'],env,wall=1800,text_cap=64<<20,file_cap=512<<20);assert m.completed(e) and e['exit_code']==0,'build did not complete normally'
  assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
  r.update(complete=True,passed=True,binary=str(BUILD/'strata'),binary_identity=ident(BUILD/'strata'),device_binary_identity=ident(BUILD/'strata-device'),compile_commands=ident(BUILD/'compile_commands.json'),cache_identity=ident(BUILD/'CMakeCache.txt'),build_json=ident(BUILD/'BUILD.json'))
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  r.update(active=owner.active is not None);r['passed']=r['passed'] and not r['active'] and all(m.completed(e) for e in owner.commands);save();print(json.dumps({k:r.get(k) for k in ['active','complete','passed','error','binary']}),flush=True)
 if not r['passed']:raise SystemExit(1)
