from pathlib import Path
import datetime,hashlib,json,os,shlex,shutil,subprocess,types

root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
base=Path(__file__).parent;out=base/'current-tasks9-link-system-tools';out.mkdir();out.chmod(0o700)
build=root/'build-sycl-upstream-jit'
old=Path('/home/yayoi/.local/state/strata-sycl/cpu-pool-scheduling-probe')
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
expected='79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223'
assert digest(build/'strata')==expected
assert digest(build/'libstrata_kernels_cpu.a')=='766e388d445dc0da3627f1eae91c8a2575785ac739cdeffebce16b2237f8220f'
assert digest(old/'production-constant9.a')=='ac0a17a817eef883083214efa941ff2d6ead3e4549c0306e91b3bf78b038577c'
assert digest(root/'src/kernels/cpu/pool.cpp')=='41a2be640f1d1823f7ba88107beba4815cb3ad6f7ffc39e6f34575b8b1b1cd32'
shutil.copy2(old/'production-constant9.a',out/'libstrata_kernels_cpu.a')
def members(p):
 result={}
 for name in subprocess.check_output(['/usr/bin/ar','t',str(p)],text=True).splitlines():
  assert name not in result
  payload=subprocess.check_output(['/usr/bin/ar','p',str(p),name])
  result[name]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
 return result
original=members(build/'libstrata_kernels_cpu.a');candidate=members(out/'libstrata_kernels_cpu.a')
left=set(original)-set(candidate);right=set(candidate)-set(original)
assert left=={'pool.cpp.o'} and right=={'native-pool-task-factor.cpp.o'},(left,right)
assert all(original[name]==candidate[name] for name in set(original)&set(candidate))
comparison=dict(scope='Exactly one CPU pool member replaces the original; all other archive members agree',original=original,candidate=candidate)
(out/'archive-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands','strata'],text=True).splitlines()
command=commands[-1];assert command.startswith(': && ') and command.endswith(' && :')
argv=shlex.split(command[5:-5]);assert argv.count('libstrata_kernels_cpu.a')==1
exe=base/'strata-current-tasks9'
argv[argv.index('-o')+1]=str(exe)
argv[argv.index('libstrata_kernels_cpu.a')]=str(out/'libstrata_kernels_cpu.a')
src=root/'sycl/tools/recover-xe.sh';mod=types.ModuleType('readonly')
exec(compile(src.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(src),'exec'),mod.__dict__)
env=mod.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
env.update(MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib')
env['PATH']='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin'
record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source_revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
 scope='Current unchanged SYCL/engine objects relinked with the previously byte-validated CPU task-factor-9 archive; build only, GPU comparison pending',
 link_argv=argv,default_binary_sha256=expected,original_cpu_archive_sha256=digest(build/'libstrata_kernels_cpu.a'),
 candidate_cpu_archive_sha256=digest(out/'libstrata_kernels_cpu.a'),
 source_pool_sha256=digest(root/'src/kernels/cpu/pool.cpp'),
 unchanged_libraries={name:digest(build/name) for name in argv if name.endswith('.a') and name!=str(out/'libstrata_kernels_cpu.a')},
 program_object_sha256=digest(build/'CMakeFiles/strata.dir/src/program/generate.cpp.o'),
 controller_sha256=digest(Path(__file__)),build_passed=False,gpu_verified=False)
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 with (out/'link.stdout').open('wb') as so,(out/'link.stderr').open('wb') as se:
  r=subprocess.run(argv,cwd=build,env=env,stdout=so,stderr=se,timeout=300)
 record['link_exit_code']=r.returncode;assert r.returncode==0
 record['candidate_binary_sha256']=digest(exe)
 deps=subprocess.run(['/usr/bin/ldd',str(exe)],capture_output=True,text=True,env=env,timeout=20)
 (out/'runtime-link.txt').write_text(deps.stdout+deps.stderr)
 assert deps.returncode==0 and 'not found' not in deps.stdout
 assert digest(build/'strata')==expected and digest(base/'strata-residency-candidate')==expected
 record['build_passed']=True
except BaseException as exc:record['error']=repr(exc)
finally:
 record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 print(json.dumps({k:v for k,v in record.items() if k not in ['link_argv','unchanged_libraries']},indent=2))
if not record['build_passed']:raise SystemExit(1)
