"""Root-owned actual Xe SYCL TU/link; no device, model or CMake mutation."""
import fcntl,importlib.util,json,shlex,shutil
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-b570-prefill-publication-20261011')
OLD=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/eval-b570-20261010')
BUILD=OLD/'build/eval-b570-icpx-v1'
HERE=Path(__file__).resolve().parent
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
s=importlib.util.spec_from_file_location('qualified_owner',OWNER);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')
def pin(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=m.sha(p))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-prefill-failclosed-linked-root-v3';m.require(not out.exists(),'new output');out.mkdir(mode=0o700)
 record=dict(active=True,complete=False,passed=False,adopted=False,fork_pristine=False,gpu_executed=False,model_executed=False,
             full262144_qualified=False,started_utc=m.utc(),controller_sha256=m.sha(__file__))
 def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);record['commands']=owner.commands
 env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
          LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib',
          MKLROOT='/opt/intel/oneapi/mkl/2026.1')
 record['environment']=env
 def run(label,args,wall=15):
  e,so,se=owner.run(label,args,env,BUILD,wall=wall,text_cap=8<<20,rss_cap=6<<30,total_cap=64<<20,file_cap=64<<20,cpu=wall)
  m.require(m.closed(e) and e['exit_code']==0,'normal owned build '+label)
  return so
 save()
 try:
  cpu=B/'xe-prefill-failclosed-cpu-root-v3/record.json';j=json.loads(cpu.read_text())
  m.require(j['passed'] and j['complete'] and not j['active'],'root CPU gate')
  record['cpu_qualification']=pin(cpu)
  record['sources']={str(W/f):pin(W/f) for f in j['source_sha256']}
  for f,p in j['source_sha256'].items():m.require(m.sha(W/f)==p,'tested exact source')
  m.require(m.sha(BUILD/'strata')=='4e68a6865161b8838886cf7805ad689fe7a05e280f963abdec5e3010c52bd036','unchanged original fork binary')
  m.require(m.sha(OLD/'src/prefill/prefill.cpp')=='919c696192311066055077bdd0fc74769bfb50854e3437b6b387732acf6d532e','unchanged original source')
  for p in (OLD/'include').rglob('*'):
   if p.is_file():m.require(m.sha(p)==m.sha(W/'include'/p.relative_to(OLD/'include')),'shared header equivalence')
  record['compile_commands']=pin(BUILD/'compile_commands.json')
  cs=[v for v in json.loads((BUILD/'compile_commands.json').read_text()) if v['file']==str(OLD/'src/prefill/prefill.cpp')]
  m.require(len(cs)==1,'one actual original TU recipe')
  argv=shlex.split(cs[0]['command']);argv[argv.index('-o')+1]=str(out/'prefill.cpp.o')
  argv[argv.index('-c')+1]=str(W/'src/prefill/prefill.cpp')
  old_include='-I'+str(OLD/'include');m.require(argv.count(old_include)==1,'one project include')
  argv[argv.index(old_include)]='-I'+str(W/'include');argv+=['-MMD','-MF',str(out/'prefill.cpp.d')]
  record['effective_compile']=argv;record['compiler']=pin(argv[0])
  recipe=run('read-link-recipe',['/usr/bin/ninja','-t','commands','strata']).read_text().splitlines()[-1]
  m.require(recipe.startswith(': && ') and recipe.endswith(' && :'),'known generated link structure')
  link=shlex.split(recipe[5:-5]);link[link.index('-o')+1]=str(out/'strata')
  cached={}
  for a in link:
   if a.endswith(('.a','.o')):cached[str(BUILD/a)]=pin(BUILD/a)
  cached[str(BUILD/'lib/libonemath.so')]=pin((BUILD/'lib/libonemath.so').resolve())
  record['cached_dependencies_before']=cached
  run('compile-SYCL-prefill',argv,180);record['object']=pin(out/'prefill.cpp.o')
  dep=(out/'prefill.cpp.d').read_text().replace('\\\n',' ')
  paths=shlex.split(dep.split(':',1)[1]);record['actual_included_dependencies']={p:pin(p) for p in sorted(set(paths))}
  archive=out/'libstrata_prefill.a';shutil.copyfile(BUILD/'libstrata_prefill.a',archive)
  names=run('member-list',['/usr/bin/ar','t',str(archive)]).read_text().splitlines()
  m.require(names.count('prefill.cpp.o')==1,'one replaced member')
  before={name:pin(run('member-before-'+str(i),['/usr/bin/ar','p',str(archive),name])) for i,name in enumerate(names)}
  run('replace',['/usr/bin/ar','r',str(archive),str(out/'prefill.cpp.o')]);run('index',['/usr/bin/ar','s',str(archive)])
  comparisons=[]
  for i,name in enumerate(names):
   p=pin(run('member-after-'+str(i),['/usr/bin/ar','p',str(archive),name]))
   expected=record['object']['sha256'] if name=='prefill.cpp.o' else before[name]['sha256']
   m.require(p['sha256']==expected,'actual archive member comparison')
   comparisons.append(dict(member=name,before=before[name],after=p,replaced=name=='prefill.cpp.o'))
  record['member_comparison']=comparisons
  m.require(link.count('libstrata_prefill.a')==1,'one prefill link input')
  link[link.index('libstrata_prefill.a')]=str(archive);record['effective_link']=link
  run('link',link,180);record['binary']=pin(out/'strata');record['archive']=pin(archive)
  for p,v in cached.items():m.require(m.sha(p)==v['sha256'],'cached inputs stable')
  for p,v in record['actual_included_dependencies'].items():m.require(m.sha(p)==v['sha256'],'actual include stable')
  for p,v in record['sources'].items():m.require(m.sha(p)==v['sha256'],'source stable')
  record.update(complete=True,passed=True,linked=True,all_other_members_unchanged=True,cached_inputs_unchanged=True)
 except BaseException as exc:record['error']=repr(exc)
 finally:record['active']=owner.active is not None;record['finished_utc']=m.utc();save()
 print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'))))
 raise SystemExit(0 if record['passed'] else 1)
