#!/usr/bin/env python3
"""Main-owned host dependency inspection only; no metric/native GPU execution.
Repository LICENSE. The sealed proposal is unchanged; no automatic build/run.
"""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,shlex,sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
D=A/'own-metric-enumeration-v1'
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
LLVM=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
OWNER=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
SNAPSHOT='288a79e2c7397ad993575bdb4dcfaebfaa79e96b7b88f403d2219769bb3364f0'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
 p=Path(p).resolve(strict=True);s=p.stat()
 assert 'prefill-gemm-only-contract-root-v1' not in p.parts
 assert 'XeStrata' not in p.parts or p.is_relative_to(LLVM)
 return dict(path=str(p),realpath=str(p),bytes=s.st_size,sha256=sha(p))
def main():
 root=B/'main-metric-source-dependency-review-v1'
 target=A/'own-metric-dependency-review-v1.json'
 seal=A/'own-metric-build-seal-v1.json'
 assert not any(p.exists() for p in [root,target,seal]) and sha(D/'SHA256SUMS')==SNAPSHOT
 assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 for l in (D/'SHA256SUMS').read_text().splitlines():
  h,n=l.split('  ',1);assert sha(D/n)==h
 candidate=json.loads((D/'manifest-candidate.json').read_text())
 base=json.loads((A/'own-vllm-resource-timeline-v2/manifest.json').read_text())
 assert candidate['original_elf_pins']==base['elf_pins']
 for p,v in candidate['original_file_pins'].items():
  assert p in base['file_pins'] and base['file_pins'][p]==v
 for p,v in {**candidate['original_file_pins'],**candidate['input_pins']}.items():
  got=pin(p);assert all(got[k]==v[k] for k in ['realpath','bytes','sha256']),p
 spec=importlib.util.spec_from_file_location('metric_dependencies_owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
  record=dict(schema='strata-metric-host-dependency-review-v1',approved=False,active=False,commands=[],GPU_executed=False,reviewer='Main',source=pin(__file__),package_sha256=SNAPSHOT,original_pins_preserved=True,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
  owner=None
  def persist():
   record['active']=bool(owner and owner.active is not None)
   (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
  compiler=str(LLVM/'bin/clang++');header=str(B/'main-primary-unitrace-build-root-v1/build/level_zero')
  env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root),LD_LIBRARY_PATH=str(LLVM/'lib'))
  common=[compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-fno-sycl','-I'+header,str(D/'enumerate.cpp')]
  def run(label,args):
   c,out,err=owner.run(label,args,env,A.parents[2],wall=60,cpu=40,rss_cap=2<<30,text_cap=8<<20,total_cap=128<<20,file_cap=8<<20)
   assert m.closed(c) and c['exit_code']==0
   for old in c['owners']+[c['direct_identity']]:
    now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
   return out.read_text(),err.read_text()
  dep,_=run('host-header-dependencies',common+['-M','-MT','metric_enumeration'])
  _,driver=run('host-link-recipe',common+['/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0','-ldl','-o',str(root/'not-built'),'-###'])
  inputs=set(shlex.split(dep.replace('\\\n',' ').split(':',1)[1]))
  link_args=[]
  for line in driver.splitlines():
   if line.lstrip().startswith('"'):
    args=shlex.split(line);link_args.append(args)
    inputs.update(x for x in args if x.startswith('/') and Path(x).is_file())
  assert link_args and any(any(x.endswith('/ld') for x in args) for args in link_args)
  for name in ['libstdc++.so','libgcc_s.so','libc.so','Scrt1.o','crti.o','crtbeginS.o','crtendS.o','crtn.o']:
   text,_=run('resolve-'+name.replace('.','-'),[compiler,'-print-file-name='+name]);p=Path(text.strip());assert p.is_absolute() and p.is_file();inputs.add(str(p))
  pins={}
  original={**candidate['original_file_pins'],**candidate['input_pins'],**base['elf_pins']}
  by_real={v['realpath']:v for v in original.values()}
  for p in sorted(inputs):
   v=pin(p)
   if v['realpath'] in by_real:assert all(v[k]==by_real[v['realpath']][k] for k in ['bytes','sha256'])
   else:pins[v['realpath']]=v
  for p,v in {**candidate['original_file_pins'],**candidate['input_pins']}.items():
   got=pin(p);assert all(got[k]==v[k] for k in ['realpath','bytes','sha256'])
  record.update(approved=True,complete=True,ownership_closed=owner.active is None and all(m.closed(c) and c['exit_code']==0 for c in owner.commands),environment=env,header_dependency_output=dep,driver_output=driver,driver_argv=link_args,pins=pins,header_and_explicit_input_count=len(inputs),scope='Actual compiler -M headers, -### explicit compile/link inputs, resolved CRT and standard linker inputs; original executable ELF mappings stay pinned. Does not attest every dynamic linker search candidate or compiler-internal anonymous code. No original pin replaced; newly recorded host build inputs only.',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  assert record['ownership_closed'];persist()
  with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  candidate.update(status='MAIN_REVIEWED_SEALED',mode='build',run_id='enum-build-v1',boot_id=record['boot_id'],reviewed_package_sha256=SNAPSHOT,additional_dependency_review=dict(approved=True,report_path=str(target),report_sha256=sha(target),pins=pins))
  with seal.open('x') as f:json.dump(candidate,f,indent=2);f.write('\n')
  print(json.dumps(dict(approved=True,GPU_executed=False,extra_input_pins=len(pins),seal=str(seal),seal_sha256=sha(seal),review_sha256=sha(target))))
if __name__=='__main__':main()
