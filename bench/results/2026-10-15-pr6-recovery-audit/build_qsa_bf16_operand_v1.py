#!/usr/bin/env python3
"""Compile the new BF16 operand kernel only; no native execution or old build mutation."""
import fcntl, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent; P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
def pin(p):
 p = Path(p)
 with p.open('rb') as f: h = hashlib.file_digest(f, 'sha256').hexdigest()
 return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=h)
def main():
 root = B/'main-qsa-bf16-operand-build-v1'; dst = A/'own-qsa-bf16-operand-build-v1.json'
 assert not root.exists() and not dst.exists()
 compiler = pin(L/'bin/clang++'); assert compiler['sha256']=='1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
 ownerpath = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
 assert pin(ownerpath)['sha256']=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 priorpath = A/'own-prompt-softmax-build-v3.json'; prior = json.loads(priorpath.read_text())
 assert prior['passed'] and prior['complete'] and prior['ownership_closed']
 frozen = {str(P/n): pin(P/n) for n in prior['source_pins'] if n.startswith(('include/','src/'))}
 for n in prior['source_pins']:
  if str(P/n) in frozen: assert frozen[str(P/n)]['sha256']==prior['source_pins'][n],n
 for p in [P/'src/kernels/sycl/qsa_bf16_attention.cpp', P/'include/strata/kernels/qsa_bf16_attention.hpp', Path(__file__), ownerpath, priorpath, L/'bin/clang++']:
  v=pin(p); frozen[v['path']]=v
 spec=importlib.util.spec_from_file_location('qsa_bf16_compile_owner',ownerpath); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); root.mkdir(mode=0o700); (root/'tmp').mkdir(); (root/'cache').mkdir()
  rec=dict(schema='strata-qsa-bf16-operand-build-v1',passed=False,complete=False,active=False,GPU_executed=False,commands=[],frozen_files=frozen,compiler=compiler,
   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),production_selected=False,
   scope='Single new kernel translation unit, not linked into any engine or executed. Existing qualified archives remain immutable.')
  owner=None
  def persist():
   rec['active']=bool(owner and owner.active is not None); (root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist); rec['commands']=owner.commands
  def verify():
   for p,v in frozen.items(): assert pin(p)==v,p
  try:
   verify(); obj=root/'qsa_bf16_attention.o'
   env=dict(PATH=str(L/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(L/'lib'),TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
   argv=[compiler['path'],'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1','-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror','-c',str(P/'src/kernels/sycl/qsa_bf16_attention.cpp'),'-o',str(obj)]
   c,out,err=owner.run('compile',argv,env,P,wall=300,cpu=600,rss_cap=8<<30,text_cap=64<<20,file_cap=128<<20,total_cap=1<<30)
   rec.update(exit_code=c['exit_code'],stdout=out.read_text(),stderr=err.read_text()); assert m.closed(c) and c['exit_code']==0
   verify(); rec.update(object=pin(obj),passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   rec['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands); persist(); dst.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(dst),passed=rec['passed'],exit_code=rec.get('exit_code'),GPU_executed=False,error=rec.get('error'))))
  return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
