#!/usr/bin/env python3
"""Compile-only narrow namespace-scope kernel-name compatibility probe."""
import fcntl,hashlib,importlib.util,json,shlex,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 root=B/'main-pr6-provider-forward-compile-v1';target=A/'own-pr6-provider-forward-compile-v1.json';assert not root.exists() and not target.exists()
 cp=A/'own-pr6-harness-configure-v4.json';assert sha(cp)=='111d8c1728c4ab78d975272b68202bb5a53eec0eaf39c0218673e32c69b708fb'
 config=json.loads(cp.read_text());assert config['passed'] and config['ownership_closed']
 owner_path=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py';assert sha(owner_path)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 spec=importlib.util.spec_from_file_location('pr6_forward_compile_owner',owner_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 build=Path(config['build_path']);rows=json.loads((build/'compile_commands.json').read_text());row=next(r for r in rows if r['file'].endswith('/harness/provider_probe.cpp'))
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
  source=root/'kernel-forward.hpp';source.write_text('// Independent project compatibility declarations, repository LICENSE.\nnamespace dpct { namespace detail { class dpct_memcpy_3d_detail; class dpct_memcpy_3d_detail_usmnone; } }\n')
  argv=shlex.split(row['command']);argv[argv.index('-o')+1]=str(root/'provider-forward.o');i=argv.index('-include');argv[i:i]=['-include',str(source)]
  record=dict(schema='strata-pr6-provider-forward-compile-v1',passed=False,complete=False,active=False,GPU_executed=False,commands=[],added_header_source=source.read_text(),added_header_sha256=sha(source),configuration_sha256=sha(cp),actual_compile_argv=argv,scope='Same provider TU and flags; only prior namespace-scope name declarations and newobjectoutput. Compile only, no link/runtime.')
  owner=None
  def persist():
   record['active']=bool(owner and owner.active is not None);(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
  try:
   for v in [*config['source_pins'].values(),*config['generated_inputs'],*config['compilers']]:assert sha(v['path'])==v['sha256'],v['path']
   c,out,err=owner.run('compile-provider-forward',argv,config['commands'][0]['environment'],build,wall=120,cpu=240,rss_cap=8<<30,text_cap=64<<20,file_cap=128<<20,total_cap=256<<20)
   record.update(exit_code=c['exit_code'],stdout=out.read_text(),stderr=err.read_text());assert m.closed(c) and c['exit_code']==0,(c['exit_code'],out.read_text()[-4096:],err.read_text()[-4096:])
   record.update(passed=True,complete=True,object_sha256=sha(root/'provider-forward.o'))
  except BaseException as e:record.update(error=repr(e))
  finally:
   record['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands);persist();target.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(target),passed=record['passed'],exit_code=record.get('exit_code'),GPU_executed=False,error=(record.get('error') or '')[:200])))
  return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
