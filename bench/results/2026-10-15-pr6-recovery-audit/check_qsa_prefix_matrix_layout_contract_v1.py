#!/usr/bin/env python3
"""Main serialized compile-only exact-source Q/KV/P layout proof."""
import fcntl,hashlib,importlib.util,json
from pathlib import Path
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def pin(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
root=B/'main-qsa-prefix-matrix-layout-contract-v1';out=A/'own-qsa-prefix-matrix-layout-contract-v1.json'
assert not root.exists() and not out.exists()
build_path=A/'own-qsa-bf16-prefix-optimized-engine-build-v1.json';build=json.loads(build_path.read_text());assert build['passed'] and build['complete'] and build['ownership_closed']
helper=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py';assert pin(helper)['sha256']=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec=importlib.util.spec_from_file_location('matrix_contract_owner',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=A/'qsa_prefix_matrix_layout_contract_v1.cpp';compiler=Path(next(x.split('=',1)[1] for x in build['configure_argv'] if x.startswith('-DCMAKE_CXX_COMPILER=')))
r=dict(passed=False,complete=False,GPU_executed=False,active=False,build_receipt=pin(build_path),probe=pin(source),controller=pin(__file__),commands=[],scope='Exact current production template instantiations: BF16sizeof2; Q/KV/P standard layout,trivially copyable,align>=4,memberoffset0/mod4,exactarraybytes; both host/device syntax compilation. Not IGC codegen, runtime address observation, speed or full-context/RESTORE.')
def verify():
 for p,v in build['frozen_files'].items():assert pin(p)['sha256']==v['sha256'],p
 assert pin(source)==r['probe'] and pin(__file__)==r['controller']
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
 def save():
  r['active']=owner.active is not None;(root/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(root/'owned',save);r['commands']=owner.commands
 argv=[str(compiler),'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1','-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror','-fsyntax-only',str(source)]
 r['argv']=argv;r['environment']=build['environment']
 try:
  verify();c,so,se=owner.run('layout-assertions',argv,build['environment'],P,wall=180,cpu=300,rss_cap=4<<30,text_cap=8<<20,total_cap=16<<20,file_cap=8<<20)
  r.update(stdout=so.read_text(),stderr=se.read_text());assert m.closed(c) and c['exit_code']==0;verify();r.update(passed=True,complete=True)
 except BaseException as e:r['error']=repr(e)
 finally:
  r['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands);save();out.write_bytes((root/'record.json').read_bytes())
print(json.dumps(dict(passed=r['passed'],complete=r['complete'],error=r.get('error'),GPU_executed=False)))
raise SystemExit(0 if r['passed'] else 1)
