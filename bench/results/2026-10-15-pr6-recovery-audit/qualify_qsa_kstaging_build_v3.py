#!/usr/bin/env python3
"""Qualify already successfully compiled V2 products; preserve failed receipts."""
import datetime,fcntl,hashlib,importlib.util,json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 root=B/'main-qsa-kstaging-build-qualification-v3';dst=A/'own-qsa-kstaging-build-v3.json';assert not root.exists() and not dst.exists()
 prior=A/'own-qsa-kstaging-build-v2.json';d=json.loads(prior.read_text())
 assert not d['passed'] and not d['complete'] and d['ownership_closed'] and not d['active'] and not d['GPU_executed']
 assert 'assert str(obj) in linkmap' in d['traceback']
 op=A/'direct_owner_transient_artifacts_v1.py';assert pin(op)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a';m=module('kstaging_build_owner',op)
 package=A/'own-vllm-xe2-qualification-v1';mp=package/'manifest.json';assert pin(mp)['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58';meta=json.loads(mp.read_text());hp=package/'main_controller.py';assert pin(hp)['sha256']==meta['package_files']['main_controller.py']['sha256'];guard=module('kstaging_build_guard',hp)
 for c in d['commands']:
  assert m.closed(c) and c['exit_code']==0
  for old in c['owners']+[c['direct_identity']]:
   now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
 assert [c['label'] for c in d['commands']]==['candidate-compile','correctness-link','perf-link','readelf-correctness']
 compiled=B/'main-qsa-kstaging-build-v2'
 products={'binary':compiled/'sycl_qsa_kstaging','perf_binary':compiled/'sycl_qsa_kstaging_perf','object':compiled/'candidate.o'}
 frozen=dict(d['frozen_files'])
 for p in [prior,Path(__file__),*products.values(),compiled/'correctness.map',compiled/'perf.map']:frozen[str(p)]=pin(p)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.no_model_owners(meta['known_model_files']);guard.driver_guard();root.mkdir(mode=0o700)
  rec=dict(schema='strata-qsa-kstaging-build-v3',passed=False,complete=False,active=False,GPU_executed=False,commands=[],boot_id=d['boot_id'],frozen_files=frozen,environment=d['environment'],source_commit=d['source_commit'],provenance=d['provenance'],base_build=pin(prior),library_inputs=d['library_inputs'],build_plans=d['build_plans'],build_output=d['build_output'],original_failures_preserved=True,scope='V2 compiler/link commands all closed normally. V2 verifier incorrectly required the original fat object path in the final host link map; SYCL instead links its unbundled temporary host object. V3 verifies named exports fall inside that object text range, absence of original archive object, dynamic libraries, source/product hashes and complete ownership. No recompile or GPU execution; V1/V2 FAIL receipts remain unchanged.',production_selected=False,model_qualified=False,dynamic={},link_map_evidence={})
  owner=None
  def save():
   rec['active']=bool(owner and owner.active is not None);rec['ownership_closed']=bool(owner and owner.active is None and all(m.closed(c) for c in owner.commands));(root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',save);rec['commands']=owner.commands
  def verify():
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
   for p,v in frozen.items():assert pin(p)==v,p
  try:
   verify()
   exports=['qsa_bf16_prefix'+suffix+'_batch' for suffix in ['', '_optimized','_subgroup','_tilepack','_kstage']]
   for label,key in [('correctness','binary'),('perf','perf_binary')]:
    c,so,se=owner.run('readelf-'+label,['/usr/bin/readelf','-d',str(products[key])],d['environment'],P,wall=30,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=16<<20);assert m.closed(c) and c['exit_code']==0
    rec['dynamic'][label]=so.read_text();needed=re.findall(r'Shared library: \[([^\]]+)\]',so.read_text());assert set(needed)=={'libsycl.so.9','libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6'}
    text=(compiled/(label+'.map')).read_text();assert 'qsa_bf16_prefix.cpp.o' not in text
    sections=list(re.finditer(r'^ \.text\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S*/tmp/candidate-[^\s]+\.o)$',text,re.M));assert len(sections)==1
    section=sections[0];start=int(section[1],16);stop=start+int(section[2],16);symbols={}
    for name in exports:
     matches=list(re.finditer(r'^\s+0x([0-9a-f]+)\s+strata::kernels::'+re.escape(name)+r'\(',text,re.M));assert len(matches)==1
     address=int(matches[0][1],16);assert start<=address<stop;symbols[name]=address
    rec['link_map_evidence'][label]=dict(map=pin(compiled/(label+'.map')),unbundled_host_text=section[0],text_range=[start,stop],exports=symbols,original_archive_object_extracted=False)
   mapping={}
   for span in [1,4,256]:
    pairs=[((i//span)%32,(i//(span*32))*span+i%span) for i in range(8192)];assert len(set(pairs))==8192 and set(pairs)=={(c,d) for c in range(32) for d in range(256)};mapping[str(span)]='8192 unique complete in-bounds producer destinations; same d*32+cell consumer.'
   rec['producer_ownership']=mapping;verify();guard.driver_guard();rec.update({k:pin(p) for k,p in products.items()});rec.update(passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc())
  finally:
   rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();dst.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(dst),passed=rec['passed'],complete=rec['complete'],error=rec.get('error'))));return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
