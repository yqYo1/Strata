
from pathlib import Path
import fcntl,json,hashlib,os,types,re,sys
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq4nl-caller-census-20261010')
OUT=B/'iq4nl-caller-census-host-contract-v1'
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
def ident(p):
 with Path(p).open('rb') as s:h=hashlib.file_digest(s,'sha256').hexdigest()
 return dict(bytes=Path(p).stat().st_size,sha256=h)
def main():
 with (B/'owned-v0141-measurement.lock').open('a') as l:
  fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
  mod=types.ModuleType('owned_host_contract');exec(compile(PARENT.read_text().split('\ndef parse_probe(',1)[0],str(PARENT),'exec'),mod.__dict__);mod.W=W
  assert not OUT.exists();OUT.mkdir();owner=mod.Owner(OUT)
  header=W/'sycl/src/prefill/iq4nl_caller_census.hpp';hp=ident(header);assert hp['sha256']=='a653161dabba83e6bd72157e2bcfbf504912c8bcc9d89f51a31aaa99b34903b3'
  src=B/'iq4nl_caller_census_host_contract_v1.cpp'
  record=dict(active=True,complete=False,passed=False,source_header=dict(path=str(header),**hp),source_fixture=ident(src),controller=ident(__file__),commands=owner.commands,GPU_executed=False,model_opened=False,full_physical_context_qualified=False,started_utc=mod.utc())
  def save():
   p=OUT/'record.json.tmp';p.write_text(json.dumps(record,indent=2)+'\n');p.replace(OUT/'record.json')
  owner.persist=save;env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
  def run(label,args):
   e,o,err=owner.run(label,args,env,wall=60,text_cap=1<<20,file_cap=1<<20);save();assert mod.completed(e) and e['exit_code']==0,(label,e);return o,err
  try:
   binary=OUT/'host-contract'
   run('compile',['/usr/bin/g++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-I'+str(header.parent),str(src),'-o',str(binary)])
   out,err=run('host-contract',[str(binary)])
   text=err.read_text();assert 'PASS' not in text and 'async_success=UNIMPLEMENTED' in text
   sections={}
   for section in text.split('HOST_CASE,')[1:]:
    label,body=section.split('\n',1);summary=[s for s in body.splitlines() if ' census scope=' in s];assert len(summary)==1 and label not in sections
    fields=dict(v.split('=',1) for v in summary[0].split()[3:]);sections[label]=fields
   assert len(sections)==14
   for label in ['valid32k','full_count_layermajor_HOST_SIMULATION']:
    d=sections[label]
    for key in ['invalid','overflow','unsupported','incomplete']:assert d[key]=='0',(label,key,d)
    assert d['gate']=='UNQUALIFIED'
   for label in ['bounds','chunk_order','private_wrong_type','type20_extent','shape_change','returned_without_selected','rows_mismatch']:assert sections[label]['invalid']=='1',label
   for label in ['add_overflow','mul_overflow']:assert sections[label]['overflow']=='1',label
   assert sections['missing_return']['incomplete']=='1' and sections['missing_return']['invalid']=='0'
   assert sections['unsupported']['unsupported']=='1'
   assert sections['pipeline_incomplete']['incomplete']=='1'
   expected='HOST_CONTRACT,cases=14,ledger_bytes=2360872,expert_bytes=96,layer_bytes=32,no_GPU=1,full_count_is_host_simulation=1\n'
   # ABI size is measured, not guessed: parse then reconcile the fixed arrays plus residual fields.
   match=re.fullmatch(r'HOST_CONTRACT,cases=14,ledger_bytes=(\d+),expert_bytes=(\d+),layer_bytes=(\d+),no_GPU=1,full_count_is_host_simulation=1\n',out.read_text());assert match
   ledger,expert,layer=map(int,match.groups());assert 24576*expert+48*layer<=ledger<4<<20
   record.update(passed=True,complete=True,cases=14,compiled_ledger_bytes=ledger,compiled_expert_bytes=expert,compiled_layer_bytes=layer,sections=sections,binary=dict(path=str(binary),**ident(binary)))
  except BaseException as e:record['error']=repr(e)
  finally:
   if ident(header)!=hp:record['passed']=False;record['final_pin_error']='header changed'
   record.update(active=owner.active is not None,finished_utc=mod.utc());record['passed']=record['passed'] and not record['active'] and all(mod.completed(c) for c in owner.commands);save()
   print(json.dumps({k:record.get(k) for k in ['passed','complete','active','cases','compiled_ledger_bytes','error']}),flush=True)
  return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
