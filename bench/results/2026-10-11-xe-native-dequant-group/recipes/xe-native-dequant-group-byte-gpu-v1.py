import argparse,fcntl,importlib.util,json,os,re,time,types
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');LLVM=W/'.tools/intel-llvm-v711/install'
O=B/'xe-llvm-v711-source-v5/direct_owner.py';H=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
DUMP=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['free','contrib-llvm'],required=True);p.add_argument('--matrix',choices=['smoke','small','full'],required=True);a=p.parse_args()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/('xe-native-dequant-group-byte-gpu-root-v1-'+a.mode+'-'+a.matrix);m.require(not out.exists(),'immutable GPU receipt');out.mkdir(mode=0o700)
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();wall=120 if a.matrix=='smoke' else (900 if a.matrix=='small' else 2400)
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),boot_id=boot,controller_sha256=m.sha(__file__),mode=a.mode,matrix=a.matrix,model_executed=False,adopted=False,performance_eligible=False,reset_executed=False,retry_count=0,gpu_submission_attempted=False,raw_text_budget_bytes=64<<20,whole_wall_budget_seconds=wall+150,probe_wall_budget_seconds=wall,observed_libraries={},scope='Bounded real unchanged individual versus grouped native dequant byte equality, complete output and source guard preservation; not component timing or global model quality')
 def save():
  for c in r['commands']:
   if c['label']!='byte-matrix' or not c.get('direct_identity'):continue
   ident=c['direct_identity'];now=m.identity(ident['pid'])
   if not now or now['start_ticks']!=ident['start_ticks']:continue
   try:
    for line in Path('/proc',str(ident['pid']),'maps').read_text().splitlines():
     at=line.find('/')
     if at<0:continue
     path=line[at:]
     if '(deleted)' in path or not any(n in Path(path).name for n in ['libsycl','libur_','libze_','libumf']):continue
     if path not in r['observed_libraries']:r['observed_libraries'][path]=dict(path=path,sha256=m.sha(Path(path)))
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;begin=time.monotonic();cursor=None;helper=None
 BASE=dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib')+(':/opt/intel/oneapi/mkl/2026.1/lib' if a.mode!='free' else ''),XDG_RUNTIME_DIR='/run/user/'+str(os.getuid()))
 def run(label,args,env=BASE,deadline=20):
  left=wall+150-(time.monotonic()-begin);m.require(left>0,'whole finite deadline')
  e,so,se=owner.run(label,args,env,W,wall=min(deadline,left),cpu=deadline*2,text_cap=64<<20,total_cap=64<<20,rss_cap=8<<30)
  m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label);return so.read_text(),se.read_text()
 def kernel(label,args):
  so,_=run(label,['/usr/bin/journalctl','-k',*args,'--no-pager','-o','json']);rows=[json.loads(s) for s in so.splitlines() if s.startswith('{')];relevant=[x for x in rows if '0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))]
  faults=[x['MESSAGE'] for x in relevant if helper.FAULT.search(x['MESSAGE'])];r[label]=dict(relevant_entries=relevant,faults=faults);m.require(not faults and not DUMP.exists(),'fault-free '+label)
 save()
 try:
  paths=dict(release=B/('xe-e32-'+a.mode+'-full-build-root-v17/record.json'),asan=B/('xe-e32-'+a.mode+'-asan-full-build-root-v14/record.json'),lint=B/'xe-e32-lints-baseline-compare-root-v12/record.json',probe_build=B/('xe-native-dequant-group-byte-build-root-v1-'+a.mode+'/record.json'),health=B/'xe-source-llvm-small-health-root-v3/record.json')
  proofs={}
  for name,path in paths.items():
   v=json.loads(path.read_text());m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'closed '+name);proofs[name]=v
  r['prerequisite_receipt_pins']={name:dict(path=str(path),sha256=m.sha(path)) for name,path in paths.items()}
  m.require(proofs['release']['source_pins']==proofs['asan']['source_pins']==proofs['probe_build']['source_pins'],'same compiled release/ASAN/probe source')
  r['source_pins']=proofs['release']['source_pins'];m.require(all(m.sha(W/f)==h for f,h in r['source_pins'].items()),'exact production source')
  m.require(all(m.sha(W/f)==h for f,h in proofs['lint']['source_pins'].items()),'exact mandatory lint source');m.require(proofs['health']['boot_id']==boot,'same qualified health boot')
  r['binary']=proofs['probe_build']['binary'];m.require(m.sha(Path(r['binary']['path']))==r['binary']['sha256'],'exact built probe');health=proofs['health'];m.require(m.sha(Path(health['binary']['path']))==health['binary']['sha256'],'actual health binary');r['health_binary']=health['binary']
  rh=B/'xe-source-llvm-census32k-v3/runtime.py';m.require(m.sha(rh)==json.loads((B/'xe-source-llvm-census32k-root-v3/counts/record.json').read_text())['runtime_helper_sha256'],'qualified exact runtime validator');rs=importlib.util.spec_from_file_location('runtime_validator',rh);runtime=importlib.util.module_from_spec(rs);rs.loader.exec_module(runtime);r['runtime_validator_sha256']=m.sha(rh)
  m.require(m.sha(H)=='51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457','exact read-only health helper');helper=types.ModuleType('diagnostic_helper');exec(compile(H.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(H),'exec'),helper.__dict__)
  adapter=health['adapter'];m.require(m.sha(Path(adapter['path']))==adapter['sha256'],'exact source adapter')
  env=dict(BASE,ONEAPI_DEVICE_SELECTOR='level_zero:gpu',STRATA_GPU_PCI='0000:05:00.0',STRATA_EVENT_MARK='barrier',SYCL_CACHE_PERSISTENT='0',UR_ADAPTERS_FORCE_LOAD=adapter['path'],UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0')
  health_env=helper.diagnostic_environment(env);r['environment']=health_env if a.matrix=='smoke' else env;r['health_environment']=health_env
  kernel('kernel-before',['-b']);so,_=run('fresh-kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager']);match=re.search(r'^-- cursor: (.+)$',so,re.M);m.require(match is not None,'fresh cursor');cursor=match[1];r['kernel_cursor_before']=cursor
  def health_run(label):
   text,_=run(label,[health['binary']['path'],'0000:05:00.0'],env=health_env,deadline=30)
   expected=['STATUS 0000:05:00.0 0','QUEUE 0000:05:00.0','H2D 65536 complete','KERNEL 16384 complete','D2H 65536 complete','PASS 0000:05:00.0: 1 round, 16384 exact words'];m.require(text.splitlines()[:6]==expected,'exact '+label);r[label]=text.splitlines()
  if a.matrix!='smoke':
   sp=B/('xe-native-dequant-group-byte-gpu-root-v1-'+a.mode+'-smoke/record.json');sm=json.loads(sp.read_text());m.require(sm['passed'] and sm['complete'] and not sm['active'] and sm['boot_id']==boot and sm['binary']==r['binary'] and all(m.closed(c) for c in sm['commands']),'same binary prior detailed smoke');r['smoke_receipt_sha256']=m.sha(sp)
  health_run('health-before');kernel('kernel-after-prehealth',['--after-cursor',cursor]);r['gpu_submission_attempted']=True;save()
  so,se=run('byte-matrix',[r['binary']['path'],'--'+a.matrix],env=r['environment'],deadline=wall)
  m.require(not re.search(r'UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\bFAIL\b|\bDIFF\b|GPU completion unconfirmed|terminate called',so+'\n'+se),'no byte/runtime error')
  lines=so.splitlines();arms=1 if a.matrix=='smoke' else (144 if a.matrix=='full' else 72)
  final=[x for x in lines if x.startswith('PASS arms=')];m.require(len(final)==1,'exact one final PASS');r['final_protocol']=final[0]
  pat=r'PASS arms=(\d+) cases=(\d+) individual_calls=(\d+) group_calls=(\d+) refusals=(\d+) readbacks=(\d+) read_bytes=(\d+) tracing_qualified=0 timing_qualified=0';match=re.fullmatch(pat,final[0]);m.require(match is not None,'strict finite pass protocol')
  counts=list(map(int,match.groups()));m.require(counts[:5]==[arms,arms*128,arms*16,arms*128,arms*26] and counts[5]==arms*131 and counts[6]<=1<<40,'exact expected finite coverage')
  compares=[x for x in lines if x.startswith('BYTE_COMPARE ')];m.require(len(compares)==arms*130 and all(x.endswith('differences=0') for x in compares),'all full byte comparisons reported');r['byte_comparison_records']=compares
  m.require(lines.count('ARM_PASS')==arms and lines.count('CASE_PASS')==arms*128 and 'EMPTY_BEFORE_RUNTIME_PASS' in lines,'complete arms/cases/zero before Runtime');r['coverage']=dict(arms=arms,group_cases=arms*128,individual_reference_calls=arms*16,refusals=arms*26,empty_pre_runtime=34,readbacks=counts[5],readback_bytes=counts[6],full_byte_compares=len(compares))
  r['runtime_mapping_validation']=runtime.validate_runtime(r['observed_libraries'],health['loaded_libraries']);kernel('kernel-after-byte-matrix',['--after-cursor',cursor]);health_run('health-after');kernel('kernel-after-health',['--after-cursor',cursor]);r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:
  if helper and cursor and owner.active is None:
   try:kernel('kernel-final',['--after-cursor',cursor]);r['kernel_gate_passed']=True
   except BaseException as exc:r['kernel_gate_error']=repr(exc)
  r['source_stable']=all(m.sha(W/f)==h for f,h in r.get('source_pins',{}).items());r['binary_stable']=bool(r.get('binary') and m.sha(Path(r['binary']['path']))==r['binary']['sha256']);r['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip();r['active']=owner.active is not None;r['passed']=bool(r['complete'] and r.get('kernel_gate_passed') and r['source_stable'] and r['binary_stable'] and r['boot_unchanged'] and not r['active'] and not r.get('error') and not r.get('kernel_gate_error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['mode','matrix','passed','error','coverage','kernel_gate_error']}));raise SystemExit(0 if r['passed'] else 1)
