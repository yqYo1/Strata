import argparse,fcntl,importlib.util,json,os,re,time,types
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');LLVM=W/'.tools/intel-llvm-v711/install'
O=B/'xe-llvm-v711-source-v5/direct_owner.py';H=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
DUMP=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['free','contrib-llvm'],required=True);p.add_argument('--matrix',choices=['smoke-host','full-host','smoke-tags','full-tags'],required=True);a=p.parse_args()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/('xe-native-dequant-group-component-gpu-root-v1-'+a.mode+'-'+a.matrix);m.require(not out.exists(),'immutable GPU receipt');out.mkdir(mode=0o700)
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();wall=240 if a.matrix.startswith('smoke-') else 600
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),boot_id=boot,controller_sha256=m.sha(__file__),mode=a.mode,matrix=a.matrix,model_executed=False,adopted=False,performance_eligible=a.matrix.startswith('full-'),instrumented_tags=a.matrix.endswith('-tags'),reset_executed=False,retry_count=0,gpu_submission_attempted=False,raw_text_budget_bytes=64<<20,whole_wall_budget_seconds=wall+150,probe_wall_budget_seconds=wall,observed_libraries={},scope='Same-work individual/group GU+Down component: all7 recorded pairs640x2560 n1/8/16, six ABBA samples/arm, host-only or separate production tags; full output/source/padding guard oracle; no model performance/global quality inference')
 def save():
  for c in r['commands']:
   if c['label']!='component' or not c.get('direct_identity'):continue
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
  paths=dict(release=B/('xe-e32-'+a.mode+'-full-build-root-v17/record.json'),asan=B/('xe-e32-'+a.mode+'-asan-full-build-root-v14/record.json'),lint=B/'xe-e32-lints-baseline-compare-root-v12/record.json',probe_build=B/('xe-native-dequant-group-component-build-root-v1-'+a.mode+'/record.json'),health=B/'xe-source-llvm-small-health-root-v3/record.json')
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
  health_env=helper.diagnostic_environment(env);r['environment']=health_env if a.matrix.startswith('smoke-') else env;r['health_environment']=health_env
  kernel('kernel-before',['-b']);so,_=run('fresh-kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager']);match=re.search(r'^-- cursor: (.+)$',so,re.M);m.require(match is not None,'fresh cursor');cursor=match[1];r['kernel_cursor_before']=cursor
  def health_run(label):
   text,_=run(label,[health['binary']['path'],'0000:05:00.0'],env=health_env,deadline=30)
   expected=['STATUS 0000:05:00.0 0','QUEUE 0000:05:00.0','H2D 65536 complete','KERNEL 16384 complete','D2H 65536 complete','PASS 0000:05:00.0: 1 round, 16384 exact words'];m.require(text.splitlines()[:6]==expected,'exact '+label);r[label]=text.splitlines()
  full_path=B/'xe-native-dequant-group-byte-gpu-root-v3-free-full/record.json';qualified=json.loads(full_path.read_text());m.require(qualified['passed'] and qualified['complete'] and not qualified['active'] and qualified['boot_id']==boot and all(m.closed(c) for c in qualified['commands']),'complete same-boot all18 full-byte guard qualification')
  m.require(qualified['source_pins']==r['source_pins'] and qualified['coverage']['group_cases']==18432,'exact current API qualified');r['all18_full_byte_receipt_sha256']=m.sha(full_path)
  if a.matrix.endswith('-tags'):
   tag_path=B/'xe-source-llvm-profiling-tags-root-v1/record.json';tag=json.loads(tag_path.read_text());m.require(tag['passed'] and tag['complete'] and not tag['active'] and tag['boot_id']==boot and all(m.closed(c) for c in tag['commands']),'actual same-boot production tag qualification');m.require(all(m.sha(W/f)==h for f,h in tag['source_pins'].items()),'exact unchanged production timing helper');r['prior_tag_qualification_sha256']=m.sha(tag_path)
  if a.matrix.startswith('full-'):
   sp=B/('xe-native-dequant-group-component-gpu-root-v1-'+a.mode+'-smoke-'+a.matrix.split('-')[1]+'/record.json');sm=json.loads(sp.read_text());m.require(sm['passed'] and sm['complete'] and not sm['active'] and sm['boot_id']==boot and sm['binary']==r['binary'] and all(m.closed(c) for c in sm['commands']),'same binary detailed component smoke');r['smoke_receipt_sha256']=m.sha(sp)
  health_run('health-before');kernel('kernel-after-prehealth',['--after-cursor',cursor]);r['gpu_submission_attempted']=True;save()
  so,se=run('component',[r['binary']['path'],'--'+a.matrix],env=r['environment'],deadline=wall)
  m.require(not re.search(r'UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\bFAIL\b|\bDIFF\b|GPU completion unconfirmed|terminate called',so+'\n'+se),'no byte/runtime error')
  lines=so.splitlines();smoke=a.matrix.startswith('smoke-');tags=a.matrix.endswith('-tags');count=12 if smoke else 252;pair_count=1 if smoke else 7
  final=[x for x in lines if x.startswith('PASS samples=')];m.require(len(final)==1,'one exact component final protocol');r['final_protocol']=final[0]
  pat=r'PASS samples=(\d+) oracle_pairs=(\d+) individual_normal_returns=(\d+) group_normal_returns=(\d+) total_logical_expert_pairs=(\d+) readbacks=(\d+) read_bytes=(\d+) component_executed=1 model_executed=0 low_level_submission_count_proved=0';match=re.fullmatch(pat,final[0]);m.require(match is not None,'strict component protocol');counts=list(map(int,match.groups()));expected=[12,1,92,64,93,8] if smoke else [252,7,10724,1316,10899,112];m.require(counts[:6]==expected and counts[6]<=1<<40,'exact finite component work')
  actual=[]
  integer_fields={'gu_type','down_type','n','tags','index','block','position','repetitions','expert_pairs','logical_normal_return_api_calls','logical_output_bytes'}
  for line in lines:
   if not line.startswith('SAMPLE '):continue
   values={}
   for item in line.split()[1:]:
    name,value=item.split('=',1);m.require(name not in values,'unique sample fields');values[name]=int(value) if name in integer_fields else (value if name=='arm' else float(value))
   expected_keys=integer_fields|{'arm','host_submit_ms','host_submit_per_expert_us','host_submit_checked_drain_ms','host_submit_checked_drain_per_expert_us','host_mark_enclosure_ms','begin_mark_submit_ms','end_mark_submit_ms','completed_stream_tag_ms','completed_stream_tag_per_expert_us'}
   m.require(set(values)==expected_keys and values['arm'] in ['individual','group'] and values['tags']==int(tags),'strict sample shape and phase')
   import math
   m.require(all(math.isfinite(x) for x in values.values() if isinstance(x,float)),'finite sample metrics')
   m.require(values['repetitions']==4 and values['expert_pairs']==4*values['n'] and values['logical_output_bytes']==values['expert_pairs']*640*2560*6,'exact same-work sample volume')
   m.require(values['logical_normal_return_api_calls']==(8 if values['arm']=='group' else 8*values['n']),'logical wrapper normal-return count')
   m.require(values['host_submit_ms']>=0 and values['host_submit_checked_drain_ms']>=values['host_submit_ms'],'host clock order')
   m.require((values['completed_stream_tag_ms']>0 and values['completed_stream_tag_per_expert_us']>0) if tags else (values['completed_stream_tag_ms']==-1 and values['completed_stream_tag_per_expert_us']==-1),'separate instrumented tag fields')
   actual.append(values)
  m.require(len(actual)==count,'every individual sample recorded');r['samples']=actual
  pairs=[(18,20),(18,42),(21,20),(21,42),(22,20),(22,42),(23,20)][:pair_count];ns=[1] if smoke else [1,8,16]
  order=[(g,d,n,arm,ix,b,pos) for g,d in pairs for n in ns for b in range(3) for pos,(arm,ix) in enumerate([('individual',2*b),('group',2*b),('group',2*b+1),('individual',2*b+1)])]
  observed=[(x['gu_type'],x['down_type'],x['n'],x['arm'],x['index'],x['block'],x['position']) for x in actual];m.require(observed==order,'exact ABBA sequence and per-arm six samples')
  compares=[x for x in lines if x.startswith('BYTE_COMPARE ')];m.require(len(compares)==(7 if smoke else 105) and all(' differences=0 ' in x for x in compares),'all exact byte comparisons and guards')
  m.require(sum(x.startswith('PAIR_PASS ') for x in lines)==pair_count,'all pairs complete')
  ledger=[x for x in lines if x.startswith(('PAIR_BEGIN ','INPUT ','BYTE_COMPARE ','TIMING_BEGIN '))];ledger_path=out/'byte-comparison-ledger.txt';ledger_path.write_text('\n'.join(ledger)+'\n');r['byte_comparison_ledger']=dict(path=str(ledger_path),sha256=m.sha(ledger_path),bytes=ledger_path.stat().st_size)
  r['coverage']=dict(samples=count,samples_per_arm_per_pair_n=6,pairs=pair_count,n=ns,repetitions_per_sample=4,complete_byte_compares=len(compares),readbacks=counts[5],readback_bytes=counts[6],global_quality_qualified=False,model_speed_qualified=False)
  r['runtime_mapping_validation']=runtime.validate_runtime(r['observed_libraries'],health['loaded_libraries']);kernel('kernel-after-byte-matrix',['--after-cursor',cursor]);health_run('health-after');kernel('kernel-after-health',['--after-cursor',cursor]);r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:
  if helper and cursor and owner.active is None:
   try:kernel('kernel-final',['--after-cursor',cursor]);r['kernel_gate_passed']=True
   except BaseException as exc:r['kernel_gate_error']=repr(exc)
  r['source_stable']=all(m.sha(W/f)==h for f,h in r.get('source_pins',{}).items());r['binary_stable']=bool(r.get('binary') and m.sha(Path(r['binary']['path']))==r['binary']['sha256']);r['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip();r['active']=owner.active is not None;r['passed']=bool(r['complete'] and r.get('kernel_gate_passed') and r['source_stable'] and r['binary_stable'] and r['boot_unchanged'] and not r['active'] and not r.get('error') and not r.get('kernel_gate_error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['mode','matrix','passed','error','coverage','kernel_gate_error']}));raise SystemExit(0 if r['passed'] else 1)
