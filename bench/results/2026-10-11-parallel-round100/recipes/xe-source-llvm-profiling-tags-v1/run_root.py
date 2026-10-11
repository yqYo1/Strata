import argparse,fcntl,importlib.util,json,re,time,types
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011');LLVM=W/'.tools/intel-llvm-v711/install';O=B/'xe-llvm-v711-source-v5/direct_owner.py';H=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh');DUMP=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
s=importlib.util.spec_from_file_location('o',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['free','contrib-llvm'],required=True);a=p.parse_args();mode=a.mode;args=[];deadline=90
BASE=dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib')+(':/opt/intel/oneapi/mkl/2026.1/lib' if mode!='free' else ''))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 proofs={'build':B/('xe-e32-'+mode+'-full-build-root-v12/record.json'),'probe':B/'xe-source-llvm-profiling-tags-build-root-v2/record.json','supervision':B/'xe-source-llvm-profiling-tags-cpu-root-v1/record.json','asan':B/('xe-e32-'+mode+'-asan-full-build-root-v9/record.json'),'lint':B/'xe-e32-lints-baseline-compare-root-v7/record.json','health':B/'xe-source-llvm-small-health-root-v3/record.json','cap':B/'xe-free-host-capabilities-root-v3/record.json'}
 verified={}
 for key,path in proofs.items():
  v=json.loads(path.read_text());m.require(v['passed'] and v['complete'] and not v['active'],'closed '+key);verified[key]=v
 m.require(all(m.sha(Path(__file__).parent/p)==h for p,h in verified['supervision']['source_pins'].items()),'qualified controller/parser byte identities')
 m.require(verified['build']['source_pins']==verified['asan']['source_pins'],'same release/ASan source pins')
 m.require(all(m.closed(c) for proof in verified.values() for c in proof['commands']),'all prerequisite process owners closed')
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();m.require(verified['health']['boot_id']==boot and verified['cap']['boot_id']==boot,'same qualified boot')
 for key in ['build','asan','lint']:m.require(all(m.sha(W/p)==h for p,h in verified[key]['source_pins'].items()),'same qualified source '+key)
 m.require(all(m.sha(W/p)==h for p,h in verified['probe']['source_pins'].items()),'actual production timer helper components');binary=verified['probe']['binaries'][mode];m.require(m.sha(Path(binary['path']))==binary['sha256'],'qualified parity binary');out=B/'xe-source-llvm-profiling-tags-root-v1'/mode;m.require(not out.exists(),'immutable new case output');out.mkdir(parents=True)
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),proof_sha256={k:m.sha(v) for k,v in proofs.items()},mode=mode,case='production-timer-tags',binary=binary,boot_id=boot,gpu_submission_attempted=False,model_executed=False,reset_executed=False,retry_count=0,raw_log_budget_bytes=32<<20,total_wall_budget_seconds=300,scope='Eight finite queue timeline samples through actual production Event helper; real copied-word oracle and expected negative controls. No exclusive-kernel service, timestamp-domain calibration, physical bandwidth roofline, full model or context claim',source_revision=verified['probe']['source_revision'],production_source_pins=verified['probe']['source_pins'],probe_sha256=verified['probe']['probe_sha256'],observed_libraries={})
 def save():
  for c in r['commands']:
   if c['label']!='gpu-parity' or not c.get('direct_identity'):continue
   ident=c['direct_identity'];q=Path('/proc')/str(ident['pid'])
   try:
    state=q.joinpath('stat').read_text().split(') ',1)[1].split()
    if int(state[19])!=ident['start_ticks']:continue
    for line in q.joinpath('maps').read_text().splitlines():
     at=line.find('/')
     if at<0:continue
     path=line[at:]
     if '(deleted)' in path or not any(name in Path(path).name for name in ('libsycl','libur_','libze_','libumf','libonemath','libmkl_sycl')):continue
     if path not in r['observed_libraries']:r['observed_libraries'][path]=dict(path=path,sha256=m.sha(Path(path)))
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save();begin=time.monotonic();cursor=None
 def run(label,argv,env=BASE,wall=20):
  m.require(time.monotonic()-begin<300,'whole diagnostic deadline');e,so,se=owner.run(label,argv,env,W,wall=min(wall,300-(time.monotonic()-begin)),cpu=int(wall)*2,text_cap=32<<20,total_cap=32<<20,rss_cap=16<<30);m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label);return so.read_text(),se.read_text()
 helper=types.ModuleType('diagnostic_helper');m.require(m.sha(H)=='51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457','health helper');exec(compile(H.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(H),'exec'),helper.__dict__)
 try:
  m.require(not DUMP.exists(),'no preexisting dump');adapter=Path(verified['health']['adapter']['path']);m.require(m.sha(adapter)==verified['health']['adapter']['sha256'],'same source adapter')
  env=helper.diagnostic_environment(dict(BASE,ONEAPI_DEVICE_SELECTOR='level_zero:gpu',STRATA_GPU_PCI='0000:05:00.0',STRATA_EVENT_MARK='barrier',SYCL_CACHE_PERSISTENT='0',UR_ADAPTERS_FORCE_LOAD=str(adapter),UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0'));r['environment']=env
  text,_=run('kernel-before',['/usr/bin/journalctl','-k','-b','--no-pager','-o','json']);rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')];rows=[v for v in rows if '0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))];r['kernel_preflight_entries']=rows;m.require(not any(helper.FAULT.search(v['MESSAGE']) for v in rows),'no boot GPU fault')
  text,_=run('kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager']);match=re.search(r'^-- cursor: (.+)$',text,re.M);m.require(match is not None,'fresh cursor');cursor=match[1];r['kernel_cursor_before']=cursor;r['gpu_submission_attempted']=True;save()
  so,se=run('gpu-parity',[binary['path'],*args],env=env,wall=deadline);r['numeric_output']=so.splitlines();from protocol import validate;r['diagnostic']=validate(so,se,mode);r['complete_samples']=len(r['diagnostic']['samples']);r['production_helper_qualified']=True;r['native_test_passed']=True
  for x in verified['health']['loaded_libraries']:m.require(x['path'] in r['observed_libraries'] and r['observed_libraries'][x['path']]['sha256']==x['sha256'],'actual same runtime mapping '+x['path'])
  m.require(all(not ('/opt/intel/oneapi/' in x and any(k in Path(x).name for k in ['libsycl','libur_','libumf'])) for x in r['observed_libraries']),'no inherited oneAPI SYCL/UR/UMF')
  m.require(all(m.sha(Path(x['path']))==x['sha256'] for x in r['observed_libraries'].values()),'runtime/driver identities stable');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:
  if cursor and owner.active is None:
   try:
    text,_=run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')];rows=[v for v in rows if '0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))];r['kernel_device_entries']=rows;r['new_fault_messages']=[v['MESSAGE'] for v in rows if helper.FAULT.search(v['MESSAGE'])];m.require(not r['new_fault_messages'] and not DUMP.exists(),'no new GPU fault/dump');r['kernel_gate_passed']=True
   except BaseException as e:r['kernel_gate_error']=repr(e)
  r['source_stable']=all(m.sha(W/p)==h for p,h in verified['build']['source_pins'].items());r['binary_stable']=m.sha(Path(binary['path']))==binary['sha256'];r['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip();r['active']=owner.active is not None;r['passed']=bool(r['source_stable'] and r['binary_stable'] and r['complete'] and r.get('kernel_gate_passed') and r['boot_unchanged'] and not r['active'] and not r.get('error') and not r.get('kernel_gate_error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['case','passed','error','native_test_passed','new_fault_messages']}));raise SystemExit(0 if r['passed'] else 1)
