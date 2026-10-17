#!/usr/bin/env python3
"""Whole-call QSA SLM row-pitch timing and full-bit checks; all arms explicit parameters."""
import fcntl,hashlib,importlib.util,json,os,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
BOOT='cfda0b6a-7c13-4fe2-bdf5-8f9202770840';DEV=Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
ADAPTED_FROM_SHA256='8092cfa8b6a6bbf6a3546364de35bb948518f1d08741c4f5d6578060a30a19c2'
src_path=A/'run_qsa_bf16_prefix_subgroup_v1.py'
HOST=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
def pin(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=h)
def main():
 assert len(sys.argv)==2 and sys.argv[1] in ['tiny','perf'];mode=sys.argv[1];backend='all'
 root=B/f'main-qsa-kpitch-all-{mode}-v1';dst=A/f'own-qsa-kpitch-all-{mode}-v1.json';assert not root.exists() and not dst.exists()
 buildpath=A/'own-qsa-kpitch-build-v1.json';build=json.loads(buildpath.read_text());assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
 predecessors=[build];prerequisite_paths=[]
 for arm in ['baseline','k','v','both','wide']:
  path=A/f'own-qsa-kpitch-{arm}-component-v1.json';d=json.loads(path.read_text());assert d['passed'] and d['complete'] and d['ownership_closed'] and not d['active'] and d['fresh_faults']==[] and not d['driver_dump_entries'] and d['boot_unchanged'];predecessors.append(d);prerequisite_paths.append(path)
 if mode=='perf':
  path=A/'own-qsa-kpitch-all-tiny-v1.json';d=json.loads(path.read_text());assert d['passed'] and d['complete'] and d['ownership_closed'] and not d['active'] and d['fresh_faults']==[] and d['binary']==build['perf_binary'];predecessors.append(d);prerequisite_paths.append(path)
 admission_path=A/'qsa-kpitch-source-admission-v1.json';admission=json.loads(admission_path.read_text());assert admission['passed'] and admission['bounded_GPU_checks_authorized'] and not admission['adoption']
 for p,v in admission['frozen_inputs'].items():assert pin(p)==v,p
 base=A/'own-vllm-resource-timeline-v3/manifest.json';assert pin(base)['sha256']=='61da9469e3f975c72d770beee77d45118c988eb6d3d82b6dbe1b568cd9ea3f1c'
 allowed=dict(json.loads(base.read_text())['elf_pins']);binary=build['perf_binary'];allowed[binary['realpath']]=binary
 frozen={**allowed,**build['frozen_files']}
 for p in [admission_path,buildpath,Path(__file__),src_path]:v=pin(p);frozen[v['path']]=v
 for path in prerequisite_paths:frozen[str(path)]=pin(path)
 ownerpath=A/'direct_owner_transient_artifacts_v1.py';assert pin(ownerpath)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
 hostcheck=A/'transient-artifact-owner-check-v1.json';hc=json.loads(hostcheck.read_text());assert hc['passed'] and hc['complete'] and hc['helper_sha256']==pin(ownerpath)['sha256']
 for q in [ownerpath,hostcheck,A/'qsa-bf16-prefix-subgroup-contract-v1.ja.md']:
  frozen[str(q)]=pin(q)
 package=A/'own-vllm-xe2-qualification-v1';metadata=package/'manifest.json'
 assert pin(metadata)['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
 meta=json.loads(metadata.read_text());helper=package/'main_controller.py'
 assert pin(helper)['sha256']==meta['package_files']['main_controller.py']['sha256']
 spec=importlib.util.spec_from_file_location('subgroup_fixture_guards',helper);guards=importlib.util.module_from_spec(spec);spec.loader.exec_module(guards)
 for q in [metadata,helper]:frozen[str(q)]=pin(q)
 assert pin(src_path)['sha256']==ADAPTED_FROM_SHA256
 spec=importlib.util.spec_from_file_location('qsa_bf16_native_owner',ownerpath);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # CPU benchmarks also contend for host submission and memory bandwidth.
  resource_guard=guards.no_model_owners(meta['known_model_files'])
  guards.driver_guard()
  for predecessor in predecessors:
   for command in predecessor['commands']:
    assert m.closed(command) and command['exit_code']==0
    for old in command['owners']+[command['direct_identity']]:
     now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
  root.mkdir(mode=0o700)
  env={**HOST,'LD_LIBRARY_PATH':str(L/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu',
   'UR_ADAPTERS_FORCE_LOAD':str(L/'lib/libur_adapter_level_zero_v2.so.0'),'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1',
   'NEOReadDebugKeys':'1','EnableDirectSubmission':'0','STRATA_SYCL_PREFILL_SAME_QUEUE':'1','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0',
   'SYCL_CACHE_PERSISTENT':'0','ZES_ENABLE_SYSMAN':'0','XDG_CACHE_HOME':str(root/'cache'),'STRATA_TEST_MAIN_GATES':'1',
   'UR_ENABLE_LAYERS':'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION','UR_LOG_TRACING':'level:info;flush:info;output:stderr',
   'UR_LOG_VALIDATION':'level:warning;flush:warning;output:stderr','UR_LOG_LOADER':'level:warning;flush:warning;output:stderr','UR_LOG_LEVEL_ZERO':'level:warning;flush:warning;output:stderr'}
  if mode=='perf':
   env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'
   env.pop('UR_LOG_TRACING')
  rec=dict(schema='strata-qsa-kpitch-perf-v1',backend=backend,mode=mode,passed=False,complete=False,active=False,GPU_executed=False,commands=[],snapshots=[],
   boot_id=BOOT,binary=binary,build=pin(buildpath),runtime_elf_pins=allowed,native_environment=env,source=pin(__file__),
   scope='Same current row-major arithmetic, SLM row-pitch span1/4/256 explicit host parameters within one process; no setenv. Fresh own random/clustered masked fixtures with8192queries. Admission+prefix pack+attention+checked drain wall timing, H2D/input setup and all D2H/output checks excluded. All output BF16 bits and input/guard checks required. No physical bandwidth/resource saturation, model quality/PP or RESTORE claim.',
   adapted_from=dict(path=str(src_path),sha256=ADAPTED_FROM_SHA256),resource_guard=resource_guard,
   numerical_contract=pin(A/'qsa-bf16-prefix-subgroup-contract-v1.ja.md'),
   production_selected=False,model_execution=False,model_PP_measured=False,physical262144_model_qualified=False,RESTORE_qualified=False,
   budget=dict(native_wall_seconds=600,child_CPU_seconds=300,session_RSS_bytes=4<<30,raw_text_bytes=64<<20),
   retention='Keep compact oracle/runtime/ownership/fault facts. Retire successful fullAPI logs once committed and no current consumer.')
  owner=None
  def persist():
   rec['active']=bool(owner and owner.active is not None);(root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);rec['commands']=owner.commands
  def run(label,argv,current=HOST,interaction=None,wall=30):
   c,so,se=owner.run(label,argv,current,P,interaction=interaction,wall=wall,cpu=300,rss_cap=4<<30,text_cap=64<<20,file_cap=64<<20,total_cap=128<<20)
   assert m.closed(c),(label,'ownership');return c,so,se
  def journal(label,args):
   c,so,se=run(label,['/usr/bin/journalctl',*args]);assert c['exit_code']==0;return so.read_text()
  def verify():
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
   assert DEV.joinpath('driver').resolve().name=='xe' and not list(DEV.glob('devcoredump*'))
   for p,v in frozen.items():assert pin(v['path'])==v,p
  def snapshot(io,entry,phase):
   identity=m.identity(io.proc.pid);assert identity and identity['start_ticks']==entry['direct_identity']['start_ticks']
   proc=Path('/proc')/str(io.proc.pid);assert str(proc.joinpath('exe').resolve())==binary['realpath']
   observed=dict(x.decode().split('=',1) for x in proc.joinpath('environ').read_bytes().split(b'\0') if x);assert observed==env
   maps=[];seen=set()
   for line in proc.joinpath('maps').read_text().splitlines():
    f=line.split(None,5)
    if len(f)!=6 or 'x' not in f[1] or not f[5].startswith('/'):continue
    assert not f[5].endswith(' (deleted)');p=Path(f[5]).resolve(strict=True);assert str(p) in allowed,'unreviewedELF '+str(p)
    v=allowed[str(p)]
    if str(p) not in seen:assert pin(p)==v;seen.add(str(p))
    st=p.stat();major,minor=[int(x,16) for x in f[3].split(':')];assert st.st_ino==int(f[4]) and os.major(st.st_dev)==major and os.minor(st.st_dev)==minor
    maps.append(dict(mapping=line,pin=v))
   assert binary['realpath'] in seen
   if phase!='before-runtime':
    for p in [L/'lib/libsycl.so.9',L/'lib/libur_adapter_level_zero_v2.so.0',Path('/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395')]:assert str(p.resolve()) in seen
   rec['snapshots'].append(dict(phase=phase,identity=identity,initial_exec_environment=observed,mappings=maps));io.send(b'X')
  cursor=None
  try:
   verify()
   c,so,se=run('readelf',['/usr/bin/readelf','-d',binary['path']]);assert c['exit_code']==0
   rec['dynamic']=so.read_text()
   needed=re.findall(r'Shared library: \[([^\]]+)\]',rec['dynamic'])
   assert set(needed)=={'libsycl.so.9','libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6'},needed
   verify();raw=journal('journal-before',['-k','-b','-n','0','--show-cursor','--no-pager']);cursor=re.search(r'-- cursor: (\S+)',raw)[1];rec['journal_before_cursor']=cursor
   holds=[];lines=[]
   def interaction(io,entry):
    for _ in range(200):
     line=io.line(seconds=90);lines.append(line)
     if line.startswith('QSA_BF16_READY '):
      phase=line.split(' ',1)[1];assert phase==['before-runtime','runtime-ready','after-cleanup'][len(holds)];snapshot(io,entry,phase);holds.append(phase)
     if line.startswith('fixture_status=PASS '):
      assert len(holds)==3;io.exit(seconds=10);return
    raise RuntimeError('missing bounded completion')
   rec['GPU_executed']=True
   c,so,se=run('qsa-bf16',[binary['path'],'--'+mode],env,interaction,wall=600)
   rec.update(native_exit_code=c['exit_code'],stdout=so.read_text(),stderr_sha256=pin(se)['sha256'],progress=lines);assert c['exit_code']==0
   assert any(re.fullmatch(r'device=.*B570.* driver=1\.17\.39395\+14',line) for line in lines)
   text=se.read_text();assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|fixture_status=FAIL|cleanup failed',text)
   count=len(re.findall(r'^\s*---> urEnqueueKernelLaunch(?:WithArgsExp)?\s*$',text,re.M));rec['UR_kernel_submissions']=count;assert count>0 or mode=='perf'
   cases=[line for line in lines if line.startswith('case ')];assert len(cases)==(1 if mode=='tiny' else 3)
   for line in cases:assert all(x in line for x in ['full_bits=PASS','guards=PASS','inputs=PASS','status=PASS'])
   samples=[line for line in lines if line.startswith('sample ')];assert len(samples)==(30 if mode=='tiny' else 90)
   for line in samples:assert all(x in line for x in ['exact_bits=PASS','guards=PASS','status=PASS'])
   rec['individual_samples']=samples
   rec.update(cases=cases,passed=True,complete=True);verify()
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc(),passed=False,complete=False)
  finally:
   try:
    assert owner.active is None
    if cursor:
     end=journal('journal-end',['-k','-b','-n','0','--show-cursor','--no-pager']);rec['journal_after_cursor']=re.search(r'-- cursor: (\S+)',end)[1]
     raw=journal('journal-final',['-k','-b','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(x) for x in raw.splitlines() if x.strip()]
     assert all(x.get('_BOOT_ID')==BOOT.replace('-','') for x in rows);rec['fresh_faults']=[x for x in rows if FAULT.search(x.get('MESSAGE',''))]
    rec['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT;rec['driver_dump_entries']=[str(x) for x in DEV.glob('devcoredump*')];rec['ownership_closed']=all(m.closed(c) for c in owner.commands)
    for c in owner.commands:
     for old in c['owners']+[c['direct_identity']]:
      now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
    if not rec['ownership_closed'] or not rec['boot_unchanged'] or rec['driver_dump_entries'] or 'fresh_faults' not in rec or rec['fresh_faults']:rec.update(passed=False,complete=False)
   except BaseException as e:rec.update(closure_error=repr(e),passed=False,complete=False)
   persist();dst.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(dst),passed=rec['passed'],complete=rec['complete'],native_exit=rec.get('native_exit_code'),GPU_executed=rec['GPU_executed'],error=rec.get('error'),closure_error=rec.get('closure_error'))))
  return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
