#!/usr/bin/env python3
"""Main-owned model-free resource sensitivity tests; all arms separate processes."""
import fcntl,hashlib,importlib.util,json,math,os,re,statistics,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2];B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
DEV=Path('/sys/bus/pci/devices/0000:05:00.0');BOOT='cfda0b6a-7c13-4fe2-bdf5-8f9202770840'
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
HOST=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
 p=Path(p);return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def check(p,v):
 p=Path(p);assert str(p.resolve())==v['realpath'] and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],str(p)
def main():
 assert len(sys.argv)==2 and sys.argv[1] in ('smoke','first','clean');mode=sys.argv[1]
 root=B/f'main-qsa-resource-{mode}-v2';target=A/f'own-qsa-resource-{mode}-v2.json';assert not root.exists() and not target.exists()
 ownerpath=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py';assert sha(ownerpath)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 base=A/'own-vllm-resource-timeline-v3/manifest.json';assert sha(base)=='61da9469e3f975c72d770beee77d45118c988eb6d3d82b6dbe1b568cd9ea3f1c'
 buildpath=A/'own-qsa-resource-build-v4.json';build=json.loads(buildpath.read_text());assert build['passed'] and build['complete'] and build['ownership_closed']
 binary=build['binary'];assert binary['sha256']=='a5a7afac5d41d8a9be6e0223b85ab55dacffca6327cef19fe7d4a56cfd212cb7'
 prerequisites=[]
 for prev in ([] if mode=='smoke' else ['smoke'] if mode=='first' else ['smoke','first']):
  p=A/f'own-qsa-resource-{prev}-v2.json';v=json.loads(p.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and v['binary']==binary;prerequisites.append(pin(p))
 allowed=dict(json.loads(base.read_text())['elf_pins']);allowed[binary['realpath']]=binary
 frozen={**allowed,**build['frozen_files']}
 for v in [pin(buildpath),pin(Path(__file__))]:frozen[v['realpath']]=v
 spec=importlib.util.spec_from_file_location('qsa_resource_native_owner',ownerpath);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
  env={**HOST,'LD_LIBRARY_PATH':str(L/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu',
   'UR_ADAPTERS_FORCE_LOAD':str(L/'lib/libur_adapter_level_zero_v2.so.0'),'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1',
   'NEOReadDebugKeys':'1','EnableDirectSubmission':'0','STRATA_SYCL_PREFILL_SAME_QUEUE':'1','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0',
   'SYCL_CACHE_PERSISTENT':'0','ZES_ENABLE_SYSMAN':'0','XDG_CACHE_HOME':str(root/'cache'),'STRATA_TEST_MAIN_GATES':'1',
   'STRATA_SYCL_PROMPT_DIRECT':'1','STRATA_SYCL_PROMPT_K_CELL_MAJOR':'0','STRATA_SYCL_PROMPT_NATIVE_KV_MIN':'0','STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP':'0'}
  traced=mode!='clean'
  if traced:env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',UR_LOG_TRACING='level:info;flush:info;output:stderr',UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
  specs=[dict(name='smoke',argv=['--smoke'],reps=0,grain=3,reuse='both')] if mode=='smoke' else [dict(name='first128',reuse='shared',grain=128,reps=1)] if mode=='first' else [dict(name=f'{arm}-r{rep}',reuse='replicated' if arm=='B' else 'shared',grain=8 if arm=='C' else 128,reps=10) for arm,rep in [('A',0),('B',0),('C',0),('C',1),('B',1),('A',1),('A',2),('C',2),('B',2)]]
  rec=dict(schema='strata-qsa-resource-controls-v2',mode=mode,passed=False,complete=False,active=False,GPU_executed=False,commands=[],cases=[],snapshots=[],
   boot_id=BOOT,binary=binary,build=pin(buildpath),prerequisites=prerequisites,runtime_elf_pins=allowed,native_environment=env,declared_order=specs,
   budgets=dict(native_wall_seconds=120 if mode!='smoke' else 90,child_cpu_seconds=240,raw_text_bytes=64<<20,private_bytes=128<<20,frame_bytes_per_process=7<<20),
   scope='Synthetic matched Mode1floatQSA, identicalvalues/math/allocation, varyaddressreuse orenqueuegrain. Not modelPP or hardwarephysicaltraffic/utilization.',model_execution=False,performance_adopted=False,physical262144_qualified=False,RESTORE_qualified=False,
   retention=dict(owner='Main',consumer='Full initial-frame cross-arm comparison and hardware resource sensitivity',next_review='At exactcohort comparison; retire successful API/fullframes with verified manifest.'))
  owner=None
  def persist():
   rec['active']=bool(owner and owner.active is not None);(root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);rec['commands']=owner.commands
  def run(label,argv,current=HOST,interaction=None,wall=30):
   c,so,se=owner.run(label,argv,current,P,interaction=interaction,wall=wall,cpu=240,rss_cap=4<<30,text_cap=64<<20,file_cap=64<<20,total_cap=128<<20)
   assert m.closed(c),(label,'ownership');return c,so,se
  def journal(label,args):
   c,so,se=run(label,['/usr/bin/journalctl',*args]);assert c['exit_code']==0;return so.read_text()
  def freeze():
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
   assert DEV.joinpath('driver').resolve().name=='xe' and not list(DEV.glob('devcoredump*'))
   for v in frozen.values():check(v['path'],v)
  def snapshot(io,entry,phase,label):
   ident=m.identity(io.proc.pid);assert ident and ident['start_ticks']==entry['direct_identity']['start_ticks']
   proc=Path('/proc')/str(io.proc.pid);assert str(proc.joinpath('exe').resolve())==binary['realpath']
   effective=dict(x.decode().split('=',1) for x in proc.joinpath('environ').read_bytes().split(b'\0') if x);assert effective==env
   maps=[];seen=set()
   for line in proc.joinpath('maps').read_text().splitlines():
    f=line.split(None,5)
    if len(f)!=6 or 'x' not in f[1] or not f[5].startswith('/'):continue
    assert not f[5].endswith(' (deleted)');p=Path(f[5]).resolve(strict=True);assert str(p) in allowed,'unreviewedELF '+str(p)
    v=allowed[str(p)]
    if str(p) not in seen:check(p,v);seen.add(str(p))
    st=p.stat();major,minor=[int(x,16) for x in f[3].split(':')];assert st.st_ino==int(f[4]) and os.major(st.st_dev)==major and os.minor(st.st_dev)==minor
    maps.append(dict(mapping=line,pin=v))
   assert binary['realpath'] in seen
   if phase!='before-runtime':assert str((L/'lib/libsycl.so.9').resolve()) in seen and str((L/'lib/libur_adapter_level_zero_v2.so.0').resolve()) in seen and '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395' in seen
   after=m.identity(io.proc.pid);assert after and after['start_ticks']==ident['start_ticks']
   rec['snapshots'].append(dict(case=label,phase=phase,identity=ident,effective_environment=effective,mappings=maps));io.send(b'X')
  cursor=None
  try:
   freeze();before=journal('journal-before',['-k','-b','-n','0','--show-cursor','--no-pager']);cursor=re.search(r'-- cursor: (\S+)',before)[1];rec['journal_before_cursor']=cursor
   for index,case in enumerate(specs):
    freeze();label='qsa-'+str(index);frame=root/(label+'.bin');lines=[];holds=[]
    argv=case.get('argv') or ['--bench','--nq','128','--reuse',case['reuse'],'--grain',str(case['grain']),'--repeats',str(case['reps']),'--warmups','2','--dump',str(frame)]
    def interaction(io,entry):
     for _ in range(500):
      line=io.line(seconds=45);lines.append(line)
      if line.startswith('QSA_RESOURCE_READY '):
       phase=line.split(' ',1)[1];assert phase==['before-runtime','runtime-ready','after-cleanup'][len(holds)]
       snapshot(io,entry,phase,label);holds.append(phase)
      if line.startswith('fixture_status=PASS'):
       assert holds==['before-runtime','runtime-ready','after-cleanup'];io.exit(seconds=10);return
     raise RuntimeError('missing bounded fixture completion')
    rec['GPU_executed']=True
    c,so,se=run(label,[binary['path'],*argv],env,interaction,wall=rec['budgets']['native_wall_seconds']);assert c['exit_code']==0
    text=se.read_text();assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|fixture_status=FAIL|cleanup failed',text)
    assert any(s=='device=Intel(R) Arc(TM) B570 Graphics driver=1.17.39395+14' for s in lines)
    samples=[]
    for line in lines:
     if not line.startswith('sample='):continue
     vals=dict(re.findall(r'(\w+)=([^ ]+)',line));sample={k:(int(v) if k=='sample' else float(v)) for k,v in vals.items() if k!='validation'}
     assert all(math.isfinite(v) and v>=0 for v in sample.values());assert abs(sample['enqueue_wall_ms']+sample['wait_wall_ms']-sample['whole_wall_ms'])<1e-7
     samples.append(sample)
    assert len(samples)==case['reps'] and [s['sample'] for s in samples]==list(range(case['reps']))
    result_lines=[s for s in lines if ' status=PASS ' in s];assert len(result_lines)==(3 if mode=='smoke' else 1)
    for line in result_lines:assert all(t in line for t in ['host_model_identity=PASS','full_finite=PASS','input_and_guards=PASS','cross_arm_bits=PASS'])
    counts=None
    if traced:
     counts=len(re.findall(r'^\s*---> urEnqueueKernelLaunch(?:WithArgsExp)?\s*$',text,re.M));expected=39 if mode=='smoke' else 1+(128//case['grain'])*(1+2+case['reps'])
     assert counts==expected,(counts,expected);assert 'UR_RESULT_SUCCESS' in text
    frame_pin=None
    if mode!='smoke':
     assert frame.stat().st_size==24+128*24*256*4;frame_pin=pin(frame)
     if rec['cases']:assert frame_pin['sha256']==rec['cases'][0]['frame']['sha256'],'external fullframe bytes hash differ'
    raw=journal('journal-after-'+str(index),['-k','-b','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(v) for v in raw.splitlines() if v.strip()]
    assert all(v.get('_BOOT_ID')==BOOT.replace('-','') for v in rows);faults=[v for v in rows if FAULT.search(v.get('MESSAGE',''))];assert not faults,faults
    rec['cases'].append(dict(spec=case,stdout=so.read_text(),stderr_sha256=sha(se),samples=samples,traced=traced,UR_kernel_submissions=counts,frame=frame_pin,exit_code=c['exit_code'],fresh_faults=faults,three_Main_gates=True));persist()
   freeze();rec.update(passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc(),passed=False,complete=False)
  finally:
   try:
    assert owner.active is None
    if cursor:
     end=journal('journal-end',['-k','-b','-n','0','--show-cursor','--no-pager']);rec['journal_after_cursor']=re.search(r'-- cursor: (\S+)',end)[1]
     raw=journal('journal-final',['-k','-b','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(v) for v in raw.splitlines() if v.strip()];rec['fresh_faults']=[v for v in rows if FAULT.search(v.get('MESSAGE',''))]
    rec['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT;rec['driver_dump_entries']=[str(v) for v in DEV.glob('devcoredump*')];rec['ownership_closed']=all(m.closed(c) for c in owner.commands)
    for c in owner.commands:
     for old in c['owners']+[c['direct_identity']]:
      now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
    if not rec['ownership_closed'] or not rec['boot_unchanged'] or rec['driver_dump_entries'] or 'fresh_faults' not in rec or rec['fresh_faults']:rec.update(passed=False,complete=False)
   except BaseException as e:rec.update(closure_error=repr(e),passed=False,complete=False)
   persist();target.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(target),passed=rec['passed'],complete=rec['complete'],cases=len(rec['cases']),mode=mode,error=(rec.get('error') or '')[:500],closure_error=rec.get('closure_error'))))
  return 0 if rec['passed'] and rec['complete'] else 1
if __name__=='__main__':sys.exit(main())
