#!/usr/bin/env python3
"""Main's model-free real64bit checked-provider oracle. No model or legacy entrypoint."""
import fcntl,hashlib,importlib.util,json,os,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
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
 root=B/'main-pr6-model-small-v2';target=A/'own-pr6-model-small-v2.json';assert not root.exists() and not target.exists()
 owner_source=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py';assert sha(owner_source)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
 base=A/'own-vllm-resource-timeline-v3/manifest.json';assert sha(base)=='61da9469e3f975c72d770beee77d45118c988eb6d3d82b6dbe1b568cd9ea3f1c'
 buildpath=A/'own-pr6-gated-harness-build-v1.json';build=json.loads(buildpath.read_text());assert build['passed'] and build['complete'] and build['ownership_closed']
 elfpath=A/'own-pr6-gated-harness-ELF-review-v1.json';elf=json.loads(elfpath.read_text());assert elf['passed'] and elf['binary']==build['binary']
 configpath=A/'own-pr6-harness-configure-v6.json';config=json.loads(configpath.read_text());assert config['passed'] and config['complete']
 binary=build['binary'];assert binary['sha256']=='56bd0fefdf117f1cd6baea6a32f9acad0c8d82ba438b6fd6b2c61cc70f3db88f'
 allowed=dict(json.loads(base.read_text())['elf_pins']);allowed[binary['realpath']]=binary
 frozen={**allowed}
 for v in [*build['immutable_inputs'],pin(buildpath),pin(elfpath),pin(configpath),pin(Path(__file__))]:frozen[v['realpath']]=v
 pack=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
 shards=list(json.loads(base.read_text())['model_stat_pins'])
 shards=sorted(shards)
 for p in [pack/'index.txt',pack/'native_experts.txt',pack/'tokenizer/vocab.json',pack/'tokenizer/merges.txt',pack/'tokenizer/token_type.json',A.parents[2]/'tools/strata_tokenizer.py']:
  v=pin(p);frozen[v['realpath']]=v
 assets={}
 for p in [*map(Path,shards),pack/'dense.bin']:
  st=p.stat();assets[str(p)]=dict(realpath=str(p.resolve()),bytes=st.st_size,mtime_ns=st.st_mtime_ns,device=st.st_dev,inode=st.st_ino)
 for p,v in json.loads(base.read_text())['model_stat_pins'].items():
  assert assets[p]['bytes']==v['bytes'] and assets[p]['mtime_ns']==v['mtime_ns']
 spec=importlib.util.spec_from_file_location('pr6_provider_native_owner',owner_source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
  env={**HOST,'LD_LIBRARY_PATH':str(L/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu',
   'UR_ADAPTERS_FORCE_LOAD':str(L/'lib/libur_adapter_level_zero_v2.so.0'),'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1',
   'NEOReadDebugKeys':'1','EnableDirectSubmission':'0','STRATA_SYCL_PREFILL_SAME_QUEUE':'1','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0',
   'SYCL_CACHE_PERSISTENT':'0','XDG_CACHE_HOME':str(root/'cache'),'MKLROOT':'/opt/intel/oneapi/mkl/2026.1','OMP_NUM_THREADS':'1',
   'STRATA_PREFILL_LAYER_MAJOR':'2','STRATA_PREFILL_COMPACT':'1','STRATA_PREFILL_MMQ':'0','STRATA_PROMPT_ATTN_XMX':'0',
   'STRATA_PREFILL_FIRST':'0','STRATA_PREFILL_LAYER_MAJOR_R_GPU':'8','STRATA_PREFILL_RING':'8',
   'STRATA_TRACE':'1','STRATA_PREFILL_TIMING':'1','STRATA_PREFILL_TRANSFER_TIMING':'1',
   'UR_ENABLE_LAYERS':'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION','UR_LOG_TRACING':'level:info;flush:info;output:stderr',
   'UR_LOG_VALIDATION':'level:warning;flush:warning;output:stderr','UR_LOG_LOADER':'level:warning;flush:warning;output:stderr','UR_LOG_LEVEL_ZERO':'level:warning;flush:warning;output:stderr'}
  rec=dict(schema='strata-pr6-model-small-v2',passed=False,complete=False,active=False,GPU_executed=False,commands=[],cases=[],snapshots=[],
   boot_id=BOOT,binary=binary,build=pin(buildpath),ELF_review=pin(elfpath),config=pin(configpath),runtime_elf_pins=allowed,native_environment=env,
   budgets=dict(native_wall_seconds=600,child_cpu_seconds=900,raw_text_bytes=64<<20,private_bytes=128<<20),
   scope='Eight tokenizer-produced model tokens in restored no-doorbell prefill-only path: actual completion/finite/state-window/lifetime/fault first gate only. No performance/modelquality approval.',
   model_execution=True,legacy_entrypoint_execution=False,async87_delivery_qualified=False,physical262144_qualified=False,RESTORE_qualified=False,
   retention=dict(owner='Main',consumer='First eight-token restored model safety/correctness gate',next_review='After extracting exact summaries, maps and any distinct failure; retire successful full API traces.'))
  owner=None
  def persist():
   rec['active']=bool(owner and owner.active is not None);(root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);rec['commands']=owner.commands
  def run(label,argv,current=HOST,interaction=None,wall=30):
   c,so,se=owner.run(label,argv,current,A.parents[2],interaction=interaction,wall=wall,cpu=900,rss_cap=104<<30,text_cap=64<<20,file_cap=64<<20,total_cap=128<<20)
   assert m.closed(c),(label,'ownership not closed');return c,so,se
  def journal(label,args):
   c,so,se=run(label,['/usr/bin/journalctl',*args]);assert c['exit_code']==0;return so.read_text()
  def freeze():
   assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
   assert DEV.joinpath('driver').resolve().name=='xe' and not list(DEV.glob('devcoredump*'))
   for p,v in frozen.items():check(p,v)
   for p,v in assets.items():
    q=Path(p);st=q.stat();assert dict(realpath=str(q.resolve()),bytes=st.st_size,mtime_ns=st.st_mtime_ns,device=st.st_dev,inode=st.st_ino)==v,p
  def snapshot(io,entry,phase,label):
   ident=m.identity(io.proc.pid);assert ident and ident['start_ticks']==entry['direct_identity']['start_ticks']
   proc=Path('/proc')/str(io.proc.pid);assert str(proc.joinpath('exe').resolve())==binary['realpath']
   effective=dict(x.decode().split('=',1) for x in proc.joinpath('environ').read_bytes().split(b'\0') if x);assert effective==env
   maps=[];seen=set()
   for line in proc.joinpath('maps').read_text().splitlines():
    f=line.split(None,5)
    if len(f)!=6 or 'x' not in f[1] or not f[5].startswith('/'):continue
    assert not f[5].endswith(' (deleted)');p=Path(f[5]).resolve(strict=True);assert str(p) in allowed,'unreviewed ELF '+str(p)
    v=allowed[str(p)]
    if str(p) not in seen:check(p,v);seen.add(str(p))
    st=p.stat();major,minor=[int(x,16) for x in f[3].split(':')];assert st.st_ino==int(f[4]) and os.major(st.st_dev)==major and os.minor(st.st_dev)==minor
    maps.append(dict(mapping=line,pin=v))
   assert binary['realpath'] in seen
   if phase!='before-runtime':
    assert str((L/'lib/libsycl.so.9').resolve()) in seen and str((L/'lib/libur_adapter_level_zero_v2.so.0').resolve()) in seen
    assert '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395' in seen
    stderr=(root/'owned'/f'{label}.stderr').read_text()
    assert 'device=Intel(R) Arc(TM) B570 Graphics driver=1.17.39395+14' in stderr
   after=m.identity(io.proc.pid);assert after and after['start_ticks']==ident['start_ticks']
   rec['snapshots'].append(dict(case=label,phase=phase,identity=ident,effective_environment=effective,mappings=maps,scope='File-backed executable mappings at explicit hold, not anonymous/unloaded-code attestation'))
   io.send(b'X')
  cursor=None
  try:
   freeze();before=journal('journal-before',['-k','-b','-n','0','--show-cursor','--no-pager']);cursor=re.search(r'-- cursor: (\S+)',before)[1];rec['journal_before_cursor']=cursor
   token_source="import sys, json\nfrom pathlib import Path\nsys.path.insert(0, '/home/yayoi/orca/workspaces/Strata/sycl/tools')\nfrom strata_tokenizer import Tokenizer\np=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/tokenizer')\nv=json.loads((p/'vocab.json').read_text())\nt=[None]*len(v)\nfor text,index in v.items():\n t[index]=text\nm=(p/'merges.txt').read_text().splitlines()\nty=json.loads((p/'token_type.json').read_text())\ntk=Tokenizer(t,m,ty)\nx=tk.encode('Explain the purpose of a linked list and give a short example.')[:8]\nassert len(x)==8\nprint(json.dumps(x))\n"
   c,so,se=run('tokenize-input',['/home/yayoi/.local/share/strata-sycl/venv/bin/python','-c',token_source],HOST,wall=60)
   assert c['exit_code']==0;ids=json.loads(so.read_text());assert len(ids)==8 and all(isinstance(x,int) and 0<=x<=2147483647 for x in ids)
   import struct
   tokenfile=root/'input-i32le.bin';tokenfile.write_bytes(struct.pack('<8i',*ids));v=pin(tokenfile);frozen[v['realpath']]=v
   rec.update(model_assets_stat_only=assets,model_asset_integrity_limit='GGUF and dense bytes/mtime/device/inode identities; not full-file SHA. Textmetadata/tokenizer and actual8token bytes fullySHA pinned.',tokenizer_output=ids,input=pin(tokenfile),protocol_progress=[])
   argv=[binary['path'],'--model-root',str(pack)]
   for shard in shards:argv+=['--shard',shard]
   argv+=['--ple-file',shards[1],'--tokens-file',str(tokenfile),'--token-count','8','--context','8192','--chunk','8192','--first-chunk','0','--residual-gpu-tokens','8','--ring-slots','8','--device','0','--kv','int8','--logit-id','0','--validation-small-input']
   for n,name in enumerate(['restored-eight-token-first-gate']):
    freeze();label='model-small-'+str(n);summary=None
    def interaction(io,entry):
     nonlocal summary
     for phase,limit in [('before-runtime',30),('runtime-ready',120),('before-prefill',240),('after-cleanup',240)]:
      expected='PR6_PREFILL_READY '+phase
      for _ in range(1000):
       line=io.line(seconds=limit)
       if line==expected:break
       rec['protocol_progress'].append(dict(before=phase,line=line))
      else:raise RuntimeError('excess protocol progress before '+phase)
      snapshot(io,entry,phase,label)
     summary=json.loads(io.line(seconds=10));assert summary['status']=='PREFILL_ONLY_COMPLETED_UNCOMPARED'
     io.exit(seconds=10)
    rec['GPU_executed']=True
    c,so,se=run(label,argv,env,interaction,wall=600)
    assert c['exit_code']==0 and summary and len([v for v in rec['snapshots'] if v['case']==label])==4
    text=se.read_text();assert 'UR_RESULT_SUCCESS' in text and not re.search(r'(?:UR|ZE)_RESULT_ERROR_|pr6_prefill_api failed:|retirement failed',text)
    expected=dict(main_gates_released=4,validation_small_input=True,measurement_scope='safety_correctness_only',tokens=8,position0=0,context=8192,chunk=8192,chunks=1,kv='int8',doorbell_initialized=False,verifier_initialized=False,drafter_initialized=False,generation=False,numerical_comparison=False,physical256k_qualified=False,historical600_requalified=False)
    for key,value in expected.items():assert summary[key]==value,(key,summary.get(key),value)
    assert summary['ple_prev']==ids[-2:]
    kernel_entries=len(re.findall(r'^\s*---> urEnqueueKernelLaunch(?:WithArgsExp)?\s*$',text,re.M));assert kernel_entries>0
    raw=journal('journal-after-'+str(n),['-k','-b','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(v) for v in raw.splitlines() if v.strip()]
    assert all(v.get('_BOOT_ID')==BOOT.replace('-','') for v in rows)
    faults=[v for v in rows if FAULT.search(v.get('MESSAGE',''))];assert not faults,faults
    rec['cases'].append(dict(name=name,summary=summary,stdout=so.read_text(),stderr_sha256=sha(se),exit_code=c['exit_code'],fresh_faults=faults,API_error_count=0,real_UR_kernel_entries=kernel_entries));persist()
   freeze();rec.update(passed=True,complete=True)
  except BaseException as e:
   import traceback
   rec.update(error=repr(e),traceback=traceback.format_exc(),passed=False,complete=False)
  finally:
   try:
    assert owner.active is None
    if cursor:
     end=journal('journal-end',['-k','-b','-n','0','--show-cursor','--no-pager']);rec['journal_after_cursor']=re.search(r'-- cursor: (\S+)',end)[1]
     raw=journal('journal-final',['-k','-b','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(v) for v in raw.splitlines() if v.strip()]
     rec['fresh_faults']=[v for v in rows if FAULT.search(v.get('MESSAGE',''))]
    rec['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
    rec['driver_dump_entries']=[str(v) for v in DEV.glob('devcoredump*')]
    rec['ownership_closed']=all(m.closed(c) for c in owner.commands)
    for c in owner.commands:
     for old in c['owners']+[c['direct_identity']]:
      now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
    if not rec['ownership_closed'] or not rec['boot_unchanged'] or rec['driver_dump_entries'] or 'fresh_faults' not in rec or rec['fresh_faults']:rec.update(passed=False,complete=False)
   except BaseException as e:rec.update(closure_error=repr(e),passed=False,complete=False)
   persist();target.write_bytes((root/'record.json').read_bytes())
  print(json.dumps(dict(receipt=str(target),passed=rec['passed'],complete=rec['complete'],cases=len(rec['cases']),fresh_fault_count=len(rec.get('fresh_faults',[])),ownership_closed=rec.get('ownership_closed'),error=(rec.get('error') or '')[:400],closure_error=rec.get('closure_error'))))
  return 0 if rec['passed'] and rec['complete'] else 1
if __name__=='__main__':sys.exit(main())
