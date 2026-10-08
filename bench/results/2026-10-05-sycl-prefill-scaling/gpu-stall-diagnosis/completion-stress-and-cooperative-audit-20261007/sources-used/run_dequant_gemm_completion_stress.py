"""One owned, fully logged actual-wrapper completion stress case; no resets."""
from pathlib import Path
import datetime,hashlib,json,os,re,selectors,sys,time,tty,types
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import OwnedGdb,process_identity
assert len(sys.argv)==2 and sys.argv[1] in ['usm','retire']
mode=sys.argv[1]
continuation=json.loads((base/'profiler-continuation.json').read_text())
assert not continuation['live_gpu_jobs']
previous=json.loads((base/'owned-upstream-refresh-2048-recheck/record.json').read_text())
health=json.loads((base/'post-upstream-refresh-2048-recheck-health/record.json').read_text())
assert not previous['active'] and not previous['cleanup']['inferior_survived'] and not previous['cleanup']['gdb_survived']
assert health['healthy'] and health['started_utc']>previous['finished_utc']
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==health['boot_id']
for name in ['owned-expanded-fp16-refresh-128-2048','owned-expanded-fp16-no-root-refresh-128-2048','owned-expanded-fp16-no-root-refresh-zero-2048','owned-upstream-refresh-2048-recheck']:
    old=json.loads((base/name/'record.json').read_text());assert not old['active']
    for k in ['inferior','debugger']:
        owner=old[k];now=process_identity(owner['pid']);assert not now or now['start_ticks']!=owner['start_ticks']
if mode=='retire':
    before=json.loads((base/'dequant-gemm-completion-stress-usm/record.json').read_text())
    assert before['passed'] and not before['active'] and not before['new_fault_messages']
    for k in ['inferior','debugger']:
        owner=before[k];now=process_identity(owner['pid']);assert not now or now['start_ticks']!=owner['start_ticks']
out=base/('dequant-gemm-completion-stress-'+mode);out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
source=root/'sycl/tools/recover-xe.sh'
module=types.ModuleType('stress_diagnostics')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),module.__dict__)
runner=module.Runner(probes)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'scope':'Actual regular IQ wrappers using real layer17/expert0 weights, model-sized512-slotUSMarena, 20000 GU/down pairs and40 complete guarded-output checks; actual GEMM/SwiGLU/down consumers with synthetic eight-row FP16 input; optional64MiB VMM warmup/retirement on samequeue; not full-model math/capacity, clean throughput or hang prevention',
   'active':True,'passed':False,'mode':mode,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'boot_id':boot,'events':[],'steps':[],'deadline_seconds':180,'no_progress_seconds':45,
   'log_limit_bytes':1024**3,'source_sha256':digest(base/'dequant_gemm_completion_stress.cpp'),
   'controller_sha256':digest(Path(__file__)),'helper_sha256':digest(root/'sycl/tools/owned_gdb.py')}
g=None;master=slave=None;cursor=None;start=time.monotonic()

def save():
    r['elapsed_seconds']=time.monotonic()-start;r['steps']=runner.calls
    if g:r.update(inferior=g.inferior,debugger=g.debugger_identity)
    (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')

try:
    old_cursor=next(s['argv'][s['argv'].index('--after-cursor')+1] for s in health['steps'] if s['label']=='health-kernel')
    text=runner.run('kernel-gap',['/usr/bin/journalctl','-k','--after-cursor',old_cursor,'--no-pager','-o','json'],seconds=5)
    rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
    r['preflight_fault_messages']=[x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and module.FAULT.search(x.get('MESSAGE',''))]
    assert not r['preflight_fault_messages']
    assert runner.run('embedding-state',['/usr/bin/systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],seconds=5).strip()=='ActiveState=inactive'
    build=json.loads((base/'dequant-gemm-completion-stress-build-v2/record.json').read_text());assert build['passed']
    binary=base/'dequant-gemm-completion-stress-build-v2/probe';assert digest(binary)==build['binary_sha256']
    r['binary_sha256']=build['binary_sha256'];r['build_record_sha256']=digest(base/'dequant-gemm-completion-stress-build-v2/record.json')
    metadata=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/native_experts.txt')
    shard=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
    row=next(s.split() for s in metadata.read_text().splitlines() if s.startswith('17 '));assert row[1:3]==['21','20'] and row[4]=='2329600'
    r['metadata_sha256']=digest(metadata);r['shard_bytes']=shard.stat().st_size;r['input_regions']=[]
    with shard.open('rb') as stream:
        for label,offset,size in [('gate',int(row[5]),704000),('up',int(row[6]),704000),('down',int(row[7]),921600)]:
            stream.seek(offset);payload=stream.read(size);assert len(payload)==size
            r['input_regions'].append(dict(label=label,offset=offset,bytes=size,sha256=hashlib.sha256(payload).hexdigest()))
    env=module.diagnostic_environment();env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0');env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
    r['environment']={k:v for k,v in env.items() if k.startswith(('SYCL_','UR_','ZE_','ZEL_','ONEAPI_','STRATA_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    args=[str(binary),str(shard),str(metadata),mode,'20000',str(out/'reference')];r['argv']=args
    cursor=module.journal_cursor(runner,'kernel-before');save()
    master,slave=os.openpty();tty.setraw(slave);os.set_blocking(master,False)
    g=OwnedGdb(args,out/'debugger',env,inferior_tty_fd=slave)
    g.command('-gdb-set may-call-functions off')
    g.command('-gdb-set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug')
    g.run();os.close(slave);slave=None
    pending=bytearray();last_progress=time.monotonic();next_save=last_progress
    with (out/'stdout.raw').open('wb') as raw,selectors.DefaultSelector() as selector:
        selector.register(master,selectors.EVENT_READ)
        def drain():
            global last_progress
            if not selector.select(.01):return False
            try:data=os.read(master,65536)
            except (OSError,BlockingIOError):return False
            if not data:return False
            raw.write(data);raw.flush();pending.extend(data)
            while b'\n' in pending:
                line,_,tail=pending.partition(b'\n');pending[:]=tail
                event=json.loads(line.decode());r['events'].append(event);last_progress=time.monotonic()
            return True
        while g.exit_code is None and g.exit_signal is None:
            g.poll(.01);drain()
            if g.stops and g.stops[-1]!='resumed' and g.exit_code is None and g.exit_signal is None:raise RuntimeError('inferior stopped: '+g.stops[-1])
            if time.monotonic()-start>r['deadline_seconds']:raise TimeoutError('bounded stress deadline')
            if time.monotonic()-last_progress>r['no_progress_seconds']:raise TimeoutError('bounded stress no-progress deadline')
            if sum(p.stat().st_size for p in (out/'debugger').glob('*.stderr'))>r['log_limit_bytes']:raise RuntimeError('bounded API log limit')
            if time.monotonic()>=next_save:save();next_save=time.monotonic()+2
        while drain():pass
        assert not pending
    assert g.exit_code==0 and g.exit_signal is None
    final=r['events'][-1];assert final['stage']=='PASS' and final['iterations']==20000 and final['full_checks']==40
    assert final['dequant_launches']==40002+(2 if mode=='retire' else 0)
    r['pipeline_heads']=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.glob('reference-pipeline-*.bin'))]
    assert len(r['pipeline_heads'])==3
    if mode=='retire':assert r['pipeline_heads']==before['pipeline_heads']
    r['reference_heads']=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.glob('reference-*.bin')) if '-pipeline-' not in p.name]
    assert len(r['reference_heads'])==2
    if mode=='retire':assert r['reference_heads']==before['reference_heads']
    r['completed']=True
except BaseException as error:
    r['error']=repr(error)
    if g:
        try:r['failure_snapshot']=g.snapshot('failure',resume=False)
        except BaseException as e:r['snapshot_error']=repr(e)
finally:
    if g:r.update(exit_code=g.exit_code,exit_signal=g.exit_signal,cleanup=g.close())
    for fd in [master,slave]:
        if fd is not None:os.close(fd)
    if cursor:
        text=runner.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],seconds=5)
        rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
        r['new_fault_messages']=[x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and module.FAULT.search(x.get('MESSAGE',''))]
    r['active']=False;r['passed']=bool(r.get('completed') and not r.get('error') and not r.get('new_fault_messages') and not r.get('cleanup',{}).get('inferior_survived') and not r.get('cleanup',{}).get('gdb_survived'))
    r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps({k:v for k,v in r.items() if k not in ['events','environment','argv','steps']},indent=2))
raise SystemExit(0 if r['passed'] else 1)
