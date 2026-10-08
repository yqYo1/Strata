"""First logged upstream-refresh model check through owned GDB/PTY; no performance/capacity claim."""
from pathlib import Path
import array
import datetime
import hashlib
import json
import math
import os
import re
import selectors
import shutil
import sys
import time
import tty
import types

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
os.chdir(root)
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity
out = base/'owned-upstream-refresh-2048';out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
source=root/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(probes)
record={'scope':'First updated-upstream SYCL executable context4096 normal-MTP 2048-token two-chunk check with unchanged FP16 path/phase waits, full Level Zero and UR logging/validation; exact whole heads, IDs and LP versus frozen pre-update control; not clean timing or256K capacity',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'active':True,'completed':False,'healthy':False,'requests':[],'snapshots':[],'steps':[],
        'deadline_seconds':600,'protocol_timeout_seconds':180,'log_limit_bytes':2*1024**3}
g=None;master=slave=None;cursor=None;pending=bytearray();current=None
raw=(out/'protocol.stdout.raw').open('wb')
events=(out/'events.jsonl').open('w',buffering=1)
started=time.monotonic();next_update=started

def save():
    record['elapsed_seconds']=time.monotonic()-started
    record['steps']=r.calls
    record['active_request']=current
    if g:record.update(inferior=g.inferior,debugger=g.debugger_identity)
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')

def event(kind,**fields):
    events.write(json.dumps({'kind':kind,'elapsed_seconds':time.monotonic()-started,**fields})+'\n')

def poll():
    global next_update
    g.poll(.01)
    if time.monotonic()-started>=record['deadline_seconds']:raise TimeoutError('bounded model job deadline')
    if g.stops and g.stops[-1]!='resumed' and g.exit_code is None and g.exit_signal is None:
        raise RuntimeError('inferior stopped: '+g.stops[-1])
    if sum(p.stat().st_size for p in (out/'debugger').glob('*.stderr'))>=record['log_limit_bytes']:
        raise RuntimeError('diagnostic log size limit')
    if time.monotonic()>=next_update:
        save();next_update=time.monotonic()+5

def line(seconds=None):
    end=time.monotonic()+(seconds or record['protocol_timeout_seconds'])
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        while time.monotonic()<end:
            if b'\n' in pending:
                value,_,tail=pending.partition(b'\n');pending[:]=tail
                return value.decode().strip()
            poll()
            if not ready.select(.02):continue
            try:data=os.read(master,65536)
            except BlockingIOError:continue
            except OSError:
                if g.exit_code is not None or g.exit_signal is not None:raise RuntimeError('engine exited before protocol reply')
                raise
            if not data:raise RuntimeError('protocol EOF')
            raw.write(data);raw.flush();pending.extend(data)
            event('stdout',bytes=len(data),raw_offset=raw.tell())
    raise TimeoutError('engine protocol deadline')

def send(data):
    end=time.monotonic()+5;view=memoryview(data)
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_WRITE)
        while view:
            poll()
            if time.monotonic()>=end:raise TimeoutError('protocol input deadline')
            if not ready.select(.02):continue
            try:n=os.write(master,view[:8192])
            except BlockingIOError:continue
            view=view[n:]

try:
    cpu=json.loads((base/'owned-main-thread-cpu/record.json').read_text())
    assert cpu['passed'] and len(cpu['cases'])==4
    current_helper=hashlib.sha256((root/'sycl/tools/owned_gdb.py').read_bytes()).hexdigest()
    assert all(v['helper_sha256']==current_helper for v in cpu['cases'] if v['name'].startswith('current-'))
    for v in cpu['cases']:
        for key in ['inferior','debugger']:
            now=process_identity(v[key]['pid'])
            assert not now or now['start_ticks']!=v[key]['start_ticks'] or now['state']=='Z'
    record['cpu_debugger_guard_sha256']=hashlib.sha256((base/'owned-main-thread-cpu/record.json').read_bytes()).hexdigest()
    health=json.loads((base/'post-full-eager-verifier-health/record.json').read_text())
    full=json.loads((base/'full-context-eager-verifier-serve/record.json').read_text())
    assert not full['active'] and not full['new_fault_messages']
    assert not full['cleanup']['inferior_survived'] and not full['cleanup']['gdb_survived']
    assert health['started_utc'] > full['finished_utc']
    for key in ['inferior','debugger']:
        old=full[key];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']

    assert health['healthy'] and health['boot_id']==record['boot_id']
    for name in ['full-context-entry-capacity','full-context-entry-csr','full-context-layer-trace-serve','full-context-expert-wait-32-serve','full-context-expert-wait-1-serve','full-context-legacy-l0-serve']:
        old=json.loads((base/name/'record.json').read_text())
        assert not old['active'] and not old['new_fault_messages']
        for key in ['inferior','debugger']:
            now=process_identity(old[key]['pid'])
            assert not now or now['start_ticks']!=old[key]['start_ticks'] or now['state']=='Z'
    for p in base.rglob('stalled-writer.json'):
        old=json.loads(p.read_text());now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks'] or now['state']=='Z'
    old_cursor=next(x['argv'][x['argv'].index('--after-cursor')+1] for x in health['steps'] if x['label']=='health-kernel')
    gap=r.run('kernel-gap',['/usr/bin/journalctl','-k','--after-cursor',old_cursor,'--no-pager','-o','json'],seconds=5)
    rows=[json.loads(s) for s in gap.splitlines() if s.startswith('{')]
    faults=[x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and m.FAULT.search(x.get('MESSAGE',''))]
    record['preflight_fault_messages']=faults;assert not faults
    assert r.run('embedding-state',['/usr/bin/systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],seconds=5).strip()=='ActiveState=inactive'
    short=json.loads((base/'owned-upstream-refresh-short/record.json').read_text())
    assert short['healthy'] and not short['active'] and not short['new_fault_messages']
    for key in ['inferior','debugger']:
        old=short[key];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
    reference=json.loads((base/'model-lease-copy-off/normal-mtp.json').read_text())
    assert reference['exit_code']==0 and len(reference['requests'])==4
    control=json.loads((base/'unitrace-model-2048-suffix-clean/record.json').read_text())
    assert control['passed'] and not control['active'] and not control['new_fault_messages'] and not control['job']['survivors']
    for flag,value in [('--max-context','4096'),('--prefill','1024'),('--expert-cache','128'),('--pcie-frac','0')]:
        reference['args'][reference['args'].index(flag)+1]=value
    reference['args'] += ['--kv','int8','--ple-io','direct','--prompt-cache','0']
    tokens=(root/'bench/results/2026-10-05-sycl-prefill-scaling/coding-context-tokens.txt').read_text().split()
    suffix=['248046','198','248045','74455','198','248068','198','248069','271']
    expected=control['requests'][0]
    reference['requests']=[dict(name='2048',request='GEN 4 logprobs=5 '+','.join(tokens[:2039]+suffix),
        ids=expected['ids'],logprobs=expected['logprobs'],first_head={'file':str(base/'unitrace-model-2048-suffix-clean/2048.head.bin')})]
    binary=root/'build-sycl-refresh-20261007/strata'
    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    build=json.loads((base/'upstream-refresh-20261007/build-record.json').read_text())
    assert build['passed'] and not build['active']
    expected=json.loads((base/'upstream-refresh-20261007/binary-receipt.json').read_text())
    assert record['binary_sha256']==expected['sha256']
    env=m.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
    env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0',STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',STRATA_PREFILL_EXPERT_WAIT_BATCH='0',STRATA_PREFILL_SYNC='1')
    env.update({k:v for k,v in control['target_environment'].items() if k.startswith('STRATA_') and k != 'STRATA_DUMP_FIRST_LOGITS'})
    env=m.diagnostic_environment(env)
    head=out/'first-head.bin';env['STRATA_DUMP_FIRST_LOGITS']=str(head)
    record['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    argv=[str(binary)]+reference['args'];record['argv']=argv
    record['reference_binary_sha256']=control['binary_sha256']
    record['reference_result_sha256']=hashlib.sha256((base/'unitrace-model-2048-suffix-clean/record.json').read_bytes()).hexdigest()
    record['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,root/'sycl/tools/owned_gdb.py',root/'sycl/src/prefill/prefill.cpp']}
    cursor=m.journal_cursor(r,'kernel-before');save()
    master,slave=os.openpty();tty.setraw(slave);os.set_blocking(master,False)
    g=OwnedGdb(argv,out/'debugger',env,inferior_tty_fd=slave)
    g.command('-gdb-set may-call-functions off')
    g.command('-gdb-set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug')
    g.run();record['startup']=[]
    while True:
        value=line();record['startup'].append(value)
        if value.startswith('ERR'):raise RuntimeError(value)
        if value.startswith('READY '):break
    os.close(slave);slave=None;save()
    for expected in reference['requests']:
        current=expected['name'];head.unlink(missing_ok=True)
        send((expected['request']+'\n').encode());event('request',name=current)
        result={'name':current,'ids':[],'logprobs':[],'protocol':[]}
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith('ERR'):raise RuntimeError(value)
            if value.startswith('T '):result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):
                assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in value.split()[1:])
                result['logprobs'].append(value)
            if value.startswith('DONE '):break
        assert len(result['ids'])==4 and len(result['logprobs'])==4
        data=head.read_bytes();values=array.array('f');values.frombytes(data)
        assert len(values)==248320 and all(map(math.isfinite,values))
        saved=out/(current+'.head.bin');saved.write_bytes(data)
        result['head']={'bytes':len(data),'floats':len(values),'sha256':hashlib.sha256(data).hexdigest()}
        result['equality']={'ids':result['ids']==expected['ids'],'logprobs':result['logprobs']==expected['logprobs'],'head':data==Path(expected['first_head']['file']).read_bytes()}
        record['requests'].append(result);save()
        assert all(result['equality'].values())
    current=None;send(b'QUIT\n');event('quit')
    end=time.monotonic()+30
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        while g.exit_code is None and g.exit_signal is None and time.monotonic()<end:
            poll()
            if ready.select(.01):
                try:data=os.read(master,65536)
                except (BlockingIOError,OSError):continue
                if data:raw.write(data);raw.flush();event('quit-stdout',bytes=len(data))
    assert g.exit_code==0 and g.exit_signal is None
    text=(out/'debugger/inferior.stderr').read_text(errors='replace')
    record['prefill_layer_ranges']=[list(map(int,x)) for x in re.findall(r'strata trace: prompt chunk \d+ of \d+, layers \[(\d+), (\d+)\)',text)]
    assert record['prefill_layer_ranges']
    record['expert_wait_batches']=[list(map(int,x)) for x in re.findall(r'strata prefill expert wait: (\d+) experts completed, layer (-?\d+), chunk (-?\d+)',text)]
    assert not record['expert_wait_batches']
    phase_pattern=r'strata prefill sync: mark (\d+) phase (.*?) done \(waited ([0-9.]+) ms\)'
    record['phase_sync_mark_count']=len(re.findall(phase_pattern,text))
    record['phase_sync_last_marks']=re.findall(phase_pattern,text)[-16:]
    assert record['phase_sync_mark_count']>0
    record['release_pairs']={'release':text.count('strata mtp decode release:'),'restore':text.count('strata mtp decode restore:')}
    assert record['release_pairs']['release']>=1 and record['release_pairs']['release']==record['release_pairs']['restore']
    record['completed']=True
except BaseException as error:
    record['error']=repr(error)
    if g:
        try:record['snapshots'].append(g.snapshot('failure',resume=False))
        except BaseException as inspect:record['snapshot_error']=repr(inspect)
finally:
    if g:
        record['exit_code']=g.exit_code;record['exit_signal']=g.exit_signal
        record['cleanup']=g.close()
    for fd in [master,slave]:
        if fd is not None:os.close(fd)
    raw.close();events.close()
    if cursor:
        text=r.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],seconds=5)
        rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
        record['new_fault_messages']=[x['MESSAGE'] for x in rows if (('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and m.FAULT.search(x.get('MESSAGE',''))) or ('strata' in x.get('MESSAGE','') and 'segfault' in x.get('MESSAGE',''))]
    record['healthy']=record['completed'] and not record.get('new_fault_messages') and not record.get('error') and not any(record.get('cleanup',{}).get(k) for k in ['inferior_survived','gdb_survived'])
    record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps({k:v for k,v in record.items() if k not in ['environment','argv','steps','prefill_layer_ranges']},indent=2))
if not record['healthy']:raise SystemExit(1)
