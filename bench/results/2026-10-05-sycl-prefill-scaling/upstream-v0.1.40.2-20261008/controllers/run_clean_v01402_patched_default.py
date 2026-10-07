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
import subprocess

base = Path(__file__).parent
observer = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
mode=sys.argv[1]; repetition=int(sys.argv[2])
assert mode in ['patched-default'] and repetition in [1,2]
root=Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0.1.40.2-20261008') if mode=='pure' else observer
os.chdir(root)
sys.path.insert(0,str(observer/'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity
out = base/f'clean-v01402-default-{mode}-r{repetition}';out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
source=observer/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(probes)
record={'scope':'Clean short first-request benchmark for unmodified upstream v0.1.40.2 or patched SYCL build; cold first-request times from DONE, finite output and bounded owned lifecycle; not diagnostic tracing, long-prompt tuning or256K capacity',
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
    current_helper=hashlib.sha256((observer/'sycl/tools/owned_gdb.py').read_bytes()).hexdigest()
    assert all(v['helper_sha256']==current_helper for v in cpu['cases'] if v['name'].startswith('current-'))
    for v in cpu['cases']:
        for key in ['inferior','debugger']:
            now=process_identity(v[key]['pid'])
            assert not now or now['start_ticks']!=v[key]['start_ticks'] or now['state']=='Z'
    record['cpu_debugger_guard_sha256']=hashlib.sha256((base/'owned-main-thread-cpu/record.json').read_bytes()).hexdigest()
    health=json.loads((base/'post-upstream-e8ca-final-float-storage-health/record.json').read_text())
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
    reference=json.loads((base/'model-lease-copy-off/normal-mtp.json').read_text())
    assert reference['exit_code']==0 and len(reference['requests'])==4
    binary=root/('build-sycl-pure-20261008/strata' if mode=='pure' else 'build-sycl-e8ca-refresh-20261007/strata')
    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    build=json.loads((base/('pure-upstream-v0.1.40.2-20261008' if mode=='pure' else 'upstream-e8ca-refresh-20261007')/'build-record.json').read_text())
    assert build['passed'] and not build['active']
    if mode=='pure':
        assert record['binary_sha256']==build['binary_sha256']
        record['source_status_before']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        assert not record['source_status_before']
    else:
        expected=json.loads((base/'upstream-e8ca-refresh-20261007/binary-receipt.json').read_text())
        assert record['binary_sha256']==expected['sha256']
    final=json.loads((base/'owned-upstream-e8ca-refresh-short-final-float-storage/record.json').read_text());assert final['healthy'] and not final['active']
    for previous_path in list(base.glob('clean-v01402-default-*/record.json'))+list(base.glob('clean-v01402-short-*/record.json')):
        if previous_path.parent==out:continue
        previous=json.loads(previous_path.read_text());assert not previous['active'] and previous['healthy']
        for key in ['inferior','debugger']:
            old=previous[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'))
    env=m.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
    env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    # Default patched algorithm configuration: no additional tuning environment.
    assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in env)
    record['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    args=list(reference['args']);args[args.index('--pcie-frac')+1]='0.55' if mode=='patched-dma' else '0'
    args[args.index('--conversation-cache-mib')+1]='0'
    argv=[str(binary)]+args;record['argv']=argv
    record['mode']=mode;record['repetition']=repetition
    record['configuration_note']='37 prompt tokens, 64 generated tokens, context128, chunk32, greedy normal MTP, five CPU workers. Cold first request per process, including JIT/graph preparation in engine times; no throughput claim for long prompts or steady state. Patched defaults without COMPACT, LAYER_MAJOR, RELEASE_DRAFT or DRAFT_VERIFY overrides; same requested cache600; upstream expands it to768 whereas patched keeps600; both counts and VRAM reported in INFO. No diagnostic validation, tracing, logits dump or extra phase synchronization. External owned GDB/PTY stays attached without interrupts while requests run.'
    record['reference_binary_sha256']='79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223'
    record['reference_result_sha256']=hashlib.sha256((base/'model-lease-copy-off/normal-mtp.json').read_bytes()).hexdigest()
    record['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',root/'sycl/src/prefill/prefill.cpp']}
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
    for expected in reference['requests'][:1]:
        current=expected['name']
        request=expected['request'].replace('GEN 4 ', 'GEN 64 ',1)
        send((request+'\n').encode());event('request',name=current)
        request_start=time.monotonic()
        result={'name':current,'ids':[],'logprobs':[],'protocol':[]}
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith('ERR'):raise RuntimeError(value)
            if value.startswith('T '):result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):
                assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in value.split()[1:])
                result['logprobs'].append(value)
            if value.startswith('DONE '):break
        assert len(result['ids'])==64 and len(result['logprobs'])==64
        fields=value.split();assert fields[0]=='DONE' and fields[1]=='64' and fields[2]=='37' and fields[5]=='length'
        result['measurement']={'generated_tokens':int(fields[1]),'prompt_tokens':int(fields[2]),'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),'wall_seconds':time.monotonic()-request_start}
        mm=result['measurement'];assert mm['prompt_ms']>0 and mm['decode_ms']>0
        mm['prefill_tok_s']=1000*mm['prompt_tokens']/mm['prompt_ms'];mm['decode_tok_s']=1000*mm['generated_tokens']/mm['decode_ms']
        result['prefix_ids_match_historical']=result['ids'][:4]==expected['ids']
        record['requests'].append(result);save()

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
    if mode=='pure':
        record['source_status_after']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        assert not record['source_status_after']
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
