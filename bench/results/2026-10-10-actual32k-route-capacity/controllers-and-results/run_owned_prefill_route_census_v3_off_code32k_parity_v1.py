"""Owned 32K update comparison: raw upstream, integrated update, qualified old control.

First-use diagnostics are separate from quiet timings. Every read is fresh.
This controller never changes sources, resets the GPU, or adopts a binary.
"""
from pathlib import Path
import datetime
import array
import struct
import shutil
import hashlib
import json
import math
import os
import re
import selectors
import subprocess
import sys
import time
import tty
import types
import fcntl


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_fresh(result):
    fields = result['protocol'][-1].split()
    assert len(fields) >= 8 and fields[0] == 'DONE'
    generated, prompt = int(fields[1]), int(fields[2])
    prompt_ms, decode_ms = float(fields[3]), float(fields[4])
    resume = [int(x.split()[1]) for x in result['protocol']
              if x.startswith(('RESUME ', 'REUSED '))]
    progress = [int(x.split()[1]) for x in result['protocol'] if x.startswith('PP ')]
    return {
        'full_32768_input': prompt == 32768,
        'fresh_resume_and_reused': bool(resume) and all(x == 0 for x in resume)
            and any(x.startswith('RESUME ') for x in result['protocol'])
            and any(x.startswith('REUSED ') for x in result['protocol']),
        'complete_prefill_prefix': bool(progress) and max(progress) == 32767,
        'visible_output_complete': 0 < generated <= 64
            and len(result['ids']) == generated and len(result['logprobs']) == generated,
        'normal_finish': fields[5] in ['length', 'stop'],
        'finite_logprobs': all(math.isfinite(float(item.rsplit(':', 1)[-1]))
            for value in result['logprobs'] for item in value.split()[1:]),
        'finite_positive_times': all(math.isfinite(x) and x > 0 for x in [prompt_ms, decode_ms]),
        'valid_mtp_counts': 0 <= int(fields[6]) <= int(fields[7]),
    }


def same_output(actual, expected):
    return {key + '_equal': actual[key] == expected[key]
            for key in ['ids', 'logprobs', 'mtp_counts', 'finish_reason']}


def normalize_reference(request):
    done=[value.split() for value in request['protocol'] if value.startswith('DONE ')]
    assert len(done)==1 and len(done[0])>=8, 'reference needs one complete DONE'
    fields=done[0]
    assert int(fields[1])==len(request['ids'])==len(request['logprobs'])==64
    assert int(fields[2])==32768 and fields[5] in ['length','stop']
    result=dict(request)
    result['finish_reason']=fields[5]
    counts=list(map(int,fields[6:8]))
    assert result.get('mtp_counts',counts)==counts
    result['mtp_counts']=counts
    return result


base=Path(__file__).parent
observer=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(observer/'sycl/tools'));sys.path.insert(0,str(base))
from owned_gdb import OwnedGdb,process_identity
from compare_live_prefill_state_v01402_v2 import compare_states
assert sys.argv[1:] in [[],['--cpu-preflight']]
root=observer.parent/'diag-sycl-prefill-service-events-20261010'
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
build_path=base/'prefill-route-census-linked-root-v4/record.json'
build=json.loads(build_path.read_text());assert build['passed'] and build['complete'] and not build['active'] and build['all_other_members_unchanged'] and build['cached_inputs_unchanged']
binary=Path(build['binary']);assert digest(binary)==build['binary_identity']['sha256']
caller=base/'prefill-route-census-v3/prefill.cpp';header=caller.parent/'prefill_route_census.hpp'
assert digest(caller)==build['caller']['sha256']=='905f43162ae079b6785584da226d950cec14b13bd413ca51ab1a48f49de6ebaf'
assert digest(header)==build['header']['sha256']=='c673382e9f4bf63919489cf00daa71bec793ff7b4584c203291380b827dc110b'
subset_path=base/'upstream-v0141-tuned-fourfresh-numerical-subset-20261009-v1.json'
assert digest(subset_path)=='95ae2a0a7e61f2240f6138f21fdcb35c1a70ec54300b22a7b99fd6b92873b4a7'
subset=json.loads(subset_path.read_text());assert subset['passed'] and not subset['active']
reference=normalize_reference(subset['requests'][0]);assert reference['math_gate_passed'] and all(reference['comparison'].values())
for k in ['first_head','prefill_state']:assert Path(reference[k]['file']).is_file()
assert digest(reference['first_head']['file'])==reference['first_head']['sha256']
state_comparator=base/'compare_live_prefill_state_v01402_v2.py';assert digest(state_comparator)=='9a4f55a4760d316045c1ec5179a946de9a9024f7e42fe09032012ceae2475a72'
cpu_path=base/'owned-main-thread-cpu/record.json';cpu=json.loads(cpu_path.read_text());assert cpu['passed'] and len(cpu['cases'])==4
for case in cpu['cases']:
 if case['name'].startswith('current-'):assert case['helper_sha256']==digest(observer/'sycl/tools/owned_gdb.py')
 for role in ['inferior','debugger']:
  old=case[role];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks'] or now['state']=='Z'
a_source=base/'coding-review-32k-tokens.txt';assert digest(a_source)=='137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
fixtures={'A':list(map(int,a_source.read_text().split()))[:32768]};assert len(fixtures['A'])==32768
args=list(subset['argv'][1:]);assert args[args.index('--max-context')+1]=='262144' and args[args.index('--prefill')+1]=='8192'
assert args[args.index('--expert-cache')+1]=='128' and args[args.index('--pool-workers')+1]=='5'
argv=[str(binary)]+args
lock=(base/'owned-v0141-measurement.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if sys.argv[1:]==['--cpu-preflight']:
 print(json.dumps({'passed':True,'gpu_executed':False,'binary':str(binary),'binary_sha256':digest(binary),'reference':reference['name'],'input_tokens':32768,'actual_batched_positions':32767,'expected_layer_rows':192}));lock.close();sys.exit(0)
out=base/'owned-prefill-route-census-v3-off-code32k-parity-r1';assert not out.exists()
assert shutil.disk_usage(base).free>8*1024**3
mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))}
assert mem['MemAvailable']>64*1024**3
out.mkdir(mode=0o700);probes=out/'probes';probes.mkdir()
source=observer/'sycl/tools/recover-xe.sh';m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(probes);os.chdir(root)
record=dict(scope='DefaultOFF runtime parity of exact v3 (dumps remain; not clean throughput): one fresh32768 A and64 greedy outputs, firsthead and all66 live-state parts against previously qualified reference. n32767 prefix plus final decode token is explicit. No extra device commands/events/waits from census. Quiet runtime; external ownership and full numerical captures, finite owner. Logged times not performance or full256K/adoption evidence.',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=boot,active=True,completed=False,healthy=False,math_gate_passed=False,census_gate_passed=False,requests=[],snapshots=[],steps=[],mode='census-v3-off',phase='parity',repetition=1,deadline_seconds=1000,protocol_timeout_seconds=700,log_limit_bytes=64*1024**2,artifact_limit_bytes=2*1024**3,RSS_inferior_limit_bytes=104*1024**3,commit='495369cf4dc6e6da564095a1c26504dd3694e3f2',compiled_engine_base_commit='7d0105f2a942ac72ca20fef69d4c2e7960653de3',root=str(root),binary_sha256=digest(binary),build_receipt_sha256=digest(build_path),source_sha256={str(p):digest(p) for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',state_comparator,caller,header]},reference_binary_sha256=subset['binary_sha256'],reference_receipt_sha256=digest(subset_path),reference_request=reference['name'],performance_eligible=False,full_lifecycle_passed=False,adopted=False,cpu_debugger_guard_sha256=digest(cpu_path),initial_memory=mem,argv=argv,environment={})
g=None;master=slave=None;cursor=None;pending=bytearray();current=None
raw=(out/'protocol.stdout.raw').open('wb');events=(out/'events.jsonl').open('w',buffering=1)
started=time.monotonic();next_update=started

def save():
    record['elapsed_seconds'] = time.monotonic() - started
    record['steps'] = r.calls
    record['active_request'] = current
    if g:
        record.update(inferior=g.inferior, debugger=g.debugger_identity)
    temporary = out / 'record.json.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(out / 'record.json')


def event(kind, **fields):
    events.write(json.dumps({'kind': kind, 'elapsed_seconds': time.monotonic() - started, **fields}) + '\n')


def captures(name, required):
    result = {}
    head = out / 'first-head.bin'
    if head.exists():
        kept = out / f'{name}.head.bin'
        assert not kept.exists()
        head.rename(kept)
        data = kept.read_bytes()
        values = array.array('f')
        values.frombytes(data)
        assert len(values) == 248320 and all(map(math.isfinite, values))
        result['first_head'] = {'file': str(kept), 'sha256': hashlib.sha256(data).hexdigest(), 'floats': len(values), 'finite': True}
    state = out / 'prefill-state.bin'
    if state.exists():
        kept = out / f'{name}.state.bin'
        assert not kept.exists()
        state.rename(kept)
        parts = []
        with kept.open('rb') as stream:
            while (header := stream.read(8)):
                assert len(header) == 8
                size = struct.unpack('=Q', header)[0]
                offset = stream.tell()
                left = size
                digest = hashlib.sha256()
                while left:
                    data = stream.read(min(left, 1048576))
                    assert data
                    digest.update(data)
                    left -= len(data)
                parts.append({'index': len(parts), 'offset': offset, 'bytes': size, 'sha256': digest.hexdigest()})
        assert len(parts) == 66 and parts[0]['bytes'] == 8
        result['prefill_state'] = {'file': str(kept), 'bytes': kept.stat().st_size, 'parts': parts}
    if required:
        assert 'first_head' in result and 'prefill_state' in result, 'missing fresh full-read capture'
    return result

def poll():
    global next_update
    g.poll(.01)
    if time.monotonic() - started >= record['deadline_seconds']:
        raise TimeoutError('bounded model job deadline')
    if g.stops and g.stops[-1] != 'resumed' and g.exit_code is None and g.exit_signal is None:
        raise RuntimeError('inferior stopped: ' + g.stops[-1])
    if sum(p.stat().st_size for p in (out / 'debugger').glob('*.stderr')) >= record['log_limit_bytes']:
        raise RuntimeError('model log size limit')
    if g.inferior:
        stat=Path('/proc',str(g.inferior['pid']),'status')
        if stat.exists():
            rss=[int(x.split()[1])*1024 for x in stat.read_text().splitlines() if x.startswith('VmRSS:')]
            if rss and rss[0]>=record['RSS_inferior_limit_bytes']:raise RuntimeError('inferior RSS budget')
    if time.monotonic() >= next_update:
        total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if total>=record['artifact_limit_bytes']:raise RuntimeError('model artifact byte budget')
        save()
        next_update = time.monotonic() + 5


def line(seconds=None):
    end = time.monotonic() + (seconds or record['protocol_timeout_seconds'])
    with selectors.DefaultSelector() as ready:
        ready.register(master, selectors.EVENT_READ)
        while time.monotonic() < end:
            if b'\n' in pending:
                value, _, tail = pending.partition(b'\n')
                pending[:] = tail
                return value.decode().strip()
            poll()
            if not ready.select(.02):
                continue
            try:
                data = os.read(master, 65536)
            except BlockingIOError:
                continue
            if not data:
                raise RuntimeError('protocol EOF')
            raw.write(data)
            raw.flush()
            pending.extend(data)
            event('stdout', bytes=len(data), raw_offset=raw.tell())
    raise TimeoutError('engine protocol deadline')


def send(data):
    end = time.monotonic() + 20
    view = memoryview(data)
    with selectors.DefaultSelector() as ready:
        ready.register(master, selectors.EVENT_WRITE)
        while view:
            poll()
            if time.monotonic() >= end:
                raise TimeoutError('protocol input deadline')
            if not ready.select(.02):
                continue
            try:
                count = os.write(master, view[:8192])
            except BlockingIOError:
                continue
            view = view[count:]


save()
try:
    assert Path('/dev/dri/renderD129').exists() and not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
    assert r.run('embedding-state',['/usr/bin/systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],seconds=5).strip()=='ActiveState=inactive'
    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    cursor=m.journal_cursor(r,'kernel-before')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'),cursor=cursor)
    env=m.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
    for key in ['LD_PRELOAD','ZET_ENABLE_METRICS','UR_LOG_TRACING']:env.pop(key,None)
    env={k:v for k,v in env.items() if not k.startswith(('UNITRACE_','XPTI_'))}
    env.pop('UR_ENABLE_LAYERS',None)
    env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0',STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8',STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_ATTN_LAYOUT='1',STRATA_PREFILL_LAYER_MAJOR='0',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0',STRATA_PREFILL_ROUTE_CENSUS='0',STRATA_DUMP_FIRST_LOGITS=str(out/'first-head.bin'),STRATA_PREFILL_DUMP_STATE=str(out/'prefill-state.bin'))
    assert not any(k in env for k in ['STRATA_PREFILL_SYNC','STRATA_PREFILL_TIMING','STRATA_TRANSFER_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_SERVICES','STRATA_VERIFY_NO_HOST','STRATA_VERIFY_DEVICE_PLAN'])
    record['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    record['prompt_fixture']={'file':str(a_source),'sha256':digest(a_source),'tokens':32768}
    master,slave=os.openpty();tty.setraw(slave);os.set_blocking(master,False)
    g=OwnedGdb(argv,out/'debugger',env,inferior_tty_fd=slave);g.command('-gdb-set may-call-functions off');g.run()
    record['startup']=[]
    while True:
        value=line();record['startup'].append(value)
        if value.startswith('ERR'):raise RuntimeError(value)
        if value.startswith('READY '):break
    os.close(slave);slave=None
    identity=process_identity(g.inferior['pid']);assert identity and identity['start_ticks']==g.inferior['start_ticks']
    actual_exe=Path('/proc',str(identity['pid']),'exe');assert actual_exe.resolve()==binary.resolve() and digest(actual_exe)==record['binary_sha256']
    actual=dict(item.decode().split('=',1) for item in Path('/proc',str(identity['pid']),'environ').read_bytes().split(b'\0') if b'=' in item)
    actual_relevant={k:v for k,v in actual.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    assert actual_relevant==record['environment'];record['actual_executable_identity']={'inferior':identity,'sha256':record['binary_sha256'],'boot_id':boot};record['actual_target_environment']=actual_relevant
    current='A-read0';result={'name':current,'fixture':'A','read_index':0,'first_process_read':True,'ids':[],'logprobs':[],'protocol':[]};record['requests'].append(result);save()
    request_start=time.monotonic();send(('GEN 64 ckpt=1 logprobs=5 '+','.join(map(str,fixtures['A']))+'\n').encode());event('request',name=current)
    while True:
        value=line();result['protocol'].append(value)
        if value.startswith('ERR'):raise RuntimeError(value)
        if value.startswith('T '):result['ids'].append(int(value.split()[1]))
        if value.startswith('LP '):result['logprobs'].append(value)
        if value.startswith('DONE '):break
    fields=value.split();result['finish_reason']=fields[5];result['mtp_counts']=list(map(int,fields[6:8]));result['validation']=validate_fresh(result)
    result['measurement']={'generated_tokens':int(fields[1]),'prompt_tokens':int(fields[2]),'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),'wall_seconds':time.monotonic()-request_start,'purpose':'logged census correctness only; excluded from clean throughput'}
    result.update(captures(current,True));state_cmp=compare_states(result['prefill_state'],reference['prefill_state'],32767);result['live_prefill_comparison']=state_cmp
    result['comparison']=same_output(result,reference)|{'first_head_equal':result['first_head']['sha256']==reference['first_head']['sha256'],'all66_live_state_parts_equal':not state_cmp['different_live_parts'],'actual8192_chunks':[int(x.split()[1]) for x in result['protocol'] if x.startswith('PP ')]==[8192,16384,24576,32767]}
    result['math_gate_passed']=all(result['validation'].values()) and all(result['comparison'].values());record['math_gate_passed']=result['math_gate_passed'];save()
    current=None;send(b'QUIT\n');event('quit');end=time.monotonic()+30
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        while g.exit_code is None and g.exit_signal is None and time.monotonic()<end:
            poll()
            if ready.select(.01):
                try:data=os.read(master,65536)
                except (BlockingIOError,OSError):continue
                if data:raw.write(data);raw.flush();event('quit-stdout',bytes=len(data))
    assert g.exit_code==0 and g.exit_signal is None
    log=out/'debugger/inferior.stderr';kept=out/'census.jsonl';rows=[];encoded=[]
    with log.open('rb') as stream,kept.open('wb') as dest:
        for value in stream:
            if value.startswith(b'{"kind":"prefill_route_'):
                j=json.loads(value);rows.append(j);dest.write(value)
                if j['kind']=='prefill_route_layer':encoded.append(value)
    layers=[x for x in rows if x['kind']=='prefill_route_layer'];receipts=[x for x in rows if x['kind']=='prefill_route_census_receipt'];assert len(receipts)==0
    assert not rows, 'exact defaultOFF must emit no census data'
    record['census_gate_passed']=True;record['defaultOFF_emits_no_census']=True
    record['engine_log']={'bytes':log.stat().st_size,'sha256':digest(log)};record['completed']=True
except BaseException as error:
    record['error']=repr(error)
    if g:
        try:
            snap=g.snapshot('failure',resume=False);record['snapshots'].append(snap)
        except BaseException as inspect:record['snapshot_error']=repr(inspect)
finally:
    if g:record['exit_code']=g.exit_code;record['exit_signal']=g.exit_signal;record['cleanup']=g.close()
    for fd in [master,slave]:
        if fd is not None:os.close(fd)
    raw.close();events.close()
    if cursor:
        try:
            text=r.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],seconds=5);rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
            record['new_fault_messages']=[x['MESSAGE'] for x in rows if (('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and m.FAULT.search(x.get('MESSAGE',''))) or ('strata' in x.get('MESSAGE','') and 'segfault' in x.get('MESSAGE',''))]
        except BaseException as error:record['kernel_gate_error']=repr(error)
    record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
    record['healthy']=record['completed'] and record['math_gate_passed'] and record['census_gate_passed'] and not record.get('new_fault_messages') and not record.get('error') and not record.get('kernel_gate_error') and not any(record.get('cleanup',{'forced':True}).values()) and record['boot_unchanged']
    record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();lock.close()
print(json.dumps({k:record.get(k) for k in ['healthy','math_gate_passed','census_gate_passed','elapsed_seconds','error','exit_code','exit_signal','new_fault_messages','cleanup','census_receipt']},indent=2))
if not record['healthy']:raise SystemExit(1)
