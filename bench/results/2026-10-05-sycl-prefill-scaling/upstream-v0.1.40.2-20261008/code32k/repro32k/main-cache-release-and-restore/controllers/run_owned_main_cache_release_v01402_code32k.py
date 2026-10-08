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
import struct

base = Path(__file__).parent
observer = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
mode=sys.argv[1]; phase=sys.argv[2]; repetition=int(sys.argv[3])
assert mode in ['main-vmm-kept-ram','main-vmm-half-ram','main-vmm-full-ram','main-vmm-full-snapshot'] and phase in ['diagnostic','state'] and repetition in [1,2,3]
previous=json.loads((base/'v01402-code32k-sequence/record.json').read_text());assert previous['passed'] and not previous['active']
default_gate=json.loads((base/'event-ack-no-root-default-v01402-gated-checks/record.json').read_text())
assert default_gate['passed'] and not default_gate['active']
qsa_gate=json.loads((base/'qsa-batch128-layout1-v01402-clean-sequence/record.json').read_text())
assert qsa_gate['passed'] and not qsa_gate['active']
layer_gate=json.loads((base/'compact-layer-v01402-state-sequence/record.json').read_text())
assert layer_gate['passed'] and not layer_gate['active']
parent_gate=json.loads((base/'layer-processing-v01402-clean-sequence/record.json').read_text())
assert parent_gate['passed'] and not parent_gate['active']
source_gate=json.loads((base/'main-cache-release-v01402-source-review/record.json').read_text())
assert source_gate['passed'] and not source_gate['active']
clean_gate=json.loads((base/'registered-vs-no-root-v01402-clean-sequence/record.json').read_text())
assert clean_gate['passed'] and not clean_gate['active']
previous=json.loads((base/'owned-v01402-code32k-patched-default-diagnostic-r1/record.json').read_text())
assert previous['healthy'] and not previous['active']
root=Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0.1.40.2-20261008') if mode=='pure' else observer
os.chdir(root)
sys.path.insert(0,str(observer/'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity
out = base/f'owned-{mode}-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
source=observer/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(probes)
record={'scope':'Same integrated fork32K input/configuration, fresh process each time, capture all248320 first-head floats after prefill and final input-token verifier window, before committing the first generation. No API traces or phase waits; not clean speed/capacity evidence.',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'active':True,'completed':False,'healthy':False,'requests':[],'snapshots':[],'steps':[],
        'deadline_seconds':900,'protocol_timeout_seconds':600,'log_limit_bytes':8*1024**3}
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
    end=time.monotonic()+20;view=memoryview(data)
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
    health=json.loads((base/'post-event-ack-cb-cnr-v01402-stall-health/record.json').read_text())
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
    binary=root/'build-sycl-event-ack-no-root-prefill-20261008/strata'
    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    build=json.loads((base/('pure-upstream-v0.1.40.2-20261008' if mode=='pure' else 'upstream-e8ca-refresh-20261007')/'build-record.json').read_text())
    assert build['passed'] and not build['active']
    if mode=='pure':
        assert record['binary_sha256']==build['binary_sha256']
        record['source_status_before']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        assert not record['source_status_before']
    else:
        receipt=base/'event-ack-no-root-prefill-v01402-build/record.json'
        expected=json.loads(receipt.read_text())
        assert expected['passed'] and not expected['active'] and expected['baseline_inputs_unchanged']
        assert record['binary_sha256']==expected['candidate_binary_sha256']
        record['candidate_build_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
        matched_receipt=base/'event-ack-registered-copy-v01402-matched-build/record.json'
        matched=json.loads(matched_receipt.read_text())
        assert matched['passed'] and not matched['active'] and matched['production_inputs_unchanged']
        assert expected['baseline_build_receipt_sha256']==hashlib.sha256(matched_receipt.read_bytes()).hexdigest()
        assert expected['baseline_binary_sha256']==matched['candidate_binary_sha256']
        private_source=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
        assert all(hashlib.sha256((private_source/path).read_bytes()).hexdigest()==value for path,value in matched['candidate_sources'].items())
        original=private_source/'sycl/src/prefill/kernels.dp.cpp'
        candidate=binary.parent/'source/kernels.dp.cpp'
        assert hashlib.sha256(original.read_bytes()).hexdigest()==expected['original_source_sha256']
        assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected['candidate_source_sha256']
        assert candidate.read_text()==original.read_text().replace('sycl::ext::oneapi::experimental::use_root_sync','')
        record['candidate_sources']=matched['candidate_sources']
        record['prefill_kernel_candidate']={'file':str(candidate),'sha256':expected['candidate_source_sha256'],'removed_sites':expected['removed_sites']}
        assert len(expected['removed_sites'])==14
    final=json.loads((base/'owned-upstream-e8ca-refresh-short-final-float-storage/record.json').read_text());assert final['healthy'] and not final['active']
    for previous_path in list(base.glob('clean-v01402-short-*/record.json'))+list(base.glob('clean-v01402-default-*/record.json'))+list(base.glob('owned-v01402*-*/record.json'))+list(base.glob('owned-event-ack-v01402-code32k-*/record.json'))+list(base.glob('owned-cb-off-v01402-code32k-*/record.json'))+list(base.glob('owned-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-profile-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-registered-copy-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-registered-copy-cb-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-host-profile-v01402-code32k-*/record.json'))+list(base.glob('owned-qsa-*-v01402-code32k-*/record.json'))+list(base.glob('owned-compact2-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-ram-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-gpu-release-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-processing-default-v01402-code32k-*/record.json'))+list(base.glob('owned-main-vmm-*-v01402-code32k-*/record.json')):
        if previous_path.parent==out:continue
        previous=json.loads(previous_path.read_text());assert not previous['active']
        known_failures={'owned-v01402-code32k-patched-default-head-r2': '74e4b01947b304ce63400cf41930dffcbcaa7482d4342770557da4a00520fae6', 'owned-event-ack-v01402-code32k-diagnostic-r1': 'b1d072a15ceaf7b30c35c6599b7df4a91f62caeb0fa8469000a980f6c3c8ed46', 'owned-event-ack-cb-cnr-v01402-code32k-state-r1': '073f0a2081f333c21254c7f1388681473bada935cacf5db56ee3d7994f856f95'}
        if previous_path.parent.name in known_failures:
            assert hashlib.sha256(previous_path.read_bytes()).hexdigest()==known_failures[previous_path.parent.name]
            assert not any(previous['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
            assert health['started_utc']>previous['finished_utc']
            record['known_terminal_failures']=known_failures
        else:assert previous['healthy']
        for key in ['inferior','debugger']:
            old=previous[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'))
    env=m.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
    env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')
    env.update(STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_ATTN_LAYOUT='1')
    env.update(STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='32767',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1')
    fractions={'main-vmm-kept-ram':'0','main-vmm-half-ram':'0.5','main-vmm-full-ram':'1','main-vmm-full-snapshot':'1'}
    env.update(STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',STRATA_PREFILL_CACHE_RESTORE='snapshot' if mode=='main-vmm-full-snapshot' else 'ram',STRATA_PREFILL_CACHE_RELEASE_FRAC=fractions[mode],STRATA_PREFILL_CACHE_VERIFY='1')
    assert not any(k in env for k in ['STRATA_PREFILL_ATTN_BATCH','STRATA_GDN_KEYHEAD','STRATA_GDN_KEYHEAD_TUNED','STRATA_PROMPT_ATTN_XMX'])
    assert 'EnableImplicitConvertionToCounterBasedEvents' not in env
    assert 'MKL_CBWR' not in env
    if mode=='patched-tuned':env.update(STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',STRATA_PREFILL_EXPERT_WAIT_BATCH='0')
    if phase=='diagnostic':env=m.diagnostic_environment(env)
    if phase!='diagnostic':assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in env)
    env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')
    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')
    assert 'STRATA_PREFILL_SYNC' not in env
    assert 'STRATA_PREFILL_TRANSFER_TIMING' not in env
    record['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission','EnableImplicitConvertionToCounterBasedEvents','MKL_CBWR']}
    args=list(reference['args'])
    for key,value in [('--pcie-frac','0'),('--max-context','33024'),('--prefill','8192'),('--expert-cache','128')]:args[args.index(key)+1]=value
    args+=['--kv','int8','--prompt-cache','0']
    args[args.index('--conversation-cache-mib')+1]='0'
    argv=[str(binary)]+args;record['argv']=argv
    record['mode']=mode;record['phase']=phase;record['repetition']=repetition
    record['scope']='Private matched32K main-cache lease comparison:64MiB segmented allocation in every arm, release fraction0/0.5/1 and immutable-RAM or GPU snapshot restoration. Same all-GPU residuals,8192 chunks, MTP decode-only release, e82fc5 executable and model arithmetic as the passed parent. Full MTP and occupied main-cache payload verification is enabled; all66 main state parts,248320 first-head floats,64IDs and every logprob must match the completed default before timing. First use is logged and validated. No full256K, speed, hang-prevention or production-adoption claim.'
    previous_cnr=json.loads((base/'event-ack-registered-copy-cb-v01402-state-sequence/record.json').read_text())
    assert previous_cnr['passed'] and not previous_cnr['active']
    record['previous_goal_turn']='progress: nine exact32K main-state/head/output controls and six clean processing-order jobs archived in86e13c1b17104e3f53b45c56f85c35c74c2a4099. All owned identities absent; production unchanged. GPU-row pair varies about11%, MTP RAM restoration repeats at248ms. Proceed to matched main-cache release/restore conditions.'
    record['configuration_note']='State and head capture are after the prompt computation and are not performance evidence. 32768 prompt tokens from the same expanded real-code fixture with a completed review question and assistant prefix, up to64 output tokens, context33024, chunk8192, int8 KV, normal MTP4, five CPU workers, cache128 requested, pcie0. FIRST0 and RING8 common to both. No prompt/decode retirement tuning for pure/patched-default. Diagnostic logs do not count as performance results. Prompt caching disabled and ckpt0 for a complete fresh read; actual cache count/workspaces/chunks/freeVRAM are captured from engine logs; upstream mixed packing differs from explicit-count fork contract.'
    record['reference_binary_sha256']='79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223'
    record['reference_result_sha256']=hashlib.sha256((base/'model-lease-copy-off/normal-mtp.json').read_bytes()).hexdigest()
    record['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',root/'sycl/src/prefill/prefill.cpp']}
    record['source_review_sha256']=hashlib.sha256((base/'main-cache-release-v01402-source-review/record.json').read_bytes()).hexdigest()
    record['host_lease_check_sha256']=hashlib.sha256((base/'main-cache-lease-v01402-host-check/record.json').read_bytes()).hexdigest()
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
        fixture=base/'coding-review-32k-tokens.txt'
        tokens=list(map(int,fixture.read_text().split()))[:32768];assert len(tokens)==32768
        request='GEN 64 ckpt=0 logprobs=5 '+','.join(map(str,tokens))
        prompt_path=out/'input-tokens.txt';prompt_path.write_text(' '.join(map(str,tokens))+'\n')
        record['prompt_fixture']={'source':str(fixture),'source_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'tokens':len(tokens),'file':str(prompt_path),'sha256':hashlib.sha256(prompt_path.read_bytes()).hexdigest()};save()
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
        fields=value.split();assert fields[0]=='DONE' and fields[2]=='32768' and fields[5] in ['length','stop']
        assert 0<int(fields[1])<=64 and len(result['ids'])==int(fields[1]) and len(result['logprobs'])==len(result['ids'])
        result['finish_reason']=fields[5]
        result['measurement']={'generated_tokens':int(fields[1]),'prompt_tokens':int(fields[2]),'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),'wall_seconds':time.monotonic()-request_start}
        mm=result['measurement'];assert mm['prompt_ms']>0 and mm['decode_ms']>0
        mm['prefill_tok_s']=1000*mm['prompt_tokens']/mm['prompt_ms'];mm['decode_tok_s']=1000*mm['generated_tokens']/mm['decode_ms']
        result['finite_logprobs']=True
        head=out/'first-head.bin';data=head.read_bytes();floats=array.array('f');floats.frombytes(data)
        assert len(floats)==248320 and all(map(math.isfinite,floats))
        result['first_head']={'file':str(head),'sha256':hashlib.sha256(data).hexdigest(),'floats':len(floats),'finite':True,'captured_at':'first verifier window row0, after32767 batched prompt tokens and held-out input token, before first-window commit'}
        state=out/'prefill-state.bin';parts=[]
        with state.open('rb') as stream:
            while header:=stream.read(8):
                assert len(header)==8
                size=struct.unpack('=Q',header)[0];offset=stream.tell();left=size;digest=hashlib.sha256()
                while left:
                    data=stream.read(min(left,1048576));assert data;digest.update(data);left-=len(data)
                parts.append({'index':len(parts),'offset':offset,'bytes':size,'sha256':digest.hexdigest()})
        assert len(parts)==66 and parts[0]['bytes']==8
        result['prefill_state']={'file':str(state),'bytes':state.stat().st_size,'parts':parts,'captured_at':'after32767 batched prompt tokens, before the held-out input token and first verifier window'}
        baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]
        result['comparison_to_default_counter_control']={
            'first_head_equal':result['first_head']['sha256']==baseline['first_head']['sha256'],
            'ids_equal':result['ids']==baseline['ids'],'logprobs_equal':result['logprobs']==baseline['logprobs'],
            'different_prefill_state_parts':[i for i,(a,c) in enumerate(zip(result['prefill_state']['parts'],baseline['prefill_state']['parts'])) if a['bytes']!=c['bytes'] or a['sha256']!=c['sha256']]}
        record['requests'].append(result);save()
        c=result['comparison_to_default_counter_control']
        assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal']), 'reject before timing: full-state/head/output mismatch'

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
    engine_log=out/'debugger/inferior.stderr'
    project=[]
    with engine_log.open(errors='replace') as log:
        for item in log:
            if item.startswith('strata '):project.append(item.rstrip())
    (out/'project-messages.txt').write_text('\n'.join(project)+'\n')
    record['engine_log_bytes']=engine_log.stat().st_size
    record['engine_log_sha256']=hashlib.file_digest(engine_log.open('rb'),'sha256').hexdigest()
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
print(json.dumps({k:record.get(k) for k in ['mode','phase','healthy','elapsed_seconds','error','exit_code','exit_signal','new_fault_messages','cleanup']} | {'measurements':[x['measurement'] for x in record['requests']]},indent=2))
if not record['healthy']:raise SystemExit(1)
