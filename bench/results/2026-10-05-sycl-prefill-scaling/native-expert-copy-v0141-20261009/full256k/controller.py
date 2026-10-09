"""Logged native expert-copy v0.1.41 physical256K gate with owned read-only native-counter failure capture."""
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
import fcntl

base = Path(__file__).parent
observer = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
root = observer.parent/'perf-sycl-prefill-native-copy-v0141-20261009'
mode=sys.argv[1]; phase=sys.argv[2]; repetition=int(sys.argv[3])
assert mode=='nativeon' and phase=='diagnostic' and repetition==1
assert sys.argv[4:] in [[], ['--cpu-preflight']]
sys.path.insert(0,str(observer/'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity
sequence_path=base/'native-expert-copy-v0141-quiet32k-comparison-sequence-v1/record.json'
assert hashlib.sha256(sequence_path.read_bytes()).hexdigest()=='50121345d4ca8d7638d0c620989e6ecbb78bc80c2634dc14dcfea593f002de5a'
sequence=json.loads(sequence_path.read_text())
assert not sequence['active'], 'native-copy32K comparison still active; no full-capacity process'
assert sequence['passed'] and len(sequence['steps'])==6
assert sequence['configuration_equal_except_explicit_native_copy_flag']
for step in sequence['steps']:
    p=Path(step['receipt']);assert hashlib.sha256(p.read_bytes()).hexdigest()==step['receipt_sha256']
    data=json.loads(p.read_text())
    assert not data['active'] and data['healthy'] and data['completed'] and data['math_gate_passed']
    assert data['exit_code']==0 and not data['exit_signal'] and not data['new_fault_messages']
    assert not any(data['cleanup'].values())
    for role in ['inferior','debugger']:
        old=data[role];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
qualified_path=base/'owned-profile-definition-v01402-full256k-diagnostic-r7/record.json'
assert hashlib.sha256(qualified_path.read_bytes()).hexdigest()=='18436b6361831d41aa19e1374454aed458e746f3e234092a1793fd15b823d2a3'
qualified=json.loads(qualified_path.read_text())
assert not qualified['active'] and qualified['healthy'] and qualified['math_gate_passed']
assert qualified['physical256k_sequence_completed'] and qualified['capacity_sequence_completed']
assert qualified['exit_code']==0 and not qualified['exit_signal'] and not any(qualified['cleanup'].values())
assert qualified['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for role in ['inferior','debugger']:
    old=qualified[role];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
numerical_path=base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json'
assert hashlib.sha256(numerical_path.read_bytes()).hexdigest()=='a96f3686903b63138a827c173e0ea5b1a8beebbee8ef9cdae46459bf9f2c9699'
qualified_numerical=json.loads(numerical_path.read_text())
assert qualified_numerical['math_gate_passed'] and not qualified_numerical['active']
reference_build_path=base/'integrated-upstream-v0.1.41-20261009-v2/build-record.json'
reference_build=json.loads(reference_build_path.read_text())
assert reference_build['passed'] and not reference_build['active'] and reference_build['commit']=='1eb89482a4afd20277ae0405780ed4f8eb98eb20'
assert reference_build['binary_sha256']=='86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
build_path=base/'native-expert-copy-v0141-private-build-v1/record.json'
assert hashlib.sha256(build_path.read_bytes()).hexdigest()=='74b48ad90996c942fd74fa826d22dfecf76145d8dec82a40cfa0d35dc9b65fad'
build=json.loads(build_path.read_text())
assert build['passed'] and not build['active'] and build['compiled_engine'] and not build['adopted']
assert build['source_head']=='6caa1421f9212750a425fe1729139ffdde6e9f9a'
binary=Path(build['binary'])
assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binary_sha256']=='2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f'
candidate_source=root/'sycl/src/prefill/prefill.cpp'
candidate_header=root/'sycl/include/dpct/device.hpp'
assert hashlib.sha256(candidate_source.read_bytes()).hexdigest()=='6399d65feea884f932a5a1079a91c80228293bfc3be371732b48413977229f84'
assert hashlib.sha256(candidate_header.read_bytes()).hexdigest()=='8805874ecb1d58d7acea8ca931154a38d0ad3d623f7c5ee8d599fa2ba56a841b'
flags_path=base/'native-expert-copy-v0141-uniform-build-flags-v1.json'
assert hashlib.sha256(flags_path.read_bytes()).hexdigest()=='54a1a491a7f00874a90af1b7776d0654b8b001e9aea3da99f6e0b1dfe0a83e53'
flags=json.loads(flags_path.read_text());assert flags['passed'] and not flags['active']
assert flags['old_compile_count']==flags['candidate_compile_count']==114 and not flags['missing_sources'] and not flags['different_flags']
assert flags['build_receipt_sha256']==hashlib.sha256(build_path.read_bytes()).hexdigest()
first_path=base/'owned-native-expert-copy-v0141-code32k-diagnostic-r1/record.json'
assert hashlib.sha256(first_path.read_bytes()).hexdigest()=='83670ee10b75d702c621e0ce029c5be1a4701ac6684d08b4f22de905704ae908'
first=json.loads(first_path.read_text())
assert not first['active'] and first['completed'] and first['healthy'] and first['math_gate_passed']
assert first['native_copy_queue_startup_gate_passed'] and all(x==1 for x in first['native_copy_queue_ordinals'])
assert first['exit_code']==0 and not first['exit_signal'] and not first['new_fault_messages'] and not any(first['cleanup'].values())
assert first['binary_sha256']==build['binary_sha256'] and first['reference_binary_sha256']==reference_build['binary_sha256']
assert len(first['requests'])==4 and all(x['math_gate_passed'] for x in first['requests'])
assert all(x['first_head']['finite'] and x['first_head']['floats']==248320 and len(x['prefill_state']['parts'])==66 for x in first['requests'])
for role in ['inferior','debugger']:
    old=first[role];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
baseline_full_path=base/'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
assert hashlib.sha256(baseline_full_path.read_bytes()).hexdigest()=='168d7896cbe3a60dfc604a08b6fe310f16d823a17d13d8a013260bfc7facb0a4'
baseline_full=json.loads(baseline_full_path.read_text())
assert not baseline_full['active'] and baseline_full['completed'] and baseline_full['healthy'] and baseline_full['math_gate_passed']
assert baseline_full['physical256k_sequence_completed'] and baseline_full['capacity_sequence_completed']
assert baseline_full['binary_sha256']==reference_build['binary_sha256']
for role in ['inferior','debugger']:
    old=baseline_full[role];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
assert first['boot_id']==baseline_full['boot_id']==qualified['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert hashlib.sha256(Path('/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0').read_bytes()).hexdigest()=='bfdc0f26bf88bd0bccd66fc25302a4b22559f8fe30e499be18529be3ed610e60'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==build['source_head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
assert shutil.disk_usage(base).free>170*1024**3
# Preserve terminal failed-history evidence; require a separate completed4-read gate.
rejected_path=base/'owned-upstream-v0141-integrated-full256k-diagnostic-r1/record.json'
assert hashlib.sha256(rejected_path.read_bytes()).hexdigest()=='2cabcde178049b9c3479632e299f54f8b2d262c3270c15dd60772babaa9e7be7'
rejected=json.loads(rejected_path.read_text())
assert not rejected['active'] and rejected['completed'] and rejected['healthy']
assert not rejected['math_gate_passed'] and rejected['exit_code']==0 and not rejected['exit_signal']
assert not rejected['new_fault_messages'] and not any(rejected['cleanup'].values())
for role in ['inferior','debugger']:
    prior=rejected[role];now=process_identity(prior['pid'])
    assert not now or now['start_ticks']!=prior['start_ticks']
subset_path=base/'upstream-v0141-tuned-fourfresh-numerical-subset-20261009-v1.json'
assert hashlib.sha256(subset_path.read_bytes()).hexdigest()=='95ae2a0a7e61f2240f6138f21fdcb35c1a70ec54300b22a7b99fd6b92873b4a7'
subset=json.loads(subset_path.read_text())
assert not subset['active'] and subset['passed'] and not subset['source_full_math_gate_passed']
assert subset['binary_sha256']==reference_build['binary_sha256'] and subset['source_commit']==reference_build['commit']
assert subset['source_receipt_sha256']==hashlib.sha256(rejected_path.read_bytes()).hexdigest()
assert [x['name'] for x in subset['requests']]==['control32k-before','alternate32k-first','control32k-repeat','alternate32k-repeat']
assert all(x['math_gate_passed'] and all(x['comparison'].values()) for x in subset['requests'])
version_cpu_path=base/'upstream-v0141-full-history-and-version-comparator-host-check-20261009-v1.json'
assert hashlib.sha256(version_cpu_path.read_bytes()).hexdigest()=='c083340eae6fc5a62ebb079fc6124a50fcc8ee875f316b40296a55a2024157a0'
version_cpu=json.loads(version_cpu_path.read_text())
assert version_cpu['passed'] and not version_cpu['active'] and version_cpu['tensor_bytes_ignored_by_new_comparator']==0
version_comparator_path=base/'compare_saved_session_versions_v0141_v1.py'
assert hashlib.sha256(version_comparator_path.read_bytes()).hexdigest()==version_cpu['comparator_sha256']=='0260db007151cde044651208dc531249985d3afb1212b1b19500b883c2eba8e0'
from compare_saved_session_versions_v0141_v1 import same_cross_version_saved_tensors
if sys.argv[4:]==['--cpu-preflight']:
    print(json.dumps({'cpu_preflight_passed':True,'gpu_executed':False,'binary_sha256':build['binary_sha256'],'quiet_sequence_sha256':hashlib.sha256(sequence_path.read_bytes()).hexdigest(),'full_reference_sha256':hashlib.sha256(baseline_full_path.read_bytes()).hexdigest()}))
    sys.exit(0)
lock=(base/'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
os.chdir(root)
out = base/f'owned-native-expert-copy-v0141-full256k-{phase}-r{repetition}';out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
source=observer/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(probes)
record={'scope':'Native copy-only queue272 physical262144 qualification with exactly the qualified DD5 request history. Separate completed4fresh32K A/B/A/B head/state/output gate required; this run uses A, SAVE, live continuation before first full read. Two fresh262140+4 through cell262143, every saved tensor byte including inactive MTP against DD5, same-version repeat/disk restore, clipped tail, refusals and later32K. Flushed API diagnostics; no performance or adoption claim.',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'active':True,'completed':False,'healthy':False,'requests':[],'snapshots':[],'steps':[],
        'deadline_seconds':10800,'protocol_timeout_seconds':5400,'log_limit_bytes':128*1024**3}
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
    health_path=base/'post-device-profile-abort-v01402-health-v1/record.json'
    assert hashlib.sha256(health_path.read_bytes()).hexdigest()=='27eda22ac64827a042db7ef90c204a644e8dbc1d56ed2c4132aae57768f91cff'
    health=json.loads(health_path.read_text())
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
    record['binary_sha256']=build['binary_sha256']
    record['build_receipt_sha256']=hashlib.sha256(build_path.read_bytes()).hexdigest()
    record['updated32k_comparison_receipt_sha256']=hashlib.sha256(sequence_path.read_bytes()).hexdigest()
    record['previous_mismatched_history_receipt_sha256']=hashlib.sha256(rejected_path.read_bytes()).hexdigest()
    record['version_identity_comparator_sha256']=hashlib.sha256(version_comparator_path.read_bytes()).hexdigest()
    record['version_identity_cpu_check_sha256']=hashlib.sha256(version_cpu_path.read_bytes()).hexdigest()
    record['qualified_DD5_full_receipt_sha256']=hashlib.sha256(qualified_path.read_bytes()).hexdigest()
    record['physical256k_sequence_completed']=False
    capture_cpu_path=base/'owned-native-counter-v01402-cpu-v1/record.json'
    assert hashlib.sha256(capture_cpu_path.read_bytes()).hexdigest()=='c93b119422938718e230d9e7c02377b7f0e6d48dc2095cb4add442f2e1025e86'
    capture_cpu=json.loads(capture_cpu_path.read_text());assert capture_cpu['passed'] and not capture_cpu['active']
    capture_path=base/'capture_owned_native_counter_v01402_v1.py'
    assert hashlib.sha256(capture_path.read_bytes()).hexdigest()==capture_cpu['capture_controller_sha256']
    assert hashlib.sha256((observer/'sycl/tools/owned_gdb.py').read_bytes()).hexdigest()==capture_cpu['owned_helper_sha256']
    from capture_owned_native_counter_v01402_v1 import capture as capture_native_counter
    record['native_counter_capture_controller_sha256']=capture_cpu['capture_controller_sha256']
    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'))
    env=m.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib'
    env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')
    env.update(STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_ATTN_LAYOUT='1')
    env.update(STRATA_PREFILL_LAYER_MAJOR='0',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0',STRATA_PREFILL_COPY_ENGINE='1',STRATA_DECODE_TIMING='1')
    env.pop('STRATA_PREFILL_HOST_TIMING',None)
    assert 'STRATA_VERIFY_PROFILE' not in env
    # Streaming uses its existing chunk-major path; no dormant layer-major leases.
    assert not any(k in env for k in [
        'STRATA_PREFILL_RELEASE_CACHE','STRATA_PREFILL_RELEASE_DRAFT',
        'STRATA_PREFILL_CACHE_ALLOC','STRATA_PREFILL_LAYER_MAJOR_R_GPU'])
    assert not any(k in env for k in ['STRATA_PREFILL_ATTN_BATCH','STRATA_GDN_KEYHEAD','STRATA_GDN_KEYHEAD_TUNED','STRATA_PROMPT_ATTN_XMX'])
    assert 'EnableImplicitConvertionToCounterBasedEvents' not in env
    assert 'MKL_CBWR' not in env
    if mode=='patched-tuned':env.update(STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',STRATA_PREFILL_EXPERT_WAIT_BATCH='0')
    env['STRATA_TRACE']='1'
    if phase=='diagnostic':env=m.diagnostic_environment(env)
    if phase!='diagnostic':assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in env)
    env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')
    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')
    assert 'STRATA_PREFILL_SYNC' not in env
    assert 'STRATA_PREFILL_TRANSFER_TIMING' not in env
    record['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission','EnableImplicitConvertionToCounterBasedEvents','MKL_CBWR']}
    record['candidate_build_receipt_sha256']=hashlib.sha256(build_path.read_bytes()).hexdigest()
    record['compatibility_change']='Only opt-in native expert-copy queue selection differs from qualified869 math. Candidate272 was uniformly rebuilt with all114 compiler commands matched. Four-fresh32K heads/live-state/output pass; this new binary still needs full-capacity proof.'
    record['decode_host_counters']='STRATA_DECODE_TIMING prints existing unconditional host counters once per request. No STRATA_VERIFY_PROFILE, GPU timestamps, added synchronization or GPU profiler. Logged diagnostic times are excluded from speed comparisons.'
    record['first_native_copy_diagnostic_receipt_sha256']=hashlib.sha256(first_path.read_bytes()).hexdigest()
    record['baseline_full_capacity_receipt_sha256']=hashlib.sha256(baseline_full_path.read_bytes()).hexdigest()
    record['uniform_flags_receipt_sha256']=hashlib.sha256(flags_path.read_bytes()).hexdigest()
    record['root']=str(root);record['source_head']=build['source_head']
    record['performance_eligible']=False;record['adopted']=False;record['full_lifecycle_passed']=False
    args=list(reference['args'])
    for key,value in [('--pcie-frac','0'),('--max-context','262144'),('--prefill','8192'),('--expert-cache','128')]:args[args.index(key)+1]=value
    args+=['--kv','int8','--kv-resident','32768','--prompt-cache','1','--prompt-cache-every','262139','--prompt-cache-root','0','--turn-token','-1']
    args[args.index('--conversation-cache-mib')+1]='0'
    argv=[str(binary)]+args;record['argv']=argv
    record['mode']=mode;record['phase']=phase;record['repetition']=repetition
    record['scope']='Native copy-only queue272 physical262144 qualification with exactly the qualified DD5 request history. Separate completed4fresh32K A/B/A/B head/state/output gate required; this run uses A, SAVE, live continuation before first full read. Two fresh262140+4 through cell262143, every saved tensor byte including inactive MTP against DD5, same-version repeat/disk restore, clipped tail, refusals and later32K. Flushed API diagnostics; no performance or adoption claim.'
    previous_cnr=json.loads((base/'event-ack-registered-copy-cb-v01402-state-sequence/record.json').read_text())
    assert previous_cnr['passed'] and not previous_cnr['active']
    record['previous_goal_turn']='Native-copy first4fresh32K full-head/live-state proof and all24 quiet reads must finish before this full-capacity process. Baseline869 full qualification does not qualify candidate272.'
    record['configuration_note']='Context262144, chunk8192, int8 KV resident32768, cache128, workers5, pcie0, MTP4. PC1/ckpt1/every262139/root0/turn-token-1 preserves qualified32K geometry and actual disk restoration. Fresh reads require RESUME0/REUSED0. Flushed diagnostic UR/ZE/ZEL/Strata logs retained; no profiler, additional phase waits, cache retirement or prefetch. Expected/current native counters are read through source-derived fields only at an owned stopped failure.'
    record['reference_binary_sha256']=qualified['binary_sha256']
    record['reference_result_sha256']=hashlib.sha256((base/'model-lease-copy-off/normal-mtp.json').read_bytes()).hexdigest()
    record['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',root/'sycl/src/prefill/prefill.cpp',candidate_header,flags_path]}
    record['source_review_sha256']=hashlib.sha256((base/'main-cache-repeat32k-v01402-source-review/record.json').read_bytes()).hexdigest()
    live_gate=json.loads((base/'live-prefill-state-v01402-host-check-v2/record.json').read_text())
    assert live_gate['passed'] and not live_gate['active']
    record['live_state_comparator_sha256']=hashlib.sha256((base/'compare_live_prefill_state_v01402_v2.py').read_bytes()).hexdigest()
    record['host_lease_check_sha256']=hashlib.sha256((base/'main-cache-lease-v01402-host-check/record.json').read_bytes()).hexdigest()
    reader_check=base/'saved-session-reader-v01402-host-check-v2/record.json'
    cpu_reader=json.loads(reader_check.read_text())
    assert cpu_reader['passed'] and not cpu_reader['active']
    assert cpu_reader['reader_sha256']==hashlib.sha256((base/'read_saved_session_v01402_v2.py').read_bytes()).hexdigest()
    record['session_reader_cpu_check_sha256']=hashlib.sha256(reader_check.read_bytes()).hexdigest()
    earlier=base/'owned-full-kv-access-v01402-full256k-diagnostic-r1/record.json'
    assert hashlib.sha256(earlier.read_bytes()).hexdigest()=='833dcb26548cd298b49f2da746fa48675f04e75d40f47c1409f5a3d3b60caf1d'
    negative=json.loads(earlier.read_text())
    assert negative['healthy'] and negative['completed'] and not negative['active'] and not negative['math_gate_passed']
    assert negative['exit_code']==0 and not negative['new_fault_messages']
    record['previous_config_rejection']={'receipt_sha256':hashlib.sha256(earlier.read_bytes()).hexdigest(),
        'reason':'PC1/default turn-token split the32K final chunk at32761+6, unlike the accepted PC0 geometry32767. Same64 IDs but different intermediate state/head/logprobs and42/69 rather than43/66 MTP counts. Normal exit, no GPU fault. Not a source-change attribution.'}
    from read_saved_session_v01402_v2 import read_session, full_kv_gate
    fixture=base/'coding-review-32k-tokens.txt'
    full_fixture=base/'full-context-copy-off/coding-context-256k-tokens.txt'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
    assert hashlib.sha256(full_fixture.read_bytes()).hexdigest()=='cc29e4427bcacb21c9f7df2d1f1a7d41897fce5bd76a8ca43d8c3c34d914dd2a'
    control=list(map(int,fixture.read_text().split()))[:32768]
    full_source=list(map(int,full_fixture.read_text().split()))
    suffix=[248046,198,248045,74455,198,248068,198,248069,271]
    def full_prompt(n):
        assert 32768<=n<=262144
        ids=full_source[:n-len(suffix)]+suffix
        assert len(ids)==n
        return ids
    full=full_prompt(262140)
    common=next(i for i,(a,b) in enumerate(zip(control,full)) if a!=b)
    assert common<32767, 'interposed control must clear the full-prefix checkpoint'
    record['fixtures']={'control_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        'full_sha256':hashlib.sha256(full_fixture.read_bytes()).hexdigest(),
                        'common_prefix_tokens':common}
    baseline=next(x for x in qualified['requests'] if x['name']=='control32k-before')
    alternate=full_prompt(32768)
    alternate_baseline=next(x for x in qualified_numerical['requests'] if x['name']=='alternate32k-first')
    assert baseline['mtp_counts']==[41,66] and alternate_baseline['mtp_counts']==[38,75]
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
    os.close(slave);slave=None
    actual_identity=process_identity(g.inferior['pid'])
    assert actual_identity and actual_identity['start_ticks']==g.inferior['start_ticks']
    assert Path(os.readlink('/proc/'+str(actual_identity['pid'])+'/exe')).resolve()==binary.resolve()
    assert hashlib.sha256(Path('/proc',str(actual_identity['pid']),'exe').read_bytes()).hexdigest()==record['binary_sha256']
    record['actual_executable_identity']={'inferior':actual_identity,'sha256':record['binary_sha256'],'boot_id':record['boot_id']}
    actual=dict(item.decode().split('=',1) for item in Path('/proc',str(actual_identity['pid']),'environ').read_bytes().split(b'\0') if b'=' in item)
    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment']={k:v for k,v in actual.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    assert argv[1:]==qualified['argv'][1:]
    save()

    from compare_live_prefill_state_v01402_v2 import compare_states

    stderr=out/'debugger/inferior.stderr'
    record['sessions']=[]
    math_ok=True
    assert int(record['startup'][-1].split()[1])==262144

    def captures(name, required):
        result={}
        head=out/'first-head.bin'
        if head.exists():
            kept=out/f'{name}.head.bin';assert not kept.exists();head.rename(kept)
            data=kept.read_bytes();values=array.array('f');values.frombytes(data)
            assert len(values)==248320 and all(map(math.isfinite,values))
            result['first_head']={'file':str(kept),'sha256':hashlib.sha256(data).hexdigest(),'floats':len(values),'finite':True}
        state=out/'prefill-state.bin'
        if state.exists():
            kept=out/f'{name}.state.bin';assert not kept.exists();state.rename(kept)
            parts=[]
            with kept.open('rb') as stream:
                while header:=stream.read(8):
                    assert len(header)==8
                    size=struct.unpack('=Q',header)[0];offset=stream.tell();left=size;digest=hashlib.sha256()
                    while left:
                        data=stream.read(min(left,1048576));assert data;digest.update(data);left-=len(data)
                    parts.append({'index':len(parts),'offset':offset,'bytes':size,'sha256':digest.hexdigest()})
            assert len(parts)==66 and parts[0]['bytes']==8
            result['prefill_state']={'file':str(kept),'bytes':kept.stat().st_size,'parts':parts}
        if required:
            assert 'first_head' in result and 'prefill_state' in result, 'missing fresh full-read capture'
        return result

    def request(name,tokens,new,expected=None,fresh=False,allow=True,tail=False):
        global current,math_ok
        current=name
        assert len(tokens)>=32768
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        start_offset=stderr.stat().st_size
        command=f'GEN {new} ckpt=1 logprobs=5 '+','.join(map(str,tokens))+'\n'
        (out/f'{name}.input-tokens.txt').write_text(' '.join(map(str,tokens))+'\n')
        result={'name':name,'input_tokens':len(tokens),'max_new':new,'ids':[],'logprobs':[],'protocol':[],
                'request_sha256':hashlib.sha256(command.encode()).hexdigest(),'allowed':allow}
        record['requests'].append(result);save();send(command.encode());event('request',name=name)
        request_started=time.monotonic()
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith('T '):result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):
                assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in value.split()[1:])
                result['logprobs'].append(value)
            if value.startswith(('DONE ','ERR ')):break
        result['diagnostic_wall_seconds']=time.monotonic()-request_started
        windows=[]
        with stderr.open('rb') as stream:
            stream.seek(start_offset)
            for item in stream:
                if item.startswith(b'strata trace: window '):
                    match=re.search(rb'window (-?\d+) (-?\d+)',item);assert match
                    windows.append(list(map(int,match.groups())))
        result['verify_windows']=windows
        assert all(pos>=0 and count>0 and pos+count<=262144 for pos,count in windows)
        if not allow:
            assert value.startswith('ERR prompt') and not result['ids'] and not result['logprobs'] and not windows
            assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
            result['math_gate_passed']=True;save();return result
        assert value.startswith('DONE '),value
        fields=value.split()
        assert int(fields[2])==len(tokens) and int(fields[1])==new and len(result['ids'])==new and len(result['logprobs'])==new
        result['mtp_counts']=list(map(int,fields[6:8]))
        result['measurement']={'generated_tokens':int(fields[1]),'prompt_tokens':int(fields[2]),
                               'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),'purpose':'diagnostic only; excluded from performance'}
        result['resume_tokens']=[int(v.split()[1]) for v in result['protocol'] if v.startswith(('RESUME ','REUSED '))]
        assert result['resume_tokens']
        if fresh:assert all(n==0 for n in result['resume_tokens']), 'a repeated full read was silently reused'
        result.update(captures(name,fresh))
        result['math_gate_passed']=True
        if expected is not None:
            comparison={'ids_equal':result['ids']==expected['ids'],
                        'logprobs_equal':result['logprobs']==expected['logprobs']}
            if fresh:
                comparison['first_head_equal']=result['first_head']['sha256']==expected['first_head']['sha256']
                state_cmp=compare_states(result['prefill_state'],expected['prefill_state'],len(tokens)-1)
                result['live_prefill_comparison']=state_cmp
                comparison['all_live_state_equal']=not state_cmp['different_live_parts']
                comparison['mtp_counts_equal']=result['mtp_counts']==expected.get('mtp_counts',[43,66])
            elif 'first_head' in result and 'first_head' in expected:
                comparison['first_head_equal']=result['first_head']['sha256']==expected['first_head']['sha256']
            result['comparison']=comparison
            result['math_gate_passed']=all(comparison.values())
        if tail:
            assert windows and max(pos+count for pos,count in windows)==262144
            result['last_executed_physical_cell']=262143
            if new==2:
                assert windows[-1][1]==2, 'expected a clipped two-token verification window'
                result['clipped_verify_tail']=2
        math_ok=math_ok and result['math_gate_passed'];save()
        return result

    def session(command,name,path,expected_tokens):
        global current
        current=name
        if command=='SAVE':assert not path.exists()
        result={'name':name,'command':command,'file':str(path),'protocol':[]}
        record['sessions'].append(result);save();send(f'{command} {path}\n'.encode())
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith(('SAVED ','RESTORED ','SERR ')):break
        assert value.startswith('SAVED ' if command=='SAVE' else 'RESTORED '),value
        fields=value.split();assert int(fields[1])==expected_tokens and int(fields[2])==path.stat().st_size
        result['tokens']=int(fields[1]);result['bytes']=int(fields[2]);result['diagnostic_ms']=float(fields[3])
        if command=='SAVE':
            result['image']=read_session(path)
            sem=result['image']['semantic']
            dead=sem['live']['state']['dead']
            with path.open('rb') as checked_file:
                checked_file.seek(dead['offset']);dead_bytes=checked_file.read(dead['bytes'])
                assert len(dead_bytes)==12*512
                for layer in range(12):
                    pooled=sem['kv'][layer]['parts']['pooled']
                    row=expected_tokens//4
                    assert row<sem['kv'][layer]['shape'][5]
                    checked_file.seek(pooled['offset']+row*512)
                    assert checked_file.read(512)==dead_bytes[layer*512:(layer+1)*512], ('noncanonical saved spare',name,layer,row)
            result['all_saved_main_indexer_spares_equal_live_dead']=True
            assert result['image']['semantic']['live']['ids']['count']==expected_tokens
            if name=='save-control32k':
                visible_ids=control+control_first['ids'][:-1]
            elif name in ['save-full256k-first','save-full256k-repeat']:
                visible_ids=full+full_first['ids'][:-1]
            else:visible_ids=None
            if visible_ids is not None:
                exact=hashlib.sha256(array.array('i',visible_ids).tobytes()).hexdigest()
                assert result['image']['semantic']['live']['ids']['sha256']==exact
                result['saved_ids_match_exact_consumed_visible_prefix']=True
        else:result['engine_checksum_and_compatibility_validated']=True
        result['passed']=True;save();return result

    def reject():
        raise ValueError('mathematical gate rejected; no timing/adoption')

    # A mismatch leaves the engine waiting for input, so QUIT can close it normally.
    try:
        control_first=request('control32k-before',control,64,baseline,fresh=True)
        if not math_ok:reject()
        control_file=out/'control32k.session.bin'
        control_saved=session('SAVE','save-control32k',control_file,32831)
        continuation=control+control_first['ids'];assert len(continuation)==32832
        resume_reference=request('resume32k-reference',continuation,64)
        assert max(resume_reference['resume_tokens'])==32831
        # The4-read numerical gate was already completed in a separate owned run.
        # Do not prime additional MTP cells before the all-byte full-state check.
        record['prior_four_fresh32k_heads_state_output_passed']=True
        record['prior_four_fresh32k_numerical_subset_sha256']=hashlib.sha256(subset_path.read_bytes()).hexdigest()
        record['matched_before_full_request_history']=[x['name'] for x in record['requests']]
        assert record['matched_before_full_request_history']==[x['name'] for x in qualified['requests'][:2]]
        save()
        previous_full=next(x for x in qualified['requests'] if x['name']=='full256k-first')
        full_first=request('full256k-first',full,4,previous_full,fresh=True,tail=True)
        if not math_ok:reject()
        full_file=out/'full256k-first.session.bin'
        full_saved=session('SAVE','save-full256k-first',full_file,262143)
        full_saved['full_capacity_gate']=full_kv_gate(full_saved['image'])
        dd5_full_saved=next(x for x in qualified['sessions'] if x['name']=='save-full256k-first')
        full_saved['all_saved_state_and_kv_bytes_equal_dd5']=same_cross_version_saved_tensors(full_saved['image'],dd5_full_saved['image'])
        full_saved['version_identity_comparison']={'old_config':9380593023437872391,'new_config':643052213586580166,'tensor_bytes_ignored':0,'known_engine_version_identity_only':True}
        math_ok=math_ok and full_saved['all_saved_state_and_kv_bytes_equal_dd5'];save()
        if not math_ok:reject()
        request('control32k-between-full-reads',control,64,baseline,fresh=True)
        if not math_ok:reject()
        full_repeat=request('full256k-repeat',full,4,full_first,fresh=True,tail=True)
        if not math_ok:reject()
        repeat_saved=session('SAVE','save-full256k-repeat',out/'full256k-repeat.session.bin',262143)
        repeat_saved['full_capacity_gate']=full_kv_gate(repeat_saved['image'])
        repeat_saved['all_saved_state_and_kv_bytes_equal']=repeat_saved['image']['semantic_sha256']==full_saved['image']['semantic_sha256']
        math_ok=math_ok and repeat_saved['all_saved_state_and_kv_bytes_equal'];save()
        if not math_ok:reject()
        clipped=full+full_first['ids'][:2];assert len(clipped)==262142
        clip_reference=request('full256k-clipped-reference',clipped,2,tail=True)
        assert max(clip_reference['resume_tokens'])==262139
        request('control32k-before-restore',control,64,baseline,fresh=True)
        if not math_ok:reject()
        session('RESTORE','restore-control32k',control_file,32831)
        resume_restored=request('resume32k-restored',continuation,64,resume_reference)
        assert max(resume_restored['resume_tokens'])==32831
        if not math_ok:reject()
        session('RESTORE','restore-full256k',full_file,262143)
        roundtrip=session('SAVE','save-full256k-restored',out/'full256k-restored.session.bin',262143)
        roundtrip['full_capacity_gate']=full_kv_gate(roundtrip['image'])
        roundtrip['all_saved_state_and_kv_bytes_equal']=roundtrip['image']['semantic_sha256']==full_saved['image']['semantic_sha256']
        math_ok=math_ok and roundtrip['all_saved_state_and_kv_bytes_equal'];save()
        if not math_ok:reject()
        clip_restored=request('full256k-clipped-restored',clipped,2,clip_reference,tail=True)
        assert max(clip_restored['resume_tokens'])==262139
        if not math_ok:reject()
        request('no-room',full_prompt(262144),1,allow=False)
        request('one-too-many',full_prompt(262142),3,allow=False)
        request('control32k-after-refusals',control,64,baseline,fresh=True)
        record['capacity_sequence_completed']=True
        record['physical256k_sequence_completed']=True
    except ValueError as rejected:
        if not math_ok:
            record['mathematical_rejection']=str(rejected)
        else:raise

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
    queue_lines=[value for value in project if value.startswith('strata prefill copy queue: ')]
    assert queue_lines, 'native copy-only queue creation required'
    prefix='strata prefill copy queue: native copy-only, ordinal '
    suffix=', index0, in-order, profiling0'
    ordinals=[]
    for value in queue_lines:
        assert value.startswith(prefix) and value.endswith(suffix)
        ordinal=value[len(prefix):-len(suffix)]
        assert ordinal.isdecimal() and int(ordinal)==1
        ordinals.append(int(ordinal))
    record['native_copy_queue_ordinals']=ordinals
    record['native_copy_queue_startup_gate_passed']=True
    record['engine_log_bytes']=engine_log.stat().st_size
    record['engine_log_sha256']=hashlib.file_digest(engine_log.open('rb'),'sha256').hexdigest()
    if mode=='nativeon':
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==build['source_head']
        record['source_status_after']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        assert not record['source_status_after']
    record['completed']=True
    record['math_gate_passed']=math_ok and bool(record.get('capacity_sequence_completed')) and bool(record.get('physical256k_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])
    # A normal mathematical rejection is terminal evidence, not a GPU-health failure.
except BaseException as error:
    record['error']=repr(error)
    if g:
        try:
            snap=g.snapshot('failure',resume=False);record['snapshots'].append(snap)
            if snap:
                record['native_counter_at_failure']=capture_native_counter(g,snap,out/'native-counter-at-failure-v1.json')
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
    record['full_lifecycle_passed']=record['healthy'] and bool(record.get('math_gate_passed')) and bool(record.get('native_copy_queue_startup_gate_passed'))
    record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps({k:record.get(k) for k in ['mode','phase','healthy','elapsed_seconds','error','exit_code','exit_signal','new_fault_messages','cleanup']} | {'measurements':[x['measurement'] for x in record['requests'] if 'measurement' in x]},indent=2))
if not record['healthy'] or not record.get('math_gate_passed'):raise SystemExit(1)
