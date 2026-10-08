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
assert mode=='full-kv-access' and phase=='diagnostic' and repetition==5
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
out = base/f'owned-{mode}-v01402-full256k-{phase}-r{repetition}';out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
source=observer/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(probes)
record={'scope':'Same integrated fork32K input/configuration, fresh process each time, capture all248320 first-head floats after prefill and final input-token verifier window, before committing the first generation. No API traces or phase waits; not clean speed/capacity evidence.',
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
    health_path=base/'post-indexer-spare-rejection-v01402-health-v1/record.json'
    assert hashlib.sha256(health_path.read_bytes()).hexdigest()=='d45fe2a9cbc28d6db83e544fc277681cd27bf71c2d53e80b1364b65906f1daa3'
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
    binary=root/'build-sycl-visible-commit-v1-20261008/strata'
    visible_receipt=base/'visible-commit-v01402-build-v1/record.json'
    visible_build=json.loads(visible_receipt.read_text())
    assert visible_build['passed'] and not visible_build['active'] and visible_build['baseline_inputs_unchanged']
    assert visible_build['cpu_cases_passed']==12
    assert visible_build['candidate_binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()
    assert visible_build['candidate_source_sha256']==hashlib.sha256((binary.parent/'source/generate.cpp').read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in visible_build['link_input_sha256'].items())
    record['visible_commit_build_receipt_sha256']=hashlib.sha256(visible_receipt.read_bytes()).hexdigest()
    record['visible_commit_source_sha256']=visible_build['candidate_source_sha256']
    spare_receipt=base/'indexer-spare-commit-v01402-build-v1/record.json'
    spare_build=json.loads(spare_receipt.read_text())
    assert spare_build['passed'] and not spare_build['active'] and spare_build['baseline_inputs_unchanged']
    assert spare_build['baseline_binary_sha256']==visible_build['candidate_binary_sha256']
    assert spare_build['visible_commit_build_receipt_sha256']==hashlib.sha256(visible_receipt.read_bytes()).hexdigest()
    binary=Path(spare_build['candidate_binary'])
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==spare_build['candidate_binary_sha256']=='7f054f8338c6552a5ae6dba548d2020ff2e3bd2661f4a41bb5a2289c56d07891'
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in spare_build['link_input_sha256'].items())
    assert hashlib.sha256((binary.parent/'source/native_qsa_indexer.dp.cpp').read_bytes()).hexdigest()==spare_build['candidate_source_sha256']
    spare_review=base/'indexer-spare-commit-v01402-source-review-v1/record.json'
    checked=json.loads(spare_review.read_text());assert checked['passed'] and not checked['active']
    assert checked['candidate_source_sha256']==spare_build['candidate_source_sha256']
    rejected_path=base/'owned-full-kv-access-v01402-full256k-diagnostic-r4/record.json'
    rejected=json.loads(rejected_path.read_text())
    assert rejected['healthy'] and rejected['completed'] and not rejected['active'] and not rejected['math_gate_passed']
    assert hashlib.sha256(rejected_path.read_bytes()).hexdigest()==spare_build['full_capacity_rejection_receipt_sha256']==checked['rejection_receipt_sha256']
    for key in ['inferior','debugger']:
        old_id=rejected[key];now=process_identity(old_id['pid']);assert not now or now['start_ticks']!=old_id['start_ticks']
    record['indexer_spare_build_receipt_sha256']=hashlib.sha256(spare_receipt.read_bytes()).hexdigest()
    record['indexer_spare_source_sha256']=spare_build['candidate_source_sha256']
    record['indexer_spare_source_review_sha256']=hashlib.sha256(spare_review.read_bytes()).hexdigest()
    access_receipt=base/'kv-access-safe-v01402-build-v2/record.json'
    access_build=json.loads(access_receipt.read_text())
    assert access_build['passed'] and not access_build['active'] and access_build['baseline_inputs_unchanged']
    assert access_build['baseline_binary_sha256']=='1441ad556cb1d8ecddf42e525a1462cc53f9251fae4734c008046e10e8b0e599'
    assert access_build['candidate_binary_sha256']==visible_build['baseline_binary_sha256']
    assert visible_build['access_build_receipt_sha256']==hashlib.sha256(access_receipt.read_bytes()).hexdigest()
    assert access_build['candidate_source_sha256']==hashlib.sha256((Path(access_build['candidate_binary']).parent/'source/layer.cpp').read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in access_build['link_input_sha256'].items())
    record['access_build_receipt_sha256']=hashlib.sha256(access_receipt.read_bytes()).hexdigest()
    record['access_source_sha256']=access_build['candidate_source_sha256']
    kv_receipt=base/'kv-stream-safe-v01402-build-v3/record.json'
    kv_build=json.loads(kv_receipt.read_text())
    assert kv_build['passed'] and not kv_build['active'] and kv_build['baseline_inputs_unchanged']
    assert kv_build['candidate_binary_sha256']==access_build['baseline_binary_sha256']
    assert access_build['kv_build_receipt_sha256']==hashlib.sha256(kv_receipt.read_bytes()).hexdigest()
    assert kv_build['baseline_binary_sha256']=='e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323'
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in kv_build['link_input_sha256'].items())
    kv_source=Path(kv_build['candidate_binary']).parent/'source/kv_stream.dp.cpp'
    assert hashlib.sha256(kv_source.read_bytes()).hexdigest()==kv_build['candidate_source_sha256']
    record['kv_build_receipt_sha256']=hashlib.sha256(kv_receipt.read_bytes()).hexdigest()
    record['kv_source_sha256']=kv_build['candidate_source_sha256']
    review=base/'kv-stream32k-v01402-source-review-v2/record.json'
    audit=json.loads(review.read_text());assert audit['passed'] and not audit['active']
    assert audit['candidate_sha256']==record['kv_source_sha256']
    record['kv_source_review_sha256']=hashlib.sha256(review.read_bytes()).hexdigest()
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
        assert kv_build['baseline_binary_sha256']==expected['candidate_binary_sha256']
        record['candidate_build_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
        assert access_build['prefill_build_receipt_sha256']==record['candidate_build_receipt_sha256']
        matched_receipt=base/'event-ack-registered-copy-v01402-matched-build/record.json'
        matched=json.loads(matched_receipt.read_text())
        assert matched['passed'] and not matched['active'] and matched['production_inputs_unchanged']
        assert expected['baseline_build_receipt_sha256']==hashlib.sha256(matched_receipt.read_bytes()).hexdigest()
        assert expected['baseline_binary_sha256']==matched['candidate_binary_sha256']
        private_source=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
        assert all(hashlib.sha256((private_source/path).read_bytes()).hexdigest()==value for path,value in matched['candidate_sources'].items())
        original=private_source/'sycl/src/prefill/kernels.dp.cpp'
        candidate=Path(expected['candidate_binary']).parent/'source/kernels.dp.cpp'
        assert hashlib.sha256(original.read_bytes()).hexdigest()==expected['original_source_sha256']
        assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected['candidate_source_sha256']
        assert candidate.read_text()==original.read_text().replace('sycl::ext::oneapi::experimental::use_root_sync','')
        record['candidate_sources']=matched['candidate_sources']
        record['prefill_kernel_candidate']={'file':str(candidate),'sha256':expected['candidate_source_sha256'],'removed_sites':expected['removed_sites']}
        assert len(expected['removed_sites'])==14
    final=json.loads((base/'owned-upstream-e8ca-refresh-short-final-float-storage/record.json').read_text());assert final['healthy'] and not final['active']
    for previous_path in list(base.glob('clean-v01402-short-*/record.json'))+list(base.glob('clean-v01402-default-*/record.json'))+list(base.glob('owned-v01402*-*/record.json'))+list(base.glob('owned-event-ack-v01402-code32k-*/record.json'))+list(base.glob('owned-cb-off-v01402-code32k-*/record.json'))+list(base.glob('owned-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-profile-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-registered-copy-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-registered-copy-cb-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-host-profile-v01402-code32k-*/record.json'))+list(base.glob('owned-qsa-*-v01402-code32k-*/record.json'))+list(base.glob('owned-compact2-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-ram-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-gpu-release-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-processing-default-v01402-code32k-*/record.json'))+list(base.glob('owned-main-vmm-*-v01402-code32k-*/record.json'))+list(base.glob('owned-kv-stream32k-v01402-code32k-*/record.json'))+list(base.glob('owned-kv-layer-major32k-*-*/record.json'))+list(base.glob('owned-kv-stream32k-quiet-*/record.json'))+list(base.glob('owned-full-kv-access-*/record.json')):
        if previous_path.parent==out:continue
        previous=json.loads(previous_path.read_text());assert not previous['active']
        known_failures={'owned-v01402-code32k-patched-default-head-r2': '74e4b01947b304ce63400cf41930dffcbcaa7482d4342770557da4a00520fae6', 'owned-event-ack-v01402-code32k-diagnostic-r1': 'b1d072a15ceaf7b30c35c6599b7df4a91f62caeb0fa8469000a980f6c3c8ed46', 'owned-event-ack-cb-cnr-v01402-code32k-state-r1': '073f0a2081f333c21254c7f1388681473bada935cacf5db56ee3d7994f856f95', 'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1': '3de5e40d7a9bb390112c8de7bc6ee29604e62f8fed45e516cbf57ffe8cb0dfdb', 'owned-full-kv-access-v01402-full256k-diagnostic-r2': 'da8657868131392654c034ed6c004176fd624e7bd065f03c4c73aa3b3969a7ac', 'owned-full-kv-access-v01402-full256k-diagnostic-r3': 'e2003090cf5556999cdc2c0fda31ba29c6e86c6bb1d85836ea9af614765a944f'}
        if previous_path.parent.name in known_failures:
            assert hashlib.sha256(previous_path.read_bytes()).hexdigest()==known_failures[previous_path.parent.name]
            assert not any(previous['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
            assert health['started_utc']>previous['finished_utc']
            if previous_path.parent.name=='owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1':
                assert previous['completed'] and previous['exit_code']==0 and not previous['math_gate_passed'] and not previous['new_fault_messages']
                record['known_mathematical_rejection']=previous_path.parent.name
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
    env.update(STRATA_PREFILL_LAYER_MAJOR='0',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0')
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
    args=list(reference['args'])
    for key,value in [('--pcie-frac','0'),('--max-context','262144'),('--prefill','8192'),('--expert-cache','128')]:args[args.index(key)+1]=value
    args+=['--kv','int8','--kv-resident','32768','--prompt-cache','1','--prompt-cache-every','262139','--prompt-cache-root','0','--turn-token','-1']
    args[args.index('--conversation-cache-mib')+1]='0'
    argv=[str(binary)]+args;record['argv']=argv
    record['mode']=mode;record['phase']=phase;record['repetition']=repetition
    record['scope']='Indexer spare repair after partial speculative commit plus Full256K occupancy/repeat/session restore/clipped verify tail/refusal/later32K correctness gate. All inputs >=32768. Diagnostic timing excluded from speed. KV streaming candidate, chunk-major, context262144, resident32768, own staging, no prefetch or layer-major cache leases. Three fresh full32768-token requests in one logged/validated normal-MTP process on private v3 KV binary. Accepted8192 chunks. Each64-output reply has a newly written head and complete raw66-part state; used cells, recurrent/PLE/indexer state, complete head/IDs/logprobs and MTP43/66 must match the accepted baseline. Only page-rounded future cells may differ and are retained separately; no exclusions at full262144 occupancy. Trace phase memory and existing-graph retirement/restoration. Diagnostic durations excluded from speed; not full256K/adoption proof.'
    previous_cnr=json.loads((base/'event-ack-registered-copy-cb-v01402-state-sequence/record.json').read_text())
    assert previous_cnr['passed'] and not previous_cnr['active']
    record['previous_goal_turn']='progress: nine exact32K main-state/head/output controls and six clean processing-order jobs archived in86e13c1b17104e3f53b45c56f85c35c74c2a4099. All owned identities absent; production unchanged. GPU-row pair varies about11%, MTP RAM restoration repeats at248ms. Proceed to matched main-cache release/restore conditions.'
    record['configuration_note']='PC1/ckpt1 for real SAVE/RESTORE; turn-token=-1 and root=0 preserve the accepted chunk geometry. A single periodic checkpoint at262139 is taken after the existing final full-input chunk; no checkpoint is taken in32K controls; fresh full reads must report RESUME0 after an interposed different32K prompt. Context262144/kv-resident32768 (main mode1, MTP ring), own stage, chunk-major0, no cache release or prefetch; identical math gates on32768/GEN64. State and head capture are after the prompt computation and are not performance evidence. 32768 prompt tokens from the same expanded real-code fixture with a completed review question and assistant prefix, up to64 output tokens, context33024, chunk8192, int8 KV, normal MTP4, five CPU workers, cache128 requested, pcie0. FIRST0 and RING8 common to both. No prompt/decode retirement tuning for pure/patched-default. Diagnostic logs do not count as performance results. Prompt caching disabled and ckpt0 for a complete fresh read; actual cache count/workspaces/chunks/freeVRAM are captured from engine logs; upstream mixed packing differs from explicit-count fork contract.'
    record['reference_binary_sha256']='79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223'
    record['reference_result_sha256']=hashlib.sha256((base/'model-lease-copy-off/normal-mtp.json').read_bytes()).hexdigest()
    record['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',root/'sycl/src/prefill/prefill.cpp']}
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
    baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]
    assert list(map(int,baseline['protocol'][-1].split()[6:8]))==[43,66]
    record['legacy_control_mtp_counts']=list(map(int,baseline['protocol'][-1].split()[6:8]))
    # The preserved actual SAVE and final window prove two invisible tokens were
    # counted/committed. The fixed code keeps the same66 offered drafts and all
    #64 IDs/logprobs, but counts41 visible accepted drafts.
    baseline['mtp_counts']=[41,66]
    record['expected_visible_control_mtp_counts']=[41,66]
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
        previous_full=next(x for x in rejected['requests'] if x['name']=='full256k-first')
        full_first=request('full256k-first',full,4,previous_full,fresh=True,tail=True)
        if not math_ok:reject()
        full_file=out/'full256k-first.session.bin'
        full_saved=session('SAVE','save-full256k-first',full_file,262143)
        full_saved['full_capacity_gate']=full_kv_gate(full_saved['image']);save()
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
    record['engine_log_bytes']=engine_log.stat().st_size
    record['engine_log_sha256']=hashlib.file_digest(engine_log.open('rb'),'sha256').hexdigest()
    if mode=='pure':
        record['source_status_after']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        assert not record['source_status_after']
    record['completed']=True
    record['math_gate_passed']=math_ok and bool(record.get('capacity_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])
    # A normal mathematical rejection is terminal evidence, not a GPU-health failure.
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
print(json.dumps({k:record.get(k) for k in ['mode','phase','healthy','elapsed_seconds','error','exit_code','exit_signal','new_fault_messages','cleanup']} | {'measurements':[x['measurement'] for x in record['requests'] if 'measurement' in x]},indent=2))
if not record['healthy'] or not record.get('math_gate_passed'):raise SystemExit(1)
