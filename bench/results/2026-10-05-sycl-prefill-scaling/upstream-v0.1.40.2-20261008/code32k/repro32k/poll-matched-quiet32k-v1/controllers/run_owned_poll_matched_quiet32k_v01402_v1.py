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
assert mode in ['dd5-control','dd5-poll-backoff'] and phase=='quiet' and repetition in [1,2]
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
out = base/f'owned-poll-matched-{mode}-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
source=observer/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(probes)
record={'scope':'Same integrated fork32K input/configuration, fresh process each time, capture all248320 first-head floats after prefill and final input-token verifier window, before committing the first generation. No API traces or phase waits; not clean speed/capacity evidence.',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'active':True,'completed':False,'healthy':False,'requests':[],'snapshots':[],'steps':[],
        'deadline_seconds':1200,'protocol_timeout_seconds':300,'log_limit_bytes':128*1024**2}
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
    health_path=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/post-device-profile-abort-v01402-health-v1/record.json')
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
    uniform_path=base/'dpct-profile-definition-v01402-build-v1/record.json'
    uniform=json.loads(uniform_path.read_text())
    assert uniform['passed'] and not uniform['active'] and not uniform['gpu_tested']
    assert uniform['baseline_binary_sha256']==spare_build['candidate_binary_sha256']
    assert uniform['baseline_inputs_unchanged'] and len(uniform['objects'])==7
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in uniform['baseline_link_input_sha256'].items())
    assert hashlib.sha256(Path(uniform['shadow_header']).read_bytes()).hexdigest()==uniform['shadow_header_sha256']
    for item in uniform['objects']:
        for path_key,hash_key in [('source','source_sha256'),('object','object_sha256'),('dependency','dependency_sha256')]:
            assert hashlib.sha256(Path(item[path_key]).read_bytes()).hexdigest()==item[hash_key]
    candidate_inputs={uniform['replaced_archives'].get(p,p):sha for p,sha in uniform['baseline_link_input_sha256'].items()}
    for old_archive,new_archive in uniform['replaced_archives'].items():
        expected_members=[x for x in uniform['archive_members'] if x['archive']==old_archive]
        actual_members=subprocess.check_output(['/usr/bin/ar','t',new_archive],text=True).splitlines()
        assert actual_members==[x['member'] for x in expected_members]
        for item in expected_members:
            data=subprocess.check_output(['/usr/bin/ar','p',new_archive,item['member']])
            assert hashlib.sha256(data).hexdigest()==item['after_sha256']
        candidate_inputs[new_archive]=hashlib.sha256(Path(new_archive).read_bytes()).hexdigest()
    record['uniform_candidate_link_input_sha256']=candidate_inputs

    binary=Path(uniform['candidate_binary'])
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==uniform['candidate_binary_sha256']=='dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
    uniform_review=base/'dpct-profile-definition-v01402-source-review-v3/record.json'
    linked_review=base/'dpct-profile-definition-linked-tu-audit-v3/record.json'
    assert all(json.loads(p.read_text())['passed'] and not json.loads(p.read_text())['active'] for p in (uniform_review,linked_review))
    abort_review=base/'full256k-watchdog-abort-v01402-review-v1/record.json'
    assert json.loads(abort_review.read_text())['passed']
    record['uniform_header_build_receipt_sha256']=hashlib.sha256(uniform_path.read_bytes()).hexdigest()
    record['uniform_header_source_review_sha256']=hashlib.sha256(uniform_review.read_bytes()).hexdigest()
    record['uniform_header_linked_review_sha256']=hashlib.sha256(linked_review.read_bytes()).hexdigest()
    record['previous_watchdog_abort_review_sha256']=hashlib.sha256(abort_review.read_bytes()).hexdigest()
    completed_path=base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json'
    assert hashlib.sha256(completed_path.read_bytes()).hexdigest()=='a96f3686903b63138a827c173e0ea5b1a8beebbee8ef9cdae46459bf9f2c9699'
    completed=json.loads(completed_path.read_text())
    assert completed['healthy'] and completed['completed'] and completed['math_gate_passed'] and not completed['active']
    failed_path=base/'owned-device-profile-v01402-code32k-diagnostic-r1/record.json'
    failed=json.loads(failed_path.read_text())
    assert not failed['active'] and not failed['healthy'] and not failed['completed'] and not failed['new_fault_messages']
    assert health['terminal_controller_receipt_sha256']==hashlib.sha256(failed_path.read_bytes()).hexdigest()
    assert health['started_utc']>failed['finished_utc']
    for old_run in [completed,failed]:
        assert old_run['boot_id']==record['boot_id']
        assert not any(old_run['cleanup'][k] for k in ['inferior_survived','gdb_survived'])
        for key in ['inferior','debugger']:
            old=old_run[key];now=process_identity(old['pid'])
            assert not now or now['start_ticks']!=old['start_ticks']
    poll_path=base/'prefill-poll-backoff-v01402-relink-v3/record.json'
    poll_build=json.loads(poll_path.read_text())
    assert poll_build['passed'] and not poll_build['active'] and not poll_build['gpu_tested']
    assert poll_build['baseline_inputs_unchanged'] and poll_build['only_prefill_link_input_replaced']
    assert poll_build['baseline_binary_sha256']==uniform['candidate_binary_sha256']
    assert poll_build['poll_template_definition_isolated_to_prefill_tu'] and poll_build['prefill_keeps_original_profiled_dpct_definition']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in poll_build['baseline_link_input_sha256'].items())
    for pkey,hkey in [('actual_prefill_object','actual_prefill_object_sha256'),('actual_shadow_header','actual_shadow_header_sha256'),('actual_dependency_file','actual_dependency_file_sha256'),('prefill_source','prefill_source_sha256')]:
        assert hashlib.sha256(Path(poll_build[pkey]).read_bytes()).hexdigest()==poll_build[hkey]
    binary=Path(poll_build['candidate_binary'])
    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert record['binary_sha256']==poll_build['candidate_binary_sha256']=='44a274d8c699829020abb8a85a331a638f40efdf5d6c959a076a58e818f75391'
    archive=binary.parent/'libstrata_prefill.a'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==poll_build['candidate_archive_sha256']
    assert subprocess.check_output(['/usr/bin/ar','t',str(archive)],text=True).splitlines()==[m['member'] for m in poll_build['prefill_archive_members']]
    for member in poll_build['prefill_archive_members']:
        assert hashlib.sha256(subprocess.check_output(['/usr/bin/ar','p',str(archive),member['member']])).hexdigest()==member['after_sha256']
    record['polling_build_receipt_sha256']=hashlib.sha256(poll_path.read_bytes()).hexdigest()
    record['unprofiled_same_base_numerical_gate_sha256']=hashlib.sha256(completed_path.read_bytes()).hexdigest()
    record['profiler_failure_receipt_sha256']=hashlib.sha256(failed_path.read_bytes()).hexdigest()
    numerical_path=base/'owned-poll-backoff-v01402-code32k-diagnostic-r1/record.json'
    assert hashlib.sha256(numerical_path.read_bytes()).hexdigest()=='fa8ae00e741469d6bef575affa2e263bcac4001d31c98422ba225dbc350f0a68'
    numerical=json.loads(numerical_path.read_text())
    assert numerical['healthy'] and numerical['completed'] and numerical['math_gate_passed'] and not numerical['active']
    assert numerical['exit_code']==0 and not numerical['exit_signal'] and not numerical['new_fault_messages']
    assert not any(numerical['cleanup'].values())
    assert numerical['binary_sha256']==poll_build['candidate_binary_sha256']
    for key in ['inferior','debugger']:
        old=numerical[key];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
    record['backoff_full32k_numerical_gate_sha256']=hashlib.sha256(numerical_path.read_bytes()).hexdigest()
    assert numerical['argv'][0]==str(binary)
    if mode=='dd5-control':binary=Path(uniform['candidate_binary'])
    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert record['binary_sha256']==(uniform['candidate_binary_sha256'] if mode=='dd5-control' else poll_build['candidate_binary_sha256'])
    record['single_variable']='Only prefill.cpp.o polling cadence differs between DD5 control and44a; all other link inputs, model settings and request order remain identical.'
    for p in sorted(base.glob('owned-poll-matched-*-v01402-code32k-quiet-r*/record.json')):
        if p.parent==out:continue
        old=json.loads(p.read_text());assert not old['active'] and old['healthy'] and old['math_gate_passed']
        assert old['exit_code']==0 and not old['new_fault_messages'] and not any(old['cleanup'].values())
        for key in ['inferior','debugger']:
            prior=old[key];now=process_identity(prior['pid'])
            assert not now or now['start_ticks']!=prior['start_ticks']


    record['adopted']=False
    final=json.loads((base/'owned-upstream-e8ca-refresh-short-final-float-storage/record.json').read_text());assert final['healthy'] and not final['active']
    for previous_path in list(base.glob('clean-v01402-short-*/record.json'))+list(base.glob('clean-v01402-default-*/record.json'))+list(base.glob('owned-v01402*-*/record.json'))+list(base.glob('owned-event-ack-v01402-code32k-*/record.json'))+list(base.glob('owned-cb-off-v01402-code32k-*/record.json'))+list(base.glob('owned-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-profile-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-registered-copy-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-registered-copy-cb-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-host-profile-v01402-code32k-*/record.json'))+list(base.glob('owned-qsa-*-v01402-code32k-*/record.json'))+list(base.glob('owned-compact2-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-ram-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-gpu-release-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-processing-default-v01402-code32k-*/record.json'))+list(base.glob('owned-main-vmm-*-v01402-code32k-*/record.json'))+list(base.glob('owned-kv-stream32k-v01402-code32k-*/record.json'))+list(base.glob('owned-kv-layer-major32k-*-*/record.json'))+list(base.glob('owned-kv-stream32k-quiet-*/record.json'))+list(base.glob('owned-full-kv-access-*/record.json')):
        if previous_path.parent==out:continue
        previous=json.loads(previous_path.read_text());assert not previous['active']
        known_failures={'owned-v01402-code32k-patched-default-head-r2': '74e4b01947b304ce63400cf41930dffcbcaa7482d4342770557da4a00520fae6', 'owned-event-ack-v01402-code32k-diagnostic-r1': 'b1d072a15ceaf7b30c35c6599b7df4a91f62caeb0fa8469000a980f6c3c8ed46', 'owned-event-ack-cb-cnr-v01402-code32k-state-r1': '073f0a2081f333c21254c7f1388681473bada935cacf5db56ee3d7994f856f95', 'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1': '3de5e40d7a9bb390112c8de7bc6ee29604e62f8fed45e516cbf57ffe8cb0dfdb', 'owned-full-kv-access-v01402-full256k-diagnostic-r2': 'da8657868131392654c034ed6c004176fd624e7bd065f03c4c73aa3b3969a7ac', 'owned-full-kv-access-v01402-full256k-diagnostic-r3': 'e2003090cf5556999cdc2c0fda31ba29c6e86c6bb1d85836ea9af614765a944f'}
        known_failures['owned-full-kv-access-v01402-full256k-diagnostic-r5']='e7e541f91ef053acebaa6b9ed2bc3139010e269f0b671d23f423869e2c7a882f'
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
    assert 'STRATA_TRACE' not in env
    if phase=='diagnostic':env=m.diagnostic_environment(env)
    if phase!='diagnostic':assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in env)
    assert not any(k in env for k in ['STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE',
        'STRATA_PREFILL_CACHE_VERIFY','STRATA_PREFILL_DRAFT_VERIFY','STRATA_PREFILL_TIMING',
        'STRATA_PROFILE','STRATA_VERIFY_PROFILE'])
    assert 'STRATA_PREFILL_SYNC' not in env
    assert 'STRATA_PREFILL_TRANSFER_TIMING' not in env
    record['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission','EnableImplicitConvertionToCounterBasedEvents','MKL_CBWR']}
    args=list(reference['args'])
    for key,value in [('--pcie-frac','0'),('--max-context','262144'),('--prefill','8192'),('--expert-cache','128')]:args[args.index(key)+1]=value
    args+=['--kv','int8','--kv-resident','32768','--prompt-cache','1','--prompt-cache-every','262139','--prompt-cache-root','0','--turn-token','-1']
    args[args.index('--conversation-cache-mib')+1]='0'
    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for k in env)
    argv=[str(binary)]+args;record['argv']=argv
    record['mode']=mode;record['phase']=phase;record['repetition']=repetition
    record['scope']='Matched quiet DD5 versus polling backoff, four complete32768-token reads A/B/A/B per process. Same262144 capacity/int8/resident32768/chunk8192/cache128/workers5/pcie0/MTP4/GEN64. First process read is reported separately from later reads. No API/validation logs, trace/profiler, state/head/payload dump, additional phase waits or prefix reuse. All64IDs/logprobs/MTP counts must equal completed DD5 numerical gate. No physical256K or adoption proof.'
    previous_cnr=json.loads((base/'event-ack-registered-copy-cb-v01402-state-sequence/record.json').read_text())
    assert previous_cnr['passed'] and not previous_cnr['active']
    record['previous_goal_turn']='Progress: both DD5 and44a completed four fresh32K state/head/output/logprob/MTP and actual disk-continuation gates. Unitrace GPU timestamps rejected after watchdog abort; post-exit device health and later44a math gate passed. All evidence archived and pushed in523ecdd0; candidate remains private.'
    record['configuration_note']=record['scope']+' PC1/ckpt1, turn-token=-1, root0 and every262139 preserve exactly the qualified geometry; no periodic checkpoint or SAVE/RESTORE is executed in these32K timing jobs. Alternating inputs clear live prefix reuse. First means first full input in this process; filesystem/driver caches are not globally cleared.'
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
    os.close(slave);slave=None
    now=process_identity(g.inferior['pid']);assert now and now['start_ticks']==g.inferior['start_ticks']
    assert Path(os.readlink('/proc/'+str(now['pid'])+'/exe')).resolve()==binary.resolve()
    assert hashlib.sha256(Path('/proc',str(now['pid']),'exe').read_bytes()).hexdigest()==record['binary_sha256']
    actual=dict(item.decode().split('=',1) for item in Path('/proc',str(now['pid']),'environ').read_bytes().split(b'\0') if b'=' in item)
    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment']={k:v for k,v in actual.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_','UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS','STRATA_TRACE','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_PREFILL_SYNC','STRATA_PREFILL_TRANSFER_TIMING'] for k in actual)
    assert record['argv'][1:]==completed['argv'][1:]==numerical['argv'][1:]

    save()
    assert int(record['startup'][-1].split()[1])==262144
    completed_by_name={v['name']:v for v in completed['requests']}
    alternate=full_prompt(32768)
    plan=[('control32k-first',control,'control32k-before'),
          ('alternate32k-first',alternate,'alternate32k-first'),
          ('control32k-repeat',control,'control32k-repeat'),
          ('alternate32k-repeat',alternate,'alternate32k-repeat')]
    for index,(name,tokens,reference_name) in enumerate(plan):
        current=name
        assert len(tokens)==32768
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        expected=completed_by_name[reference_name]
        command='GEN 64 ckpt=1 logprobs=5 '+','.join(map(str,tokens))+'\n'
        prompt_path=out/(name+'.input-tokens.txt')
        prompt_path.write_text(' '.join(map(str,tokens))+'\n')
        result={'name':name,'process_read':index+1,'first_process_read':index==0,
                'input_tokens':len(tokens),'request_sha256':hashlib.sha256(command.encode()).hexdigest(),
                'prompt_file':str(prompt_path),'prompt_sha256':hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                'ids':[],'logprobs':[],'protocol':[]}
        record['requests'].append(result);save()
        started_request=time.monotonic();result['start_epoch_us']=time.time_ns()/1000
        send(command.encode());event('request',name=name)
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith('ERR'):raise RuntimeError(value)
            if value.startswith('T '):result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):
                assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in value.split()[1:])
                result['logprobs'].append(value)
            if value.startswith('DONE '):break
        result['end_epoch_us']=time.time_ns()/1000
        fields=value.split()
        assert int(fields[1])==64 and int(fields[2])==32768 and fields[5]=='length'
        assert len(result['ids'])==len(result['logprobs'])==64
        result['mtp_counts']=list(map(int,fields[6:8]))
        reused=[int(v.split()[1]) for v in result['protocol'] if v.startswith(('RESUME ','REUSED '))]
        result['resume_tokens']=reused
        result['comparison']={'ids_equal':result['ids']==expected['ids'],
                              'logprobs_equal':result['logprobs']==expected['logprobs'],
                              'mtp_counts_equal':result['mtp_counts']==expected['mtp_counts'],
                              'complete_fresh_prefill':bool(reused) and all(n==0 for n in reused)}
        result['math_gate_passed']=all(result['comparison'].values())
        mm={'generated_tokens':64,'prompt_tokens':32768,'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),
            'wall_seconds':time.monotonic()-started_request,'purpose':'quiet matched full32K comparison'}
        assert mm['prompt_ms']>0 and mm['decode_ms']>0
        mm['prefill_tok_s']=32768000/mm['prompt_ms'];mm['decode_tok_s']=64000/mm['decode_ms']
        result['measurement']=mm;save()
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        if not result['math_gate_passed']:break
    record['quiet_four_fresh_reads_completed']=len(record['requests'])==4 and all(v['math_gate_passed'] for v in record['requests'])
    record['full_capacity_sequence_completed']=False

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
    record['math_gate_passed']=bool(record['quiet_four_fresh_reads_completed'])
    assert record['math_gate_passed'], 'quiet output/MTP/freshness gate rejected; no performance acceptance'
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
