from pathlib import Path
import datetime, hashlib, json, shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro = parent/'code32k/repro32k'
out = repro/'event-ack-no-profile-cb-cnr'
assert not out.exists()
out.mkdir()

def copy(source, destination):
    target = out/destination
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source,target)

private = []
names = ['owned-event-ack-no-profile-cb-cnr-v01402-code32k-diagnostic-r1',
         'owned-event-ack-no-profile-cb-cnr-v01402-code32k-state-r1',
         'owned-event-ack-no-profile-cb-cnr-v01402-code32k-state-r2']
for name in names:
    directory = base/name
    record = json.loads((directory/'record.json').read_text())
    assert not record['active'] and record['healthy'] and record['completed']
    assert record['exit_code']==0 and not record['new_fault_messages']
    assert not any(record['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
    assert len(record['requests'])==1
    request = record['requests'][0]
    assert request['measurement']['prompt_tokens']==32768
    c = request['comparison_to_production_cnr_control']
    assert not c['different_prefill_state_parts'] and all(c[key] for key in ['first_head_equal','ids_equal','logprobs_equal'])
    for name2 in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','input-tokens.txt']:
        path = directory/name2
        if path.exists():copy(path,'runs/'+name+'/'+name2)
    for name2 in ['inferior-argv.json','inferior-environment.json']:
        path = directory/'debugger'/name2
        if path.exists():copy(path,'runs/'+name+'/debugger/'+name2)
    for path in (directory/'probes').glob('*'):
        if path.is_file():copy(path,'runs/'+name+'/probes/'+path.name)
    for path in [directory/'debugger/inferior.stderr',directory/'first-head.bin',directory/'prefill-state.bin']:
        with path.open('rb') as stream:value=hashlib.file_digest(stream,'sha256').hexdigest()
        private.append({'file':str(path),'bytes':path.stat().st_size,'sha256':value})
sequence = base/'event-ack-no-profile-cb-cnr-v01402-state-sequence/record.json'
record = json.loads(sequence.read_text())
assert not record['active'] and record['passed'] and len(record['steps'])==2
copy(sequence,'sequence.json')
audit = base/'event-ack-no-profile-copy-v01402-source-audit/record.json'
assert not json.loads(audit.read_text())['adoption_allowed']
copy(audit,'source-audit.json')
for name in ['event-ack-no-profile-copy-v01402-build','event-ack-no-profile-copy-v01402-v2-build']:
    directory = base/name
    for path in directory.glob('*'):
        if path.is_file():copy(path,'build/'+name+'/'+path.name)
candidate = root/'build-sycl-event-ack-no-profile-copy-v2-20261008/source'
for name in ['prefill.cpp','event_completion.hpp','event_completion_test.cpp']:
    copy(candidate/name,'private-source/'+name)
for name in ['build_event_ack_no_profile_copy_v01402.py','build_event_ack_no_profile_copy_v01402_v2.py',
             'prepare_event_ack_no_profile_v01402_controller.py','prepare_event_ack_no_profile_v01402_controller_v2.py',
             'run_owned_event_ack_no_profile_cb_cnr_v01402_code32k.py',
             'run_event_ack_no_profile_cb_cnr_v01402_state_sequence.py',
             'audit_event_ack_no_profile_copy_v01402.py','archive_v01402_event_ack_no_profile.py']:
    copy(base/name,'controllers/'+name)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'README.md').write_text('''# Nonprofiling private copy queue: three32K state/head comparisons

The private binary4c184a9dfe0632ca1e79b8dfe4273b21f9eee0a420ee48ab1b28c07d57269a9b
retains the actual memcpy event for source reuse and creates an owned in-order
copy queue in the compute queue's context/device. Copy profiling is requested
only when STRATA_PREFILL_TRANSFER_TIMING is set. Compute queue properties and
GPU arithmetic are unchanged. Implicit counter conversion is disabled and
MKL_CBWR=AUTO remains set as in the previous production CNR control.

The logged first use and two fresh unlogged state/head processes each read the
same32,768-token code fixture with8192-token chunks, int8 KV and normal MTP4.
All three complete64finite-logprob outputs, exit normally and leave no owned
inferior/debugger. No new xe fault is recorded. All66prefill state parts,
all248,320first-head float bytes, all64IDs and every protocol logprob match
one another and the production CNR logged control. The [sequence](sequence.json)
passes both full comparisons. All timings are excluded from speed comparisons
because these runs capture full state/head and the first also logs API calls.

This is a narrow mechanism experiment, not an adopted queue implementation.
The [source audit](source-audit.json) identifies that this directly owned queue
is absent from device_ext::_queues: existing global waits and default-queue
sync_barrier no longer include it. Normal run/relayout/release have explicit
copy waits, but that does not validate every cache/graph retirement path.
drain_pipeline itself only joins successors. A separate full private rebuild
uses a registered copy-queue factory to preserve the global-wait contract.
Neither full262,144-cell serving nor stall prevention is proved here.

The initial source build fails because an initializer-list comma reaches the
single-argument DPCT_CHECK_ERROR macro. The second private build adds the needed
parentheses and passes compile/link and the identical CPU ring/lifetime suite,
including ASan/UBSan. The first controller preparer fails before writing either
GPU controller because it expects one build-receipt path occurrence, where
there are two; its v2 replaces both. Those executed versions are preserved.

Raw multi-GiB API logs and583,631,432-byte state files stay in private state
storage. Their byte counts and SHA256 digests are in private-artifacts.json.
The archived records include exact argv/environment, fixture and binary hashes,
full protocol output, pre/post kernel journal probes and ownership cleanup.
Production sources/objects/binary remain unchanged by the private builds.
''')
with (repro/'README.md').open('a') as stream:
    stream.write('''

The [nonprofiling private copy queue](event-ack-no-profile-cb-cnr/README.md)
completes three32K comparisons with identical prefill state, full head and
output, without new xe faults. It is not adopted: its directly owned queue is
outside the existing global-wait registry. A registered factory is separately
rebuilt and validated before considering any clean timing or production use.
''')
for directory in [out,repro,parent/'code32k',parent]:
    path = directory/'manifest.json'
    meta = json.loads(path.read_text()) if path.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    meta.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                revision_reason='Append three32K private nonprofiling copy comparisons and explicit global-wait limitation; preserve prior raw results.',
                files={str(p.relative_to(directory)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.rglob('*')) if p.is_file() and p!=path})
    path.write_text(json.dumps(meta,indent=2)+'\n')
    for name, item in meta['files'].items():
        path2 = directory/name
        assert path2.stat().st_size==item['bytes'] and hashlib.sha256(path2.read_bytes()).hexdigest()==item['sha256']
    print(directory.name,len(meta['files']))
