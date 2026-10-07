from pathlib import Path
import datetime, hashlib, json, shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro = parent/'code32k/repro32k'
out = repro/'event-ack-registered-copy-cb-cnr'
sequence = base/'event-ack-registered-copy-cb-cnr-v01402-state-sequence/record.json'
seq = json.loads(sequence.read_text())
assert not seq['active'] and seq['passed'] and len(seq['steps'])==2
assert not out.exists()
out.mkdir()

def copy(source,destination):
    target=out/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
def private_file(path):
    with path.open('rb') as stream:value=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'file':str(path),'bytes':path.stat().st_size,'sha256':value}

private=[]
names=['owned-event-ack-registered-copy-cb-cnr-v01402-code32k-diagnostic-r1',
       'owned-event-ack-registered-copy-cb-cnr-v01402-code32k-state-r1',
       'owned-event-ack-registered-copy-cb-cnr-v01402-code32k-state-r2']
for name in names:
    directory=base/name;r=json.loads((directory/'record.json').read_text())
    assert not r['active'] and r['healthy'] and r['completed'] and r['exit_code']==0
    assert not r['new_fault_messages'] and not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
    assert len(r['requests'])==1 and r['requests'][0]['measurement']['prompt_tokens']==32768
    c=r['requests'][0]['comparison_to_production_cnr_control']
    assert not c['different_prefill_state_parts'] and all(c[key] for key in ['first_head_equal','ids_equal','logprobs_equal'])
    for name2 in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','input-tokens.txt']:
        p=directory/name2
        if p.exists():copy(p,'runs/'+name+'/'+name2)
    for name2 in ['inferior-argv.json','inferior-environment.json']:
        p=directory/'debugger'/name2
        if p.exists():copy(p,'runs/'+name+'/debugger/'+name2)
    for p in (directory/'probes').glob('*'):
        if p.is_file():copy(p,'runs/'+name+'/probes/'+p.name)
    for p in [directory/'debugger/inferior.stderr',directory/'first-head.bin',directory/'prefill-state.bin']:
        private.append(private_file(p))
copy(sequence,'sequence.json')
builds=['event-ack-registered-copy-v01402-build','event-ack-registered-copy-v01402-v2-build',
        'event-ack-registered-copy-v01402-v3-build','event-ack-registered-copy-v01402-matched-build']
for name in builds:
    directory=base/name;r=json.loads((directory/'record.json').read_text());assert not r['active']
    for p in directory.glob('*'):
        if not p.is_file():continue
        if p.name in ['strata-before-config-match'] or p.stat().st_size>1048576:
            private.append(private_file(p))
            if p.suffix in ['.stdout','.stderr']:
                # Preserve an exact diagnostic excerpt; the complete compiler
                # output is retained privately and hashed above.
                data=p.read_bytes();lines=data.splitlines(keepends=True)
                indexes=[i for i,line in enumerate(lines) if b'FAILED:' in line or b'error:' in line]
                chosen=set(range(max(0,len(lines)-12),len(lines)))
                for i in indexes:chosen.update(range(max(0,i-2),min(len(lines),i+5)))
                target=out/'build'/name/(p.name+'.excerpt');target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(b''.join(lines[i] for i in sorted(chosen)))
        else:copy(p,'build/'+name+'/'+p.name)
matched=json.loads((base/builds[-1]/'record.json').read_text())
assert matched['passed'] and matched['production_inputs_unchanged'] and not matched['selected_configuration_differences']
source=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
for path,value in matched['candidate_sources'].items():
    p=source/path;assert private_file(p)['sha256']==value;copy(p,'private-source/'+path)
for name in ['build_event_ack_registered_copy_v01402.py','build_event_ack_registered_copy_v01402_v2.py',
             'build_event_ack_registered_copy_v01402_v3.py','match_event_ack_registered_copy_v01402_config.py',
             'prepare_event_ack_registered_copy_v01402_controller.py',
             'prepare_event_ack_registered_copy_v01402_controller_v3.py',
             'prepare_event_ack_registered_copy_v01402_controller_matched.py',
             'run_owned_event_ack_registered_copy_cb_cnr_v01402_code32k_matched.py',
             'run_event_ack_registered_copy_cb_cnr_v01402_state_sequence_matched.py',
             'archive_v01402_event_ack_registered_copy.py']:
    copy(base/name,'controllers/'+name)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'README.md').write_text('''# Registered nonprofiling copy queue: matched32K state/head controls

On Arc B57010GiB / Ryzen5 5600X /128GiB RAM, kernel7.0.0-38,
NEO26.31.39395.14, oneAPI2026.1.1, the private binary
f02213fc440a1f11b64857fceaa028ab43eeee7fdfb40e5cec33abca411a9705
retains actual DMA events until source reuse and requests profiling for its
registered in-order copy queue only when transfer timing is requested.
Compute queue properties, GPU arithmetic and explicit release waits remain
unchanged. The factory inserts into device_ext::_queues, so device-wide waits
and sync_barrier on the default queue still include this copy queue. This
avoids the global-wait gap in the earlier directly owned queue experiment.

All translation units are rebuilt against the same changed header. The final
configure argv matches the production build, including the C/C++ compilers,
IQ2_S GCC groups, shared pinned ggml checkout and precise floating-point flags.
The [matched-build receipt](build/event-ack-registered-copy-v01402-matched-build/record.json)
compares the selected Strata/GGML/compiler settings with no differences and
verifies production source/header/binary hashes are unchanged. The completion
header is byte-identical to the previously sanitized CPU ring/lifetime suite.

The first logged check and two fresh unlogged state/head captures all use the
same32,768-token fixture,8192-token chunks, int8 KV and normal MTP4.
All three complete64finite-logprob outputs, normal exit and owned cleanup;
no new xe fault is recorded. All66prefill state parts, all248,320first-head
float bytes, all64IDs and every protocol logprob match one another and the
production CNR logged control. The [sequence](sequence.json) passes both
comparisons. Implicit counter conversion remains disabled and MKL_CBWR=AUTO
remains set as in that control; these settings alone previously failed.

These are equality checks, not clean timing jobs. State/head dumping is enabled
in all three, and the first also captures UR/Level Zero diagnostics. No speed
claim is based on these times. A stable32K comparison does not prove general
stall prevention, default-environment reproduction or full262,144-cell serving.
The candidate remains private and unadopted while those gates are open.

The first full source archive omitted third_party/ggml headers and fails before
linking. The second includes those headers but leaves CMake's C compiler at
system cc, which rejects -fp-model=precise. Both complete failures are preserved.
The third selects icx and completes the full rebuild, but its default IQ2_S
GCC option is off; before any GPU use, the configuration-matching controller
preserves that binary then reruns the exact production configure argv and
incrementally rebuilds. Its snapshot hash remains in the matching receipt.
Only the matched build is used by the GPU controllers. Earlier generated
v2/v3 model controllers are not executed. No package/service/reset/rebind or
reboot change is made by these controllers.

Private multi-GiB API logs,583,631,432-byte state files, compiler logs over1MiB
and the earlier binary snapshot have byte counts and SHA256 hashes in
private-artifacts.json. Public records retain exact argv/environment, fixture
and binary hashes, full protocol output, kernel journal and ownership cleanup.
Raw excerpts retain their original bytes. Manifests verify archived contents.
''')
with (repro/'README.md').open('a') as stream:
    stream.write('''

The [registered copy-queue candidate](event-ack-registered-copy-cb-cnr/README.md)
preserves the existing global-wait contract and passes three matched32K full
state/head/output checks with no new xe faults. Every translation unit uses
the same changed header and production compile settings match. This is still
a private correctness candidate; clean timing, default-environment repeats and
full262,144-cell serving remain separate gates, with no adoption claimed.
''')
for directory in [out,repro,parent/'code32k',parent]:
    path=directory/'manifest.json';meta=json.loads(path.read_text()) if path.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    meta.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                revision_reason='Append registered nonprofiling copy candidate with three exact32K comparisons, matched configuration and preserved failed builds.',
                files={str(p.relative_to(directory)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.rglob('*')) if p.is_file() and p!=path})
    path.write_text(json.dumps(meta,indent=2)+'\n')
    for name,item in meta['files'].items():
        p=directory/name;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    print(directory.name,len(meta['files']))
