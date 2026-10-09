"""Archive terminal qualification metadata; leave large private dumps untouched."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess

b=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-server-env-v0141-2026-10-09')
scope=Path('bench/results/2026-10-05-sycl-prefill-scaling/native-expert-copy-v0141-20261009')
out=root/scope/'full256k'
run=b/'owned-native-expert-copy-v0141-full256k-diagnostic-r1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((run/'record.json').read_text())
assert not r['active'], 'owned full256K still active; no archive mutation admitted'
assert r['completed'] and r['healthy'] and r['math_gate_passed'] and r['full_lifecycle_passed']
assert r['physical256k_sequence_completed'] and r['capacity_sequence_completed']
assert r['exit_code']==0 and not r['exit_signal'] and not r['new_fault_messages'] and not any(r['cleanup'].values())
assert r['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert r['binary_sha256']=='2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f'
assert r['source_head']=='6caa1421f9212750a425fe1729139ffdde6e9f9a'
assert r['native_copy_queue_startup_gate_passed'] and r['native_copy_queue_ordinals'] and all(x==1 for x in r['native_copy_queue_ordinals'])
assert len(r['requests'])==12 and all(q['math_gate_passed'] for q in r['requests'])
assert len(r['sessions'])==6 and all(q['passed'] for q in r['sessions'])
for role in ['inferior','debugger']:
    old=r[role];p=Path('/proc')/str(old['pid'])/'stat'
    if p.exists():
        text=p.read_text();assert int(text[text.rfind(')')+2:].split()[19])!=old['start_ticks']
sessions={q['name']:q for q in r['sessions']}
first=sessions['save-full256k-first'];repeat=sessions['save-full256k-repeat'];restored=sessions['save-full256k-restored']
assert first['all_saved_state_and_kv_bytes_equal_dd5'] and repeat['all_saved_state_and_kv_bytes_equal'] and restored['all_saved_state_and_kv_bytes_equal']
assert first['full_capacity_gate']['ignored_tensor_bytes']==0 and first['full_capacity_gate']['last_physical_cell']==262143
assert first['bytes']==repeat['bytes']==restored['bytes']==4239591088
assert r['actual_executable_identity']['sha256']==r['binary_sha256']
assert r['actual_target_environment']['STRATA_PREFILL_COPY_ENGINE']=='1'
assert r['actual_target_environment']['STRATA_DECODE_TIMING']=='1' and 'STRATA_VERIFY_PROFILE' not in r['actual_target_environment']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='eafeb059602541f749cbb7e734ae17452a86c9e2'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
assert not out.exists()
controller=b/'run_owned_native_expert_copy_v0141_full256k_v1.py'
assert sha(controller)=='adc183010b01bd3e270f6966babc5ba4d2422b846c06d28a39efb9fafbff2f9d'
generator=b/'prepare_owned_native_expert_copy_v0141_full256k_v1.py'
assert sha(generator)=='35078fdf601082d83bd6a41b3d4b6077f3fc42a4a7edfaadad087f908ed7ed31'
files={
    'record.json':run/'record.json',
    'controller.py':controller,
    'controller-generator.py':generator,
    'controller-preparation.json':b/'native-expert-copy-v0141-full-controller-preparation-v1.json',
    'cpu-preflight.json':b/'native-expert-copy-v0141-full-cpu-preflight-v1.json',
    'launch-admission.json':b/'native-expert-copy-v0141-full-launch-admission-v1.json',
    'project-messages.txt':run/'project-messages.txt',
    'protocol.stdout.raw':run/'protocol.stdout.raw',
    'archive-controller.py':Path(__file__),
}
assert sha(files['cpu-preflight.json'])=='cd869ce63cf3db53a29f02e813b4260795e7ead381bee2c9f23bbdc497122308'
assert all(p.is_file() for p in files.values())
# The owned controller already scanned/hashed the API log once at termination.
# This archive copies only its recorded identity, without reading it again.
log=run/'debugger/inferior.stderr'
assert log.stat().st_size==r['engine_log_bytes']
out.mkdir()
for name,p in files.items():
    shutil.copyfile(p,out/name)
    assert sha(p)==sha(out/name)
private={'scope':'Large artifacts remain private. Identity comes from the terminal owned controller; this archive does not rescan the API log or tensor dumps. RESTORE records refer to the saved file and do not contain a new image identity.','engine_api_log':{'file':str(log),'bytes':r['engine_log_bytes'],'sha256':r['engine_log_sha256']},'sessions':[{k:q[k] for k in ['name','tokens','bytes','file']}|{'image_identity':{k:q.get('image',{})[k] for k in ['sha256','semantic_sha256'] if k in q.get('image',{})}} for q in r['sessions']]}
(out/'private-artifact-identities.json').write_text(json.dumps(private,indent=2)+'\n')
readme='''# Native expert-copy full context qualification

On 2026-10-09 JST, Arc B570 10 GiB / Ryzen 5 5600X / 128 GiB RAM completed the native copy-only candidate's own 262,144-cell lifecycle. The uniformly rebuilt binary is `2721f8ef...`, source `6caa1421...`; flags, argv, runtime selection and source hashes are recorded. This uses the qualified fixture and request history from the DD5 reference and integrated upstream v0.1.41 reference.

All 12 requests and six session operations passed. Two fresh inputs of 262,140 tokens produced four tokens through physical cell 262,143. First heads, all 66 live intermediate/state parts, output IDs, LP5 scores and MTP counts matched the references. The first saved session contains 4,239,591,088 bytes; every saved tensor byte, including inactive MTP data and spare cells, matches the DD5 reference with only the already qualified version/config identity difference. No tensor bytes are ignored. Repeated and actually restored sessions match the first saved state.

The repeated session's complete file hash differs because its checkpoint LRU-use counter changes from 1 to 2. The reader explicitly excludes that counter, file offsets and derived checksums from its semantic identity, while independently validating the checksums. The parsed comparison finds no other saved semantic difference. The actually restored session's complete file hash equals the first file hash. This qualification asserts all saved tensor bytes and relevant state, not identical LRU bookkeeping in every file.

The sequence also covers an ordinary 32K input and its restored continuation, a clipped two-token tail, refusal of output at context 262,144, refusal of three output tokens when only two cells remain, and another fresh 32K input after those refusals. Both owned processes exited normally with code 0, no survivors or forced cleanup, and no new GPU fault in the captured kernel interval. The native queue metadata uses copy-only ordinal 1 throughout. This is observed success for this configuration, not a guarantee against every driver fault.

Flushed Level Zero / UR diagnostics and the existing host decode counters are enabled here; these elapsed times are excluded from speed comparisons. The separate [quiet 32K comparison](../quiet32k/README.md) measured later prefill at 541.376 tokens/s versus 449.782 (+20.364%), with a separate same-source candidate-off control. LP5 scoring is included; decode variation did not establish a small change. The full qualification now passes for native272/ring8. The opt-in engine change remains on its tuning branch; the later stream-all host fix and larger rings require their own GPU checks and full lifecycle qualification.

The archive contains the original terminal receipt, protocol, small project messages, controller, CPU admission and launch records. Large API/tensor/session files remain private; their recorded identities are preserved without a second API-log scan. `record.json` retains historical `adopted: false` and the actual private artifact paths. See `manifest.json` for the copied artifact hashes.
'''
(out/'README.md').write_text(readme)
manifest={'active':False,'archived':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipt_sha256':sha(run/'record.json'),'source_head':r['source_head'],'binary_sha256':r['binary_sha256'],'full_lifecycle_passed':True,'adopted':False,'scope':'Terminal qualification archive only; no engine/source/default changes or additional GPU operation.','files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir())}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name,identity in manifest['files'].items():assert sha(out/name)==identity['sha256']
parent=root/scope/'README.md'
old=parent.read_text()
needle='The change is opt-in and retains existing math, CPU work, completion checks and ring barriers. All owned runs finish normally. The candidate remains unadopted until its own full262144 lifecycle qualification; that gate is pending. The first diagnostic archive preserves its historical first-use scope.'
assert old.count(needle)==1
parent.write_text(old.replace(needle,'The change is opt-in and retains existing math, CPU work, completion checks and ring barriers. All owned runs finish normally. Its own [full262144 lifecycle qualification](full256k/README.md) now passes: two fresh full inputs, all saved tensor bytes, actual restoration, clipped tails, capacity refusals and a later32K input. The opt-in engine remains on its tuning branch; this archive does not change defaults. The first diagnostic archive preserves its historical first-use scope.'))
proof={'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(out),'manifest_sha256':sha(out/'manifest.json'),'receipt_sha256':sha(run/'record.json'),'archive_controller_sha256':sha(Path(__file__)),'files':len(manifest['files'])+1,'gpu_executed':False,'engine_changed':False,'api_log_rescanned':False}
p=b/'native-expert-copy-v0141-full-archive-proof-v1.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
