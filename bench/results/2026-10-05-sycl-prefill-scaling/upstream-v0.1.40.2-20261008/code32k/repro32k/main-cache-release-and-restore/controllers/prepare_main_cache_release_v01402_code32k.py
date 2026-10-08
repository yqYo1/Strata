from pathlib import Path
import ast, datetime, hashlib, json

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
private=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
source=base/'run_owned_layer_gpu_release_v01402_code32k.py'
target=base/'run_owned_main_cache_release_v01402_code32k.py'
audit=base/'main-cache-release-v01402-source-review'
assert not target.exists() and not audit.exists()
cpu=json.loads((base/'main-cache-lease-v01402-host-check/record.json').read_text())
assert cpu['passed'] and not cpu['active'] and len(cpu['cases'])==19 and all(c['passed'] for c in cpu['cases'])
parent=json.loads((base/'layer-processing-v01402-clean-sequence/record.json').read_text())
assert parent['passed'] and not parent['active']
files=['sycl/src/prefill/prefill.cpp','sycl/src/core/expert_cache.cpp','sycl/src/core/verify.cpp','sycl/src/core/vmm.cpp','sycl/src/program/generate.cpp','sycl/include/strata/sycl_execution_policy.hpp']
hashes={str(private/f):hashlib.sha256((private/f).read_bytes()).hexdigest() for f in files}
assert hashes[str(private/files[0])]==cpu['source_sha256']
for f in files[1:]: assert (private/f).read_bytes()==(root/f).read_bytes()
text=source.read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
replace("mode=='layer2-gpu-release-layout1'", "mode in ['main-vmm-kept-ram','main-vmm-half-ram','main-vmm-full-ram','main-vmm-full-snapshot']")
replace("assert layer_gate['passed'] and not layer_gate['active']", "assert layer_gate['passed'] and not layer_gate['active']\nparent_gate=json.loads((base/'layer-processing-v01402-clean-sequence/record.json').read_text())\nassert parent_gate['passed'] and not parent_gate['active']\nsource_gate=json.loads((base/'main-cache-release-v01402-source-review/record.json').read_text())\nassert source_gate['passed'] and not source_gate['active']")
replace("+list(base.glob('owned-layer2-gpu-release-layout1-v01402-code32k-*/record.json')):", "+list(base.glob('owned-layer2-gpu-release-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-processing-default-v01402-code32k-*/record.json'))+list(base.glob('owned-main-vmm-*-v01402-code32k-*/record.json')):")
replace("    assert not any(k in env for k in ['STRATA_PREFILL_ATTN_BATCH','STRATA_GDN_KEYHEAD','STRATA_GDN_KEYHEAD_TUNED','STRATA_PREFILL_RELEASE_CACHE','STRATA_PROMPT_ATTN_XMX'])", "    fractions={'main-vmm-kept-ram':'0','main-vmm-half-ram':'0.5','main-vmm-full-ram':'1','main-vmm-full-snapshot':'1'}\n    env.update(STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',STRATA_PREFILL_CACHE_RESTORE='snapshot' if mode=='main-vmm-full-snapshot' else 'ram',STRATA_PREFILL_CACHE_RELEASE_FRAC=fractions[mode],STRATA_PREFILL_CACHE_VERIFY='1')\n    assert not any(k in env for k in ['STRATA_PREFILL_ATTN_BATCH','STRATA_GDN_KEYHEAD','STRATA_GDN_KEYHEAD_TUNED','STRATA_PROMPT_ATTN_XMX'])")
replace("record['source_review_sha256']=hashlib.sha256((base/'layer-gpu-release-v01402-source-review/record.json').read_bytes()).hexdigest()", "record['source_review_sha256']=hashlib.sha256((base/'main-cache-release-v01402-source-review/record.json').read_bytes()).hexdigest()\n    record['host_lease_check_sha256']=hashlib.sha256((base/'main-cache-lease-v01402-host-check/record.json').read_bytes()).hexdigest()")
start=text.index("    record['scope']='Private32K layer-major experiment")
end=text.index('\n',start)
text=text[:start]+"    record['scope']='Private matched32K main-cache lease comparison:64MiB segmented allocation in every arm, release fraction0/0.5/1 and immutable-RAM or GPU snapshot restoration. Same all-GPU residuals,8192 chunks, MTP decode-only release, e82fc5 executable and model arithmetic as the passed parent. Full MTP and occupied main-cache payload verification is enabled; all66 main state parts,248320 first-head floats,64IDs and every logprob must match the completed default before timing. First use is logged and validated. No full256K, speed, hang-prevention or production-adoption claim.'"+text[end:]
replace("record['previous_goal_turn']='progress: ten completed32K host-profile/attention/state/clean jobs archived in1c9b2a23dc3b0494c4b8f22b132b1b6d2e18ecd5; all identities absent and production binary unchanged. Attention batching gave no useful gain; return to phase storage and processing order.'", "record['previous_goal_turn']='progress: nine exact32K main-state/head/output controls and six clean processing-order jobs archived in86e13c1b17104e3f53b45c56f85c35c74c2a4099. All owned identities absent; production unchanged. GPU-row pair varies about11%, MTP RAM restoration repeats at248ms. Proceed to matched main-cache release/restore conditions.'")
ast.parse(text);target.write_text(text);audit.mkdir(mode=0o700)
review={'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Review of actual compiled private main-cache lease, callbacks and SYCL segment lifetime before32K model execution. CPU ownership/error evidence is included; not all runtime UB absent, actual mapping, full256K or performance proof.','binary_sha256':'e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323','source_sha256':hashes,'host_check_sha256':hashlib.sha256((base/'main-cache-lease-v01402-host-check/record.json').read_bytes()).hexdigest(),'parent_controller_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'review':[
'All four arms request128 experts with the same64MiB-segment main-cache allocation,8192-token chunks and32767 GPU residual rows. Fraction0 is the matched backing control;0.5 rounds the retained bytes up to a whole physical segment;1 releases all backing. Physical padding is distinct from logical payload and will be recorded rather than assumed.',
'The opt-in STRATA_PREFILL_RELEASE_CACHE allocation selects segmentation without --vram-elastic. Interactive elastic resizing remains explicitly rejected by require_sycl_cache_policy; no interactive command is used.',
'Main-cache suspend requires the cache graph-retirement/refresh callbacks. It drains its queue, resolves the complete immutable Arena source on the issuer using the current residency table, validates each source/destination extent, snapshots or verifies the payload, drains all registered device queues, retires graphs and only then shrinks physical backing. Partial-shrink active state enables rollback.',
'RAM spans cover only the released portion, including a partial expert at the kept boundary. Snapshot copies the logical tail from GPU before unmapping. Restore grows the original allocation, requires unchanged slots/reserved VA, copies the snapshot or immutable RAM spans, zeros uncovered logical bytes, waits, verifies bytes if requested and refreshes decoder/verifier cache pointers before readiness.',
'Verifier retirement waits pending commit and both queues before removing ordinary/no-reset/batch/commit/boundary graphs. Rebuild updates the cache base and records every legal verifier-window graph. Recording/finalization does not execute the model window; runtime full-state/head gates still check this independently.',
'Layer cache/residual storage is reclaimed after queues and staging drain and before main/MTP weight restoration. Original callbacks/cache/residency/R pointers are restored on scope exit. The existing MTP lease preserves dense/state/KV and immutable host sources.',
'The actual CacheLease CPU extraction matches current production and the compiled private source.19 ASan/UBSan cases check mixed layer order, mid-expert partial boundary, adaptive residency, no-release/partial/full RAM and snapshot, prevalidation, partial-shrink rollback and grow/refresh retry. Fakes do not establish SYCL mapping/hang safety.',
'Full262144 context cannot keep every residual row on10GiB: observed40960 bytes per row gives10737377280 bytes for262143 rows before KV/dense/cache/scratch. Fixed32767-row tests are32K only. Dynamic prefix budgeting and the older repeated-prefill retained-memory/rollback failure remain pending; no production default is changed.'
]}
(audit/'record.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'controller':str(target),'review':str(audit/'record.json')},indent=2))
