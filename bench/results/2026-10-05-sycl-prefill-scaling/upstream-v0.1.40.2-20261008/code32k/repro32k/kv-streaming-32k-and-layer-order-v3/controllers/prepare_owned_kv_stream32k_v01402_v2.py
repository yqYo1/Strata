"""Freeze a new owned streaming gate; preserve every earlier controller/receipt."""
from pathlib import Path
import datetime,hashlib,json,difflib,ast
base=Path(__file__).parent
root=Path("/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05")
parent=base/'run_owned_main_cache_repeat32k_v01402.py'
target=base/'run_owned_kv_stream32k_v01402_v2.py'
assert not target.exists()
text=parent.read_text();changed=text
def replace(a,b):
    global changed
    assert changed.count(a)==1,a[:100]
    changed=changed.replace(a,b)
replace("assert mode=='main-vmm-full-ram-repeat32k'", "assert mode=='kv-stream32k'")
replace("binary=root/'build-sycl-event-ack-no-root-prefill-20261008/strata'", """binary=root/'build-sycl-kv-stream-safe-v3-20261008/strata'
    kv_receipt=base/'kv-stream-safe-v01402-build-v3/record.json'
    kv_build=json.loads(kv_receipt.read_text())
    assert kv_build['passed'] and not kv_build['active'] and kv_build['baseline_inputs_unchanged']
    assert kv_build['candidate_binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()
    assert kv_build['baseline_binary_sha256']=='e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323'
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in kv_build['link_input_sha256'].items())
    kv_source=binary.parent/'source/kv_stream.dp.cpp'
    assert hashlib.sha256(kv_source.read_bytes()).hexdigest()==kv_build['candidate_source_sha256']
    record['kv_build_receipt_sha256']=hashlib.sha256(kv_receipt.read_bytes()).hexdigest()
    record['kv_source_sha256']=kv_build['candidate_source_sha256']
    review=base/'kv-stream32k-v01402-source-review-v2/record.json'
    audit=json.loads(review.read_text());assert audit['passed'] and not audit['active']
    assert audit['candidate_sha256']==record['kv_source_sha256']
    record['kv_source_review_sha256']=hashlib.sha256(review.read_bytes()).hexdigest()""")
replace("assert record['binary_sha256']==expected['candidate_binary_sha256']", "assert kv_build['baseline_binary_sha256']==expected['candidate_binary_sha256']")
replace("candidate=binary.parent/'source/kernels.dp.cpp'", "candidate=Path(expected['candidate_binary']).parent/'source/kernels.dp.cpp'")
replace("list(base.glob('owned-main-vmm-*-v01402-code32k-*/record.json')):", "list(base.glob('owned-main-vmm-*-v01402-code32k-*/record.json'))+list(base.glob('owned-kv-stream32k-v01402-code32k-*/record.json')):")
replace("env.update(STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='32767',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1')", "env.update(STRATA_PREFILL_LAYER_MAJOR='0',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0')")
replace("""    fractions={'main-vmm-full-ram-repeat32k':'1'}
    env.update(STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_ALLOC='vmm',STRATA_PREFILL_CACHE_RESTORE='snapshot' if mode=='main-vmm-full-snapshot' else 'ram',STRATA_PREFILL_CACHE_RELEASE_FRAC=fractions[mode],STRATA_PREFILL_CACHE_VERIFY='1')""", """    # Streaming uses its existing chunk-major path; no dormant layer-major leases.
    assert not any(k in env for k in [
        'STRATA_PREFILL_RELEASE_CACHE','STRATA_PREFILL_RELEASE_DRAFT',
        'STRATA_PREFILL_CACHE_ALLOC','STRATA_PREFILL_LAYER_MAJOR_R_GPU'])""")
replace("('--max-context','33024')","('--max-context','262144')")
replace("args+=['--kv','int8','--prompt-cache','0']", "args+=['--kv','int8','--kv-resident','32768','--prompt-cache','0']")
replace("if request_index==0:result['math_gate_passed']=result['math_gate_passed'] and not c['different_prefill_state_parts']", "# Rounded future cells are unwritten in streaming; every used byte and moving spare must match.")
replace("record['scope']='Three fresh full32768-token", "record['scope']='KV streaming candidate, chunk-major, context262144, resident32768, own staging, no prefetch or layer-major cache leases. Three fresh full32768-token")
replace("in one logged/validated normal-MTP process on unchanged e82fc5 binary. Accepted8192 chunks, all32767 GPU residual rows, full main/MTP decode-only RAM release.", "in one logged/validated normal-MTP process on private v3 KV binary. Accepted8192 chunks.")
replace("record['configuration_note']='State and head", "record['configuration_note']='Context262144/kv-resident32768 (main mode1, MTP ring), own stage, chunk-major0, no cache release or prefetch; identical math gates on32768/GEN64. State and head")
ast.parse(changed);target.write_text(changed)
out=base/'kv-stream32k-v01402-source-review-v2';out.mkdir(mode=0o700)
compiled=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
files=['sycl/src/core/layer.cpp','sycl/src/core/session.cpp',
       'sycl/src/core/conversation_snapshot.cpp','sycl/src/prefill/prefill.cpp',
       'sycl/src/program/generate.cpp','include/strata/kernels/kv_stream.hpp',
       'sycl/include/strata/sycl_allocation.hpp']
sources={}
for f in files:
    p=compiled/f;q=root/f
    if f=='sycl/src/prefill/prefill.cpp':
        expected=json.loads((base/'event-ack-registered-copy-v01402-matched-build/record.json').read_text())['candidate_sources'][f]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    else:assert p.read_bytes()==q.read_bytes(),f
    sources[f]=hashlib.sha256(p.read_bytes()).hexdigest()
kv=root/'build-sycl-kv-stream-safe-v3-20261008/source/kv_stream.dp.cpp'
receipt=dict(active=False,passed=True,gpu_tested=False,adopted=False,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    candidate_sha256=hashlib.sha256(kv.read_bytes()).hexdigest(),
    source_sha256=sources,source_authority='Pinned registered-copy compiled overlay; prefill differs from production by validated actual-event completion and registered copy queue. All KV call-path logic reviewed in the compiled overlay; do not substitute current worktree source.',controller_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    parent_controller_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    reviewed=[
        'layer.cpp kv_plan/qsa_state_init: mode1 main resident8192pages, full-context host-USM identity backing; MTP mode2 fixed ring. checked_usm rejects null. Host backing allocated once at startup and not freed/reused while captured graphs run. Serve drains all registered queues then _Exit; no session rebuild in this gate.',
        'prefill.cpp run_layer_major rejects mode1: select existing run_impl chunk-major0, no layer-major cache leases or GPU residual allocation.',
        'take_stage: one full-context identity layer (264MiB int8 at256K); STAGE_OWN=1 avoids borrowed expert placement changes and shares tracked owned allocation list. Identity upload completes before temporary host vector dies.',
        'prefetch0: staging/upload/append/attention on the same in-order compute queue. Prefix is immutable and every used new row is appended to host and stage before attention. No cross-queue prefetch enabled in initial gate.',
        'Prefill release drains compute/copy/stager/KV queues before tracked owned buffers. Snapshot occurs after compute/copy drains; no concurrent host CPU reads of device-written authoritative host USM.',
        'conversation_snapshot selects authoritative host pools for mode1/2. Restore invalidates streaming map or restores bounded MTP ring before later attention. Full restore is a later required gate, not proven by initial prefill captures.',
        'v3 resolver keeps1024WG/SG32, atomically loads page claims and stores shared hit metadata. Full group barriers separate victim phases; minimum slots>=1024 keeps sweep IDs unique. Byte representation copies remove device/host type-punning. First execution requires device capabilities and runtime launch validation.',
        'Initial capture compares all66 main-state parts using the existing fail-closed positive-geometry comparator; only unwritten page-rounded future cells may differ. Pooled spare row is live. First full head/all64IDs/logprobs and MTP43/66 unchanged. Full256K occupancy permits no padding exclusions.'],
    limitations=['Source review and CPU compilation are not GPU correctness or absence-of-UB proof.',
                 'Host KV allocations currently live until process exit rather than explicit SessionState RAII. No state reconstruction/reallocation is exercised here.',
                 'Prefetch, hybrid/Q4, multi-GPU, elastic VMM, larger chunks and restore API not activated by this gate.',
                 'At256K a larger stage and sparse indexer change allocation sizes; full occupancy/repeat/restore/tail/refusal/later>=32K remain required.'])
(out/'record.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'controller.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile=str(parent),tofile=str(target))))
print(json.dumps({'passed':True,'controller':str(target),'sha256':receipt['controller_sha256'],'candidate':receipt['candidate_sha256']},indent=2))
