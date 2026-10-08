from pathlib import Path
import ast, datetime, hashlib, json

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
source=base/'run_owned_event_ack_no_root_default_v01402_code32k.py'
target=base/'run_owned_compact_layer_v01402_code32k.py'
audit=base/'compact-layer-v01402-source-review'
assert not target.exists() and not audit.exists()
text=source.read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace("mode=='event-ack-no-root-prefill-default'", "mode in ['compact2-layout1','layer2-ram-layout1']")
replace("out = base/f'owned-event-ack-no-root-prefill-default-v01402-code32k-{phase}-r{repetition}'", "out = base/f'owned-{mode}-v01402-code32k-{phase}-r{repetition}'")
replace("clean_gate=json.loads", "default_gate=json.loads((base/'event-ack-no-root-default-v01402-gated-checks/record.json').read_text())\nassert default_gate['passed'] and not default_gate['active']\nqsa_gate=json.loads((base/'qsa-batch128-layout1-v01402-clean-sequence/record.json').read_text())\nassert qsa_gate['passed'] and not qsa_gate['active']\nif mode=='layer2-ram-layout1':\n    compact_gate=json.loads((base/'compact2-layout1-v01402-state-sequence/record.json').read_text())\n    assert compact_gate['passed'] and not compact_gate['active']\nclean_gate=json.loads")
replace("+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json')):", "+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-host-profile-v01402-code32k-*/record.json'))+list(base.glob('owned-qsa-*-v01402-code32k-*/record.json'))+list(base.glob('owned-compact2-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-ram-layout1-v01402-code32k-*/record.json')):")
replace("env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')", "env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')\n    env.update(STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_ATTN_LAYOUT='1')\n    if mode=='layer2-ram-layout1':env.update(STRATA_PREFILL_LAYER_MAJOR='2')\n    assert not any(k in env for k in ['STRATA_PREFILL_ATTN_BATCH','STRATA_GDN_KEYHEAD','STRATA_GDN_KEYHEAD_TUNED','STRATA_PREFILL_RELEASE_CACHE','STRATA_PREFILL_RELEASE_DRAFT','STRATA_PREFILL_LAYER_MAJOR_R_GPU','STRATA_PREFILL_LAYER_MAJOR_R_INPLACE','STRATA_PROMPT_ATTN_XMX'])")
start=text.index("    record['scope']='Private registered")
end=text.index('\n',start)
text=text[:start]+"    record['scope']='Private32K processing-order experiment on unchanged e82fc5 executable. COMPACT2 reuses completed phase storage, QSA layout1 retains SG32 and typed transpose loads, batch remains32. First compact-only controls precede layer-major2, whose inter-layer residual rows are in RAM and whose full-layer expert cache is temporary. Decode/main cache and MTP weights remain backed for this initial32K step. Ordered math/model/KV/queues/graphs unchanged. All66state parts,248320 first-head floats,64IDs and all logprobs must match default32K before clean timing. First use logs UR/L0 and validation. No full256K or general hang-prevention claim.'"+text[end:]
replace("record['previous_goal_turn']='progress: six exact32K candidate state/head checks, archived in de7a38edf14f634e3db2af6e0a0cd89148c75e21; executable and process ownership revalidated before this test'", "record['previous_goal_turn']='progress: ten completed32K host-profile/attention/state/clean jobs archived in1c9b2a23dc3b0494c4b8f22b132b1b6d2e18ecd5; all identities absent and production binary unchanged. Attention batching gave no useful gain; return to phase storage and processing order.'")
replace("owned-event-ack-no-root-prefill-cb-v01402-code32k-diagnostic-r1/record.json", "owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json")
replace("result['comparison_to_counter_off_control']", "result['comparison_to_default_counter_control']")
replace("        record['requests'].append(result);save()", "        record['requests'].append(result);save()\n        c=result['comparison_to_default_counter_control']\n        assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal']), 'reject before timing: full-state/head/output mismatch'")
replace("    cursor=m.journal_cursor(r,'kernel-before');save()", "    record['source_review_sha256']=hashlib.sha256((base/'compact-layer-v01402-source-review/record.json').read_bytes()).hexdigest()\n    cursor=m.journal_cursor(r,'kernel-before');save()")
ast.parse(text)
target.write_text(text)
audit.mkdir(mode=0o700)
private=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
paths=['sycl/src/prefill/prefill.cpp','sycl/src/prefill/kernels.dp.cpp','sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp','sycl/src/program/generate.cpp','sycl/src/core/mtp.cpp']
review={
    'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Source review for compact phase-storage and layer-major processing experiment, not runtime UB or capacity proof.',
    'binary_sha256':'e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323',
    'source_sha256':{str(private/p):hashlib.sha256((private/p).read_bytes()).hexdigest() for p in paths},
    'common_changes':{'STRATA_PREFILL_COMPACT':'2','STRATA_PREFILL_ATTN_LAYOUT':'1'},
    'second_step_change':{'STRATA_PREFILL_LAYER_MAJOR':'2'},
    'review':[
        'Prefill::init rejects an out-of-order compute queue. Count and carve use the same Alloc/take sequence; the region is max(HC,PLE,GDN,QSA,MoE), with bounds/null failure rather than overlapping live arrays within each phase.',
        'Emb/mixed/bo share storage across successive consumers, not within the same kernel. Persistent R, injection values, half-inputs, steps and child output remain outside HC phase storage. Fused next-HC norm is disabled before PLE invalidates shared storage.',
        'Conv completes reading qkv before recurrence writes y there. Recurrence input hbuf and y16 are disjoint from y; static_assert3*ZV<=2*C bounds the two outputs in old qkv storage.',
        'QSA split finishes all input-query reads before merge writes the same query rows; the in-order queue orders split/merge and subsequent batches. Layout1 uses required SG32, original reductions and scalar reconstruction from transposed local floats rather than unrelated-type float4 pointer loads.',
        'Compact expert scratch receives ne rows at row0; ne<=T is checked before gather. Grouped down outputs keep original o0 order. In-order gather/dequant/GEMM/SwiGLU/down finish each scratch use before the next expert overwrites it.',
        'The key-head copy helper containing local float4 pointer casts is inactive: gdn_keyhead_ok and tuned branches require absent KEYHEAD/KEYHEAD_TUNED environment flags. No new GDN kernel is enabled here.',
        'Layer-major requires single-GPU compact FP16, all KV pages on device, no peer/helper/preload. A separate temporary layer cache holds one whole layer, with all512 expert pointers resolved on issuer before staging. Actual-DMA generation completion protects host buffer reuse. Copy and compute/stager drain before the next layer overwrites cache.',
        'Layer-major host residual callback reads its old rows before downloading into the same range, and waits completion. Chunk boundaries, GEMM row counts and per-layer ordered state updates are unchanged. Each layer resets PLE prev to request-entry values; last-layer MTP/on_chunk callback retains logical token order.',
        'The restoration guard drains queues and staging before restoring original public callbacks/cache/residency/phase pointers. Temporary layer cache is freed before decode. Main cache/MTP weights remain backed for this first step, so no graph retirement/remap/rebuild occurs yet.',
        'Full262144 repeated release/restoration error propagation and retained memory budget remain unresolved from an older candidate. No full-context adoption follows from32K results; next release-fraction/RAM-versus-snapshot steps need distinct exact-state and capacity gates.'
    ],
    'parent_controller_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
}
(audit/'record.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'passed':True,'controller':str(target),'review':str(audit/'record.json')},indent=2))
