from pathlib import Path
import ast, datetime, hashlib, json

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
source = base/'run_owned_event_ack_no_root_default_v01402_code32k.py'
target = base/'run_owned_qsa_batch128_layout1_v01402_code32k.py'
audit = base/'qsa-batch128-layout1-v01402-source-review'
assert not target.exists() and not audit.exists()
text = source.read_text()
def replace(old, new):
    global text
    assert text.count(old) == 1, old
    text = text.replace(old, new)
replace("mode=='event-ack-no-root-prefill-default'", "mode=='qsa-batch128-layout1'")
replace("out = base/f'owned-event-ack-no-root-prefill-default-v01402-code32k-{phase}-r{repetition}'", "out = base/f'owned-qsa-batch128-layout1-v01402-code32k-{phase}-r{repetition}'")
replace("clean_gate=json.loads", "default_gate=json.loads((base/'event-ack-no-root-default-v01402-gated-checks/record.json').read_text())\nassert default_gate['passed'] and not default_gate['active']\nprofile_gate=json.loads((base/'host-profile32k-v01402-sequence/record.json').read_text())\nassert profile_gate['passed'] and not profile_gate['active']\nclean_gate=json.loads")
replace("+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json')):", "+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-host-profile-v01402-code32k-*/record.json'))+list(base.glob('owned-qsa-batch128-layout1-v01402-code32k-*/record.json')):")
replace("env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')", "env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')\n    env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='1')")
start = text.index("    record['scope']='Private registered")
end = text.index('\n', start)
text = text[:start]+"    record['scope']='Private32K scheduling experiment on the same e82fc5 executable: attention query batch32 to128 and transpose-query layout1 at the unchanged SG32/workgroup256. No arithmetic, reduction tree, KV format, kernel source, queue, graph or lifetime change. Match all66prefill-state parts,248320 first-head floats,64IDs and all logprobs before any clean timing. First run uses UR/L0 logs/validation; every fresh process reads32768 tokens. Not a full256K or hang-prevention claim.'"+text[end:]
replace("record['previous_goal_turn']='progress: six exact32K candidate state/head checks, archived in de7a38edf14f634e3db2af6e0a0cd89148c75e21; executable and process ownership revalidated before this test'", "record['previous_goal_turn']='progress: thirteen exact32K/scheduling checks archived in32f600e22f2039e36dd11c7b75d214aff8d0bcd1; three further terminal32K host-only profiler/control jobs retain exact full-state/head/output matches.'")
replace("owned-event-ack-no-root-prefill-cb-v01402-code32k-diagnostic-r1/record.json", "owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json")
replace("result['comparison_to_counter_off_control']", "result['comparison_to_default_counter_control']")
replace("        record['requests'].append(result);save()", "        record['requests'].append(result);save()\n        c=result['comparison_to_default_counter_control']\n        assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal']), 'reject before timing: state/head/output mismatch'")
replace("    cursor=m.journal_cursor(r,'kernel-before');save()", "    record['source_review_sha256']=hashlib.sha256((base/'qsa-batch128-layout1-v01402-source-review/record.json').read_bytes()).hexdigest()\n    cursor=m.journal_cursor(r,'kernel-before');save()")
ast.parse(text)
target.write_text(text)
audit.mkdir(mode=0o700)
private_source=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
paths=['sycl/src/prefill/prefill.cpp','sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp','sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp']
review={
    'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Source review for an environment-only experiment, not runtime UB proof or speed evidence.',
    'binary_sha256':'e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323',
    'source_sha256':{str(private_source/p):hashlib.sha256((private_source/p).read_bytes()).hexdigest() for p in paths},
    'changes':{'STRATA_PREFILL_ATTN_BATCH':'128','STRATA_PREFILL_ATTN_LAYOUT':'1'},
    'review':[
        'Batch allocation and execution share prompt_attn_batch(T); batch is bounded1..1024 and min(T), final8191 chunk has127-query tail. Scratch allocation has128 copies of the existing cap-based per-query stride; no cross-query reuse inside a dispatch.',
        'Layout1 retains required subgroup32 and local range(1,1,256). Query, KV-head and64-cell chunk identify disjoint scratch partitions. Merge uses the same per-query stride and the original ascending chunk loop.',
        'Transpose initialization maps i to(i%8)*32+(i%256)/8, a bijection of each256-element query head. Each lane reconstructs the same eight scalar operands as the original contiguous layout, preserving the dot expression and XOR16/8/4/2/1 reductions. The transpose branch avoids local float-array to float4 pointer casts.',
        'Early return n_here<=0 is workgroup-uniform. Cell skip is subgroup-uniform; head loops execute equally inside each subgroup; every workgroup barrier is reached by all remaining items. Local sq/sp/srow initialization precedes use through barriers.',
        'No global/root-group synchronization is used by this kernel; variant launch has empty experimental properties. Bounds and integer products remain64-bit for per-query offsets. Geometry, supported cap and page table remain those already verified by the baseline32K configuration.',
        'SG16/layout4 is deliberately outside this experiment. No tensor/XMX attention, changed reduction order, cell masking approximation, prefill retirement or graph update is enabled.',
        'Full-state/first-head/output comparisons and healthy owned exit are hard gates before timing. Full262144-cell occupancy/repeat/restore remains a separate pending production gate.'
    ],
    'parent_controller_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
}
(audit/'record.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'passed':True,'controller':str(target),'audit':str(audit/'record.json')},indent=2))
