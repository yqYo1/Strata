from pathlib import Path
import ast, hashlib, json

base=Path(__file__).parent
source=base/'run_owned_qsa_batch128_layout1_v01402_code32k_clean.py'
target=base/'run_owned_layer_processing_v01402_code32k_clean.py'
assert not target.exists()
text=source.read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace("mode in ['qsa-default','qsa-batch128-layout1']", "mode in ['processing-default','layer2-ram-layout1','layer2-gpu-release-layout1']")
replace("qsa-batch128-layout1-v01402-state-sequence/record.json", "layer-gpu-release-v01402-state-sequence/record.json")
replace("assert gate['passed'] and not gate['active']", "assert gate['passed'] and not gate['active']\nlayer_gate=json.loads((base/'compact-layer-v01402-state-sequence/record.json').read_text())\nassert layer_gate['passed'] and not layer_gate['active']")
replace("+list(base.glob('owned-qsa-*-v01402-code32k-*/record.json')):", "+list(base.glob('owned-qsa-*-v01402-code32k-*/record.json'))+list(base.glob('owned-compact2-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-layer2-*-layout1-v01402-code32k-*/record.json'))+list(base.glob('owned-processing-default-v01402-code32k-*/record.json')):")
replace("if mode=='qsa-batch128-layout1':env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='1')", "if mode!='processing-default':env.update(STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_ATTN_LAYOUT='1',STRATA_PREFILL_LAYER_MAJOR='2')\n    if mode=='layer2-gpu-release-layout1':env.update(STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='32767',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1')\n    assert 'STRATA_PREFILL_DRAFT_VERIFY' not in env\n    assert 'STRATA_PREFILL_ATTN_BATCH' not in env")
start=text.index("    record['scope']='Clean32K ABBA timing")
end=text.index('\n',start)
text=text[:start]+"    record['scope']='Clean32K comparison of default processing, compact2/layout1 layer-major with RAM residuals, and all-GPU residuals with MTP decode-only release/immutable-RAM restore. Same private e82fc5 executable, model, ordered math, SG32 and normal MTP4. Three completed full-state/head/output controls per configuration, including MTP payload checks before/after release, precede timing. Payload verification/debug validation/dumps/transfer profiling/extra waits/profilers are absent here. Main cache remains backed. Compare all64IDs and every logprob to the logged default control. No full256K/general hang-prevention/adoption claim.'"+text[end:]
replace("record['previous_goal_turn']='progress: three host-only32K profiler/control jobs and three attention128/layout1 full-state/head/output controls completed before this clean comparison; older controls archived in32f600e22f2039e36dd11c7b75d214aff8d0bcd1.'", "record['previous_goal_turn']='progress: 32K host profile and negative attention batching result archived in1c9b2a23dc3b0494c4b8f22b132b1b6d2e18ecd5. Nine fresh32K full-state/head controls now gate processing-order/release timing; no completed gate is re-run without a new concern.'")
ast.parse(text)
target.write_text(text)
print(json.dumps({'controller':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
