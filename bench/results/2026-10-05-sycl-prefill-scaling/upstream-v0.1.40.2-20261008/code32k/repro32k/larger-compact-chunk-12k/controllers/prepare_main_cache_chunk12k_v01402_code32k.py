"""Prepare one logged32K correctness check with a larger compact chunk."""
from pathlib import Path
import ast, datetime, hashlib, json

base=Path(__file__).parent
source=base/'run_owned_main_cache_release_v01402_code32k.py'
target=base/'run_owned_main_cache_chunk12k_v01402_code32k.py'
out=base/'main-cache-chunk12k-v01402-source-review'
assert not target.exists() and not out.exists()
text=source.read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
replace("mode in ['main-vmm-kept-ram','main-vmm-half-ram','main-vmm-full-ram','main-vmm-full-snapshot']", "mode=='main-vmm-full-ram-chunk12k'")
replace("fractions={'main-vmm-kept-ram':'0','main-vmm-half-ram':'0.5','main-vmm-full-ram':'1','main-vmm-full-snapshot':'1'}", "fractions={'main-vmm-full-ram-chunk12k':'1'}")
replace("'--prefill','8192'", "'--prefill','12288'")
replace("    if phase=='diagnostic':env=m.diagnostic_environment(env)", "    env['STRATA_TRACE']='1'\n    if phase=='diagnostic':env=m.diagnostic_environment(env)")
replace("    record['source_review_sha256']=hashlib.sha256((base/'main-cache-release-v01402-source-review/record.json').read_bytes()).hexdigest()", "    record['source_review_sha256']=hashlib.sha256((base/'main-cache-chunk12k-v01402-source-review/record.json').read_bytes()).hexdigest()\n    budget=json.loads((base/'compact-chunk-v01402-host-accounting-v2/record.json').read_text())\n    assert budget['passed'] and not budget['active'] and budget['observed_8192_accounting_matches']\n    complete=json.loads((base/'main-cache-release-v01402-clean-sequence/record.json').read_text())\n    assert complete['passed'] and not complete['active']\n    record['chunk_budget_sha256']=hashlib.sha256((base/'compact-chunk-v01402-host-accounting-v2/record.json').read_bytes()).hexdigest()")
start=text.index("    record['scope']='Private matched32K main-cache lease comparison")
end=text.index('\n',start)
text=text[:start]+"    record['scope']='Private32K correctness check of12288-token compact-hc chunks using the unchanged e82fc5 candidate and full main/MTP decode-only RAM release. Same32768 fixture,33024 context,128 cache slots,all-GPU residual rows and model math knobs. First configuration logs LevelZero/UR validation and phase memory trace. Complete main-state/head/IDs/logprobs equality must pass before any quiet timing; altered GEMM shapes may fail that gate. Snapshot/payload validation enabled. Fixed32767 rows are not a full262144 default; no production adoption.'"+text[end:]
# Always quit normally after receiving output, including a mathematical rejection.
old="        assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal']), 'reject before timing: full-state/head/output mismatch'"
new="        record['math_gate_passed']=not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal'])"
replace(old,new)
replace("    record['completed']=True", "    record['completed']=True\n    assert record['math_gate_passed'], 'reject before timing: full-state/head/output mismatch'")
ast.parse(text)
target.write_text(text)
out.mkdir(mode=0o700)
record={'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Source/accounting review only before one logged32K model check. No arithmetic rewrite, extra GPU allocation helper, runtime capacity guarantee or speed claim.',
        'source_controller_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'budget_sha256':hashlib.sha256((base/'compact-chunk-v01402-host-accounting-v2/record.json').read_bytes()).hexdigest(),
        'review':['Actual helper accounting matches the completed8192 owned carve after accounting for its single contiguous ring, rather than the borrowed-ring per-slot guard padding. TwoMiB owned estimates are engine budget arithmetic, not measured SYCL granularity.',
                  '12288 compact-hc accounted bytes2738729216; all-GPU residual new bytes838819840 because first12288 scratch rows are reused. Engine estimated net increment708837376B over8192.16K estimates an increment1417674752B and is not launched.',
                  'Run-layer-major validates all KV residency and available GPU residual capacity before allocation. Its existing RAII callbacks drain queues/staging, free temporary layer/residual buffers and restore main/MTP mappings on exit.',
                  'The changed chunk boundaries preserve token/layer/PLE ordering but may select different GEMM numerical accumulation. All66 main-state parts and248320 first-head floats must match the hard8192 baseline before timing.',
                  'One logged diagnostic uses existing immutable payload checks and memory phase logs. Its duration is excluded from performance comparison. On mismatch receive the full reply and QUIT before marking the gate rejected; no reset/rebind/service changes.']}
(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'controller':str(target),'review':str(out/'record.json')},indent=2))
