from pathlib import Path
import ast, hashlib, json

base=Path(__file__).parent
source=base/'run_owned_main_cache_release_v01402_code32k.py'
target=base/'run_owned_main_cache_release_v01402_code32k_clean.py'
assert not target.exists()
text=source.read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
replace("phase in ['diagnostic','state'] and repetition in [1,2,3]", "phase=='clean' and repetition in [1,2]")
replace("assert source_gate['passed'] and not source_gate['active']", "assert source_gate['passed'] and not source_gate['active']\nmain_gate=json.loads((base/'main-cache-release-v01402-state-sequence/record.json').read_text())\nassert main_gate['passed'] and not main_gate['active']")
replace("    env.update(STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='32767',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1')", "    env.update(STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='32767',STRATA_PREFILL_LAYER_MAJOR_R_INPLACE='1')")
replace("STRATA_PREFILL_CACHE_RELEASE_FRAC=fractions[mode],STRATA_PREFILL_CACHE_VERIFY='1')", "STRATA_PREFILL_CACHE_RELEASE_FRAC=fractions[mode])\n    assert 'STRATA_PREFILL_DRAFT_VERIFY' not in env and 'STRATA_PREFILL_CACHE_VERIFY' not in env")
replace("    env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')\n    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')\n", "    assert 'STRATA_DUMP_FIRST_LOGITS' not in env and 'STRATA_PREFILL_DUMP_STATE' not in env\n")
start=text.index("        head=out/'first-head.bin';")
end=text.index("    current=None;send(b'QUIT",start)
text=text[:start]+"        baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]\n        result['comparison_to_logged_control']={'ids_equal':result['ids']==baseline['ids'],'logprobs_equal':result['logprobs']==baseline['logprobs']}\n        record['requests'].append(result);save()\n        assert all(result['comparison_to_logged_control'].values()),'reject clean timing: output mismatch'\n\n"+text[end:]
start=text.index("    record['scope']='Private matched32K")
end=text.index('\n',start)
text=text[:start]+"    record['scope']='Clean32K comparison of identical64MiB segmented-cache allocation with fraction0/0.5/1 RAM or full GPU-snapshot restoration. Three passed fresh full-state/head/output controls per arm precede timing. Same e82fc5 executable,32767 all-GPU residual rows,8192 chunks, MTP4 and arithmetic. No diagnostic/profiler/validation/state/head/payload checks/transfer timing or extra waits. Compare all64 IDs/logprobs to the completed logged default. Graph retirement/remap/copy/recapture remain included in prefill time. No full256K/adoption/general hang-prevention claim.'"+text[end:]
ast.parse(text);target.write_text(text)
print(json.dumps({'controller':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
