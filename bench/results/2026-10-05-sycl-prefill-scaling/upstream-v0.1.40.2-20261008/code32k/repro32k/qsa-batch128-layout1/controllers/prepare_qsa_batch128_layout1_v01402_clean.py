from pathlib import Path
import ast, hashlib, json

base=Path(__file__).parent
source=base/'run_owned_event_ack_no_root_prefill_cb_v01402_code32k_clean.py'
target=base/'run_owned_qsa_batch128_layout1_v01402_code32k_clean.py'
assert not target.exists()
text=source.read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace("mode=='event-ack-no-root-prefill-cb'", "mode in ['qsa-default','qsa-batch128-layout1']")
replace("event-ack-no-root-prefill-cb-v01402-state-sequence/record.json", "qsa-batch128-layout1-v01402-state-sequence/record.json")
replace("owned-event-ack-no-root-prefill-cb-v01402-code32k-diagnostic-r1/record.json", "owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json")
replace("assert gate['passed'] and not gate['active']", "assert gate['passed'] and not gate['active']\ndefault_gate=json.loads((base/'event-ack-no-root-default-v01402-gated-checks/record.json').read_text())\nassert default_gate['passed'] and not default_gate['active']")
replace("out = base/f'owned-event-ack-no-root-prefill-cb-v01402-code32k-{phase}-r{repetition}'", "out = base/f'owned-{mode}-v01402-code32k-{phase}-r{repetition}'")
replace("+list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-clean-*/record.json')):", "+list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-clean-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-host-profile-v01402-code32k-*/record.json'))+list(base.glob('owned-qsa-*-v01402-code32k-*/record.json')):")
replace("env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8',EnableImplicitConvertionToCounterBasedEvents='0')", "env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')\n    assert 'EnableImplicitConvertionToCounterBasedEvents' not in env\n    if mode=='qsa-batch128-layout1':env.update(STRATA_PREFILL_ATTN_BATCH='128',STRATA_PREFILL_ATTN_LAYOUT='1')\n    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE'] for k in env)")
start=text.index("    record['scope']='Clean32K timing")
end=text.index('\n',start)
text=text[:start]+"    record['scope']='Clean32K ABBA timing on the same private e82fc5 executable, default attention32/layout0 versus attention128/layout1 (subgroup32 unchanged). Both have three completed exact full state/head checks with default counter conversion and no CNR. No UR/Level Zero debug logging, parameter validation, extra waits, state/head dumps, transfer timing or profiler. An uninterrupted owned GDB/PTY observer is common to both; require64 IDs and all logprobs equal the logged default control. No full256K or general hang-prevention claim; candidate not adopted.'"+text[end:]
replace("record['previous_goal_turn']='progress: six exact32K candidate state/head checks, archived in de7a38edf14f634e3db2af6e0a0cd89148c75e21; executable and process ownership revalidated before this test'", "record['previous_goal_turn']='progress: three host-only32K profiler/control jobs and three attention128/layout1 full-state/head/output controls completed before this clean comparison; older controls archived in32f600e22f2039e36dd11c7b75d214aff8d0bcd1.'")
ast.parse(text)
target.write_text(text)
print(json.dumps({'controller':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
