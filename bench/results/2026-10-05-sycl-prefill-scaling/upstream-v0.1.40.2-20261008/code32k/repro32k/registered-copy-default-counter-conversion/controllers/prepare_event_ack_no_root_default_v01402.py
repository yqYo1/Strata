"""Isolate default counter conversion after the clean 32K scheduling comparison."""
from pathlib import Path

base=Path(__file__).parent
old_mode='event-ack-no-root-prefill-cb'
new_mode='event-ack-no-root-prefill-default'
target=base/'run_owned_event_ack_no_root_default_v01402_code32k.py'
assert not target.exists()
text=(base/'run_owned_event_ack_no_root_prefill_cb_v01402_code32k.py').read_text()
text=text.replace("'"+old_mode+"'","'"+new_mode+"'").replace(old_mode+'-v01402',new_mode+'-v01402')
def change(old,new):
 global text
 assert text.count(old)==1,(old,text.count(old))
 text=text.replace(old,new)
change("list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))",
       "list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))")
change("env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8',EnableImplicitConvertionToCounterBasedEvents='0')",
       "env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')\n    assert 'EnableImplicitConvertionToCounterBasedEvents' not in env")
change("baseline=json.loads((base/'owned-event-ack-registered-copy-cb-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]",
       "baseline=json.loads((base/'owned-event-ack-no-root-prefill-cb-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]")
text=text.replace('comparison_to_registered_no_cnr_control','comparison_to_counter_off_control')
lines=text.splitlines()
i=next(i for i,line in enumerate(lines) if line.startswith("    record['scope']="))
lines[i]="    record['scope']='Private registered nonprofiling copy candidate with14 independent prefill root-sync properties removed. Keep the same executable, kernel bodies, arithmetic, ND-ranges, subgroup attributes, queues, barriers, release drains and graph code. Remove only EnableImplicitConvertionToCounterBasedEvents=0; MKL_CBWR remains absent. First logged32K use followed by two fresh32K full state/head controls compares all66state parts,248320head floats,64IDs and every protocol logprob with the completed counter-off scheduling control. Diagnostic/state capture times are not performance evidence. Other common process-local settings remain explicit in the environment. No general hang-prevention/full256K claim; candidate not adopted.'"
text='\n'.join(lines)+'\n'
needle="previous=json.loads((base/'v01402-code32k-sequence/record.json').read_text());assert previous['passed'] and not previous['active']"
change(needle,needle+"\nclean_gate=json.loads((base/'registered-vs-no-root-v01402-clean-sequence/record.json').read_text())\nassert clean_gate['passed'] and not clean_gate['active']")
compile(text,str(target),'exec');target.write_text(text)
seq_target=base/'run_event_ack_no_root_default_v01402_state_sequence.py'
assert not seq_target.exists()
seq=(base/'run_event_ack_no_root_prefill_cb_v01402_state_sequence.py').read_text()
seq=seq.replace("'"+old_mode+"'","'"+new_mode+"'").replace(old_mode+'-v01402',new_mode+'-v01402')
seq=seq.replace('run_owned_event_ack_no_root_prefill_cb_v01402_code32k.py',target.name)
seq=seq.replace('comparison_to_registered_no_cnr_control','comparison_to_counter_off_control')
seq=seq.replace('registered-copy non-CNR counter-off control','no-root registered-copy non-CNR counter-off control')
compile(seq,str(seq_target),'exec');seq_target.write_text(seq)
print(target);print(seq_target)
