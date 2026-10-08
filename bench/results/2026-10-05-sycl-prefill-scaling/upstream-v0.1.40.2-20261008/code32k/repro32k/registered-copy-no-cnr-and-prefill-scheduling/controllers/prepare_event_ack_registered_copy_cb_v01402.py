"""Isolate copy completion from oneMKL CNR, without changing the executable."""
from pathlib import Path
import json

base = Path(__file__).parent
old_mode = 'event-ack-registered-copy-cb-cnr'
new_mode = 'event-ack-registered-copy-cb'
target = base/'run_owned_event_ack_registered_copy_cb_v01402_code32k.py'
assert not target.exists()
previous = json.loads((base/'event-ack-registered-copy-cb-cnr-v01402-state-sequence/record.json').read_text())
assert previous['passed'] and not previous['active']
text = (base/'run_owned_event_ack_registered_copy_cb_cnr_v01402_code32k_matched.py').read_text().replace(old_mode,new_mode)

def change(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)

change("list(base.glob('owned-event-ack-registered-copy-cb-v01402-code32k-*/record.json'))",
       "list(base.glob('owned-event-ack-registered-copy-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-registered-copy-cb-v01402-code32k-*/record.json'))")
change("env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8',EnableImplicitConvertionToCounterBasedEvents='0',MKL_CBWR='AUTO')",
       "env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8',EnableImplicitConvertionToCounterBasedEvents='0')\n    assert 'MKL_CBWR' not in env")
change('Only implicit counter conversion disabled and MKL_CBWR=AUTO as in completed production32K controls. Compare all prefill state/head bytes and output to the production CNR control and fresh candidate repeats.',
       'Only implicit counter conversion is disabled; MKL_CBWR is absent. The same executable previously completed three exact32K comparisons with CNR enabled. Compare all prefill state/head bytes and output to the original production counter-conversion-off control with no CNR, and to fresh candidate repeats. This isolates the need for CNR; default counter conversion is still a separate gate.')
change("baseline=json.loads((base/'owned-cnr-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]",
       "baseline=json.loads((base/'owned-cb-off-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]")
text=text.replace('comparison_to_production_cnr_control','comparison_to_production_cb_off_control')
needle="    record['configuration_note']="
assert text.count(needle)==1
text=text.replace(needle,"    previous_cnr=json.loads((base/'event-ack-registered-copy-cb-cnr-v01402-state-sequence/record.json').read_text())\n    assert previous_cnr['passed'] and not previous_cnr['active']\n    record['previous_goal_turn']='progress: six exact32K candidate state/head checks, archived in de7a38edf14f634e3db2af6e0a0cd89148c75e21; executable and process ownership revalidated before this test'\n"+needle)
compile(text,str(target),'exec');target.write_text(text)

seq_target = base/'run_event_ack_registered_copy_cb_v01402_state_sequence.py'
assert not seq_target.exists()
seq = (base/'run_event_ack_registered_copy_cb_cnr_v01402_state_sequence_matched.py').read_text().replace(old_mode,new_mode)
seq = seq.replace('run_owned_event_ack_registered_copy_cb_cnr_v01402_code32k_matched.py',target.name)
seq = seq.replace('comparison_to_production_cnr_control','comparison_to_production_cb_off_control')
seq = seq.replace('production CNR control','production non-CNR counter-off control')
compile(seq,str(seq_target),'exec');seq_target.write_text(seq)
print(target)
print(seq_target)
