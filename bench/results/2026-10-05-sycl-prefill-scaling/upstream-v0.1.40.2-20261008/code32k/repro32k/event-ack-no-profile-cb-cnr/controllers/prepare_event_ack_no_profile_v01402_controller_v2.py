"""Create immutable owned32K controllers for the next private copy-queue candidate."""
from pathlib import Path
import hashlib
import json

base = Path(__file__).parent
old_mode = 'event-ack-cb-cnr'
new_mode = 'event-ack-no-profile-cb-cnr'
target = base/'run_owned_event_ack_no_profile_cb_cnr_v01402_code32k.py'
assert not target.exists()
text = (base/'run_owned_event_ack_cb_cnr_v01402_code32k.py').read_text()
text = text.replace(old_mode, new_mode)

def change(old, new, count=1):
    global text
    assert text.count(old) == count, (old, text.count(old), count)
    text = text.replace(old, new)

change("post-event-ack-v01402-stall-health/record.json",
       "post-event-ack-cb-cnr-v01402-stall-health/record.json")
change("build-sycl-event-ack-v2-20261008/strata",
       "build-sycl-event-ack-no-profile-copy-v2-20261008/strata")
change("event-ack-v01402-v2-build/record.json",
       "event-ack-no-profile-copy-v01402-v2-build/record.json", count=2)
# The global mode replacement must not rename the existing failed-candidate glob.
change("list(base.glob('owned-event-ack-no-profile-cb-cnr-v01402-code32k-*/record.json'))",
       "list(base.glob('owned-event-ack-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-profile-cb-cnr-v01402-code32k-*/record.json'))")
names = ['owned-v01402-code32k-patched-default-head-r2',
         'owned-event-ack-v01402-code32k-diagnostic-r1',
         'owned-event-ack-cb-cnr-v01402-code32k-state-r1']
known = {}
for name in names:
    path = base/name/'record.json'
    record = json.loads(path.read_text())
    assert not record['active'] and not record['healthy']
    assert not any(record['cleanup'][key] for key in ['inferior_survived', 'gdb_survived'])
    known[name] = hashlib.sha256(path.read_bytes()).hexdigest()
old_known = "known_failures={'owned-v01402-code32k-patched-default-head-r2': '74e4b01947b304ce63400cf41930dffcbcaa7482d4342770557da4a00520fae6', 'owned-event-ack-v01402-code32k-diagnostic-r1': 'b1d072a15ceaf7b30c35c6599b7df4a91f62caeb0fa8469000a980f6c3c8ed46'}"
change(old_known, 'known_failures='+repr(known))
change("Private actual-DMA event acknowledgement candidate with only implicit counter conversion disabled and MKL_CBWR=AUTO, both used by five completed production32K controls. Compare all prefill state/head bytes within candidate, and against the production CNR control that matched its first state repeat (but not its second repeat). These settings do not establish reproduction or prevent all stalls. No performance/full256K claim; candidate not adopted.",
       "Private actual-DMA event acknowledgement candidate with an owned same-context in-order copy queue, no copy-event profiling unless STRATA_PREFILL_TRANSFER_TIMING is requested. Compute queues, arithmetic, barriers and release drains unchanged. Only implicit counter conversion disabled and MKL_CBWR=AUTO as in completed production32K controls. Compare all prefill state/head bytes and output to the production CNR control and fresh candidate repeats. No performance, reproduction, hang-prevention or full256K claim; candidate not adopted.")
change("    assert 'STRATA_PREFILL_SYNC' not in env",
       "    assert 'STRATA_PREFILL_SYNC' not in env\n    assert 'STRATA_PREFILL_TRANSFER_TIMING' not in env")
compile(text, str(target), 'exec')
target.write_text(text)

seq_target = base/'run_event_ack_no_profile_cb_cnr_v01402_state_sequence.py'
assert not seq_target.exists()
seq = (base/'run_event_ack_cb_cnr_v01402_state_sequence.py').read_text()
seq = seq.replace(old_mode, new_mode)
seq = seq.replace('run_owned_event_ack_cb_cnr_v01402_code32k.py', target.name)
needle = ' baseline=request(first)\n'
assert seq.count(needle) == 1
seq = seq.replace(needle, needle+" comparison=baseline['comparison_to_production_cnr_control']\n assert not comparison['different_prefill_state_parts'] and comparison['first_head_equal'] and comparison['ids_equal'] and comparison['logprobs_equal'],'first candidate state/head differs from production CNR control; reject before repetitions'\n")
compile(seq, str(seq_target), 'exec')
seq_target.write_text(seq)
print(target)
print(seq_target)
