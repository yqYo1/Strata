from pathlib import Path
import json

base = Path(__file__).parent
old_mode = 'event-ack-no-profile-cb-cnr'
new_mode = 'event-ack-registered-copy-cb-cnr'
target = base/'run_owned_event_ack_registered_copy_cb_cnr_v01402_code32k_v3.py'
assert not target.exists()
text = (base/'run_owned_event_ack_no_profile_cb_cnr_v01402_code32k.py').read_text().replace(old_mode,new_mode)

def change(old,new,count=1):
    global text
    assert text.count(old)==count,(old,text.count(old),count)
    text=text.replace(old,new)

change('build-sycl-event-ack-no-profile-copy-v2-20261008/strata',
       'build-sycl-event-ack-registered-copy-v3-20261008/build/strata')
change('event-ack-no-profile-copy-v01402-v2-build/record.json',
       'event-ack-registered-copy-v01402-v3-build/record.json',count=2)
change("list(base.glob('owned-event-ack-registered-copy-cb-cnr-v01402-code32k-*/record.json'))",
       "list(base.glob('owned-event-ack-no-profile-cb-cnr-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-registered-copy-cb-cnr-v01402-code32k-*/record.json'))")
old = "        record['candidate_build_receipt_sha256']=hashlib.sha256((base/'event-ack-registered-copy-v01402-v3-build/record.json').read_bytes()).hexdigest()"
new = old+"\n        private_source=binary.parent.parent/'source'\n        assert all(hashlib.sha256((private_source/path).read_bytes()).hexdigest()==value for path,value in expected['candidate_sources'].items())\n        record['candidate_sources']=expected['candidate_sources']"
change(old,new)
change('an owned same-context in-order copy queue, no copy-event profiling unless STRATA_PREFILL_TRANSFER_TIMING is requested. Compute queues, arithmetic, barriers and release drains unchanged.',
       'a registered same-context in-order copy queue, no copy-event profiling unless STRATA_PREFILL_TRANSFER_TIMING is requested. The factory remains in device_ext::_queues so device-wide/default-queue waits retain their original contract. Every translation unit is rebuilt with the same modified header. Compute queues, arithmetic, barriers and release drains unchanged.')
compile(text,str(target),'exec')
target.write_text(text)

seq_target = base/'run_event_ack_registered_copy_cb_cnr_v01402_state_sequence_v3.py'
assert not seq_target.exists()
seq = (base/'run_event_ack_no_profile_cb_cnr_v01402_state_sequence.py').read_text().replace(old_mode,new_mode)
seq = seq.replace('run_owned_event_ack_no_profile_cb_cnr_v01402_code32k.py',target.name)
compile(seq,str(seq_target),'exec')
seq_target.write_text(seq)
print(target)
print(seq_target)
