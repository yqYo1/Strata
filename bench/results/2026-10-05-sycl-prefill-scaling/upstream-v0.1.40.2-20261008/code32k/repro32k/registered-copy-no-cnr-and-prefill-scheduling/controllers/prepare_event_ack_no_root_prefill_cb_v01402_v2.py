"""Create an immutable, logged-first 32K scheduling-only experiment."""
from pathlib import Path
import hashlib
import json

base = Path(__file__).parent
old_mode = 'event-ack-registered-copy-cb'
new_mode = 'event-ack-no-root-prefill-cb'
target = base/'run_owned_event_ack_no_root_prefill_cb_v01402_code32k.py'
assert not target.exists()
previous = json.loads((base/'event-ack-registered-copy-cb-v01402-state-sequence/record.json').read_text())
assert previous['passed'] and not previous['active']
build_path = base/'event-ack-no-root-prefill-v01402-build/record.json'
build = json.loads(build_path.read_text())
assert build['passed'] and not build['active'] and build['baseline_inputs_unchanged']
text = (base/'run_owned_event_ack_registered_copy_cb_v01402_code32k.py').read_text().replace("'"+old_mode+"'","'"+new_mode+"'").replace(old_mode+'-v01402',new_mode+'-v01402')

def change(old, new):
    global text
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)

change("binary=root/'build-sycl-event-ack-registered-copy-v3-20261008/build/strata'",
       "binary=root/'build-sycl-event-ack-no-root-prefill-20261008/strata'")
change("        expected=json.loads((base/'event-ack-registered-copy-v01402-matched-build/record.json').read_text())\n        assert expected['passed'] and not expected['active'] and expected['production_inputs_unchanged']\n        assert record['binary_sha256']==expected['candidate_binary_sha256']\n        record['candidate_build_receipt_sha256']=hashlib.sha256((base/'event-ack-registered-copy-v01402-matched-build/record.json').read_bytes()).hexdigest()\n        private_source=binary.parent.parent/'source'\n        assert all(hashlib.sha256((private_source/path).read_bytes()).hexdigest()==value for path,value in expected['candidate_sources'].items())\n        record['candidate_sources']=expected['candidate_sources']",
       "        receipt=base/'event-ack-no-root-prefill-v01402-build/record.json'\n        expected=json.loads(receipt.read_text())\n        assert expected['passed'] and not expected['active'] and expected['baseline_inputs_unchanged']\n        assert record['binary_sha256']==expected['candidate_binary_sha256']\n        record['candidate_build_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()\n        matched_receipt=base/'event-ack-registered-copy-v01402-matched-build/record.json'\n        matched=json.loads(matched_receipt.read_text())\n        assert matched['passed'] and not matched['active'] and matched['production_inputs_unchanged']\n        assert expected['baseline_build_receipt_sha256']==hashlib.sha256(matched_receipt.read_bytes()).hexdigest()\n        assert expected['baseline_binary_sha256']==matched['candidate_binary_sha256']\n        private_source=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'\n        assert all(hashlib.sha256((private_source/path).read_bytes()).hexdigest()==value for path,value in matched['candidate_sources'].items())\n        original=private_source/'sycl/src/prefill/kernels.dp.cpp'\n        candidate=binary.parent/'source/kernels.dp.cpp'\n        assert hashlib.sha256(original.read_bytes()).hexdigest()==expected['original_source_sha256']\n        assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected['candidate_source_sha256']\n        assert candidate.read_text()==original.read_text().replace('sycl::ext::oneapi::experimental::use_root_sync','')\n        record['candidate_sources']=matched['candidate_sources']\n        record['prefill_kernel_candidate']={'file':str(candidate),'sha256':expected['candidate_source_sha256'],'removed_sites':expected['removed_sites']}\n        assert len(expected['removed_sites'])==14")
change("list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-*/record.json'))",
       "list(base.glob('owned-event-ack-registered-copy-cb-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-*/record.json'))")
scope_start = "    record['scope']='Private actual-DMA event acknowledgement candidate"
lines = text.splitlines()
found = [i for i, line in enumerate(lines) if line.startswith(scope_start)]
assert len(found)==1
lines[found[0]] = "    record['scope']='Private scheduling-only experiment on the previously validated registered nonprofiling copy/event-completion candidate. Remove only14 use_root_sync properties from the prefill kernel translation unit; arithmetic, kernel bodies, ND-ranges, subgroups, queues, barriers, drains and graph code unchanged. Implicit counter conversion remains disabled; MKL_CBWR absent. Compare all66 prefill state parts, all248320 first-head floats, all generated IDs and protocol logprobs with the stable registered-copy non-CNR control. First use has flushed UR/Level Zero logs and parameter validation. State/head dumps and diagnostic logs exclude these executions from performance evidence. No claim of hang prevention, default-backend compatibility or full256K safety; candidate not adopted.'"
text='\n'.join(lines)+'\n'
change("previous_cnr=json.loads((base/'event-ack-registered-copy-cb-cnr-v01402-state-sequence/record.json').read_text())",
       "previous_cnr=json.loads((base/'event-ack-registered-copy-cb-v01402-state-sequence/record.json').read_text())")
change("baseline=json.loads((base/'owned-cb-off-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]",
       "baseline=json.loads((base/'owned-event-ack-registered-copy-cb-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]")
text=text.replace('comparison_to_production_cb_off_control','comparison_to_registered_no_cnr_control')
compile(text,str(target),'exec')
target.write_text(text)

seq_target=base/'run_event_ack_no_root_prefill_cb_v01402_state_sequence.py'
assert not seq_target.exists()
seq=(base/'run_event_ack_registered_copy_cb_v01402_state_sequence.py').read_text().replace("'"+old_mode+"'","'"+new_mode+"'").replace(old_mode+'-v01402',new_mode+'-v01402')
seq=seq.replace('run_owned_event_ack_registered_copy_cb_v01402_code32k.py',target.name)
seq=seq.replace('comparison_to_production_cb_off_control','comparison_to_registered_no_cnr_control')
seq=seq.replace('production non-CNR counter-off control','registered-copy non-CNR counter-off control')
compile(seq,str(seq_target),'exec')
seq_target.write_text(seq)
print(json.dumps({'controller':str(target),'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'sequence':str(seq_target),'sequence_sha256':hashlib.sha256(seq_target.read_bytes()).hexdigest()}))
