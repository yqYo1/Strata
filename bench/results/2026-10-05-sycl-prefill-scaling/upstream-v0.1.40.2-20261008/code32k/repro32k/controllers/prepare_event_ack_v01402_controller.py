"""Create a new bounded controller, preserving every previously executed file."""
from pathlib import Path
import hashlib
base=Path(__file__).parent
source=base/'run_owned_v01402_code32k_heads.py'
target=base/'run_owned_event_ack_v01402_code32k.py'
assert not target.exists()
text=source.read_text()
def change(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
change('import subprocess\n','import subprocess\nimport struct\n')
change("assert mode=='patched-default' and phase=='head' and repetition in [1,2,3]", "assert mode=='event-ack' and phase in ['diagnostic','state'] and repetition in [1,2,3]")
change("previous=json.loads((base/f'owned-v01402-code32k-{mode}-diagnostic-r1/record.json').read_text())", "previous=json.loads((base/'owned-v01402-code32k-patched-default-diagnostic-r1/record.json').read_text())")
change("out = base/f'owned-v01402-code32k-{mode}-{phase}-r{repetition}'", "out = base/f'owned-event-ack-v01402-code32k-{phase}-r{repetition}'")
change("health=json.loads((base/'post-upstream-e8ca-final-float-storage-health/record.json').read_text())", "health=json.loads((base/'post-v01402-code32k-head-stall-health/record.json').read_text())")
change("binary=root/('build-sycl-pure-20261008/strata' if mode=='pure' else 'build-sycl-e8ca-refresh-20261007/strata')", "binary=root/'build-sycl-event-ack-v2-20261008/strata'")
change("expected=json.loads((base/'upstream-e8ca-refresh-20261007/binary-receipt.json').read_text())\n        assert record['binary_sha256']==expected['sha256']", "expected=json.loads((base/'event-ack-v01402-v2-build/record.json').read_text())\n        assert expected['passed'] and not expected['active'] and expected['production_inputs_unchanged']\n        assert record['binary_sha256']==expected['candidate_binary_sha256']\n        record['candidate_build_receipt_sha256']=hashlib.sha256((base/'event-ack-v01402-v2-build/record.json').read_bytes()).hexdigest()")
change("list(base.glob('owned-v01402*-*/record.json'))", "list(base.glob('owned-v01402*-*/record.json'))+list(base.glob('owned-event-ack-v01402-code32k-*/record.json'))")
failed=base/'owned-v01402-code32k-patched-default-head-r2/record.json'
failed_sha=hashlib.sha256(failed.read_bytes()).hexdigest()
old="previous=json.loads(previous_path.read_text());assert not previous['active'] and previous['healthy']"
new=f"""previous=json.loads(previous_path.read_text());assert not previous['active']
        if previous_path==base/'owned-v01402-code32k-patched-default-head-r2/record.json':
            assert hashlib.sha256(previous_path.read_bytes()).hexdigest()=='{failed_sha}'
            assert not any(previous['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
            assert health['started_utc']>previous['finished_utc']
            record['known_terminal_failure_sha256']='{failed_sha}'
        else:assert previous['healthy']"""
change(old,new)
change("assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in env)", "if phase!='diagnostic':assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in env)")
change("env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')", "env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')\n    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')")
change("record['configuration_note']='Head capture is after the prompt computation and is not performance evidence.", "record['scope']='Private actual-DMA event acknowledgement candidate. First use with flushed UR/Level Zero logs; all32K prompt state and first head captured after prefill. No reset/rebind/service/reboot. Not performance/capacity evidence.'\n    record['configuration_note']='State and head capture are after the prompt computation and are not performance evidence.")
needle="        record['requests'].append(result);save()"
replacement="""        state=out/'prefill-state.bin';parts=[]
        with state.open('rb') as stream:
            while header:=stream.read(8):
                assert len(header)==8
                size=struct.unpack('=Q',header)[0];offset=stream.tell();left=size;digest=hashlib.sha256()
                while left:
                    data=stream.read(min(left,1048576));assert data;digest.update(data);left-=len(data)
                parts.append({'index':len(parts),'offset':offset,'bytes':size,'sha256':digest.hexdigest()})
        assert len(parts)==66 and parts[0]['bytes']==8
        result['prefill_state']={'file':str(state),'bytes':state.stat().st_size,'parts':parts,'captured_at':'after32767 batched prompt tokens, before the held-out input token and first verifier window'}
        record['requests'].append(result);save()"""
change(needle,replacement)
compile(text,str(target),'exec');target.write_text(text)
print(target)
