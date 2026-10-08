from pathlib import Path
import hashlib
base=Path(__file__).parent
target=base/'run_owned_cb_off_v01402_code32k.py';assert not target.exists()
text=(base/'run_owned_v01402_code32k_heads.py').read_text()
def change(old,new):
 global text
 assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
change('import subprocess\n','import subprocess\nimport struct\n')
change("assert mode=='patched-default' and phase=='head' and repetition in [1,2,3]", "assert mode=='patched-default' and phase in ['diagnostic','state'] and repetition in [1,2,3]")
change("out = base/f'owned-v01402-code32k-{mode}-{phase}-r{repetition}'", "out = base/f'owned-cb-off-v01402-code32k-{phase}-r{repetition}'")
change("health=json.loads((base/'post-upstream-e8ca-final-float-storage-health/record.json').read_text())", "health=json.loads((base/'post-event-ack-v01402-stall-health/record.json').read_text())")
change("list(base.glob('owned-v01402*-*/record.json'))", "list(base.glob('owned-v01402*-*/record.json'))+list(base.glob('owned-event-ack-v01402-code32k-*/record.json'))+list(base.glob('owned-cb-off-v01402-code32k-*/record.json'))")
known={name:hashlib.sha256((base/name/'record.json').read_bytes()).hexdigest() for name in ['owned-v01402-code32k-patched-default-head-r2','owned-event-ack-v01402-code32k-diagnostic-r1']}
change("previous=json.loads(previous_path.read_text());assert not previous['active'] and previous['healthy']", f"""previous=json.loads(previous_path.read_text());assert not previous['active']
        known_failures={known!r}
        if previous_path.parent.name in known_failures:
            assert hashlib.sha256(previous_path.read_bytes()).hexdigest()==known_failures[previous_path.parent.name]
            assert not any(previous['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
            assert health['started_utc']>previous['finished_utc']
            record['known_terminal_failures']=known_failures
        else:assert previous['healthy']""")
change("env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')", "env.update(STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8',EnableImplicitConvertionToCounterBasedEvents='0')")
change("assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in env)", "if phase!='diagnostic':assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in env)")
change("env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')", "env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')\n    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')")
change("['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']", "['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission','EnableImplicitConvertionToCounterBasedEvents']")
change("record['configuration_note']='Head capture is after the prompt computation and is not performance evidence.", "record['scope']='Same production integrated fork32K, only NEO implicit conversion to counter-based events disabled. This tests the profiling counter event path seen in both stopped jobs. First use fully logged. State/head captures exclude this run from speed comparisons. No claim of bug cause, prevention or full256K validation.'\n    record['configuration_note']='State and head capture are after the prompt computation and are not performance evidence.")
change("        record['requests'].append(result);save()", """        state=out/'prefill-state.bin';parts=[]
        with state.open('rb') as stream:
            while header:=stream.read(8):
                assert len(header)==8
                size=struct.unpack('=Q',header)[0];offset=stream.tell();left=size;digest=hashlib.sha256()
                while left:
                    data=stream.read(min(left,1048576));assert data;digest.update(data);left-=len(data)
                parts.append({'index':len(parts),'offset':offset,'bytes':size,'sha256':digest.hexdigest()})
        assert len(parts)==66 and parts[0]['bytes']==8
        result['prefill_state']={'file':str(state),'bytes':state.stat().st_size,'parts':parts,'captured_at':'after32767 batched prompt tokens, before held-out input token/first verifier window'}
        record['requests'].append(result);save()""")
compile(text,str(target),'exec');target.write_text(text);print(target)
