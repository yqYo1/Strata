"""Prepare two dump-free 32K controllers and a gated ABBA comparison."""
from pathlib import Path

base=Path(__file__).parent
modes={
 'event-ack-registered-copy-cb':'run_owned_event_ack_registered_copy_cb_v01402_code32k.py',
 'event-ack-no-root-prefill-cb':'run_owned_event_ack_no_root_prefill_cb_v01402_code32k.py',
}
for mode,source in modes.items():
 target=base/(Path(source).stem+'_clean.py')
 assert not target.exists()
 text=(base/source).read_text()
 old=f"assert mode=='{mode}' and phase in ['diagnostic','state'] and repetition in [1,2,3]"
 new=f"""assert mode=='{mode}' and phase=='clean' and repetition in [1,2]
gate=json.loads((base/'{mode}-v01402-state-sequence/record.json').read_text())
assert gate['passed'] and not gate['active']
baseline_clean=json.loads((base/'owned-{mode}-v01402-code32k-diagnostic-r1/record.json').read_text())
assert baseline_clean['healthy'] and not baseline_clean['active']
baseline_clean=baseline_clean['requests'][0]"""
 assert text.count(old)==1
 text=text.replace(old,new)
 for line in ["    env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')\n", "    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')\n"]:
  assert text.count(line)==1
  text=text.replace(line,'')
 start=text.index("        head=out/'first-head.bin';data=head.read_bytes();")
 end=text.index("        record['requests'].append(result);save()",start)
 replacement="""        result['comparison_to_logged_control']={'ids_equal':result['ids']==baseline_clean['ids'],'logprobs_equal':result['logprobs']==baseline_clean['logprobs']}
        assert all(result['comparison_to_logged_control'].values()), 'clean32K output differs from fully captured control'
"""
 text=text[:start]+replacement+text[end:]
 # Preserve every earlier model's ownership check and check both clean arms.
 old="    for previous_path in "
 assert text.count(old)==1
 lines=text.splitlines()
 i=next(i for i,line in enumerate(lines) if line.startswith(old))
 lines[i]=lines[i][:-1]+"+list(base.glob('owned-event-ack-registered-copy-cb-v01402-code32k-clean-*/record.json'))+list(base.glob('owned-event-ack-no-root-prefill-cb-v01402-code32k-clean-*/record.json')):"
 scope=[i for i,line in enumerate(lines) if line.startswith("    record['scope']=")]
 assert len(scope)==1
 lines[scope[0]]="    record['scope']='Clean32K timing on a private registered nonprofiling copy candidate, with or without14 prefill use_root_sync properties. Each arm is gated by three completed exact full state/head32K checks. No UR/Level Zero debug logging, parameter validation, extra waits, state/head dumps, transfer profiling or profiler. An uninterrupted owned GDB/PTY observer is common to every arm. Require all64 IDs and protocol logprobs equal the fully captured control. Implicit counter conversion remains disabled; MKL_CBWR absent. No upstream-equivalence, default-backend or full256K claim; candidates not adopted.'"
 note=next(i for i,line in enumerate(lines) if line.startswith("    record['configuration_note']='State and head capture"))
 lines[note]=lines[note].replace('State and head capture are after the prompt computation and are not performance evidence.','Clean timing: no state/head dumps, API traces or parameter validation.')
 text='\n'.join(lines)+'\n'
 compile(text,str(target),'exec')
 target.write_text(text)
 print(target)
