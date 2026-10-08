from pathlib import Path
base=Path(__file__).parent
target=base/'run_owned_cb_off_v01402_code32k_clean.py';assert not target.exists()
text=(base/'run_owned_cb_off_v01402_code32k.py').read_text()
old="assert mode=='patched-default' and phase in ['diagnostic','state'] and repetition in [1,2,3]"
new="""assert mode=='patched-default' and phase=='clean' and repetition in [1,2]
gate=json.loads((base/'cb-off-v01402-state-sequence/record.json').read_text())
assert gate['passed'] and not gate['active']
baseline_cb=json.loads((base/'owned-cb-off-v01402-code32k-diagnostic-r1/record.json').read_text())
assert baseline_cb['healthy'] and not baseline_cb['active']
baseline_cb=baseline_cb['requests'][0]"""
assert text.count(old)==1;text=text.replace(old,new)
for line in ["    env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')\n","    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')\n"]:
 assert text.count(line)==1;text=text.replace(line,'')
start=text.index("        head=out/'first-head.bin';data=head.read_bytes();")
end=text.index("        record['requests'].append(result);save()",start)
text=text[:start]+"        assert result['ids']==baseline_cb['ids'] and result['logprobs']==baseline_cb['logprobs'], 'clean32K output differs from the fully captured control'\n"+text[end:]
old="record['configuration_note']='State and head capture are after the prompt computation and are not performance evidence."
new="record['scope']='Clean32K timing of the production fork with implicit counter event conversion disabled, gated by three full state/head32K passes. No logging, validation, extra waits, state or head capture. Reject if all64 generated IDs/logprobs differ from the logged control. This does not prove full256K capacity or parity against upstream.'\n    record['configuration_note']='Clean timing: no state/head dumps or debug logging."
assert text.count(old)==1;text=text.replace(old,new)
compile(text,str(target),'exec');target.write_text(text);print(target)
