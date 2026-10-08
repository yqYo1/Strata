"""Keep chunk geometry while enabling a final full-prefix checkpoint."""
from pathlib import Path
import ast
import datetime
import difflib
import hashlib
import json

base=Path(__file__).parent
parent=base/'run_owned_full_kv_access_v01402_v1.py'
target=base/'run_owned_full_kv_access_v01402_v2.py'
assert not target.exists()
text=original=parent.read_text()
def replace(a,b):
    global text
    assert text.count(a)==1,a[:100]
    text=text.replace(a,b)
replace("assert mode=='full-kv-access' and phase=='diagnostic' and repetition==1", "assert mode=='full-kv-access' and phase=='diagnostic' and repetition==2")
replace("'--prompt-cache-every','0','--prompt-cache-root','0'", "'--prompt-cache-every','262139','--prompt-cache-root','0','--turn-token','-1'")
replace("record['configuration_note']='PC1/ckpt1 for real SAVE/RESTORE; no periodic/root checkpoints;", "record['configuration_note']='PC1/ckpt1 for real SAVE/RESTORE; turn-token=-1 and root=0 preserve the accepted chunk geometry. A single periodic checkpoint at262139 is taken after the existing final full-input chunk; no checkpoint is taken in32K controls;")
replace("    cursor=m.journal_cursor(r,'kernel-before');save()", """    reader_check=base/'saved-session-reader-v01402-host-check-v1/record.json'
    cpu_reader=json.loads(reader_check.read_text())
    assert cpu_reader['passed'] and not cpu_reader['active']
    assert cpu_reader['reader_sha256']==hashlib.sha256((base/'read_saved_session_v01402_v1.py').read_bytes()).hexdigest()
    record['session_reader_cpu_check_sha256']=hashlib.sha256(reader_check.read_bytes()).hexdigest()
    earlier=base/'owned-full-kv-access-v01402-full256k-diagnostic-r1/record.json'
    assert hashlib.sha256(earlier.read_bytes()).hexdigest()=='833dcb26548cd298b49f2da746fa48675f04e75d40f47c1409f5a3d3b60caf1d'
    negative=json.loads(earlier.read_text())
    assert negative['healthy'] and negative['completed'] and not negative['active'] and not negative['math_gate_passed']
    assert negative['exit_code']==0 and not negative['new_fault_messages']
    record['previous_config_rejection']={'receipt_sha256':hashlib.sha256(earlier.read_bytes()).hexdigest(),
        'reason':'PC1/default turn-token split the32K final chunk at32761+6, unlike the accepted PC0 geometry32767. Same64 IDs but different intermediate state/head/logprobs and42/69 rather than43/66 MTP counts. Normal exit, no GPU fault. Not a source-change attribution.'}
    cursor=m.journal_cursor(r,'kernel-before');save()""")
replace("{'measurements':[x['measurement'] for x in record['requests']]}", "{'measurements':[x['measurement'] for x in record['requests'] if 'measurement' in x]}")
ast.parse(text)
assert not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and '\\n' in n.value for n in ast.walk(ast.parse(text)))
target.write_text(text)
out=base/'full-kv-access-v01402-source-review-v2';out.mkdir(mode=0o700)
receipt=json.loads((base/'full-kv-access-v01402-source-review-v1/record.json').read_text())
receipt.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               controller_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
               parent_controller_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               corrected_configuration={'turn_token':-1,'prompt_cache_root':0,'prompt_cache_every':262139,
                  'checkpoint_boundary':'after existing final full chunk; same8192/8192/.../8187 geometry, no32K split'})
receipt['reviewed']=[v for v in receipt['reviewed'] if not v.startswith('SAVE requires')]
receipt['reviewed'].append('SAVE requires PC>0 and a completed ckpt1 request. Default turn-token causes a prefill split, so explicitly disable turn/root boundaries. on_chunk checks periodic threshold only after the existing chunk finishes:262139 records the complete full-input checkpoint without changing chunk geometry. No checkpoint splits or periodic snapshots in32K controls.')
(out/'record.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'controller.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(parent),tofile=str(target))))
print(json.dumps({'passed':True,'controller':str(target),'sha256':receipt['controller_sha256']},indent=2))
