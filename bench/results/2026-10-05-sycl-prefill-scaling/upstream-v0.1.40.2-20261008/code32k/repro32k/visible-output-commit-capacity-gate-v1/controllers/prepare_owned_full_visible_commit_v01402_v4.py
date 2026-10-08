"""Validate the actual legacy protocol schema on CPU, before model startup."""
from pathlib import Path
import ast,datetime,difflib,hashlib,json
base=Path(__file__).parent
parent=base/'run_owned_full_visible_commit_v01402_v3.py'
target=base/'run_owned_full_visible_commit_v01402_v4.py';assert not target.exists()
original=parent.read_text();text=original
health_path=base/'post-controller-assert-v01402-health-v2/record.json'
health=json.loads(health_path.read_text());assert health['healthy'] and not health['active']
old_path=base/'owned-full-kv-access-v01402-full256k-diagnostic-r3/record.json'
old=json.loads(old_path.read_text());assert old['error']=="KeyError('mtp_counts')" and not old['active'] and not old['requests']
assert health['terminal_controller_receipt_sha256']==hashlib.sha256(old_path.read_bytes()).hexdigest()
def replace(a,b):
    global text
    assert text.count(a)==1,a[:100]
    text=text.replace(a,b)
replace('repetition==3','repetition==4')
replace("health_path=base/'post-save-assert-v01402-health-v1/record.json'", "health_path=base/'post-controller-assert-v01402-health-v2/record.json'")
replace("'8f0293fa08502d8a70a3002826e0f484b7859d275d980f3becf215f657126d4f'", repr(hashlib.sha256(health_path.read_bytes()).hexdigest()))
a="'owned-full-kv-access-v01402-full256k-diagnostic-r2': 'da8657868131392654c034ed6c004176fd624e7bd065f03c4c73aa3b3969a7ac'}"
b=a[:-1]+", 'owned-full-kv-access-v01402-full256k-diagnostic-r3': "+repr(hashlib.sha256(old_path.read_bytes()).hexdigest())+'}'
replace(a,b)
replace("assert baseline['mtp_counts']==[43,66]", "assert list(map(int,baseline['protocol'][-1].split()[6:8]))==[43,66]")
replace("record['legacy_control_mtp_counts']=baseline['mtp_counts']", "record['legacy_control_mtp_counts']=list(map(int,baseline['protocol'][-1].split()[6:8]))")
# Fixture/schema failures must not allocate a model or stop an idle GPU process.
start=text.index('    from read_saved_session_v01402_v2 import read_session, full_kv_gate')
end=text.index("    stderr=out/'debugger/inferior.stderr'",start)
block=text[start:end]
text=text[:start]+text[end:]
where=text.index("    cursor=m.journal_cursor(r,'kernel-before');save()")
text=text[:where]+block+text[where:]
# Execute the CPU-only fixture and reference block now as an additional guard.
space={'base':base,'hashlib':hashlib,'json':json,'record':{}}
import sys
sys.path.insert(0,str(base))
exec(compile('\n'.join(v[4:] if v.startswith('    ') else v for v in block.splitlines()),str(target)+'::cpu_preflight','exec'),space)
assert len(space['control'])==32768 and len(space['full'])==262140
assert space['baseline']['mtp_counts']==[41,66]
assert len(space['baseline']['ids'])==64 and len(space['baseline']['logprobs'])==64 and len(space['baseline']['prefill_state']['parts'])==66
ast.parse(text)
assert not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and '\\n' in n.value for n in ast.walk(ast.parse(text)))
target.write_text(text)
out=base/'full-visible-commit-v01402-source-review-v4';out.mkdir(mode=0o700)
receipt=json.loads((base/'full-visible-commit-v01402-source-review-v3/record.json').read_text())
receipt.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               controller_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
               parent_controller_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               cpu_preflight_passed=True,legacy_control_schema='MTP counts read from final DONE protocol fields6/7; no optional derived metadata assumed',
               prior_controller_metadata_rejection_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest())
(out/'record.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'controller.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(parent),tofile=str(target))))
print(json.dumps({'passed':True,'controller':str(target),'sha256':receipt['controller_sha256']},indent=2))
