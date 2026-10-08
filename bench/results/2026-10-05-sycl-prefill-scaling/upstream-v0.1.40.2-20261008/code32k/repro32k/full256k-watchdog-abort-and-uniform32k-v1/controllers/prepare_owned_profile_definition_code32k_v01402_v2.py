"""Use the real uniform-build receipt fields and verify every copied archive member."""
from pathlib import Path
import ast, hashlib, json
base=Path(__file__).parent
old=base/'run_owned_profile_definition_code32k_v01402_v1.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='990253aa2112e8c7e119de75919bac626bc336ea845a634a69b3870e76572ec9'
s=old.read_text()
a="    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in uniform['link_input_sha256'].items())"
assert s.count(a)==1
b='''    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in uniform['baseline_link_input_sha256'].items())
    assert hashlib.sha256(Path(uniform['shadow_header']).read_bytes()).hexdigest()==uniform['shadow_header_sha256']
    for item in uniform['objects']:
        for path_key,hash_key in [('source','source_sha256'),('object','object_sha256'),('dependency','dependency_sha256')]:
            assert hashlib.sha256(Path(item[path_key]).read_bytes()).hexdigest()==item[hash_key]
    candidate_inputs={uniform['replaced_archives'].get(p,p):sha for p,sha in uniform['baseline_link_input_sha256'].items()}
    for old_archive,new_archive in uniform['replaced_archives'].items():
        expected_members=[x for x in uniform['archive_members'] if x['archive']==old_archive]
        actual_members=subprocess.check_output(['/usr/bin/ar','t',new_archive],text=True).splitlines()
        assert actual_members==[x['member'] for x in expected_members]
        for item in expected_members:
            data=subprocess.check_output(['/usr/bin/ar','p',new_archive,item['member']])
            assert hashlib.sha256(data).hexdigest()==item['after_sha256']
        candidate_inputs[new_archive]=hashlib.sha256(Path(new_archive).read_bytes()).hexdigest()
    record['uniform_candidate_link_input_sha256']=candidate_inputs
'''
s=s.replace(a,b)
ast.parse(s)
p=base/'run_owned_profile_definition_code32k_v01402_v2.py';assert not p.exists();p.write_text(s)
print(json.dumps({'prepared':True,'controller':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_launched':False,
                 'v1_not_executed':'CPU receipt-key review found link_input_sha256 absent; use baseline_link_input_sha256 plus verified candidate archives.'},indent=2))
