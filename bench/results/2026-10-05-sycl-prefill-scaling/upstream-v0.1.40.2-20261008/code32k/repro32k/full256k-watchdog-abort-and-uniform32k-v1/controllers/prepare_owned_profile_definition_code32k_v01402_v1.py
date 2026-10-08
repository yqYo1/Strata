"""Prepare a separate logged >=32K numerical gate from the immutable full controller."""
from pathlib import Path
import ast, hashlib, json

base=Path(__file__).parent
old=base/'run_owned_full_indexer_spare_v01402_v5.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='110a24bb000b1fb65c66241d2f0ef8a0402caf4ac7e6af39ab97a7d5ceab3ff8'
s=old.read_text()
def replace(a,b):
    global s
    assert s.count(a)==1,(a[:100],s.count(a))
    s=s.replace(a,b)
replace("repetition==5", "repetition==6")
replace("out = base/f'owned-{mode}-v01402-full256k-{phase}-r{repetition}'", "out = base/f'owned-profile-definition-v01402-code32k-{phase}-r{repetition}'")
replace("health_path=base/'post-indexer-spare-rejection-v01402-health-v1/record.json'", "health_path=base/'post-full256k-abort-v01402-health-v1/record.json'")
replace("=='d45fe2a9cbc28d6db83e544fc277681cd27bf71c2d53e80b1364b65906f1daa3'", "=='760491d1de59dc7d18936abc0afc4609b7332b12bdb93aec14d51638ddfa2c60'")
replace("        if previous_path.parent.name in known_failures:", "        known_failures['owned-full-kv-access-v01402-full256k-diagnostic-r5']='e7e541f91ef053acebaa6b9ed2bc3139010e269f0b671d23f423869e2c7a882f'\n        if previous_path.parent.name in known_failures:")
needle="    final=json.loads((base/'owned-upstream-e8ca-refresh-short-final-float-storage/record.json').read_text());assert final['healthy'] and not final['active']"
added='''    uniform_path=base/'dpct-profile-definition-v01402-build-v1/record.json'
    uniform=json.loads(uniform_path.read_text())
    assert uniform['passed'] and not uniform['active'] and not uniform['gpu_tested']
    assert uniform['baseline_binary_sha256']==spare_build['candidate_binary_sha256']
    assert uniform['baseline_inputs_unchanged'] and len(uniform['objects'])==7
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in uniform['link_input_sha256'].items())
    binary=Path(uniform['candidate_binary'])
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==uniform['candidate_binary_sha256']=='dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
    uniform_review=base/'dpct-profile-definition-v01402-source-review-v3/record.json'
    linked_review=base/'dpct-profile-definition-linked-tu-audit-v3/record.json'
    assert all(json.loads(p.read_text())['passed'] and not json.loads(p.read_text())['active'] for p in (uniform_review,linked_review))
    abort_review=base/'full256k-watchdog-abort-v01402-review-v1/record.json'
    assert json.loads(abort_review.read_text())['passed']
    record['uniform_header_build_receipt_sha256']=hashlib.sha256(uniform_path.read_bytes()).hexdigest()
    record['uniform_header_source_review_sha256']=hashlib.sha256(uniform_review.read_bytes()).hexdigest()
    record['uniform_header_linked_review_sha256']=hashlib.sha256(linked_review.read_bytes()).hexdigest()
    record['previous_watchdog_abort_review_sha256']=hashlib.sha256(abort_review.read_bytes()).hexdigest()
    record['compatibility_change']='Only seven originally unprofiled TUs recompiled with the uniform DPCT header; translated profiling branches and explicit copy-queue factory unchanged. No polling change or claimed stall fix.'
    record['adopted']=False
'''
replace(needle,added+needle)
start=s.index("    record['scope']='Indexer spare repair")
end=s.index('\n',start)
s=s[:start]+"    record['scope']='Logged uniform-DPCT-header numerical gate: four fresh32768-token reads alternating two inputs, all head/used-state/output/logprob/MTP comparisons, actual32K SAVE/RESTORE continuation. Diagnostic durations excluded; no full-capacity, speed or stall-fix claim.'"+s[end:]
start=s.index("    record['previous_goal_turn']=")
end=s.index('\n',start)
s=s[:start]+"    record['previous_goal_turn']='progress: first repaired full image passed; repeated full aborted in host watchdog at layer33/chunk196608. Post-exit device health passes; cause of pending counter events unresolved. Test separate standard-definition repair before profiling.'"+s[end:]
start=s.index("        previous_full=next(x for x in rejected['requests']")
end=s.index("        record['capacity_sequence_completed']=True",start)+len("        record['capacity_sequence_completed']=True")
block='''        alternate=full_prompt(32768)
        alternate_first=request('alternate32k-first',alternate,64,fresh=True)
        if not math_ok:reject()
        request('control32k-repeat',control,64,control_first,fresh=True)
        if not math_ok:reject()
        session('RESTORE','restore-control32k',control_file,32831)
        restored=request('resume32k-restored',continuation,64,resume_reference)
        assert max(restored['resume_tokens'])==32831
        if not math_ok:reject()
        request('alternate32k-repeat',alternate,64,alternate_first,fresh=True)
        if not math_ok:reject()
        record['code32k_sequence_completed']=True
        record['full_capacity_sequence_completed']=False'''
s=s[:start]+block+s[end:]
replace("bool(record.get('capacity_sequence_completed'))", "bool(record.get('code32k_sequence_completed'))")
ast.parse(s)
p=base/'run_owned_profile_definition_code32k_v01402_v1.py'
assert not p.exists();p.write_text(s)
print(json.dumps({'prepared':True,'controller':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'gpu_launched':False},indent=2))
