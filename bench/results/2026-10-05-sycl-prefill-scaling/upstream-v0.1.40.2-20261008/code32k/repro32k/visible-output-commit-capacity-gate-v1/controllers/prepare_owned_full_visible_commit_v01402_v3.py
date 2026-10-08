"""Use the bounded visible-prefix commit candidate for the full-capacity gate."""
from pathlib import Path
import ast,datetime,difflib,hashlib,json
base=Path(__file__).parent
parent=base/'run_owned_full_kv_access_v01402_v2.py'
target=base/'run_owned_full_visible_commit_v01402_v3.py';assert not target.exists()
original=parent.read_text();text=original
def replace(a,b):
    global text
    assert text.count(a)==1,a[:100]
    text=text.replace(a,b)
replace("repetition==2", "repetition==3")
replace("binary=root/'build-sycl-kv-access-safe-v2-20261008/strata'", """binary=root/'build-sycl-visible-commit-v1-20261008/strata'
    visible_receipt=base/'visible-commit-v01402-build-v1/record.json'
    visible_build=json.loads(visible_receipt.read_text())
    assert visible_build['passed'] and not visible_build['active'] and visible_build['baseline_inputs_unchanged']
    assert visible_build['cpu_cases_passed']==12
    assert visible_build['candidate_binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()
    assert visible_build['candidate_source_sha256']==hashlib.sha256((binary.parent/'source/generate.cpp').read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in visible_build['link_input_sha256'].items())
    record['visible_commit_build_receipt_sha256']=hashlib.sha256(visible_receipt.read_bytes()).hexdigest()
    record['visible_commit_source_sha256']=visible_build['candidate_source_sha256']""")
replace("assert access_build['candidate_binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()", "assert access_build['candidate_binary_sha256']==visible_build['baseline_binary_sha256']\n    assert visible_build['access_build_receipt_sha256']==hashlib.sha256(access_receipt.read_bytes()).hexdigest()")
replace("(binary.parent/'source/layer.cpp')", "(Path(access_build['candidate_binary']).parent/'source/layer.cpp')")
replace("health_path=base/'post-chunk12k-v01402-health/record.json'", "health_path=base/'post-save-assert-v01402-health-v1/record.json'")
replace("'bfb8b9fd6f525c716e8998732e1694268b2af80c90640a1d38c42b272dc536bf'", "'8f0293fa08502d8a70a3002826e0f484b7859d275d980f3becf215f657126d4f'")
a="'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1': '3de5e40d7a9bb390112c8de7bc6ee29604e62f8fed45e516cbf57ffe8cb0dfdb'}"
b="'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1': '3de5e40d7a9bb390112c8de7bc6ee29604e62f8fed45e516cbf57ffe8cb0dfdb', 'owned-full-kv-access-v01402-full256k-diagnostic-r2': 'da8657868131392654c034ed6c004176fd624e7bd065f03c4c73aa3b3969a7ac'}"
replace(a,b)
text=text.replace('saved-session-reader-v01402-host-check-v1','saved-session-reader-v01402-host-check-v2')
text=text.replace('read_saved_session_v01402_v1','read_saved_session_v01402_v2')
replace("baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]", """baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]
    assert baseline['mtp_counts']==[43,66]
    record['legacy_control_mtp_counts']=baseline['mtp_counts']
    # The preserved actual SAVE and final window prove two invisible tokens were
    # counted/committed. The fixed code keeps the same66 offered drafts and all
    #64 IDs/logprobs, but counts41 visible accepted drafts.
    baseline['mtp_counts']=[41,66]
    record['expected_visible_control_mtp_counts']=[41,66]""")
replace("assert result['image']['semantic']['live']['ids']['count']==expected_tokens", """assert result['image']['semantic']['live']['ids']['count']==expected_tokens
            if name=='save-control32k':
                visible_ids=control+control_first['ids'][:-1]
            elif name in ['save-full256k-first','save-full256k-repeat']:
                visible_ids=full+full_first['ids'][:-1]
            else:visible_ids=None
            if visible_ids is not None:
                exact=hashlib.sha256(array.array('i',visible_ids).tobytes()).hexdigest()
                assert result['image']['semantic']['live']['ids']['sha256']==exact
                result['saved_ids_match_exact_consumed_visible_prefix']=True""")
replace("    record['scope']='Full256K occupancy", "    record['scope']='Visible-output commit fix plus Full256K occupancy")
ast.parse(text)
assert not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and '\\n' in n.value for n in ast.walk(ast.parse(text)))
target.write_text(text)
out=base/'full-visible-commit-v01402-source-review-v3';out.mkdir(mode=0o700)
receipt=json.loads((base/'full-kv-access-v01402-source-review-v2/record.json').read_text())
receipt.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               controller_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
               parent_controller_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               session_reader_sha256=hashlib.sha256((base/'read_saved_session_v01402_v2.py').read_bytes()).hexdigest(),
               visible_commit_build_receipt_sha256=hashlib.sha256((base/'visible-commit-v01402-build-v1/record.json').read_bytes()).hexdigest())
receipt['reviewed'].append('Actual32K SAVE at64 emitted predictions contained32833 consumed tokens, two more than the expected32831, because accepted window rows were committed before length/EOS clipping. The new helper clips the accepted prefix before commit on serial/pipeline serve and CLI. VerifierT and emitted numerical rows stay unchanged. Initial32K must retain all head/state/IDs/logprobs and66 offered drafts; visible accepted count is41 instead of43. Saved token IDs must exactly equal prompt+outputs excluding the last output.')
receipt['reviewed'].append('The disk trailer is payload-hash followed by end magic, as implemented in the actual codec. Reader v2 corrects the initial synthetic assumption and is checked on the actual619404764-byte saved file. Engine RESTORE, not the evidence reader, validates checksum/compatibility.')
(out/'record.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'controller.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(parent),tofile=str(target))))
print(json.dumps({'passed':True,'controller':str(target),'sha256':receipt['controller_sha256']},indent=2))
