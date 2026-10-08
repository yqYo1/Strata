"""Prepare three normal-MTP32K full rereads in one logged serve process."""
from pathlib import Path
import ast, datetime, hashlib, json

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
private=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
source=base/'run_owned_main_cache_release_v01402_code32k.py'
target=base/'run_owned_main_cache_repeat32k_v01402.py'
out=base/'main-cache-repeat32k-v01402-source-review'
assert not target.exists() and not out.exists()
host=base/'live-prefill-state-v01402-host-check-v2/record.json'
cpu=json.loads(host.read_text());assert cpu['passed'] and not cpu['active'] and len(cpu['cases'])==34
parent=json.loads((base/'main-cache-release-v01402-clean-sequence/record.json').read_text())
assert parent['passed'] and not parent['active']
negative=base/'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1/record.json'
assert hashlib.sha256(negative.read_bytes()).hexdigest()=='3de5e40d7a9bb390112c8de7bc6ee29604e62f8fed45e516cbf57ffe8cb0dfdb'
n=json.loads(negative.read_text());assert n['completed'] and n['exit_code']==0 and not n['math_gate_passed'] and not n['new_fault_messages']
health=base/'post-chunk12k-v01402-health/record.json'
assert hashlib.sha256(health.read_bytes()).hexdigest()=='bfb8b9fd6f525c716e8998732e1694268b2af80c90640a1d38c42b272dc536bf'

text=source.read_text()
def replace(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
replace("mode in ['main-vmm-kept-ram','main-vmm-half-ram','main-vmm-full-ram','main-vmm-full-snapshot']", "mode=='main-vmm-full-ram-repeat32k'")
replace("fractions={'main-vmm-kept-ram':'0','main-vmm-half-ram':'0.5','main-vmm-full-ram':'1','main-vmm-full-snapshot':'1'}", "fractions={'main-vmm-full-ram-repeat32k':'1'}")
replace("health=json.loads((base/'post-event-ack-cb-cnr-v01402-stall-health/record.json').read_text())", "health_path=base/'post-chunk12k-v01402-health/record.json'\n    assert hashlib.sha256(health_path.read_bytes()).hexdigest()=='bfb8b9fd6f525c716e8998732e1694268b2af80c90640a1d38c42b272dc536bf'\n    health=json.loads(health_path.read_text())")
old="'owned-event-ack-cb-cnr-v01402-code32k-state-r1': '073f0a2081f333c21254c7f1388681473bada935cacf5db56ee3d7994f856f95'}"
new="'owned-event-ack-cb-cnr-v01402-code32k-state-r1': '073f0a2081f333c21254c7f1388681473bada935cacf5db56ee3d7994f856f95', 'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1': '3de5e40d7a9bb390112c8de7bc6ee29604e62f8fed45e516cbf57ffe8cb0dfdb'}"
replace(old,new)
replace("            record['known_terminal_failures']=known_failures", "            if previous_path.parent.name=='owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1':\n                assert previous['completed'] and previous['exit_code']==0 and not previous['math_gate_passed'] and not previous['new_fault_messages']\n                record['known_mathematical_rejection']=previous_path.parent.name\n            record['known_terminal_failures']=known_failures")
replace("'deadline_seconds':900,'protocol_timeout_seconds':600,'log_limit_bytes':8*1024**3", "'deadline_seconds':1800,'protocol_timeout_seconds':600,'log_limit_bytes':20*1024**3")
replace("    if phase=='diagnostic':env=m.diagnostic_environment(env)", "    env['STRATA_TRACE']='1'\n    if phase=='diagnostic':env=m.diagnostic_environment(env)")
replace("    record['source_review_sha256']=hashlib.sha256((base/'main-cache-release-v01402-source-review/record.json').read_bytes()).hexdigest()", "    record['source_review_sha256']=hashlib.sha256((base/'main-cache-repeat32k-v01402-source-review/record.json').read_bytes()).hexdigest()\n    live_gate=json.loads((base/'live-prefill-state-v01402-host-check-v2/record.json').read_text())\n    assert live_gate['passed'] and not live_gate['active']\n    record['live_state_comparator_sha256']=hashlib.sha256((base/'compare_live_prefill_state_v01402_v2.py').read_bytes()).hexdigest()")
start=text.index("    record['scope']='Private matched32K main-cache lease comparison")
end=text.index('\n',start)
text=text[:start]+"    record['scope']='Three fresh full32768-token requests in one logged/validated normal-MTP process on unchanged e82fc5 binary. Accepted8192 chunks, all32767 GPU residual rows, full main/MTP decode-only RAM release. Each64-output reply has a newly written head and complete raw66-part state; used cells, recurrent/PLE/indexer state, complete head/IDs/logprobs and MTP43/66 must match the accepted baseline. Only page-rounded future cells may differ and are retained separately; no exclusions at full262144 occupancy. Trace phase memory and existing-graph retirement/restoration. Diagnostic durations excluded from speed; not full256K/adoption proof.'"+text[end:]
replace("    for expected in reference['requests'][:1]:\n        current=expected['name']", "    from compare_live_prefill_state_v01402_v2 import compare_states\n    for request_index in range(3):\n        current=f'read-{request_index+1}'")
replace("        send((request+'\\n').encode());event('request',name=current)", "        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists(), 'stale capture exists before request'\n        send((request+'\\n').encode());event('request',name=current)")
replace("        head=out/'first-head.bin';data=head.read_bytes();floats=array.array('f');floats.frombytes(data)", "        head=out/'first-head.bin';assert head.exists(), 'missing fresh head capture'\n        preserved=out/f'{current}.head.bin';assert not preserved.exists();head.rename(preserved);head=preserved\n        data=head.read_bytes();floats=array.array('f');floats.frombytes(data)")
replace("        state=out/'prefill-state.bin';parts=[]", "        state=out/'prefill-state.bin';assert state.exists(), 'missing fresh state capture'\n        preserved=out/f'{current}.state.bin';assert not preserved.exists();state.rename(preserved);state=preserved\n        parts=[]")
replace("        assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal']), 'reject before timing: full-state/head/output mismatch'", "        live=compare_states(result['prefill_state'],baseline['prefill_state'],32767)\n        result['live_prefill_comparison']=live\n        result['mtp_counts']=list(map(int,result['protocol'][-1].split()[6:8]))\n        result['no_prompt_reuse']=any(line in ['RESUME 0','REUSED 0'] for line in result['protocol']) and not any(line.startswith(('RESUME ','REUSED ')) and line.split()[1]!='0' for line in result['protocol'])\n        result['math_gate_passed']=not live['different_live_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal']) and result['mtp_counts']==[43,66] and result['no_prompt_reuse']\n        if request_index==0:result['math_gate_passed']=result['math_gate_passed'] and not c['different_prefill_state_parts']\n        save()\n        if not result['math_gate_passed']:break")
replace("    record['completed']=True", "    record['completed']=True\n    record['math_gate_passed']=len(record['requests'])==3 and all(req['math_gate_passed'] for req in record['requests'])\n    assert record['math_gate_passed'], 'reject before timing: repeated live-state/head/output mismatch'")
ast.parse(text);target.write_text(text);out.mkdir(mode=0o700)
files=['sycl/src/program/generate.cpp','sycl/src/core/conversation_snapshot.cpp','sycl/src/core/session.cpp','include/strata/kernels/qsa.hpp','sycl/src/prefill/prefill.cpp','sycl/src/core/verify.cpp','sycl/src/core/mtp.cpp']
sources={f:hashlib.sha256((private/f).read_bytes()).hexdigest() for f in files}
assert all((private/f).read_bytes()==(root/f).read_bytes() for f in files if f!='sycl/src/prefill/prefill.cpp')
review={'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sources,
        'host_comparator_check_sha256':hashlib.sha256(host.read_bytes()).hexdigest(),
        'source_controller_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Source/lifetime/comparison review before first same-process32K repeated main-cache lease. No source/binary mutation or speed/fullcontext claim.',
        'review':['GEN64 ckpt=0 with prompt-cache0 and conversation-cache0 sets resume0. The serve path calls session_zero and waits before a full reread; each request initializes first_window=true and writes its head at that first window.',
                  'Consume/rename the fixed dump paths after every reply and require absence before next request. Never reuse a prior first-head/state file as repeated-request evidence.',
                  'conversation_snapshot::layout rounds upto to whole four-cell pages. Its include_index argument controls pooled/indexer state, not whole-capacity capture. Pooled includes its moving spare row and must match. For32767 cells the final page has one future cell per KV head; only those data/scales bytes are excluded from live comparison. Raw hashes and every excluded range are retained. GDN/PLE/tails/dead/block positions and all pooled rows remain exact. At262144 cells there is no rounded future-cell exclusion.',
                  'CPU34 controls check both head tail cells, changed preceding live cells and previous-page cells, pooled spare, wrong metadata, and full occupancy. V1 incorrect page-count units are caught before GPU use; only corrected v2 is imported.',
                  'All previous ownership/fault/embedding/runtime/source/binary guards remain. The sole newly audited terminal rejection has exact hash3de5e40..., normal exit/no XE/no survivors, followed by pinned same-boot exact-word health.',
                  'Keep accepted8192 geometry and complete main/MTP payload verification. Queue drains and verifier graph destruction precede main unmap; subsequent requests now exercise already-used graph retirement. Restore frees temporary layer/residual VRAM first, remaps exact addresses/payloads and captures legal verifier windows before decode.',
                  'First same-process configuration uses flushed UR/LevelZero API/validation logs and STRATA_TRACE phase memory. It has bounded1800s process,600s per-reply and20GiB log deadlines. No diagnostic duration enters a speed comparison; terminate normally on a mathematical rejection before sending another request.']}
(out/'record.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'controller':str(target),'source_review':str(out/'record.json')},indent=2))
