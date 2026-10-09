"""Prepare a phase-specific decode experiment from the qualified owned controller.

CPU only. The model/runtime ownership and bounded failure handling are retained.
The measured request always starts from the same qualified 32K checkpoint.
"""
from pathlib import Path
import ast
import hashlib
import json

b = Path(__file__).parent
parent = b / 'run_owned_native_expert_copy_v0141_quiet32k_v1.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == '0904960ef87f84eee795fb5b5105fd20e5468bcb8bc3c2c9c432627dc95700c8'
source = parent.read_text()

def replace(old, new):
    global source
    assert source.count(old) == 1, (old[:100], source.count(old))
    source = source.replace(old, new)

replace("assert mode in ['baseline','nativeoff','nativeon'] and phase == 'clean' and repetition in [1,2]",
        "assert mode in ['baseline','nativeoff','nativeon'] and phase in ['clean','diagnostic'] and 1 <= repetition <= 6")
replace("review_head='fe96101e2c3c112d1b7a36caaff66ceda80334ff' if mode=='baseline' else candidate_head",
        "review_head='0ed619e5239bebc87f018cd2f9b8443e37105c72' if mode=='baseline' else candidate_head")
replace("out=base/f'owned-native-expert-copy-v0141-code32k-{mode}-clean-r{repetition}'",
        "out=base/f'owned-native-copy-decode-repeat-v0141-{mode}-{phase}-r{repetition}'")

# Bind the existing full-capacity checkpoint and continuation, before any GPU use.
anchor = "assert shutil.disk_usage(base).free>96*1024**3"
admission = '''
full_path=base/'owned-native-expert-copy-v0141-full256k-diagnostic-r1/record.json'
assert digest(full_path)=='7f206dc65c01d05192d54211e6ba1f52eb34dd029f25e9a7c2235b1a2567d533'
native_full=json.loads(full_path.read_text());terminal_owned(native_full)
assert native_full['full_lifecycle_passed'] and native_full['physical256k_sequence_completed']
assert native_full['binary_sha256']=='2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f'
sync_path=base/'owned-native-stream-order-v0141-code32k-ring8-diagnostic-r1/record.json'
assert digest(sync_path)=='ddaf94e2fdce81adc83d0955290d9cd4a3fd774f5bda9be93f7a2ab79badcc9c'
terminal_owned(json.loads(sync_path.read_text()))
saved=next(x for x in native_full['sessions'] if x['name']=='save-control32k')
checkpoint=Path(saved['file'])
assert checkpoint.stat().st_size==saved['bytes']==619343700
assert digest(checkpoint)==saved['image']['sha256']=='96a96b4946c495977198c0b915be243960f74fa2f5f2c3dfd5079eb25f36bc5e'
assert saved['tokens']==32831
resume_reference=next(x for x in native_full['requests'] if x['name']=='resume32k-restored')
resume_reference=dict(resume_reference)
resume_reference['finish_reason']=resume_reference['protocol'][-1].split()[5]
assert resume_reference['mtp_counts']==[46,51] and len(resume_reference['ids'])==len(resume_reference['logprobs'])==64
assert resume_reference['resume_tokens']==[32831,32831]
assert same_output(resume_reference,next(dict(x,finish_reason=x['protocol'][-1].split()[5]) for x in new_full['requests'] if x['name']=='resume32k-restored'))==dict.fromkeys(['ids_equal','logprobs_equal','mtp_counts_equal','finish_reason_equal'],True)
for prior in base.glob('owned-native-copy-decode-repeat-v0141-*/record.json'):
    terminal_owned(json.loads(prior.read_text()))
assert shutil.disk_usage(base).free>64*1024**3
'''
replace(anchor, admission.strip())
replace("'deadline_seconds':1500,'protocol_timeout_seconds':300,'log_limit_bytes':128*1024**2,",
        "'deadline_seconds':1500,'protocol_timeout_seconds':450,'log_limit_bytes':(16*1024**3 if phase=='diagnostic' else 128*1024**2),")
replace("    env.pop('STRATA_PREFILL_COPY_ENGINE',None)",
        "    env.pop('STRATA_PREFILL_COPY_ENGINE',None)\n    env['STRATA_DECODE_TIMING']='1'\n    assert 'STRATA_VERIFY_PROFILE' not in env")
replace("    if phase == 'diagnostic':\n        env = m.diagnostic_environment(env)",
        "    if phase == 'diagnostic':\n        env['STRATA_TRACE']='1'\n        env = m.diagnostic_environment(env)")
replace("    seen = {}\n    math_ok = True\n    for read_index, key in enumerate(['A', 'B', 'A', 'B']):", '''    assert record['startup'][-1].split()[1]=='262144'
    record['scope']='Phase-specific decode comparison: qualified baseline869, candidate272 flag0, candidate272 flag1. Prime each process with the same fresh32768 A/64-output prefill, retaining its workspace and queue. Before every decode trial RESTORE the same verified32831-token checkpoint, then generate64 from32832 input tokens. One warm trial excluded; twelve measured repeats per quiet process. Six order-balanced independent process blocks. Exact IDs/logprobs/MTP counts required; restoration and prefill timings excluded from decode statistics.'
    record['checkpoint_identity']={'file':str(checkpoint),'sha256':saved['image']['sha256'],'bytes':saved['bytes'],'tokens':saved['tokens'],'source_receipt_sha256':digest(full_path)}
    record['sessions']=[]
    record['configuration_note']='Context262144, resident32768, chunk8192, expert-cache128, LP5, greedy MTP4. All modes share argv/environment except executable and explicit prefill-only queue flag. Priming creates the actual prefill queue/workspace. Each decode begins from the same saved numerical state; no PP chunks or prefix growth. Warm trial reported separately. Quiet host counters add no GPU timestamps/waits. Interleaved independent processes are the units for uncertainty; repeated requests within a process are not independent process samples.'
    continuation=fixtures['A']+reference_reads[0]['ids']
    assert len(continuation)==32832
    math_ok=True

    def restore(trial):
        global current
        current=f'restore-{trial}'
        item={'name':current,'command':'RESTORE','file':str(checkpoint),'protocol':[]}
        record['sessions'].append(item);save()
        send(f'RESTORE {checkpoint}\\n'.encode())
        while True:
            value=line();item['protocol'].append(value)
            if value.startswith(('RESTORED ','SERR ')):break
        fields=value.split()
        assert fields[0]=='RESTORED' and int(fields[1])==32831 and int(fields[2])==619343700,value
        item.update(tokens=int(fields[1]),bytes=int(fields[2]),restore_ms=float(fields[3]),passed=True)
        save()

    trials=2 if phase=='diagnostic' else 13
    for read_index in range(1+trials):
        key='A'
        priming=read_index==0
        if not priming:restore(read_index-1)
        tokens=fixtures['A'] if priming else continuation''')
replace("        current = f'{key}-read{read_index}'",
        "        current = 'prefill-prime' if priming else f'decode-trial{read_index-1}'")
replace("        result = {'name': current, 'fixture': key, 'read_index': read_index, 'first_process_read': read_index == 0, 'ids': [], 'logprobs': [], 'protocol': []}",
        "        result = {'name':current,'fixture':key,'read_index':read_index,'phase_kind':'prefill-prime' if priming else 'decode-only','warmup_decode':read_index==1,'timing_eligible':phase=='clean' and read_index>=2,'first_process_read':priming,'ids':[],'logprobs':[],'protocol':[]}")
replace("send(('GEN 64 ckpt=1 logprobs=5 ' + ','.join(map(str, fixtures[key])) + '\\n').encode())",
        "send(('GEN 64 ckpt=1 logprobs=5 ' + ','.join(map(str, tokens)) + '\\n').encode())")
replace("        result['validation'] = validate_fresh(result)", '''        if priming:
            result['validation']=validate_fresh(result)
        else:
            result['validation']={
                'fixed_restored_input':int(fields[2])==32832,
                'same_resume_and_reused':result['resume_tokens']==[32831,32831],
                'no_prefill_chunks':not any(x.startswith('PP ') for x in result['protocol']),
                'exact64_visible_outputs':int(fields[1])==len(result['ids'])==len(result['logprobs'])==64,
                'normal_finish':fields[5]=='length',
                'finite_logprobs':all(math.isfinite(float(x.rsplit(':',1)[-1])) for line_value in result['logprobs'] for x in line_value.split()[1:]),
                'finite_positive_times':math.isfinite(float(fields[4])) and float(fields[4])>0,
            }''')
start = source.index("        assert [int(x.split()[1]) for x in result['protocol'] if x.startswith('PP ')]")
end = source.index("        result['math_gate_passed'] = all(checks)", start)
source = source[:start] + '''        if priming:
            assert [int(x.split()[1]) for x in result['protocol'] if x.startswith('PP ')]==[8192,16384,24576,32767]
        expected=reference_reads[0] if priming else resume_reference
        result['qualified_reference_output_comparison']=same_output(result,expected)
        checks.extend(result['qualified_reference_output_comparison'].values())
''' + source[end:]
replace("        save()\n\n    current = None", "        save()\n        if not math_ok:break\n\n    current = None")
replace("record['math_gate_passed'] = math_ok and len(record['requests']) == 4",
        "record['math_gate_passed'] = math_ok and len(record['requests'])==1+trials and len(record['sessions'])==trials\n    record['phase_specific_decode_sequence_completed']=record['math_gate_passed']")
replace("'performance_eligible':True,'full_lifecycle_passed':False,'adopted':False,",
        "'performance_eligible':phase=='clean','full_lifecycle_passed':False,'adopted':False,")

# Confirm the established ownership, protocol I/O, deadlines and cleanup remain.
original = ast.parse(parent.read_text())
generated = ast.parse(source)
unchanged = ['digest','validate_fresh','same_output','normalize_reference','terminal_owned','save','event','captures','poll','line','send']
def functions(tree):
    return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
of,gf=functions(original),functions(generated)
assert all(of[name]==gf[name] for name in unchanged)
target = b/'run_owned_native_copy_decode_repeat_v0141_v1.py'
assert not target.exists()
target.write_text(source)
receipt={'passed':True,'active':False,'gpu_executed':False,'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'controller':str(target),'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'unchanged_owned_functions':unchanged,'quiet_trials_per_process':13,'warmup_trials_excluded':1,'measured_trials_per_process':12,'planned_independent_blocks':6,'planned_modes':['baseline','nativeoff','nativeon'],'checkpoint_sha256':'96a96b4946c495977198c0b915be243960f74fa2f5f2c3dfd5079eb25f36bc5e','phase_adoption_rule':'Prefill and decode judged separately. No decode change accepted by offsetting a penalty with faster prefill. Paired independent process blocks and within-process repeats reported separately; an interval including zero does not establish equivalence.'}
(b/'native-copy-decode-repeat-v0141-controller-preparation-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
