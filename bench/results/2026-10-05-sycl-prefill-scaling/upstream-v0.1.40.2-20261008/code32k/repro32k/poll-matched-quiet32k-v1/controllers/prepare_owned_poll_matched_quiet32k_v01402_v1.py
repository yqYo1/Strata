"""Prepare matched quiet DD5/backoff jobs after both full32K numerical gates."""
from pathlib import Path
import ast,hashlib

base=Path(__file__).parent
parent=base/'run_owned_poll_backoff_code32k_v01402_v1.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='90291c5b12e48951597664fed7a1a9e4172cbd5ecf96b8cd3a7a0c1d225e71a0'
text=parent.read_text()
def change(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
change("assert mode=='poll-backoff32k' and phase=='diagnostic' and repetition==1",
       "assert mode in ['dd5-control','dd5-poll-backoff'] and phase=='quiet' and repetition in [1,2]")
change("out = base/f'owned-poll-backoff-v01402-code32k-{phase}-r{repetition}'",
       "out = base/f'owned-poll-matched-{mode}-v01402-code32k-{phase}-r{repetition}'")
change("'deadline_seconds':3600,'protocol_timeout_seconds':900,'log_limit_bytes':64*1024**3",
       "'deadline_seconds':1200,'protocol_timeout_seconds':300,'log_limit_bytes':128*1024**2")
needle="    record['compatibility_change']='DD5 plus only the CPU-tested prefill polling cadence:32 failed checks yield, then request10us host sleep; completion query/ownership/generation/cancellation/deadline unchanged. No GPU profiler or claimed stall fix.'"
change(needle,'''    numerical_path=base/'owned-poll-backoff-v01402-code32k-diagnostic-r1/record.json'
    assert hashlib.sha256(numerical_path.read_bytes()).hexdigest()=='fa8ae00e741469d6bef575affa2e263bcac4001d31c98422ba225dbc350f0a68'
    numerical=json.loads(numerical_path.read_text())
    assert numerical['healthy'] and numerical['completed'] and numerical['math_gate_passed'] and not numerical['active']
    assert numerical['exit_code']==0 and not numerical['exit_signal'] and not numerical['new_fault_messages']
    assert not any(numerical['cleanup'].values())
    assert numerical['binary_sha256']==poll_build['candidate_binary_sha256']
    for key in ['inferior','debugger']:
        old=numerical[key];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
    record['backoff_full32k_numerical_gate_sha256']=hashlib.sha256(numerical_path.read_bytes()).hexdigest()
    assert numerical['argv'][0]==str(binary)
    if mode=='dd5-control':binary=Path(uniform['candidate_binary'])
    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert record['binary_sha256']==(uniform['candidate_binary_sha256'] if mode=='dd5-control' else poll_build['candidate_binary_sha256'])
    record['single_variable']='Only prefill.cpp.o polling cadence differs between DD5 control and44a; all other link inputs, model settings and request order remain identical.'
    for p in sorted(base.glob('owned-poll-matched-*-v01402-code32k-quiet-r*/record.json')):
        if p.parent==out:continue
        old=json.loads(p.read_text());assert not old['active'] and old['healthy'] and old['math_gate_passed']
        assert old['exit_code']==0 and not old['new_fault_messages'] and not any(old['cleanup'].values())
        for key in ['inferior','debugger']:
            prior=old[key];now=process_identity(prior['pid'])
            assert not now or now['start_ticks']!=prior['start_ticks']
''')
change("    env['STRATA_TRACE']='1'", "    assert 'STRATA_TRACE' not in env")
change("    env['STRATA_DUMP_FIRST_LOGITS']=str(out/'first-head.bin')\n    env['STRATA_PREFILL_DUMP_STATE']=str(out/'prefill-state.bin')",'''    assert not any(k in env for k in ['STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE',
        'STRATA_PREFILL_CACHE_VERIFY','STRATA_PREFILL_DRAFT_VERIFY','STRATA_PREFILL_TIMING',
        'STRATA_PROFILE','STRATA_VERIFY_PROFILE'])''')
change("    record['scope']='First logged polling-backoff numerical gate on DD5: four fresh32768-token reads alternating two inputs; all head/used-state/output/logprob/MTP comparisons against completed DD5 and repeated reads; actual32K SAVE/RESTORE continuation. Diagnostic durations excluded; no full-capacity, speed or stall-fix claim.'",
       "    record['scope']='Matched quiet DD5 versus polling backoff, four complete32768-token reads A/B/A/B per process. Same262144 capacity/int8/resident32768/chunk8192/cache128/workers5/pcie0/MTP4/GEN64. First process read is reported separately from later reads. No API/validation logs, trace/profiler, state/head/payload dump, additional phase waits or prefix reuse. All64IDs/logprobs/MTP counts must equal completed DD5 numerical gate. No physical256K or adoption proof.'")
change("    record['configuration_note']='PC1/ckpt1 for real SAVE/RESTORE; turn-token=-1 and root=0 preserve the accepted chunk geometry. A single periodic checkpoint at262139 is taken after the existing final full-input chunk; no checkpoint is taken in32K controls; fresh full reads must report RESUME0 after an interposed different32K prompt. Context262144/kv-resident32768 (main mode1, MTP ring), own stage, chunk-major0, no cache release or prefetch; identical math gates on32768/GEN64. State and head capture are after the prompt computation and are not performance evidence. 32768 prompt tokens from the same expanded real-code fixture with a completed review question and assistant prefix, up to64 output tokens, context33024, chunk8192, int8 KV, normal MTP4, five CPU workers, cache128 requested, pcie0. FIRST0 and RING8 common to both. No prompt/decode retirement tuning for pure/patched-default. Diagnostic logs do not count as performance results. Prompt caching disabled and ckpt0 for a complete fresh read; actual cache count/workspaces/chunks/freeVRAM are captured from engine logs; upstream mixed packing differs from explicit-count fork contract.'",
       "    record['configuration_note']=record['scope']+' PC1/ckpt1, turn-token=-1, root0 and every262139 preserve exactly the qualified geometry; no periodic checkpoint or SAVE/RESTORE is executed in these32K timing jobs. Alternating inputs clear live prefix reuse. First means first full input in this process; filesystem/driver caches are not globally cleared.'")
change("    record['previous_goal_turn']='progress: first repaired full image passed; repeated full aborted in host watchdog at layer33/chunk196608. Post-exit device health passes; cause of pending counter events unresolved. Test separate standard-definition repair before profiling.'",
       "    record['previous_goal_turn']='Progress: both DD5 and44a completed four fresh32K state/head/output/logprob/MTP and actual disk-continuation gates. Unitrace GPU timestamps rejected after watchdog abort; post-exit device health and later44a math gate passed. All evidence archived and pushed in523ecdd0; candidate remains private.'")
needle="    record['actual_target_environment']={k:v for k,v in actual.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}"
change(needle,needle+'''
    assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_','UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS','STRATA_TRACE','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_PREFILL_SYNC','STRATA_PREFILL_TRANSFER_TIMING'] for k in actual)
    assert record['argv'][1:]==completed['argv'][1:]==numerical['argv'][1:]
''')
start=text.index('    from compare_live_prefill_state_v01402_v2 import compare_states\n\n    stderr=')
end=text.index("    current=None;send(b'QUIT\\n');event('quit')",start)
bench='''    assert int(record['startup'][-1].split()[1])==262144
    completed_by_name={v['name']:v for v in completed['requests']}
    alternate=full_prompt(32768)
    plan=[('control32k-first',control,'control32k-before'),
          ('alternate32k-first',alternate,'alternate32k-first'),
          ('control32k-repeat',control,'control32k-repeat'),
          ('alternate32k-repeat',alternate,'alternate32k-repeat')]
    for index,(name,tokens,reference_name) in enumerate(plan):
        current=name
        assert len(tokens)==32768
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        expected=completed_by_name[reference_name]
        command='GEN 64 ckpt=1 logprobs=5 '+','.join(map(str,tokens))+'\\n'
        prompt_path=out/(name+'.input-tokens.txt')
        prompt_path.write_text(' '.join(map(str,tokens))+'\\n')
        result={'name':name,'process_read':index+1,'first_process_read':index==0,
                'input_tokens':len(tokens),'request_sha256':hashlib.sha256(command.encode()).hexdigest(),
                'prompt_file':str(prompt_path),'prompt_sha256':hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                'ids':[],'logprobs':[],'protocol':[]}
        record['requests'].append(result);save()
        started_request=time.monotonic();result['start_epoch_us']=time.time_ns()/1000
        send(command.encode());event('request',name=name)
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith('ERR'):raise RuntimeError(value)
            if value.startswith('T '):result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):
                assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in value.split()[1:])
                result['logprobs'].append(value)
            if value.startswith('DONE '):break
        result['end_epoch_us']=time.time_ns()/1000
        fields=value.split()
        assert int(fields[1])==64 and int(fields[2])==32768 and fields[5]=='length'
        assert len(result['ids'])==len(result['logprobs'])==64
        result['mtp_counts']=list(map(int,fields[6:8]))
        reused=[int(v.split()[1]) for v in result['protocol'] if v.startswith(('RESUME ','REUSED '))]
        result['resume_tokens']=reused
        result['comparison']={'ids_equal':result['ids']==expected['ids'],
                              'logprobs_equal':result['logprobs']==expected['logprobs'],
                              'mtp_counts_equal':result['mtp_counts']==expected['mtp_counts'],
                              'complete_fresh_prefill':bool(reused) and all(n==0 for n in reused)}
        result['math_gate_passed']=all(result['comparison'].values())
        mm={'generated_tokens':64,'prompt_tokens':32768,'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),
            'wall_seconds':time.monotonic()-started_request,'purpose':'quiet matched full32K comparison'}
        assert mm['prompt_ms']>0 and mm['decode_ms']>0
        mm['prefill_tok_s']=32768000/mm['prompt_ms'];mm['decode_tok_s']=64000/mm['decode_ms']
        result['measurement']=mm;save()
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        if not result['math_gate_passed']:break
    record['quiet_four_fresh_reads_completed']=len(record['requests'])==4 and all(v['math_gate_passed'] for v in record['requests'])
    record['full_capacity_sequence_completed']=False

'''
text=text[:start]+bench+text[end:]
change("    record['math_gate_passed']=math_ok and bool(record.get('code32k_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])",
       "    record['math_gate_passed']=bool(record['quiet_four_fresh_reads_completed'])\n    assert record['math_gate_passed'], 'quiet output/MTP/freshness gate rejected; no performance acceptance'")
ast.parse(text)
target=base/'run_owned_poll_matched_quiet32k_v01402_v1.py';assert not target.exists();target.write_text(text)
print(target,hashlib.sha256(target.read_bytes()).hexdigest())
