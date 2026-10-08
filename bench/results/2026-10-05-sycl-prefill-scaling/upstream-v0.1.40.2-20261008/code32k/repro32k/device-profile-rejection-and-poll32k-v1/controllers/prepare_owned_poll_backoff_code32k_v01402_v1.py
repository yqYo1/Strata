"""Prepare the first logged32K gate for the DD5-based polling candidate."""
from pathlib import Path
import ast,hashlib,json

base=Path(__file__).parent
parent=base/'run_owned_profile_definition_code32k_v01402_v2.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='597c6022a603c1902f0d75052e183940b20177f1f3f66d5439af7c75961a826b'
text=parent.read_text()
def change(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
change("assert mode=='full-kv-access' and phase=='diagnostic' and repetition==6",
       "assert mode=='poll-backoff32k' and phase=='diagnostic' and repetition==1")
change("out = base/f'owned-profile-definition-v01402-code32k-{phase}-r{repetition}'",
       "out = base/f'owned-poll-backoff-v01402-code32k-{phase}-r{repetition}'")
change("'deadline_seconds':10800,'protocol_timeout_seconds':5400,'log_limit_bytes':128*1024**3",
       "'deadline_seconds':3600,'protocol_timeout_seconds':900,'log_limit_bytes':64*1024**3")
health=base/'post-device-profile-abort-v01402-health-v1/record.json'
hr=json.loads(health.read_text());assert hr['healthy'] and not hr['active']
change("base/'post-full256k-abort-v01402-health-v1/record.json'",repr(str(health)))
# A string would lose Path.read_bytes(), so preserve the original Path form.
text=text.replace('health_path='+repr(str(health)),"health_path=Path("+repr(str(health))+")")
change("'760491d1de59dc7d18936abc0afc4609b7332b12bdb93aec14d51638ddfa2c60'",
       repr(hashlib.sha256(health.read_bytes()).hexdigest()))
needle="    record['compatibility_change']='Only seven originally unprofiled TUs recompiled with the uniform DPCT header; translated profiling branches and explicit copy-queue factory unchanged. No polling change or claimed stall fix.'"
change(needle,'''    completed_path=base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json'
    assert hashlib.sha256(completed_path.read_bytes()).hexdigest()=='a96f3686903b63138a827c173e0ea5b1a8beebbee8ef9cdae46459bf9f2c9699'
    completed=json.loads(completed_path.read_text())
    assert completed['healthy'] and completed['completed'] and completed['math_gate_passed'] and not completed['active']
    failed_path=base/'owned-device-profile-v01402-code32k-diagnostic-r1/record.json'
    failed=json.loads(failed_path.read_text())
    assert not failed['active'] and not failed['healthy'] and not failed['completed'] and not failed['new_fault_messages']
    assert health['terminal_controller_receipt_sha256']==hashlib.sha256(failed_path.read_bytes()).hexdigest()
    assert health['started_utc']>failed['finished_utc']
    for old_run in [completed,failed]:
        assert old_run['boot_id']==record['boot_id']
        assert not any(old_run['cleanup'][k] for k in ['inferior_survived','gdb_survived'])
        for key in ['inferior','debugger']:
            old=old_run[key];now=process_identity(old['pid'])
            assert not now or now['start_ticks']!=old['start_ticks']
    poll_path=base/'prefill-poll-backoff-v01402-relink-v3/record.json'
    poll_build=json.loads(poll_path.read_text())
    assert poll_build['passed'] and not poll_build['active'] and not poll_build['gpu_tested']
    assert poll_build['baseline_inputs_unchanged'] and poll_build['only_prefill_link_input_replaced']
    assert poll_build['baseline_binary_sha256']==uniform['candidate_binary_sha256']
    assert poll_build['poll_template_definition_isolated_to_prefill_tu'] and poll_build['prefill_keeps_original_profiled_dpct_definition']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in poll_build['baseline_link_input_sha256'].items())
    for pkey,hkey in [('actual_prefill_object','actual_prefill_object_sha256'),('actual_shadow_header','actual_shadow_header_sha256'),('actual_dependency_file','actual_dependency_file_sha256'),('prefill_source','prefill_source_sha256')]:
        assert hashlib.sha256(Path(poll_build[pkey]).read_bytes()).hexdigest()==poll_build[hkey]
    binary=Path(poll_build['candidate_binary'])
    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert record['binary_sha256']==poll_build['candidate_binary_sha256']=='44a274d8c699829020abb8a85a331a638f40efdf5d6c959a076a58e818f75391'
    archive=binary.parent/'libstrata_prefill.a'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==poll_build['candidate_archive_sha256']
    assert subprocess.check_output(['/usr/bin/ar','t',str(archive)],text=True).splitlines()==[m['member'] for m in poll_build['prefill_archive_members']]
    for member in poll_build['prefill_archive_members']:
        assert hashlib.sha256(subprocess.check_output(['/usr/bin/ar','p',str(archive),member['member']])).hexdigest()==member['after_sha256']
    record['polling_build_receipt_sha256']=hashlib.sha256(poll_path.read_bytes()).hexdigest()
    record['unprofiled_same_base_numerical_gate_sha256']=hashlib.sha256(completed_path.read_bytes()).hexdigest()
    record['profiler_failure_receipt_sha256']=hashlib.sha256(failed_path.read_bytes()).hexdigest()
    record['compatibility_change']='DD5 plus only the CPU-tested prefill polling cadence:32 failed checks yield, then request10us host sleep; completion query/ownership/generation/cancellation/deadline unchanged. No GPU profiler or claimed stall fix.'
''')
change("    argv=[str(binary)]+args;record['argv']=argv",'''    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for k in env)
    argv=[str(binary)]+args;record['argv']=argv''')
change("    record['scope']='Logged uniform-DPCT-header numerical gate: four fresh32768-token reads alternating two inputs, all head/used-state/output/logprob/MTP comparisons, actual32K SAVE/RESTORE continuation. Diagnostic durations excluded; no full-capacity, speed or stall-fix claim.'",
       "    record['scope']='First logged polling-backoff numerical gate on DD5: four fresh32768-token reads alternating two inputs; all head/used-state/output/logprob/MTP comparisons against completed DD5 and repeated reads; actual32K SAVE/RESTORE continuation. Diagnostic durations excluded; no full-capacity, speed or stall-fix claim.'")
change("    os.close(slave);slave=None;save()",'''    os.close(slave);slave=None
    now=process_identity(g.inferior['pid']);assert now and now['start_ticks']==g.inferior['start_ticks']
    assert Path(os.readlink('/proc/'+str(now['pid'])+'/exe')).resolve()==binary.resolve()
    assert hashlib.sha256(Path('/proc',str(now['pid']),'exe').read_bytes()).hexdigest()==record['binary_sha256']
    actual=dict(item.decode().split('=',1) for item in Path('/proc',str(now['pid']),'environ').read_bytes().split(b'\\0') if b'=' in item)
    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment']={k:v for k,v in actual.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    save()''')
change("        control_first=request('control32k-before',control,64,baseline,fresh=True)",
       "        completed_by_name={v['name']:v for v in completed['requests']}\n        control_first=request('control32k-before',control,64,completed_by_name['control32k-before'],fresh=True)")
change("        resume_reference=request('resume32k-reference',continuation,64)",
       "        resume_reference=request('resume32k-reference',continuation,64,completed_by_name['resume32k-reference'])")
change("        alternate_first=request('alternate32k-first',alternate,64,fresh=True)",
       "        alternate_first=request('alternate32k-first',alternate,64,completed_by_name['alternate32k-first'],fresh=True)")
ast.parse(text)
target=base/'run_owned_poll_backoff_code32k_v01402_v1.py';assert not target.exists();target.write_text(text)
assert "health_path=Path(" in text
print(target,hashlib.sha256(target.read_bytes()).hexdigest())
