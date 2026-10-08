"""Prepare an immutable, owned 32K GPU timestamp gate from the completed DD5 gate."""
from pathlib import Path
import ast
import hashlib

base = Path(__file__).parent
parent = base / 'run_owned_profile_definition_code32k_v01402_v2.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == '597c6022a603c1902f0d75052e183940b20177f1f3f66d5439af7c75961a826b'
text = parent.read_text()

def change(old, new):
    global text
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)

change("assert mode=='full-kv-access' and phase=='diagnostic' and repetition==6",
       "assert mode=='device-profile32k' and phase=='diagnostic' and repetition==1")
change("out = base/f'owned-profile-definition-v01402-code32k-{phase}-r{repetition}'",
       "out = base/f'owned-device-profile-v01402-code32k-{phase}-r{repetition}'")
change("'deadline_seconds':10800,'protocol_timeout_seconds':5400,'log_limit_bytes':128*1024**3",
       "'deadline_seconds':3600,'protocol_timeout_seconds':900,'log_limit_bytes':64*1024**3")
needle = "    binary=Path(uniform['candidate_binary'])"
change(needle, needle + "\n    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()")
needle = "    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')"
change(needle, '''    profiler_gate_path=base/'unitrace-device-options-v01402-v1/record.json'
    profiler_gate=json.loads(profiler_gate_path.read_text())
    assert profiler_gate['passed'] and not profiler_gate['active']
    profiler_source=base/'pti-gpu-profiler-source'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=profiler_source,text=True).strip()==profiler_gate['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=profiler_source,text=True)
    assert all(hashlib.sha256((profiler_source/path).read_bytes()).hexdigest()==sha for path,sha in profiler_gate['source_sha256'].items())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in profiler_gate['profiler_sha256'].items())
    record['profiler_options_review_sha256']=hashlib.sha256(profiler_gate_path.read_bytes()).hexdigest()
    record['profiler_sha256']=profiler_gate['profiler_sha256']
    completed_path=base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json'
    assert hashlib.sha256(completed_path.read_bytes()).hexdigest()=='a96f3686903b63138a827c173e0ea5b1a8beebbee8ef9cdae46459bf9f2c9699'
    completed=json.loads(completed_path.read_text())
    assert completed['healthy'] and completed['completed'] and completed['math_gate_passed'] and not completed['active']
    assert completed['boot_id']==record['boot_id'] and completed['exit_code']==0 and not completed['new_fault_messages']
    assert not any(completed['cleanup'][k] for k in ['forced','inferior_survived','gdb_survived'])
    for key in ['inferior','debugger']:
        old=completed[key];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
    record['unprofiled_same_binary_numerical_gate_sha256']=hashlib.sha256(completed_path.read_bytes()).hexdigest()
    assert shutil.disk_usage(base).free>160*1024**3
''' + needle)
change("    argv=[str(binary)]+args;record['argv']=argv", '''    model_argv=[str(binary)]+args
    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for k in env)
    argv=[str(base/'unitrace-build-system-cc/build/unitrace'),'-d','--chrome-device-logging',
          '--output-dir-path',str(out),'-o',str(out/'device-summary.txt')]+model_argv
    record['model_argv']=model_argv
    record['argv']=argv
    record['profile_changes_target_gpu_event_properties']=True
    record['profiling_collection']='Always enabled from process initialization through QUIT; no pause/resume transitions, metrics, KMD, sampling or added application waits. GPU timestamp events alter append/event handling; numerical gate required. Recorded durations are not clean throughput.' ''')
change("    record['scope']='Logged uniform-DPCT-header numerical gate: four fresh32768-token reads alternating two inputs, all head/used-state/output/logprob/MTP comparisons, actual32K SAVE/RESTORE continuation. Diagnostic durations excluded; no full-capacity, speed or stall-fix claim.'",
       "    record['scope']='First logged unitrace GPU timestamp gate on DD5: four fresh32768-token reads alternating two inputs, full head/used-state/output/logprob/MTP comparisons against the completed unprofiled DD5 run, actual32K SAVE/RESTORE continuation. Diagnostic/profiled durations excluded; no full-capacity, speed or stall-fix claim.'")
change("    os.close(slave);slave=None;save()", '''    os.close(slave);slave=None
    now=process_identity(g.inferior['pid'])
    assert now and now['start_ticks']==g.inferior['start_ticks']
    assert Path(os.readlink('/proc/'+str(now['pid'])+'/exe')).resolve()==binary.resolve()
    assert hashlib.sha256(Path('/proc',str(now['pid']),'exe').read_bytes()).hexdigest()==record['binary_sha256']
    actual=dict(item.decode().split('=',1) for item in Path('/proc',str(now['pid']),'environ').read_bytes().split(b'\\0') if b'=' in item)
    record['actual_target_environment']={k:v for k,v in actual.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZET_','ZEL_','UNITRACE_','XPTI_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','LD_PRELOAD','NEOReadDebugKeys','EnableDirectSubmission']}
    assert actual.get('UNITRACE_DeviceTiming')==actual.get('UNITRACE_ChromeDeviceLogging')=='1'
    assert actual.get('ZE_ENABLE_TRACING_LAYER')=='1'
    assert str(base/'unitrace-build-system-cc/build/libunitrace_tool.so') in actual.get('LD_PRELOAD','')
    assert not any(actual.get(k)=='1' for k in ['UNITRACE_KernelMetrics','UNITRACE_MetricQuery','UNITRACE_StallSampling','UNITRACE_ChromeKmdLogging','UNITRACE_StartPaused','UNITRACE_ChromeCallLogging','UNITRACE_HostTiming','ZET_ENABLE_METRICS'])
    save()''')
change("        record['requests'].append(result);save();send(command.encode());event('request',name=name)",
       "        result['start_epoch_us']=time.time_ns()/1000\n        record['requests'].append(result);save();send(command.encode());event('request',name=name,epoch_us=result['start_epoch_us'])")
change("        result['diagnostic_wall_seconds']=time.monotonic()-request_started",
       "        result['end_epoch_us']=time.time_ns()/1000\n        result['diagnostic_wall_seconds']=time.monotonic()-request_started")
change("        control_first=request('control32k-before',control,64,baseline,fresh=True)",
       "        completed_by_name={v['name']:v for v in completed['requests']}\n        control_first=request('control32k-before',control,64,completed_by_name['control32k-before'],fresh=True)")
change("        alternate_first=request('alternate32k-first',alternate,64,fresh=True)",
       "        alternate_first=request('alternate32k-first',alternate,64,completed_by_name['alternate32k-first'],fresh=True)")
ast.parse(text)
target = base / 'run_owned_device_profile32k_v01402_v1.py'
assert not target.exists()
target.write_text(text)
print(target, hashlib.sha256(target.read_bytes()).hexdigest())
