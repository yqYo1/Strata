"""Prepare logged host-only profiling, a quiet profile and a matched32K control."""
from pathlib import Path

base=Path(__file__).parent
old_mode='event-ack-no-root-prefill-default'
new_mode='event-ack-no-root-host-profile'
target=base/'run_owned_host_profile32k_v01402.py';assert not target.exists()
text=(base/'run_owned_event_ack_no_root_default_v01402_code32k.py').read_text()
text=text.replace("'"+old_mode+"'","'"+new_mode+"'").replace(old_mode+'-v01402',new_mode+'-v01402')
def change(old,new):
 global text
 assert text.count(old)==1,(old,text.count(old))
 text=text.replace(old,new)
change("phase in ['diagnostic','state'] and repetition in [1,2,3]", "phase in ['diagnostic','state','control'] and repetition==1")
change("list(base.glob('owned-event-ack-no-root-host-profile-v01402-code32k-*/record.json'))", "list(base.glob('owned-event-ack-no-root-prefill-default-v01402-code32k-*/record.json'))+list(base.glob('owned-event-ack-no-root-host-profile-v01402-code32k-*/record.json'))")
needle="clean_gate=json.loads((base/'registered-vs-no-root-v01402-clean-sequence/record.json').read_text())\nassert clean_gate['passed'] and not clean_gate['active']"
change(needle,needle+"\ndefault_gate=json.loads((base/'event-ack-no-root-default-v01402-gated-checks/record.json').read_text())\nassert default_gate['passed'] and not default_gate['active']")
needle="    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')"
change(needle,"""    audit_path=base/'unitrace-host-options-v01402/record.json'
    audit=json.loads(audit_path.read_text());assert audit['passed'] and not audit['active']
    profiler_source=base/'pti-gpu-profiler-source'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=profiler_source,text=True).strip()==audit['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=profiler_source,text=True)
    assert all(hashlib.sha256((profiler_source/path).read_bytes()).hexdigest()==value for path,value in audit['source_sha256'].items())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in audit['profiler_sha256'].items())
    record['host_options_audit_sha256']=hashlib.sha256(audit_path.read_bytes()).hexdigest()
    record['profiler_sha256']=audit['profiler_sha256']
"""+needle)
change("argv=[str(binary)]+args;record['argv']=argv", """argv=[str(binary)]+args;record['model_argv']=argv.copy()
    assert not any(key.startswith(('UNITRACE_','XPTI_')) or key in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for key in env)
    if phase!='control':
        argv=[str(base/'unitrace-build-system-cc/build/unitrace'),'-h','--chrome-call-logging','--output-dir-path',str(out),'-o',str(out/'host-summary.txt')]+argv
    record['argv']=argv""")
needle="    os.close(slave);slave=None;save()"
change(needle,"""    os.close(slave);slave=None
    now=process_identity(g.inferior['pid']);assert now and now['start_ticks']==g.inferior['start_ticks']
    assert Path(os.readlink('/proc/'+str(now['pid'])+'/exe')).resolve()==binary.resolve()
    actual=dict(item.decode().split('=',1) for item in Path('/proc',str(now['pid']),'environ').read_bytes().split(b'\\0') if b'=' in item)
    selected={key:value for key,value in actual.items() if key.startswith(('STRATA_','SYCL_','UR_','ZE_','ZET_','ZEL_','UNITRACE_','XPTI_','ONEAPI_')) or key in ['LD_LIBRARY_PATH','LD_PRELOAD','NEOReadDebugKeys','EnableDirectSubmission','EnableImplicitConvertionToCounterBasedEvents','MKL_CBWR']}
    record['actual_target_environment']=selected
    if phase!='control':
        assert actual.get('UNITRACE_HostTiming')==actual.get('UNITRACE_ChromeCallLogging')=='1'
        assert actual.get('UNITRACE_LogToFile')==actual.get('UNITRACE_TraceOutputDirPath')=='1'
        assert str(base/'unitrace-build-system-cc/build/libunitrace_tool.so') in actual.get('LD_PRELOAD','')
        assert actual.get('ZE_ENABLE_TRACING_LAYER')=='1'
        assert not any(actual.get(key)=='1' for key in ['UNITRACE_DeviceTiming','UNITRACE_DeviceTimeline','UNITRACE_KernelSubmission','UNITRACE_ChromeKernelLogging','UNITRACE_ChromeDeviceLogging','UNITRACE_KernelMetrics','UNITRACE_MetricQuery','ZET_ENABLE_METRICS'])
    else:assert not any(key.startswith(('UNITRACE_','XPTI_')) or key=='LD_PRELOAD' for key in actual)
    record['profile_changes_target_gpu_event_properties']=False
    save()""")
change("baseline=json.loads((base/'owned-event-ack-no-root-prefill-cb-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]", "baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]")
text=text.replace('comparison_to_counter_off_control','comparison_to_default_counter_control')
change("events.write(json.dumps({'kind':kind,'elapsed_seconds':time.monotonic()-started,**fields})", "events.write(json.dumps({'kind':kind,'epoch_us':time.time_ns()/1000,'elapsed_seconds':time.monotonic()-started,**fields})")
change("        send((request+'\\n').encode());event('request',name=current)", "        request_epoch_us=time.time_ns()/1000\n        send((request+'\\n').encode());event('request',name=current)")
change("result={'name':current,'ids':[],'logprobs':[],'protocol':[]}", "result={'name':current,'start_epoch_us':request_epoch_us,'ids':[],'logprobs':[],'protocol':[],'protocol_timestamps':[]}")
change("value=line();result['protocol'].append(value)", "value=line();result['protocol'].append(value);result['protocol_timestamps'].append({'text':value,'epoch_us':time.time_ns()/1000})")
change("        record['requests'].append(result);save()", """        result['end_epoch_us']=result['protocol_timestamps'][-1]['epoch_us']
        comparison=result['comparison_to_default_counter_control']
        assert not comparison['different_prefill_state_parts'] and all(comparison[key] for key in ['first_head_equal','ids_equal','logprobs_equal']), 'profile/control differs from full32K default-counter state/head/output control'
        record['requests'].append(result);save()""")
lines=text.splitlines()
i=next(i for i,line in enumerate(lines) if line.startswith("    record['scope']="))
lines[i]="    record['scope']='32K host-API profiling of the unchanged stable no-root registered-copy candidate using unitrace host timing and Chrome call trace only. No GPU kernel tracing, timestamps, metrics, event-pool/append instrumentation or extra waits. Default counter conversion and no CNR; same model/math/configuration as three completed exact controls. First use additionally captures flushed UR/Level Zero diagnostics and parameter validation; a subsequent quiet host profile and unprofiled control both retain identical full state/head capture for parity and overhead assessment. Every job must match all66state parts,248320head floats,64IDs and every logprob. These captured/profiled durations are not clean throughput or full256K evidence; candidate remains private.'"
i=next(i for i,line in enumerate(lines) if line.startswith("    record['previous_goal_turn']="))
lines[i]="    record['previous_goal_turn']='progress:13 completed32K state/output/control jobs, four accepted private timing records, default counter gate passed, all archived and pushed in32f600e22f2039e36dd11c7b75d214aff8d0bcd1; PID and binary identities revalidated'"
text='\n'.join(lines)+'\n'
compile(text,str(target),'exec');target.write_text(text);print(target)
