"""Root-owned isolated IQ4NL host-task async contract; no production/model/device-fault qualification."""
from pathlib import Path
import csv,fcntl,hashlib,json,math,os,re,statistics,struct,sys,types
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-prefill-iq4nl-async-boundary-20261010')
OUT=B/'prefill-iq4nl-async-parity-runtime-v1'
HEAD='000f072da7d3f5b179173e81028cacb5f528e0af'
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
PARENT_HASH='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ident(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sha(PARENT)==PARENT_HASH
        owner_source=PARENT.read_text()
        module=types.ModuleType('bounded_private_iq4nl_async_owner');exec(compile(owner_source,str(PARENT)+'[timing-limits]','exec'),module.__dict__);module.W=W
        build_path=B/'prefill-iq4nl-async-parity-cpu-build-v1/record.json';build=json.loads(build_path.read_text())
        assert build['passed'] and build['complete'] and not build['active'] and build['source_head']==HEAD
        binary=Path(build['binary']['path']);assert ident(binary)=={k:build['binary'][k] for k in ['bytes','sha256']}
        assert module.subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==HEAD
        assert not module.subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
        for rel,pin in build['source_pins'].items():assert ident(W/rel)==pin
        primitive_path=B/'gdn-gate-factor-probe-runtime-v2/record.json';primitive=json.loads(primitive_path.read_text())
        assert primitive['passed'] and primitive['complete'] and not primitive['active']
        pins=primitive['runtime_pins']
        for path,pin in pins.items():assert sha(path)==pin['sha256'] and str(Path(path).resolve())==pin['realpath']
        assert not OUT.exists();OUT.mkdir(mode=0o700);owner=module.Owner(OUT)
        helper=W/'sycl/tools/recover-xe.sh';assert sha(helper)==module.RECOVER_HASH
        helper_python=helper.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        environment_module=types.ModuleType('pure_diagnostic_environment');exec(compile(helper_python,str(helper),'exec'),environment_module.__dict__)
        clean=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        env=environment_module.diagnostic_environment(dict(clean,LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:'+build['environment']['LD_LIBRARY_PATH']+':/usr/lib/x86_64-linux-gnu'))
        env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',SYCL_CACHE_PERSISTENT='0',EnableDirectSubmission='0')
        record=dict(active=True,complete=False,passed=False,started_utc=module.utc(),controller_sha256=sha(__file__),owner_sha256=PARENT_HASH,
            source_head=HEAD,build_receipt=ident(build_path),binary=build['binary'],environment=env,runtime_pins=pins,commands=owner.commands,
            GPU_runtime_launched=False,model_opened=False,adopted=False,performance_eligible=False,full_physical_KV_lifecycle=False,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope='Isolated completed host-origin async error on production-handler queue; production boundary unintegrated, device fault and recovery unqualified')
        def save():
            temp=OUT/'record.json.tmp'
            with temp.open('w') as stream:json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            temp.replace(OUT/'record.json')
        owner.persist=save
        def run(label,args,environment,wall=120):
            entry,out,err=owner.run(label,args,environment,wall=wall,text_cap=32<<20,file_cap=16<<20);save()
            assert module.completed(entry) and entry['exit_code']==0,(label,entry)
            return entry,out,err
        dump=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
        record['limits']=dict(AS_each_bytes=16<<30,RSS_session_sampled_bytes=2<<30,CPU_each_seconds=[120,121],GPU_wall_seconds=240,text_cap_bytes=32<<20,file_cap_each_bytes=16<<20)
        try:
            assert not dump.exists(),'existing GPU fault';record['devcoredump_before']=False
            _,out,_=run('journal-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'],clean,wall=5)
            match=re.search(r'^-- cursor: (.+)$',out.read_text(),re.M);assert match;cursor=match[1];record['kernel_cursor']=cursor
            _,dependencies,dependency_errors=run('dynamic-dependencies',['/usr/bin/ldd',str(binary)],env,wall=20)
            assert dependency_errors.stat().st_size==0 and 'not found' not in dependencies.read_text()
            dependency_pins={}
            for line in dependencies.read_text().splitlines():
                match=re.search(r'(?:=>\s*)?(/\S+)\s+\(',line)
                if match:
                    path=Path(match[1]);dependency_pins[str(path)]=dict(realpath=str(path.resolve()),**ident(path))
            assert dependency_pins;record['resolved_dependency_pins']=dependency_pins;save()
            entry,refuse_out,refuse_err=owner.run('prequeue-CLI-admission',[str(binary)],env,wall=10,text_cap=32<<20,file_cap=16<<20);save()
            assert module.completed(entry) and entry['exit_code']==2 and refuse_out.stat().st_size==0 and refuse_err.read_text()=='usage: prefill_iq4nl_async_parity --run\n'
            record['host_CLI_rejection_passed']=True;record['GPU_runtime_launched']=True;save()
            _,out,err=run('iq4nl-async-probe',[str(binary),'--run'],env,wall=240)
            rows=list(csv.reader(out.read_text().splitlines()));assert all(rows)
            stages=['IDENTITY_8086_e20c_0000:05:00.0_root','upload_input','upload_output','private_submit','existing_completion_wait','later_drain','readback','input_readback']
            expected=[]
            for fault in [False,True]:
                begin=stages[:4]+(['host_task_submit'] if fault else [])+stages[4:]
                expected+=[['STAGE',s] for s in begin]
                expected+=[['CASE','fault' if fault else 'control','completed_wait','1','boundary_ok','0' if fault else '1','callback','0' if fault else '1','generic_retry','0','private_calls','1','host_task','1' if fault else '0','later_drain','normal']]
                expected+=[['STAGE','free_buffers'],['STAGE','queue_teardown']]
            expected+=[['TERMINAL','pass','host_origin_async_boundary_only','2_cases','production_unintegrated','device_fault_unqualified','adoption_false']]
            assert rows==expected,'missing/unexpected/unordered output'
            assert sum(r[0]=='STAGE' for r in rows)==21
            trace=err.read_text();assert trace and 'urEnqueueKernelLaunch' in trace and 'zeCommandListAppendLaunchKernel' in trace
            errors=[line for line in trace.splitlines() if line.startswith('asynchronous SYCL error:')]
            assert len(errors)==1 and 'IQ4NL_HOST_TASK_ONE_SHOT' in errors[0],('handler receipts',errors)
            assert not any(line.startswith(('FAIL,','FAIL_STOP,')) for line in trace.splitlines())
            record['host_origin_async_result']=dict(cases=2,stage_records=21,control_callback_count=1,fault_callback_count=0,private_calls_each=1,generic_retry_count=0,host_task_count=1,completed_wait=True,completed_boundary_false_for_fault=True,concrete_error_receipt=errors[0],later_drain_without_redelivery=True,full_input_immutable=True,full_output_canaries=True,terminal_after_explicit_USM_and_queue_teardown=True,production_integrated=False,device_fault_qualified=False,recovery_qualified=False,model=False,performance=False,fullKV=False,adopted=False)
            _,interval,interval_err=run('kernel-interval',['/usr/bin/journalctl','-k','--after-cursor='+cursor,'--no-pager','-o','short-monotonic'],clean,wall=5)
            assert interval_err.stat().st_size==0
            faults=[line for line in interval.read_text().splitlines() if re.search(r'\bxe\b|i915|GPU HANG|devcoredump',line,re.I)]
            record.update(journal_status='complete',visible_kernel_GPU_entries=faults,devcoredump_after=dump.exists());assert not faults and not dump.exists(),'new GPU fault'
            record.update(passed=True,complete=True)
        except BaseException as error:record['error']=type(error).__name__+': '+str(error)
        finally:
            if record.get('GPU_runtime_launched') and record.get('journal_status')!='complete':
                if owner.active is None and record.get('kernel_cursor'):
                    try:
                        _,interval,interval_err=run('fault-observation-after-failure',['/usr/bin/journalctl','-k','--after-cursor='+record['kernel_cursor'],'--no-pager','-o','short-monotonic'],clean,wall=5)
                        assert interval_err.stat().st_size==0
                        record.update(journal_status='complete',visible_kernel_GPU_entries=[line for line in interval.read_text().splitlines() if re.search(r'\bxe\b|i915|GPU HANG|devcoredump',line,re.I)],devcoredump_after=dump.exists())
                    except BaseException as error:record['fault_observation_error']=repr(error)
                else:record['fault_observation_error']='unknown owned active state or kernel cursor unavailable'
            try:
                assert ident(binary)=={k:build['binary'][k] for k in ['bytes','sha256']}
                assert module.subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==HEAD
                assert not module.subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
                for rel,pin in build['source_pins'].items():assert ident(W/rel)==pin
                for path,pin in pins.items():assert sha(path)==pin['sha256']
                for path,pin in record.get('resolved_dependency_pins',{}).items():assert sha(path)==pin['sha256'] and str(Path(path).resolve())==pin['realpath']
                assert sha(PARENT)==PARENT_HASH
                assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
            except BaseException as error:record['final_pin_error']=repr(error);record['passed']=False
            record.update(active=owner.active is not None,finished_utc=module.utc());record['passed']=record['passed'] and not record['active'] and all(module.completed(c) for c in owner.commands);save()
            print(json.dumps(dict(receipt=str(OUT/'record.json'),passed=record['passed'],active=record['active'],error=record.get('error'))),flush=True)
        return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
