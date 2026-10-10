"""Root-owned balanced direct IQ4NL qualification/clean service; no model/adoption."""
from pathlib import Path
import csv,fcntl,hashlib,json,math,os,re,statistics,struct,sys,types
import importlib.util
MODE=sys.argv[1] if len(sys.argv)==2 else None
assert MODE in ["qualify","service"]
parser_path=Path("/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/validate_iq4nl_service_output_v1.py")
spec=importlib.util.spec_from_file_location("iq4nl_service_parser",parser_path)
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq4nl-event-receipt-20261010')
OUT=B/('iq4nl-service-runtime-'+MODE+'-v1')
HEAD='9c28dec59b04b3d92e1dc8a4862103089f908f25'
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
        module=types.ModuleType('bounded_private_iq4nl_owner');exec(compile(owner_source,str(PARENT)+'[timing-limits]','exec'),module.__dict__);module.W=W
        build_path=B/'iq4nl-service-cpu-build-v1/record.json';build=json.loads(build_path.read_text())
        assert build['passed'] and build['complete'] and not build['active'] and build['source_head']==HEAD
        correction_path=B/'iq4nl-service-build-identity-revalidation-v1.json'
        correction=json.loads(correction_path.read_text())
        assert correction['passed'] and correction['offline_only'] and correction['original_build_receipt']==ident(build_path)
        assert correction['source_head']==HEAD and correction['original_binary_identity_invalid']
        actual_binary=correction['actual_binary'];binary=Path(actual_binary['path'])
        assert ident(binary)=={k:actual_binary[k] for k in ['bytes','sha256']}
        if MODE=='service':
            qualification=json.loads((B/'iq4nl-service-runtime-qualify-v1/record.json').read_text())
            assert qualification['passed'] and qualification['complete'] and not qualification['active']
            assert qualification['source_head']==HEAD and qualification['binary']==actual_binary
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
        env=dict(clean,LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:'+build['environment']['LD_LIBRARY_PATH']+':/usr/lib/x86_64-linux-gnu')
        if MODE=='qualify':env=environment_module.diagnostic_environment(env)
        else:
            forbidden=('ZEL_','ZE_ENABLE_','UR_LOG_','UR_ENABLE_LAYERS','SYCL_PI_TRACE','STRATA_TRACE','PTI_')
            assert not any(key.startswith(forbidden) for key in env)
        env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',SYCL_CACHE_PERSISTENT='0',EnableDirectSubmission='0')
        record=dict(active=True,complete=False,passed=False,started_utc=module.utc(),controller_sha256=sha(__file__),owner_sha256=PARENT_HASH,
            source_head=HEAD,build_receipt=ident(build_path),build_identity_correction=ident(correction_path),binary=actual_binary,parser=ident(parser_path),mode=MODE,environment=env,runtime_pins=pins,commands=owner.commands,
            GPU_runtime_launched=False,model_opened=False,adopted=False,performance_eligible=False,full_physical_KV_lifecycle=False,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope='Fixed balanced synthetic IQ4NL helper component, clean service or logged qualification; no model/fullKV/adoption')
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
            assert dependency_errors.stat().st_size==0 and 'not found' not in dependencies.read_text(),'unresolved dependency'
            dependency_pins={}
            for line in dependencies.read_text().splitlines():
                match=re.search(r'(?:=>\s*)?(/\S+)\s+\(',line)
                if match:
                    path=Path(match[1]);dependency_pins[str(path)]=dict(realpath=str(path.resolve()),**ident(path))
            assert dependency_pins;record['resolved_dependency_pins']=dependency_pins;save()
            _,host_out,host_err=run('host-contract',[str(binary),'--host-only'],env,wall=30)
            host_result=parser.validate(host_out,'host-only')
            assert host_err.stat().st_size==0
            record['host_contract_result']=host_result;save()
            for label,argv,expected_error in [
                ('no-default',[str(binary)],'FAIL,usage_prefill_iq4nl_service_--host-only_or_--qualify_or_--service_no_default\n'),
                ('bad-mode',[str(binary),'--invalid'],'FAIL,invalid_mode_no_queue_created\n'),
                ('extra-argument',[str(binary),'--host-only','extra'],'FAIL,usage_prefill_iq4nl_service_--host-only_or_--qualify_or_--service_no_default\n')]:
                entry,rejected_out,rejected_err=owner.run(label,argv,env,wall=30,text_cap=32<<20,file_cap=16<<20);save()
                assert module.completed(entry) and entry['exit_code']==1
                assert rejected_out.stat().st_size==0 and rejected_err.read_text()==expected_error
            record['host_rejections_passed']=3
            record['GPU_runtime_launched']=True;save()
            _,out,err=run('iq4nl-service',[str(binary),'--'+MODE],env,wall=240)
            record['component_result']=parser.validate(out,MODE)
            if MODE=='qualify':
                trace=err.read_text()
                assert trace and 'urEnqueueKernelLaunch' in trace and 'zeCommandListAppendLaunchKernel' in trace
            else:assert err.stat().st_size==0,'unexpected stderr in clean performance process'
            save()
            _,interval,interval_err=run('kernel-interval',['/usr/bin/journalctl','-k','--after-cursor='+cursor,'--no-pager','-o','short-monotonic'],clean,wall=5)
            assert interval_err.stat().st_size==0
            faults=[line for line in interval.read_text().splitlines() if re.search(r'\bxe\b|i915|GPU HANG|devcoredump',line,re.I)]
            record.update(journal_status='complete',visible_kernel_GPU_entries=faults,devcoredump_after=dump.exists());assert not faults and not dump.exists(),'new GPU fault'
            record.update(passed=True,complete=True,performance_eligible=(MODE=='service'))
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
                assert ident(binary)=={k:actual_binary[k] for k in ['bytes','sha256']}
                assert module.subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==HEAD
                assert not module.subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
                for rel,pin in build['source_pins'].items():assert ident(W/rel)==pin
                for path,pin in pins.items():assert sha(path)==pin['sha256']
                for path,pin in record.get('resolved_dependency_pins',{}).items():assert sha(path)==pin['sha256'] and str(Path(path).resolve())==pin['realpath']
                assert sha(PARENT)==PARENT_HASH
                assert ident(parser_path)==record['parser'] and ident(correction_path)==record['build_identity_correction']
                assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
            except BaseException as error:record['final_pin_error']=repr(error);record['passed']=False
            record.update(active=owner.active is not None,finished_utc=module.utc());record['passed']=record['passed'] and not record['active'] and all(module.completed(c) for c in owner.commands);save()
            print(json.dumps(dict(receipt=str(OUT/'record.json'),passed=record['passed'],active=record['active'],error=record.get('error'))),flush=True)
        return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
