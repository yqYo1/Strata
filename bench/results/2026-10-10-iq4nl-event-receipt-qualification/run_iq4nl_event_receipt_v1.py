"""Root-owned exact private IQ4NL synthetic qualification; no model/performance/adoption."""
from pathlib import Path
import csv,fcntl,hashlib,json,math,os,re,statistics,struct,sys,types
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq4nl-event-receipt-20261010')
OUT=B/'iq4nl-event-receipt-runtime-v1'
HEAD='9e54ed18c0e374289293838b026e3e69b1c1fe70'
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
        build_path=B/'iq4nl-event-receipt-cpu-build-v1/record.json';build=json.loads(build_path.read_text())
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
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope='Synthetic exact IQ4NL A/B, independent RNE and exact-event receipt qualification; no batched service/model/performance/fullKV/adoption')
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
            host_line='HOST_PASS,14_roundtrips,6_boundaries,7_private_rejections,no_queue_no_GPU'
            assert host_out.read_text()==host_line+'\n' and host_err.stat().st_size==0
            record['host_contract_passed']=True;save()
            record['GPU_runtime_launched']=True;save()
            _,out,err=run('iq4nl-probe',[str(binary)],env,wall=240)
            rows=list(csv.reader(out.read_text().splitlines()));assert all(rows),'empty stdout record'
            record['stdout_record_counts']={tag:sum(r[0]==tag for r in rows) for tag in sorted({r[0] for r in rows})};save()
            # Build the full ordered transcript independently; not PASS-only admission.
            expected_tags=['HOST_PASS','IDENTITY','RECEIPT_QUEUE','FP16_CONFIG','STAGE','STAGE','STAGE']+['CONSTRUCTOR']*20+['CONSTRUCTOR_RESULT']
            expected_stages=['constructor_upload','constructor_device','constructor_readback'];scales=[0,1,0x3ff,0x400,0x401,0x3c00,0x7bff,0x8000,0x8001,0x83ff,0x8400,0x8401,0xbc00,0xfbff]
            for n in [256,512,1638400]:
                for scale in scales:
                    for repeat in range(2):
                        expected_tags+=['STAGE']*3+['RECEIPT']*2+['STAGE','CASE'];expected_stages+=['upload','generic','private','readback']
            expected_tags+=['STAGE','PASS','STAGE','TERMINAL'];expected_stages+=['final_drain','post_USM_teardown_drain']
            assert [r[0] for r in rows]==expected_tags,'missing/unexpected/unordered record'
            assert rows[0]==host_line.split(',') and rows[1]==['IDENTITY','8086','e20c','0000:05:00.0','root']
            queue=rows[2]
            assert queue[:4]==['RECEIPT_QUEUE','profiling=1','in_order=1','backend=level_zero'] and len(queue)==5
            assert queue[4].startswith('resolution_ns=') and queue[4].split('=',1)[1].isdecimal()
            timer_resolution=int(queue[4].split('=',1)[1]);assert timer_resolution<2**64
            assert rows[3][:5]==['FP16_CONFIG','advertised_capabilities_only','fp16','1','flags']
            assert all(v.isdecimal() for v in rows[3][5:]);flags=[int(v) for v in rows[3][5:]];assert flags==sorted(set(flags))
            assert [r for r in rows if r[0]=='STAGE']==[['STAGE',name] for name in expected_stages] and len(expected_stages)==341
            ctor=[r for r in rows if r[0]=='CONSTRUCTOR']
            xs=[0.,-0.,1.,-1.,1.00048828125,1.00146484375,-1.00048828125,-1.00146484375,65504.,-65504.,65520.,-65520.,65536.,-65536.,2.**-24,-2.**-24,2.**-25,3.*2.**-25,1023.*2.**-24,2047.*2.**-25]
            halfbits=[0,0x8000,0x3c00,0xbc00,0x3c00,0x3c02,0xbc00,0xbc02,0x7bff,0xfbff,0x7c00,0xfc00,0x7c00,0xfc00,1,0x8001,0,2,0x3ff,0x400]
            for i,(row,x,h) in enumerate(zip(ctor,xs,halfbits)):
                fbits=struct.unpack('<I',struct.pack('<f',x))[0]
                assert row==['CONSTRUCTOR',f'index={i}',f'float_bits={fbits:08x}',f'IEEE_RNE_reference={h:04x}',f'observed={h:04x}'],('constructor',row)
            assert [r for r in rows if r[0]=='CONSTRUCTOR_RESULT']==[['CONSTRUCTOR_RESULT','20_words','independent_RNE_bits_equal','target_specific_observation','portable_contract_unqualified']]
            cases=[r for r in rows if r[0]=='CASE'];assert len(cases)==84;index=0;active=0
            for n in [256,512,1638400]:
                for scale in scales:
                    for repeat in range(2):
                        row=cases[index];index+=1
                        values=dict(n=str(n),rows=str(2560 if n==1638400 else 0),cols=str(640 if n==1638400 else 0),scale=f'{scale:04x}',**{'class':'synthetic_negative_robustness' if scale&0x8000 else 'synthetic_finite'},repeat=str(repeat),active=str(n),ab='0',oracle_generic='0',oracle_private='0',all_guards_checked='1')
                        assert row==['CASE']+[k+'='+v for k,v in values.items()],('case',row)
                        active+=n
            assert active==45896704
            receipts=[r for r in rows if r[0]=='RECEIPT'];assert len(receipts)==168
            parsed_receipts=[]
            for scope in range(1,85):
                n=[256,512,1638400][(scope-1)//28]
                for arm_index,arm in enumerate(['generic','private']):
                    row=receipts[(scope-1)*2+arm_index]
                    assert row[:7]==['RECEIPT',f'scope={scope}',f'serial={arm_index+1}',f'arm={arm}','type=20',f'n={n}','metadata_match=1'] and len(row)==10
                    values=[]
                    for key,value in zip(['submit','start','end'],row[7:]):
                        assert value.startswith(key+'=') and value.split('=',1)[1].isdecimal()
                        item=int(value.split('=',1)[1]);assert item<2**64;values.append(item)
                    assert values[0]<=values[1]<=values[2]
                    parsed_receipts.append(dict(scope=scope,serial=arm_index+1,arm=arm,n=n,submit_ns=values[0],start_ns=values[1],end_ns=values[2]))
            record['event_receipt_result']=dict(receipts=168,generic=84,private=84,one_to_one_expected_metadata=True,timestamp_order_non_strict=True,timer_resolution_ns=timer_resolution,raw_receipts=parsed_receipts,
                source_direct_submit_identity=True,diagnostic_only=True,batched_service_qualified=False,production_call_census=False,guard_event_fault_injection=False,capture_loss_injection=False,profiling_query_failure_injection=False,model_performance=False)
            assert [r for r in rows if r[0]=='PASS']==[['PASS','cases=84','active_words=45896704','synthetic_only=1','model=0','performance=0','fullKV=0','adoption=0']]
            assert rows[-1]==['TERMINAL','pass','synthetic_only','model_false','performance_false','fullKV_false','adoption_false']
            trace=err.read_text();assert trace and 'urEnqueueKernelLaunch' in trace and 'zeCommandListAppendLaunchKernel' in trace,'missing requested runtime diagnostic'
            record['synthetic_result']=dict(cases=84,active_words_each_arm=active,stage_records=341,constructor_inputs=20,bitwise_AB=True,bitwise_independent_RNE=True,input_immutable=True,full_guards_checked=True,terminal_after_teardown=True,half_fp_config_advertised_flags=flags,portable_FP_contract_qualified=False,model=False,performance=False,fullKV=False,adopted=False)
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
