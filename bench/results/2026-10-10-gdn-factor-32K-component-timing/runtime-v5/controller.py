"""Root-owned first diagnostic real GDN component comparison; no model/timing."""
from pathlib import Path
import csv,fcntl,hashlib,json,math,os,re,statistics,sys,types
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-factor-component-timing-20261010')
OUT=B/'gdn-factor-component-timing-quad-runtime-v5'
HEAD='5ab8ab0a45b98a987b7ec77cb8b19007a35ea5d9'
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
        for old,new in [('(16 << 30, 16 << 30)','(24 << 30, 24 << 30)'),('(120, 121)','(600, 601)'),('rss <= 2 << 30','rss <= 4 << 30')]:
            assert owner_source.count(old)==1;owner_source=owner_source.replace(old,new)
        module=types.ModuleType('bounded_real_gdn_owner');exec(compile(owner_source,str(PARENT)+'[timing-limits]','exec'),module.__dict__);module.W=W
        build_path=B/'gdn-factor-component-timing-cpu-build-v2/record.json';build=json.loads(build_path.read_text())
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
        env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',SYCL_CACHE_PERSISTENT='0',EnableDirectSubmission='0',STRATA_GDN_PREFILL_GATE_FACTOR='1',STRATA_GDN_QUAD='1',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='0',STRATA_GDN_PIPELINE='1')
        record=dict(active=True,complete=False,passed=False,started_utc=module.utc(),controller_sha256=sha(__file__),owner_sha256=PARENT_HASH,
            source_head=HEAD,build_receipt=ident(build_path),binary=build['binary'],environment=env,runtime_pins=pins,commands=owner.commands,
            GPU_runtime_launched=False,model_opened=False,adopted=False,performance_eligible=False,full_physical_KV_lifecycle=False,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope='Synthetic real GDN paired32K component service; quad-selector configuration; no model wall/performance adoption')
        def save():
            temp=OUT/'record.json.tmp'
            with temp.open('w') as stream:json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            temp.replace(OUT/'record.json')
        owner.persist=save
        def run(label,args,environment,wall=120):
            entry,out,err=owner.run(label,args,environment,wall=wall,text_cap=8<<20,file_cap=4<<20);save()
            assert module.completed(entry) and entry['exit_code']==0,(label,entry)
            return entry,out,err
        dump=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
        try:
            assert not dump.exists(),'existing GPU fault';record['devcoredump_before']=False
            _,out,_=run('journal-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'],clean,wall=5)
            match=re.search(r'^-- cursor: (.+)$',out.read_text(),re.M);assert match;cursor=match[1];record['kernel_cursor']=cursor
            # Dynamic dependency check uses the build's pinned oneAPI search path, no device initialization.
            _,dependencies,dependency_errors=run('dynamic-dependencies',['/usr/bin/ldd',str(binary)],env,wall=20)
            assert dependency_errors.stat().st_size==0 and 'not found' not in dependencies.read_text(),'unresolved runtime dependency'
            dependency_pins={}
            for line in dependencies.read_text().splitlines():
                match=re.search(r'(?:=>\s*)?(/\S+)\s+\(',line)
                if match:
                    path=Path(match[1]);dependency_pins[str(path)]=dict(realpath=str(path.resolve()),**ident(path))
            assert dependency_pins,'no dependency identities'
            record['resolved_dependency_pins']=dependency_pins;save()
            # Host-only branch precedes queue construction; preserve its trace separately.
            _,host_out,host_err=run('host-contract',[str(binary),'--host-only'],env,wall=30)
            assert host_out.read_text().strip()=='HOST_PASS,empty_markers,independent_empty_API_calls,4,invalid_length_stride_variant_encoding,no_queue_construction,no_GPU_submission'
            assert host_err.stat().st_size==0,'host-only unexpectedly invoked runtime/logging or failed'
            record['host_contract_passed']=True;save()
            record['GPU_runtime_launched']=True;save()
            _,out,err=run('real-gdn-probe',[str(binary)],env,wall=240)
            rows=list(csv.reader(out.read_text().splitlines()))
            assert rows[0]==['IDENTITY','LevelZero','8086','e20c','0000:05:00.0','root','passed']
            chunks=[row for row in rows if row and row[0]=='CHUNK_PASS']
            lengths=[1,3,5,31,32,33,255,256,257,3,257]
            assert len(chunks)==22
            for sequence in range(2):
                offset=0
                for index,live in enumerate(lengths):
                    row=chunks[sequence*11+index];assert len(row)==17 and row[1::2]==['offset','live','variant','first','state','conv_history','y','half']
                    values=dict(zip(row[1::2],row[2::2]));assert values['offset']==str(offset) and values['live']==str(live) and values['variant']=='0'
                    assert values['first']==('factor' if index%2 else 'log')
                    for key in ['state','conv_history','y','half']:assert values[key].isdecimal() and int(values[key])<1<<64
                    offset+=live
            stages=[row for row in rows if row and row[0]=='STAGE'];assert len(stages)==267
            assert rows[-1]==['TERMINAL','pass','real_producer_conv_recurrence_bitwise','synthetic_only','model_unqualified','full_KV_lifecycle_unqualified','adopted_false']
            record['component_result']=dict(chunks=22,stage_records=267,total_tokens_per_initialization=sum(lengths),initializations=2,bitwise=True,conv_history_carried=True,recurrence_state_carried=True,requested_route="quad",quiet_native_launch_trace=False)
            old_diag=json.loads((B/'gdn-factor-real-quad-runtime-v4/record.json').read_text())
            assert old_diag['passed'] and old_diag['complete'] and not old_diag['active']
            old_chunks=[r for r in csv.reader((B/'gdn-factor-real-quad-runtime-v4/real-gdn-probe.stdout').read_text().splitlines()) if r and r[0]=='CHUNK_PASS']
            assert chunks==old_chunks,'changed default actual state/output fingerprints'
            quiet={k:v for k,v in env.items() if not k.startswith('ZEL_') and k not in {'STRATA_TRACE','STRATA_PREFILL_TIMING','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO','UR_LOG_TRACING','UR_ENABLE_LAYERS','ZE_DEBUG','ZE_ENABLE_VALIDATION_LAYER','ZE_ENABLE_PARAMETER_VALIDATION','ZE_ENABLE_LOADER_DEBUG_TRACE','SYCL_PI_TRACE','SYCL_UR_TRACE','SYCL_TRACE','LD_PRELOAD','LD_DEBUG'}}
            record['quiet_environment']=quiet
            record['limits']=dict(AS_each_bytes=24<<30,RSS_session_sampled_bytes=4<<30,CPU_each_seconds=[600,601],bench_wall_seconds=600,text_cap_bytes=8<<20,file_cap_each_bytes=4<<20)
            good=[str(binary),'--bench','32768','--chunk','8192','--repeats','3']
            rejections=[('zero-chunk',[str(binary),'--bench','32768','--chunk','0','--repeats','3'],quiet),('nondivisor-chunk',[str(binary),'--bench','32768','--chunk','3','--repeats','3'],quiet),('too-few-repeats',[str(binary),'--bench','32768','--chunk','8192','--repeats','2'],quiet),('dirty-ZEL-known',good,dict(quiet,ZEL_ENABLE_LOADER_LOGGING='0')),('dirty-ZEL-unknown',good,dict(quiet,ZEL_FUTURE_DIAGNOSTIC='0')),('dirty-legacy-loader',good,dict(quiet,ZE_ENABLE_LOADER_DEBUG_TRACE='0'))]
            for label,args,environment in rejections:
                entry,reject_out,reject_err=owner.run(label,args,environment,wall=10,text_cap=8<<20,file_cap=4<<20);save()
                assert module.completed(entry) and entry['exit_code']==1 and reject_out.stat().st_size==0 and reject_err.read_text().startswith('FAIL '),(label,entry)
            record['host_admission_rejections']=len(rejections);save()
            _,bench_out,bench_err=run('quiet-32K-bench',good,quiet,wall=600)
            assert bench_err.stat().st_size==0,'quiet benchmark stderr not empty'
            bench_rows=list(csv.reader(bench_out.read_text().splitlines()))
            assert bench_rows[0]==['IDENTITY','LevelZero','8086','e20c','0000:05:00.0','root','passed']
            assert bench_rows[-1]==rows[-1]
            scope_rows=[row for row in bench_rows if row[0]=='BENCH_SCOPE']
            assert len(scope_rows)==1 and scope_rows[0][1::2]==['synthetic_only','model','fullKV','performance_adopted','prefix','chunk','repeats','interval']
            scope=dict(zip(scope_rows[0][1::2],scope_rows[0][2::2]));assert scope['synthetic_only']=='true' and scope['model']=='false' and scope['fullKV']=='false' and scope['performance_adopted']=='false' and scope['prefix']=='32768' and scope['chunk']=='8192' and scope['repeats']=='3'
            assert sum(row[0]=='STAGE' for row in bench_rows)==25
            assert [row for row in bench_rows if row[0]=='BENCH_WARMUP_COMPLETE']==[['BENCH_WARMUP_COMPLETE','32768','both_arms','excluded','reset_before_every_sample']]
            bench_chunks=[row for row in bench_rows if row[0]=='BENCH_CHUNK'];assert len(bench_chunks)==24
            samples=[row for row in bench_rows if row[0]=='BENCH_SAMPLE'];assert len(samples)==6
            parsed={};individual=[]
            for row in bench_chunks:
                keys=['sample','chunk','offset','live','arm','position','variant','seconds','state','history','y','half'];assert row[1::2]==keys
                values=dict(zip(row[1::2],row[2::2]));sample=int(values['sample']);chunk=int(values['chunk']);arm=values['arm']
                assert 0<=sample<3 and 0<=chunk<4 and arm in ['log','factor']
                assert values['offset']==str(chunk*8192) and values['live']=='8192' and values['variant']=='0'
                assert int(values['position'])==int((arm=='factor')!=bool((sample+chunk)%2))
                seconds=float(values['seconds']);assert math.isfinite(seconds) and seconds>0
                for name in ['state','history','y','half']:assert values[name].isdecimal() and int(values[name])<1<<64
                key=(sample,chunk,arm);assert key not in parsed;parsed[key]=values;individual.append(values)
            for sample in range(3):
                for chunk in range(4):
                    a=parsed[sample,chunk,'log'];b=parsed[sample,chunk,'factor']
                    for name in ['state','history','y','half']:assert a[name]==b[name]
                    for arm in ['log','factor']:
                        for name in ['state','history','y','half']:assert parsed[sample,chunk,arm][name]==parsed[0,chunk,arm][name]
            aggregates=[]
            for row in samples:
                assert row[1::2]==['sample','arm','prefix','chunks','seconds','paired_chunk_bits','warmup_excluded']
                v=dict(zip(row[1::2],row[2::2]));sample=int(v['sample']);arm=v['arm'];assert v['prefix']=='32768' and v['chunks']=='4' and v['paired_chunk_bits']=='true' and v['warmup_excluded']=='true'
                observed=float(v['seconds']);derived=sum(float(parsed[sample,c,arm]['seconds']) for c in range(4));assert math.isfinite(observed) and math.isclose(observed,derived,rel_tol=1e-12,abs_tol=1e-12)
                aggregates.append(v)
            assert len({(v['sample'],v['arm']) for v in aggregates})==6
            complete=[row for row in bench_rows if row[0]=='BENCH_COMPLETE'];assert complete==[['BENCH_COMPLETE','prefix','32768','repeats','3','arm_samples','6','paired_chunks','12','synthetic_only','true','model','false','fullKV','false','performance_adopted','false']]
            arms={arm:[float(v['seconds']) for v in aggregates if v['arm']==arm] for arm in ['log','factor']}
            record['timing_result']=dict(prefix=32768,chunk=8192,repeats=3,individual_intervals=individual,individual_aggregates=aggregates,service_seconds_medians={arm:statistics.median(v) for arm,v in arms.items()},scope='Sum of timed actual GDN component service intervals; upload/readback/setup/check work excluded; correctness between pairs affects cache conditions; requestedquad profile, no quiet native launch trace; not model wall or throughput',full_physical_KV=False,model=False,adopted=False)

            _,interval,interval_err=run('kernel-interval',['/usr/bin/journalctl','-k','--after-cursor='+cursor,'--no-pager','-o','short-monotonic'],clean,wall=5)
            assert interval_err.stat().st_size==0,'incomplete journal interval'
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
                for rel,pin in build['source_pins'].items():assert ident(W/rel)==pin
                for path,pin in pins.items():assert sha(path)==pin['sha256']
                for path,pin in record.get('resolved_dependency_pins',{}).items():assert sha(path)==pin['sha256'] and str(Path(path).resolve())==pin['realpath']
                assert sha(PARENT)==PARENT_HASH
                assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
            except BaseException as error:record['final_pin_error']=repr(error);record['passed']=False
            record.update(active=owner.active is not None,finished_utc=module.utc());record['passed']=record['passed'] and not record['active'];save()
            print(json.dumps(dict(receipt=str(OUT/'record.json'),passed=record['passed'],active=record['active'],error=record.get('error'))),flush=True)
        return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
