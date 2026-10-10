"""Root-owned first diagnostic real GDN component comparison; no model/timing."""
from pathlib import Path
import csv,fcntl,hashlib,json,os,re,sys,types
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-prefill-gate-factor-20261010')
OUT=B/'gdn-factor-real-runtime-v2'
HEAD='622a53eb9551aa4edca3720279a17cc1d2f2be19'
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
PARENT_HASH='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ident(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sha(PARENT)==PARENT_HASH
        module=types.ModuleType('bounded_real_gdn_owner');exec(compile(PARENT.read_text(),str(PARENT),'exec'),module.__dict__);module.W=W
        build_path=B/'gdn-factor-real-cpu-build-v2/record.json';build=json.loads(build_path.read_text())
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
        env=environment_module.diagnostic_environment(dict(clean,LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib:/opt/intel/oneapi/umf/1.1/lib:/usr/lib/x86_64-linux-gnu'))
        env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',SYCL_CACHE_PERSISTENT='0',EnableDirectSubmission='0',STRATA_GDN_PREFILL_GATE_FACTOR='1',STRATA_GDN_QUAD='0',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='0',STRATA_GDN_PIPELINE='1')
        record=dict(active=True,complete=False,passed=False,started_utc=module.utc(),controller_sha256=sha(__file__),owner_sha256=PARENT_HASH,
            source_head=HEAD,build_receipt=ident(build_path),binary=build['binary'],environment=env,runtime_pins=pins,commands=owner.commands,
            GPU_runtime_launched=False,model_opened=False,adopted=False,performance_eligible=False,full_physical_KV_lifecycle=False,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope='Synthetic real producer/conv/recurrence/FP32+FP16 norm comparison,current pipeline route only')
        def save():
            temp=OUT/'record.json.tmp'
            with temp.open('w') as stream:json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            temp.replace(OUT/'record.json')
        owner.persist=save
        def run(label,args,environment,wall=120):
            entry,out,err=owner.run(label,args,environment,wall=wall,text_cap=64<<20,file_cap=32<<20);save()
            assert module.completed(entry) and entry['exit_code']==0,(label,entry)
            return entry,out,err
        dump=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
        try:
            assert not dump.exists(),'existing GPU fault';record['devcoredump_before']=False
            _,out,_=run('journal-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'],clean,wall=5)
            match=re.search(r'^-- cursor: (.+)$',out.read_text(),re.M);assert match;cursor=match[1];record['kernel_cursor']=cursor
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
            record['component_result']=dict(chunks=22,stage_records=267,total_tokens_per_initialization=sum(lengths),initializations=2,bitwise=True,conv_history_carried=True,recurrence_state_carried=True,pipeline_only=True)
            _,interval,interval_err=run('kernel-interval',['/usr/bin/journalctl','-k','--after-cursor='+cursor,'--no-pager','-o','short-monotonic'],clean,wall=5)
            assert interval_err.stat().st_size==0,'incomplete journal interval'
            faults=[line for line in interval.read_text().splitlines() if re.search(r'\bxe\b|i915|GPU HANG|devcoredump',line,re.I)]
            record.update(journal_status='complete',visible_kernel_GPU_entries=faults,devcoredump_after=dump.exists());assert not faults and not dump.exists(),'new GPU fault'
            record.update(passed=True,complete=True)
        except BaseException as error:record['error']=type(error).__name__+': '+str(error)
        finally:
            try:
                assert ident(binary)=={k:build['binary'][k] for k in ['bytes','sha256']}
                for rel,pin in build['source_pins'].items():assert ident(W/rel)==pin
                for path,pin in pins.items():assert sha(path)==pin['sha256']
                assert sha(PARENT)==PARENT_HASH
                assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
            except BaseException as error:record['final_pin_error']=repr(error);record['passed']=False
            record.update(active=owner.active is not None,finished_utc=module.utc());record['passed']=record['passed'] and not record['active'];save()
            print(json.dumps(dict(receipt=str(OUT/'record.json'),passed=record['passed'],active=record['active'],error=record.get('error'))),flush=True)
        return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
