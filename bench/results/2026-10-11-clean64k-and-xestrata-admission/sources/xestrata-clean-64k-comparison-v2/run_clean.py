"""One root-owned direct fresh64K clean arm, with LP5 protocol accounting.

No GDB/profiler/dumps/API validation in target; timing validity is not quality.
Root owns all invocation/testing and selects three paired alternating arms.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import types
from direct_owner import Owner, closed, identity, require, sha, utc

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
CONTRACT=B/'xestrata-diagnostic-contract-root-v3/record.json'
CONTRACT_SHA='c16fb0f8b096384ccf2bb798505b311cfacbd398b8e0f5f8a10c7b6a7ba432d4'
XE_SHA='4e68a6865161b8838886cf7805ad689fe7a05e280f963abdec5e3010c52bd036'
BASE_SHA='81375b88f043f6b791bd2fee5df7b512e25b584ad5631e0b915d5ac354db237c'
PROFILE_SHA='8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'
HEALTH=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools/recover-xe.sh')
HEALTH_SHA='2f90005f79d2309747d917cff51dc53d8ee6192d68cb1feca074f84af8b0796c'
BASE_ORIGINAL_SHA='dd929ca6c954163f0116ce0492dbfbdb316d727761e1cbc31f27f39eda3826da'
BASE_ADMISSION_SCOPE='32K execution/math only; original event-timing qualification remains invalid'
EXPECTED32=[8192,16384,24576,32767]
EXPECTED64=[8192,16384,24576,32768,40960,49152,57344,65535]


def validate(result,n=65536):
    require(n in (32768,65536),'admitted protocol geometry')
    lines=result['protocol']; done=[v.split() for v in lines if v.startswith('DONE ')]
    require(len(done)==1 and len(done[0])>=16,'one full DONE')
    d=done[0]; generated=int(d[1])
    require(0<generated<=64 and len(result['ids'])==len(result['logprobs'])==generated,'complete output count')
    require(d[5] in ('length','stop') and (d[5]!='length' or generated==64),'honest earlystop/length')
    require(int(d[2])==n and int(d[8])==0 and int(d[14])==n,'complete fresh prompt/read_n')
    require(0<=int(d[6])<=int(d[7]),'draft counts')
    require(all(math.isfinite(float(d[i])) and float(d[i])>0 for i in (3,4)),'positive finite phase times')
    for tag in ('RESUME','REUSED'):
        require([int(v.split()[1]) for v in lines if v.startswith(tag+' ')]==[0],'fresh '+tag)
    pp=[v.split() for v in lines if v.startswith('PP ')]
    require([int(v[1]) for v in pp]==(EXPECTED64 if n==65536 else EXPECTED32),'exact chunk/end positions')
    require(all(len(v)==5 and int(v[2])==n and all(math.isfinite(float(x)) and float(x)>=0 for x in v[3:]) for v in pp),'PP fullinput totals/finite')
    require(all(0<=v<248320 for v in result['ids']),'output ID domain')
    # Require every token immediately accompanied by LP5; no duplicate/lost LP.
    outputs=[v for v in lines if v.startswith(('T ','LP '))]
    require(len(outputs)==2*generated and all(outputs[2*i].startswith('T ') and outputs[2*i+1].startswith('LP ') for i in range(generated)),'T/LP pairing')
    for value in result['logprobs']:
        f=value.split(); require(len(f)==7 and math.isfinite(float(f[1])),'finite LP5')
        for v in f[2:]:
            token,lp=v.split(':'); require(0<=int(token)<248320 and math.isfinite(float(lp)),'LP top ID/finite')
    return dict(prompt_tokens=n,generated_tokens=generated,actual_batched_positions=n-1,
                prompt_ms=float(d[3]),decode_ms=float(d[4]),finish_reason=d[5],
                mtp_counts=list(map(int,d[6:8])),fresh_complete_finite=True,
                prefill_tps=n*1000/float(d[3]),decode_tps=generated*1000/float(d[4]))


def load_gate(path,expected,kind):
    require(path is not None and expected is not None and re.fullmatch('[0-9a-f]{64}',expected),'explicit diagnostic path/hash')
    require(sha(path)==expected,'diagnostic receipt hash')
    data=json.loads(path.read_text())
    if kind=='baseline':
        require(data.get('schema')=='baseline-math-execution-admission-v1','explicit baseline execution/math admission schema')
        require(data.get('binary_sha256')==BASE_SHA and data.get('engine_normal_exit') is True
                and data.get('owned_closed') is True and data.get('math_gate_passed') is True,
                'baseline admission exact linked engine/math/closure')
        require(data.get('qualification_scope')==BASE_ADMISSION_SCOPE
                and data.get('event_timing_qualified') is False,'original timing remains invalid')
        provenance=data['original_receipt']
        require(provenance.get('sha256')==BASE_ORIGINAL_SHA,'only exact original failed timing receipt admitted')
        original_path=Path(provenance['path'])
        require(original_path.is_absolute() and sha(original_path)==BASE_ORIGINAL_SHA,'reverify immutable original failed receipt')
        original=json.loads(original_path.read_text())
        require(original.get('active') is False and original.get('exit_code')==0
                and original.get('exit_signal') is None and original.get('boot_unchanged') is True,
                'original engine normal0/boot/closure')
        require(original.get('binary_sha256')==BASE_SHA and original.get('math_gate_passed') is True,
                'original exact engine/math gate')
        require(original.get('completed') is False and original.get('healthy') is False
                and original.get('error')=='AssertionError()','original overall status remains FAILED')
        require(original.get('new_fault_messages')==[] and isinstance(original.get('cleanup'),dict)
                and all(v is False for v in original['cleanup'].values()),'original nofault/no forced cleanup')
        require(not any(original.get(k) for k in ('kernel_gate_error','cleanup_error','fd_cleanup_error')),
                'no other original failure')
        requests=original.get('requests')
        require(isinstance(requests,list) and bool(requests)
                and json.dumps(data.get('requests'),sort_keys=True)==json.dumps(requests,sort_keys=True),
                'derivative requests identical to original')
        for request in requests:
            for key in ('validation','comparison'):
                evidence=request.get(key)
                require(isinstance(evidence,dict) and bool(evidence) and all(v is True for v in evidence.values()),
                        'every original request validation/comparison true')
            validate(request,32768)
        return data
    require(data.get('active') is False and data.get('completed') is True and data.get('exit_code')==0 and not data.get('exit_signal'),'closed normal diagnostic')
    require('new_fault_messages' in data and not data['new_fault_messages'] and not any(data.get('cleanup',{'unknown':True}).values()),'complete nofault/cleanup diagnostic')
    require(not any(data.get(k) for k in ('error','kernel_gate_error','cleanup_error','fd_cleanup_error')),'diagnostic no error')
    require(kind=='xe' and data.get('binary_sha256')==XE_SHA,'diagnostic exact current fork binary')
    if kind=='xe':
        require(data.get('diagnostic_passed') is True and data.get('boot_unchanged') is True,'first rawfork finite healthy')
        validate(data['request'],32768)
    # Rawfork has no independent head/state hooks: intentionally no such gate.
    return data


def preflight(options,gates):
    require(sha(CONTRACT)==CONTRACT_SHA,'frozen CPUv3 contract')
    contract=json.loads(CONTRACT.read_text()); require(contract['passed'] and not contract['active'] and contract['completed'],'CPU contract closed')
    require(re.fullmatch('[0-9a-f]{64}',options.fixture_sha or ''),'explicit fixture SHA256')
    require(options.fixture.is_absolute() and options.fixture.is_file() and options.fixture.stat().st_size<=1<<20,'bounded explicit fixture')
    require(sha(options.fixture)==options.fixture_sha,'fixture exact hash')
    text=options.fixture.read_text(); words=text.split()
    require(len(words)==65536 and all(re.fullmatch('[0-9]{1,6}',v) for v in words),'exact65536 no truncation')
    tokens=list(map(int,words)); require(all(0<=v<248320 for v in tokens),'fixture token domain')
    adapter_path=Path(contract['build_adapter']['path'])
    require(sha(adapter_path)==contract['build_adapter']['sha256'],'pristine build adapter')
    adapter=json.loads(adapter_path.read_text()); require(adapter['passed'] and not adapter['active'] and adapter['completed'],'build closed')
    xe=Path(adapter['binary']); baseline=B/'prefill-gemm-only-linked-root-v1/strata'
    require(sha(xe)==XE_SHA and sha(baseline)==BASE_SHA,'both current binary identities')
    root=Path(adapter['root'])
    for name,expected in adapter['source_sha256'].items(): require(sha(root/name)==expected,'pristine source pin '+name)
    require(sha(adapter['original_build_receipt']['path'])==adapter['original_build_receipt']['sha256'],'original build receipt')
    args=list(contract['argv'][1:]); profile=Path(args[args.index('--expert-profile')+1])
    require(sha(profile)==PROFILE_SHA,'exact common expert profile')
    for key,value in {'--max-context':'262144','--prefill':'8192','--expert-cache':'128','--pool-workers':'5','--spec':'4','--kv-resident':'32768','--short-read':'0'}.items():
        require(args.count(key)==1 and args[args.index(key)+1]==value,'fixed CLI '+key)
    require('--no-prefill-borrow' in args and '--serve' in args and '--greedy' in args,'fixed owned serve/greedy')
    require(sha(HEALTH)==HEALTH_SHA,'health helper pin')
    if gates:
        load_gate(options.fork_diagnostic,options.fork_diagnostic_sha,'xe')
        load_gate(options.baseline_diagnostic,options.baseline_diagnostic_sha,'baseline')
        host=json.loads(options.owner_qualification.read_text())
        require(host.get('passed') is True and host.get('active') is False and host.get('complete') is True,'closed CPU owner qualification')
        require(host['controller_sha256']==sha(__file__) and host['owner_sha256']==sha(Path(__file__).with_name('direct_owner.py')),'owner tests for exact controller bytes')
        require(host.get('checked')==['normal-interactive','nonzero','timeout','log-limit','reparented-new-PGID'],'all owner CPU gates')
    binary=xe if options.arm=='xe' else baseline
    require(os.access(binary,os.X_OK),'executable admission')
    return contract,adapter,root,binary,[str(binary)]+args,tokens


def environments(arm):
    # Explicit allowlist from creation. Never clone os.environ or log it.
    require(os.geteuid()!=0,'ordinary-account controller required')
    base=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',HOME='/home/yayoi',USER='yayoi',LOGNAME='yayoi')
    for key in ('XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS'):
        value=os.environ.get(key)
        if value is not None:
            require(len(value)<4096,'desktop environment bound'); base[key]=value
    runtime=dict(base,LD_LIBRARY_PATH=':'.join(['/opt/intel/oneapi/compiler/2026.1/lib',
                 '/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib','/opt/intel/oneapi/umf/1.1/lib',
                 '/usr/lib/x86_64-linux-gnu','/opt/intel/oneapi/mkl/2026.1/lib']),
                 SYCL_CACHE_PERSISTENT='0',UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',
                 UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                 NEOReadDebugKeys='1',EnableDirectSubmission='0')
    target=dict(runtime,STRATA_PREFILL_FIRST='0',STRATA_PREFILL_RING='8')
    if arm=='baseline':
        target.update(STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_ATTN_LAYOUT='1',STRATA_PREFILL_LAYER_MAJOR='0',
                      STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0')
    require(not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_','UNITRACE_','XPTI_')) or k in ('UR_ENABLE_LAYERS','STRATA_TRACE') for k in target),'quiet target environment')
    return base,runtime,target


def host_checks(out):
    out.mkdir(mode=0o700); owner=Owner(out/'processes')
    record=dict(active=True,complete=False,passed=False,commands=owner.commands,controller_sha256=sha(__file__),
                owner_sha256=sha(Path(__file__).with_name('direct_owner.py')),gpu_executed=False,model_executed=False)
    def save(): (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    owner.persist=save
    env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    try:
        def interact(io,entry):
            require(io.line(2)=='READY','CPU ready'); io.send(b'GEN\n',2)
            require(io.line(2)=='DONE','CPU done'); io.send(b'QUIT\n',2); io.exit(2)
        cases=[('normal-interactive','import sys\nprint("READY",flush=True)\nassert input()=="GEN"\nprint("DONE",flush=True)\nassert input()=="QUIT"',3,8<<20,interact),
               ('nonzero','raise SystemExit(2)',3,8<<20,None),
               ('timeout','import time;time.sleep(20)',.15,8<<20,None),
               ('log-limit','import os,time;os.write(2,b"x"*262144);time.sleep(20)',3,65536,None),
               ('reparented-new-PGID','import os,time\np=os.fork()\nif p==0:\n os.setpgid(0,0);time.sleep(20)\nelse:\n time.sleep(.05);os._exit(0)',3,8<<20,None)]
        for name,code,wall,cap,interaction in cases:
            e,_,_=owner.run(name,['/usr/bin/python3','-c',code],env,out,interaction,wall=wall,text_cap=cap,rss_cap=2<<30,cpu=10)
            require(e['direct_child_reaped'] and e['session_empty'] and not e['errors'],'CPU ownership closure')
            if name=='normal-interactive': require(closed(e) and e['exit_code']==0,'CPU normal')
            elif name=='nonzero': require(closed(e) and e['exit_code']==2,'CPU nonzero retained')
            else: require(e.get('error') and e['cleanup'] and not closed(e),'CPU failure cleanup')
        record.update(complete=True,passed=True,checked=[v[0] for v in cases])
    except BaseException as e: record['error']=repr(e)
    finally:
        record['active']=owner.active is not None; save()
    print(json.dumps(dict(passed=record['passed'],record=str(out/'record.json'))))
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('host-checks','cpu-preflight','run'))
    parser.add_argument('--arm',choices=('baseline','xe')); parser.add_argument('--repetition',type=int,choices=(1,2,3))
    parser.add_argument('--output',type=Path); parser.add_argument('--fixture',type=Path); parser.add_argument('--fixture-sha')
    parser.add_argument('--fork-diagnostic',type=Path); parser.add_argument('--fork-diagnostic-sha')
    parser.add_argument('--baseline-diagnostic',type=Path); parser.add_argument('--baseline-diagnostic-sha')
    parser.add_argument('--owner-qualification',type=Path)
    options=parser.parse_args()
    if options.output is not None:
        require(options.output.is_absolute() and options.output.parent.resolve().is_relative_to(B) and not options.output.exists(),'new private output under B')
    if options.mode=='host-checks':
        require(options.output is not None,'hostcheck output')
        with (B/'owned-v0141-measurement.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return host_checks(options.output)
    require(options.arm and options.repetition and options.fixture,'explicit arm/repetition/fixture')
    contract,adapter,root,binary,argv,tokens=preflight(options,options.mode=='run')
    if options.mode=='cpu-preflight':
        print(json.dumps(dict(passed=True,gpu_executed=False,tokens=len(tokens),argv=argv,
                              fixture_sha256=options.fixture_sha,requested_ring=8,effective_ring_source_contract=16 if options.arm=='xe' else 8)))
        return 0
    require(options.output is not None and options.owner_qualification is not None,'run output/CPU owner gate')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=options.output; out.mkdir(mode=0o700)
        require(shutil.disk_usage(B).free>8<<30,'artifact free capacity')
        memory={v.split(':')[0]:int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith(('MemTotal:','MemAvailable:'))}
        require(memory['MemAvailable']>64<<30,'host memory capacity')
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record=dict(active=True,completed=False,passed=False,timing_valid=False,quality_qualified=False,math_equivalence=False,
                    adopted=False,performance_eligible=False,full262144_lifecycle_qualified=False,arm=options.arm,repetition=options.repetition,
                    fixture=dict(path=str(options.fixture),sha256=options.fixture_sha,tokens=65536),
                    source_scope='whole unchanged fork vs current baseline; LP5 protocol, not exclusive compute',
                    binary=dict(path=str(binary),sha256=sha(binary)),argv=argv,boot_id=boot,started_utc=utc(),initial_memory=memory,
                    contract_sha256=CONTRACT_SHA,profile_sha256=PROFILE_SHA,controller_sha256=sha(__file__),
                    baseline_admission_scope=BASE_ADMISSION_SCOPE,baseline_original_receipt_sha256=BASE_ORIGINAL_SHA,
                    baseline_original_status='FAILED',baseline_original_event_timing_qualified=False,
                    owner_sha256=sha(Path(__file__).with_name('direct_owner.py')),health_helper_sha256=HEALTH_SHA,
                    diagnostic_gates={str(p):sha(p) for p in (options.fork_diagnostic,options.baseline_diagnostic,options.owner_qualification)},
                    requested_ring=8,effective_ring_source_contract=16 if options.arm=='xe' else 8,effective_ring_runtime_measured=False,
                    limits=dict(process_wall_seconds=1000,protocol_line_seconds=700,send_seconds=20,QUIT_seconds=30,
                                combined_log_bytes=64<<20,session_RSS_bytes=104<<30,artifact_bytes=2<<30,CPU_seconds=1000))
        def save():
            body=json.dumps(record,indent=2)+'\n'; require(len(body.encode())<2<<20,'compact receipt cap')
            temp=out/'record.json.tmp'; temp.write_text(body); temp.replace(out/'record.json')
        owner=Owner(out/'processes',save); record['commands']=owner.commands
        base,runtime,target=environments(options.arm); record['environment']=target
        health=types.ModuleType('private_health')
        exec(compile(HEALTH.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(HEALTH),'exec'),health.__dict__)
        # Keep original check_health/diagnostic functions, replace only their
        # environment supplier with an explicit dictionary; no inherited secrets.
        health.health_environment=lambda:dict(runtime)
        runstart=time.monotonic()
        def remaining():
            seconds=1000-(time.monotonic()-runstart)
            require(seconds>0,'whole controller deadline')
            return seconds
        class Runner:
            def __init__(self,phase):
                self.output=out/phase; self.output.mkdir(); self.number=0
            def run(self,label,arguments,seconds=20,env=None):
                self.number+=1
                e,so,se=owner.run(self.output.name+'-'+str(self.number)+'-'+label,arguments,dict(base if env is None else env),root,
                                   wall=min(seconds,remaining()),text_cap=8<<20,rss_cap=2<<30,cpu=max(5,int(seconds)))
                require(closed(e) and e['exit_code']==0,'health utility complete normal0 '+label)
                return so.read_text()
        cursor=None; model_attempted=False
        save()
        try:
            fault=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
            require(not fault.exists(),'existing GPU dump: no retry/reset')
            before=Runner('health-before')
            require(before.run('embedding',['/usr/bin/systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],5).strip()=='ActiveState=inactive','embedding inactive')
            cursor=health.journal_cursor(before,'kernel-before')
            health.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),before,Path('/home/yayoi/.local/bin/strata-xe-health'),cursor=cursor)
            preflight(options,True) # recheck executable/source/fixture/gates under lock
            def interaction(io,entry):
                startup=[]; record['startup']=startup
                while True:
                    value=io.line(); startup.append(value)
                    require(len(startup)<1000 and not value.startswith('ERR'),'startup protocol')
                    if value.startswith('READY '): require(value.split()[1]=='262144','READY context'); break
                ready=time.monotonic(); entry['READY_monotonic']=ready
                now=identity(entry['direct_identity']['pid'])
                require(now and now['start_ticks']==entry['direct_identity']['start_ticks'],'actual inferior identity')
                exe=Path('/proc',str(now['pid']),'exe')
                require(exe.resolve()==binary.resolve() and sha(exe)==record['binary']['sha256'],'actual target executable')
                # Filter nothing: compare exactly with the explicit allowlist
                # passed to exec. A foreign inherited variable fails admission.
                actual=dict(v.decode().split('=',1) for v in Path('/proc',str(now['pid']),'environ').read_bytes().split(b'\0') if b'=' in v)
                require(actual==target,'actual full target environment equals explicit allowlist')
                record['actual_target_environment']=dict(target); record['actual_target_identity']=now
                info=next(v for v in startup if v.startswith('INFO ')); settings=dict(v.split('=',1) for v in info.split()[1:] if '=' in v)
                require(all(settings.get(k)==v for k,v in {'context':'262144','kv_resident':'32768','expert_slots':'128','pool_workers':'5','spec':'4'}.items()),'effective common parameters')
                record['startup_vram_free_mib']=int(settings.get('vram_free_mib','-1'))
                require(record['startup_vram_free_mib']>=512,'startup VRAM reserve below512MiB; no GEN launch')
                result=dict(ids=[],logprobs=[],protocol=[]); record['request']=result
                begin=time.monotonic(); io.send(('GEN 64 ckpt=1 logprobs=5 '+','.join(map(str,tokens))+'\n').encode()); sent=time.monotonic()
                while True:
                    value=io.line(); result['protocol'].append(value)
                    require(len(result['protocol'])<2000 and not value.startswith('ERR'),'request protocol bound/failure')
                    if value.startswith('T '): result['ids'].append(int(value.split()[1]))
                    if value.startswith('LP '): result['logprobs'].append(value)
                    if value.startswith('DONE '): break
                done=time.monotonic(); result['validation']=validate(result)
                record['sample']=dict(host_spawn_seconds=entry['spawn_return_monotonic']-entry['spawn_begin_monotonic'],
                                     host_start_to_READY_seconds=ready-entry['spawn_begin_monotonic'],
                                     host_send_seconds=sent-begin,host_send_begin_to_DONE_seconds=done-begin,
                                     host_send_end_to_DONE_seconds=done-sent,**result['validation'])
                save(); io.send(b'QUIT\n'); io.exit(30)
            model_attempted=True
            entry,so,se=owner.run('target',argv,target,root if options.arm=='xe' else B,interaction,wall=remaining())
            record['model_result']=dict(exit_code=entry.get('exit_code'),owned_closed=closed(entry))
            require(closed(entry) and entry['exit_code']==0,'normal owned model0/no survivors')
            preflight(options,True)
            record['completed']=True
        except BaseException as e: record['error']=repr(e)
        finally:
            if owner.active is None and cursor:
                try:
                    after=Runner('health-after')
                    interval=after.run('kernel-interval',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],5)
                    rows=[json.loads(v) for v in interval.splitlines() if v.startswith('{')]
                    def is_fault(row):
                        message=row.get('MESSAGE','')
                        hardware=('0000:05:00.0' in message or re.search(r'\bxe\b',message))
                        return bool((hardware and health.FAULT.search(message))
                                    or ('strata' in message and 'segfault' in message))
                    record['new_fault_messages']=[v['MESSAGE'] for v in rows if is_fault(v)]
                    require(not record['new_fault_messages'] and not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists(),'new GPU fault/dump')
                    require(after.run('embedding',['/usr/bin/systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],5).strip()=='ActiveState=inactive','embedding still inactive')
                    health.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),after,Path('/home/yayoi/.local/bin/strata-xe-health'))
                    record['health_after_passed']=True
                except BaseException as e: record['health_after_error']=repr(e)
            record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
            record['active']=owner.active is not None
            record['timing_valid']=bool(record['completed'] and record.get('health_after_passed') and record['boot_unchanged'] and not record['active'] and not record.get('error') and not record.get('health_after_error'))
            record['passed']=record['timing_valid']; record['performance_eligible']=record['timing_valid']; record['model_launch_attempted']=model_attempted
            record['elapsed_including_health_seconds']=time.monotonic()-runstart; record['finished_utc']=utc(); save()
        print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],timing_valid=record['timing_valid'],quality_qualified=False,error=record.get('error'))))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
