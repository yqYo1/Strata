#!/usr/bin/env python3
"""Main-owned production-wrapper GDN3 qualification. Repository LICENSE (MIT).

The same complete source/build supplies the engine and this fixture. Keep GPU
component evidence separate from model PP,range,handoff and context adoption.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import gdn_model_build as built_module
import kernel_trace
import proc_observer
import runtime_probe as runtime
import ur_validation as validation

built=built_module
retained= built.retained
base,P,B,RESULT=retained.base,retained.P,retained.B,retained.RESULT
require,digest,write=base.require,base.digest,base.write
ROOT=built.ROOT
BUILD,CHECK,SPEED=built.KEY,'gdn16_model_component_device_v1','gdn16_model_component_speed_v1'
CONTROLLER=Path(__file__).resolve()


def save(root,key,row):
    write(root/'record.json',row)
    data=json.loads(RESULT.read_text());data[key]=base.compact(row);write(RESULT,data)


def controls(row):
    require(base.controls(row['source_commit'],controller=CONTROLLER)==row['controller_inputs'],
            'production-wrapper cohort controls changed')
    _,native=built_module.native_inputs(row['native_source_commit'])
    require({n:hashlib.sha256(v).hexdigest() for n,v in native.items()}==row['native_sources'],
            'frozen production model/fixture source changed')


def cases(speed):
    return [dict(name='host-canary',args=[],clean=False,trace=False),
            dict(name='clean-smoke' if speed else 'pipeline-smoke',args=['--smoke'],clean=speed,trace=not speed)] + (
        [dict(name='resident-'+str(i),args=['--bench',str(i)],clean=True,trace=False) for i in range(3)] if speed else
        [dict(name='pipeline-general',args=['--validate'],clean=False,trace=False)])


def prerequisites(speed):
    model=built.verify();data=base.committed(RESULT)
    refs=dict(build=dict(path=str(ROOT/'record.json'),sha256=digest(ROOT/'record.json')))
    binary=model['binaries'][built.COMPONENT]
    if speed:
        path=ROOT/'component-device-v1/record.json';device=json.loads(path.read_text())
        require(data[CHECK]==base.compact(device) and device['passed'] and device['complete'] and
                device['ownership_closed'] and not device['active'] and not device['fresh_faults'] and
                device['binary']==binary,'same-build production-wrapper qualification required')
        controls(device);refs['device']=dict(path=str(path),sha256=digest(path))
    gate=data['official_prefill_strategies_device_v2']
    return model,refs,gate,retained.runtime.inventory(gate['runtime_manifest'])


def prepare_run(speed):
    key,root=(SPEED,ROOT/'component-speed-v1') if speed else (CHECK,ROOT/'component-device-v1')
    require(key not in base.committed(RESULT) and not root.exists(),'one fresh cohort')
    built,refs,gate,_=prerequisites(speed); root.mkdir(mode=0o700)
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    row=dict(stage='prepared',source_commit=revision,controller_inputs=base.controls(revision,controller=CONTROLLER),
        native_sources=built['native_sources'],native_source_commit=built['source_commit'],binary=built['binaries'][built_module.COMPONENT],
        prerequisite_records=refs,**{k:gate[k] for k in ('environment','canary','native_validation_probe','runtime_manifest','boot','driver')},
        cases=cases(speed),processes=[],active=False,ownership_closed=True,passed=False,complete=False,
        GPU_executed=False,model_executed=False,inference_adopted=False,commands_executed=[],
        budgets=dict(wall=600,cpu=2400,rss_cap=8<<30,text_cap=8<<20,file_cap=64<<20,total_cap=96<<20),whole_wall_seconds=2400,whole_bytes=96<<20,
        plan=dict(scope='Actual production Gemm FP32/FP16 wrappers plus packed resident two-projection+conv+GDN3+RMS/gating component;K2560,C10240,V6144. No model PP,TG,dequantization,PCIe,output projection or physical262144model qualification.',
            precision='Default oneMKL compute mode unset in both arms. Half C uses half1/0 scalars;FP32 C uses float1/0. Typed storage uses SYCL half casts;all65536half promotions and253964signed F32 rounding/subnormal/overflow/NaN cases are checked on GPU before pipeline cases. Non-NaN bits match host RNE exactly,NaN classification only. FP32 control keeps its original software output rounding. FP32 persistent state/history,gates,reductions/accumulation. Passing diagnostics does not establish vendor accumulator type or real-model quality/range.',
            numerical='Whole typed conv matches FP32 conv followed by host RNE. Typed recurrence state/history/final FP16 output match rounded-storage FP32 reference exactly,including1/2/tail splits. 64FP64 packed-producer spots/case bounded by2gamma32*absdot plus final-half-rounding. All finite outputs,guards,immutable inputs. Original FP32 fused-vs-columns smoke bit-exact. 18synchronous GDN geometry/epsilon/null/alias/alignment/host-pointer refusals plus15production vendor-wrapper shape/overflow/null/alias/alignment/device-span/initialization no-write refusals. FP32-vs-FP16 drift is recorded diagnostically,not an extra-precision adoption criterion.',
            timing='Three fresh processes;two4096/8192-row cases. Alternating/reversed arm order by process/round;first shape calls separate,four warmups,five paired64call batches. State/history resets,allocation,copies,CPU oracles outside timing. Host interval includes complete twoGEMM+conv/history+recurrence/RMS submissions,drain every4calls and final checked wait;no returned-MKL-event whole-operation assumption.',
            eligibility='Each steady batch>=100ms;median of5empty controls<=5%of each batch. No resize/retry. Follow-up requires all3paired independent-process median ratios<1 at both lengths;no minimum percentage gain. Component success alone does not authorize a model trial.',
            safety='Original in-order Runtime;copy offload/direct submission disabled and preexec environment immutable. At most4compound calls before drain. Canary,small traced smoke,warning-only general shapes before separate clean speed processes.',
            changed_mechanism='Raw oneMKL producer calls replaced with actual production Gemm wrappers in both arms;new full model generation also contains opt-in long-prefill caller/arena integration. Prior raw-MKL speed gate passed;this cohort must verify real wrapper cost before any model execution.',
            retention='Keep useful samples,numerical maxima and decisions. Retire closed raw logs/traces after extracting names/provenance. Keep selected engine;new component generation has only this imminent comparison consumer.'))
    save(root,key,row); print('Prepared '+key+';commit before GPU execution.',flush=True)


def analyze(case,stdout,folder,pid):
    if case['name']=='host-canary':
        require(json.loads(stdout)==dict(pid=pid,initialized=0,checked=46,expected=46,version=(1<<32)-1,closed=0),'host canary differs')
        return dict(host_only=True)
    require(stdout.count('fixture_status=PASS;packed resident GDN3 only')==1,'fixture incomplete')
    rows=[json.loads(s) for s in stdout.splitlines() if s.startswith('{')]
    require(rows and all(v.get('passed') is True for v in rows),'invalid fixture row')
    bench=case['name'].startswith('resident-'); smoke=case['args']==['--smoke']
    expected=([(4096,'general'),(8192,'general')] if bench else [(1,'general'),(17,'general')] if smoke else
        [(t,'general') for t in (1,2,3,127,129,8191,8192)]+[(129,p) for p in ('zero','cancel','onehot')])
    canonical=[v for v in rows if v['kind']=='canonical']
    require([(v['T'],v['pattern'],v['repeats']) for v in canonical]==
        [(t,p,r) for t,p in expected for r in ((1,64) if bench else (1,))] and all(
            v['exact_rounded_storage_state_history_output'] and v['guards'] and
            v['split_1_2_tail']==(v['repeats']==1 and v['T']>=4) for v in canonical),'canonical coverage differs')
    producer=[v for v in rows if v['kind']=='packed_producers']
    require([v['T'] for v in producer]==[t for t,_ in expected] and all(v['FP64_spots']==64 and
            0<=v['max_bound_fraction']<=1 for v in producer),'packed producer coverage differs')
    require(len(rows)==(49 if bench else 10 if smoke else 33),'unexpected result rows')
    conversion=[v for v in rows if v['kind']=='conversion']
    require(conversion==[dict(kind='conversion',half_inputs=65536,float_inputs=253964,finite_bits_exact=True,nan_classes=True,passed=True)],'complete conversion corpus required')
    result=dict(conversion=conversion[0],canonical=canonical,packed_producers=producer,
        FP32_difference=[v for v in rows if v['kind']=='FP32_difference'])
    if not bench:
        require([v['cases'] for v in rows if v['kind']=='rejection']==[18],'no-write refusal coverage differs')
        require([v['cases'] for v in rows if v['kind']=='vendor_rejection']==[15],'vendor wrapper refusal coverage differs')
        result['vendor_rejections']=15
        require(len([v for v in rows if v['kind']=='legacy'])==(1 if smoke else 0),'legacy check differs')
    else:
        timing=[v for v in rows if v['kind']=='timing']; require(len(timing)==38,'timing coverage differs')
        for t in (4096,8192):
            selected=[v for v in timing if v['T']==t]; order=int(case['args'][1])
            require([v['arm'] for v in selected if v['phase']=='first_shape_call']==
                    [('FP32','FP16')[(i+order)%2] for i in range(2)],'first-call order differs')
            steady=[v for v in selected if v['phase']=='steady']
            require([(v['round'],v['arm'],v['calls']) for v in steady]==
                [(r,('FP32','FP16')[(i+r+order)%2],64) for r in range(5) for i in range(2)],'steady order differs')
            empty=[v['host_ms'] for v in selected if v['phase']=='timer_control']
            require(len(empty)==5 and all(v>=0 for v in empty),'empty controls missing')
            for v in steady: v['eligible']=v['host_ms']>=100 and statistics.median(empty)<=.05*v['host_ms']
        result['timing']=timing
    if case['trace']:
        result['trace']=dict(path=str(folder/'ur.trace'),sha256=digest(folder/'ur.trace'),bytes=(folder/'ur.trace').stat().st_size,
            **kernel_trace.launch_census((folder/'ur.trace').read_text()))
    return result


def decisions(row):
    out=[]
    for t in (4096,8192):
        pairs=[]
        for proc in row['processes']:
            if not proc['name'].startswith('resident-'): continue
            timing=[v for v in proc['details']['timing'] if v['T']==t and v['phase']=='steady']
            medians={arm:statistics.median(v['host_ms']/v['calls'] for v in timing if v['arm']==arm) for arm in ('FP32','FP16')}
            pairs.append(dict(process=proc['name'],milliseconds=medians,ratio=medians['FP16']/medians['FP32'],eligible=all(v['eligible'] for v in timing)))
        require(len(pairs)==3,'three fresh process pairs required')
        import math
        out.append(dict(T=t,process_medians=pairs,paired_geomean_time_ratio=math.exp(sum(math.log(p['ratio']) for p in pairs)/3),
            follow_up_eligible=all(p['eligible'] and p['ratio']<1 for p in pairs)))
    return out


def run(speed):
    key,root=(SPEED,ROOT/'component-speed-v1') if speed else (CHECK,ROOT/'component-device-v1'); row=json.loads((root/'record.json').read_text())
    require(base.committed(RESULT)[key]==base.compact(row) and row['stage']=='prepared' and not (root/'owned').exists(),'one frozen fresh cohort')
    controls(row);built,refs,gate,providers=prerequisites(speed)
    require(row['binary']==built['binaries'][built_module.COMPONENT] and row['prerequisite_records']==refs and row['cases']==cases(speed) and
            all(row[k]==gate[k] for k in ('environment','canary','native_validation_probe','runtime_manifest','boot','driver')),'cohort envelope changed')
    frozen,owner,active,mapped_all=copy.deepcopy(row),None,None,set()
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    deadline=time.monotonic()+row['whole_wall_seconds']
    def persist(): write(root/'execution.json',dict(active_case=active,commands=owner.commands if owner else [],active=bool(owner and owner.active)))
    try:
        owner=base.supervision.Owner(root/'owned',persist)
        for case in row['cases']:
            active=case['name']; folder=root/active;folder.mkdir(mode=0o700)
            for part in ('tmp','cache'): (folder/part).mkdir()
            env=dict(row['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),OMP_NUM_THREADS='1')
            if case['clean']:
                for name in tuple(env):
                    if name.startswith('UR_LOG_') or name in ('UR_ENABLE_LAYERS','LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER'): del env[name]
            else: env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'ur.trace'),UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
            asset=row['canary'] if active=='host-canary' else row['binary'];mapped,env_seen,observer=set(),False,proc_observer.ProcObserver()
            def observe(pid):
                nonlocal env_seen
                raw=Path('/proc',str(pid),'environ').read_bytes()
                if not raw: raise ProcessLookupError('empty retiring environment')
                actual=dict(v.split(b'=',1) for v in raw.split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode())==v.encode() for k,v in env.items()) and set(actual)-{k.encode() for k in env}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment differs')
                seen=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                if not seen: raise ProcessLookupError('empty retiring maps')
                require(seen<=set(providers)|{str(Path(asset['path']).resolve())},'unplanned provider: '+repr(seen-set(providers)))
                mapped.update(seen);env_seen=True
            def monitor(io,entry):
                io.proc.stdin.close();next_sample=0.
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll();require(time.monotonic()<deadline,'whole cohort deadline')
                    if time.monotonic()>=next_sample:
                        observer.sample(lambda:observe(io.proc.pid),lambda:bool(io.exit_watch.poll(0)))
                        require(base.supervision.artifact_bytes(root)<=row['whole_bytes'],'whole cohort byte budget');next_sample=time.monotonic()+.1
                    if select.select([io.proc.stdout],[],[],.01)[0]:io.read();io.pending.clear()
                io.exit(1)
            row['GPU_execution_requested']=row.get('GPU_execution_requested',False) or active!='host-canary'
            entry,out,err=owner.run(active,[asset['path'],*case['args']],env,P,interaction=monitor,**row['budgets'])
            stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
            require(base.supervision.closed(entry) and entry['exit_code']==0,'native process failed '+active+': '+stderr[-3000:])
            if not case['clean']:
                activation=validation.preload_evidence(stderr,entry['pid']);require(activation['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'pre-main loader differs')
                require(not (folder/'validation.log').exists() or (folder/'validation.log').stat().st_size==0,'parameter validation not quiet')
            details=analyze(case,stdout,folder,entry['pid'])
            if active!='host-canary':
                required={env['UR_ADAPTERS_FORCE_LOAD'],env['ZE_ENABLE_ALT_DRIVERS'],'/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0'}
                if not case['clean']:required.update((env['LD_PRELOAD'],env['STRATA_UR_PROBE_EXPECTED_LOADER']))
                require(env_seen and required<=mapped,'actual provider observation incomplete: '+repr(required-mapped));row['GPU_executed']=True
            check=dict(host_probe_accepted=True);validation.finish(check,owner,cursor,row['boot'])
            require(check['passed'] and check['driver_after']==row['driver'],'fault/ownership gate failed')
            row['processes'].append(dict(name=active,passed=True,exit_code=0,clean_speed=case['clean'],pre_main_parameter_validation=not case['clean'],
                environment_observed=env_seen,proc_observation=observer.summary(),ownership_closed=True,details=details,
                environment_sha256=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()))
            mapped_all.update(mapped);persist();print('Passed '+active,flush=True)
        controls(frozen);prerequisites(speed);require(base.committed(RESULT)[key]==base.compact(frozen),'frozen plan changed')
        if speed:row['component_decisions']=decisions(row)
        row.update(stage='resident-speed-closed' if speed else 'general-check-passed',complete=True,host_probe_accepted=True)
    except BaseException as error:row.update(stage='failed',complete=False,host_probe_accepted=False,failed_case=active,error=repr(error))
    finally:
        if owner:validation.finish(row,owner,cursor,row['boot'])
        if row.get('driver_after')!=row['driver']:row['passed']=False
        if not row.get('passed'):row['stage']='failed'
        if row.get('GPU_execution_requested') and not row['GPU_executed']:row['GPU_executed']=None
        row.update(active=bool(owner and owner.active),observed_provider_count=len(mapped_all),ordinary_runtime_teardown_qualified=False,commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None,'GPU ownership unresolved');save(root,key,row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','GPU_executed','fresh_faults','error','component_decisions')}),flush=True)
    require(row['passed'],'pipeline cohort failed')


if __name__=='__main__':
    modes=('prepare-build','build','verify','prepare-check','check','prepare-speed','speed')
    require(len(sys.argv)==2 and sys.argv[1] in modes,'use '+'|'.join(modes))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        mode=sys.argv[1]
        if mode=='prepare-build':built_module.prepare()
        elif mode=='build':built_module.build()
        elif mode=='verify':print(json.dumps(dict(binaries=built_module.verify()['binaries'],GPU_executed=False)))
        elif mode.startswith('prepare-'):prepare_run(mode=='prepare-speed')
        else:run(mode=='speed')
