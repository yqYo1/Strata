#!/usr/bin/env python3
"""Bounded existing-image event/collector qualification and scheduling attribution.

Main-authored; repository LICENSE. No rebuild, model trial or revised E2 gate.
Source -> prepare -> commit plan -> host -> commit -> next (one native child).
"""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import current_pti_admission as pti
import engine_once
import prefill_upload_overlap_reuse as native
import request_profile_reader as trace_reader
import ur_validation

component=native.component
P,B,RESULT=native.P,native.B,native.RESULT
ROOT=B/'prefill-upload-overlap-profile-v2'
KEY='prefill_upload_overlap_profile_v2'
PREFIXES=['tools/sycl','tests/core/test_upload_overlap_reuse.py',
          'tests/core/test_upload_overlap_profile.py','AGENTS.md','docs/ARTIFACT_RETENTION.md','docs/BENCHMARKS.md']
require,sha,signature=native.require,native.sha,native.signature
write=component.write
BUDGET=dict(wall=120,cpu=240,rss_cap=4<<30,text_cap=32<<20,file_cap=32<<20,total_cap=512<<20)
HOST=dict(wall=90,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=4<<20,total_cap=512<<20)
EXPECTED_SAFETY={'metadata':(0,0,0),'forward':(10,2,1),'reverse':(12,4,2),
    'permanent':(16,6,2),'cancel':(7,1,1),'cancel-throw':(7,1,1),'refuse':(6,1,1)}


def collector_proof():
    keys=['unitrace_build','unitrace_session_owner','unitrace_indexed_collector']
    # verify_tools accepts this scoped container. Do not call the old whole-file
    # admission wrapper or bind unrelated research/result rows to the new run.
    tools=pti.verify_tools({k:ur_validation.committed(RESULT,(k,)) for k in keys})
    host=ur_validation.committed(RESULT,('unitrace_current_admission_host_v3',))
    environment=ur_validation.committed(RESULT,('unitrace_current_environment_probe_v3',))
    require(host['passed'] and host['complete'] and host['ownership_closed']
        and signature(tools)==signature(host['admission']['tools'])
        and environment['passed'] and environment['complete'] and environment['ownership_closed']
        and environment['fresh_faults']==[],'existing exact collector/launcher/source qualification required')
    return tools


def trace_events(document,pid,begin,end):
    events=document['traceEvents'];require(len(events)<=50000,'bounded component trace events required')
    labels={e['pid']:e.get('args',{}).get('name','') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    require(labels.get(pid,'').startswith('HOST<'),'owned non-MPI host metadata required')
    # Pinned PTI chromelogger.h:1036-1055 emits BOTH markers on the GPU
    # track: H2D sink(t),D2H source(s). With no PMI_RANK/PMIX_RANK,the
    # category suffix is getpid(). Host API endpoints are absent in this mode.
    # This associates device records with the owned process;it does NOT
    # reconstruct a host API/enqueue timeline or physical transfer direction.
    markers={}
    for event in events:
        match=re.fullmatch(r'Flow_(H2D|D2H)_(\d+)_(\d+)',event.get('cat',''))
        if not match or int(match[3])!=pid:continue
        direction=match[1]
        if event.get('ph')!=('t' if direction=='H2D' else 's'):continue
        require(type(event.get('id')) in (int,str) and str(event['id'])==match[2],
            'device marker/category work identity differs')
        when=trace_reader.nanoseconds(event.get('ts'));require(begin<=when<=end,'device marker outside RAW lifetime')
        key=(direction,event.get('pid'),event.get('tid'),match[2])
        markers.setdefault(key,[]).append(when)
    found=[];ids=set()
    for event in events:
        if event.get('cat')!='gpu_op':continue
        require(event.get('ph')=='X','complete GPU event required')
        low=trace_reader.nanoseconds(event.get('ts'));duration=trace_reader.nanoseconds(event.get('dur'))
        require(begin<=low<=low+duration<=end,'GPU event outside actual owned RAW lifetime')
        raw_id=event.get('args',{}).get('id');identifier=str(raw_id)
        require(type(raw_id) in (int,str) and re.fullmatch(r'\d+',identifier)
            and identifier not in ids,'unique GPU work identity required')
        for direction in ['H2D','D2H']:
            hits=markers.get((direction,event.get('pid'),event.get('tid'),identifier),[])
            require(len(hits)==1 and abs(hits[0]-low)<=1,
                'unique owned-process GPUside device-only marker required')
        ids.add(identifier);label=labels.get(event.get('pid'),'')
        require(label.startswith('DEVICE<') and '0:5:0:0' in label,'actual B570 device lane required')
        name=event.get('name');require(type(name) is str and name,'named GPU work required')
        found.append(dict(name=name,begin=low,end=low+duration))
    return found


def benchmark_attribution(events,arm):
    require(arm in ['serial','overlap'],'explicit source-pinned arm required')
    copies=sorted([e for e in events if e['name'].startswith('zeCommandListAppendMemoryCopy')
                   and e['name'].endswith('[1971200]')],key=lambda e:e['begin'])
    kernels=sorted([e for e in events if 'DenseKernel' in e['name']],key=lambda e:e['begin'])
    require(len(copies)==513 and len(kernels)==514,'exact warmup plus512timed copy/kernel calls required')
    require(all(e['end']>e['begin'] for e in copies+kernels),'positive transfer/kernel intervals required')
    copies,kernels=copies[1:],kernels[2:]
    tolerance=2000 # ns; predeclared inter-engine timestamp edge tolerance, reported below.
    margins=[]
    for i,(copy,kernel) in enumerate(zip(copies,kernels)):
        if i:
            margins += [copy['begin']-copies[i-1]['end'],kernel['begin']-kernels[i-1]['end'],
                        copy['begin']-max(copies[i-1]['end'],kernels[i-1]['end'])]
        if arm=='serial':margins.append(kernel['begin']-copy['end'])
    require(all(x>=-tolerance for x in margins),'observed stream/readiness/one-lookahead dependency violation')
    c=[(e['begin'],e['end']) for e in copies];k=[(e['begin'],e['end']) for e in kernels]
    cu,ku,all_u=trace_reader.union_ns(c),trace_reader.union_ns(k),trace_reader.union_ns(c+k)
    return dict(timed_copy_calls=512,timed_dense_calls=512,requested_timed_H2D_bytes=1009254400,
        copy_union_ns=cu,kernel_union_ns=ku,copy_kernel_overlap_ns=cu+ku-all_u,
        useful_GPU_union_ns=all_u,timed_device_span_ns=max(c[-1][1],k[-1][1])-min(c[0][0],k[0][0]),
        minimum_dependency_margin_ns=min(margins),dependency_tolerance_ns=tolerance,
        actual_operation_counts_and_one_lookahead_checked=True,
        limits='Requested payload and device intervals,not physical bus/VRAM traffic,saturation or model throughput.')


def compact(row):
    result={k:v for k,v in row.items() if k not in ['project','native_proof','tools_proof','commands']}
    result['project']={k:v for k,v in row['project'].items() if k!='source_sha256'}
    result['native_proof_sha256']=signature(row['native_proof']);result['tools_proof_sha256']=signature(row['tools_proof'])
    result['cases']=[{k:v for k,v in x.items() if k!='environment'} |
        dict(environment_sha256=signature(x['environment'])) for x in row['cases']]
    return result


def save(row):
    write(ROOT/'record.json',row);data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def check(row,committed=True):
    if committed:require(signature(compact(row))==signature(ur_validation.committed(RESULT,(KEY,))),
        'commit exact typed profile plan/stage')
    pin=component.original.coherent.build.common.source.pin(P,PREFIXES)
    require(pin['source_sha256']==row['project']['source_sha256'],'frozen profile controller/host inputs changed')
    require(all(sha(image['path'])==image['sha256'] for image in row['interpreters'].values()),
        'profile controller/host interpreter changed')
    basis,proof=native.verify(expected=row['native_proof'])
    require(signature(collector_proof())==signature(row['tools_proof']),'collector source/assets changed')
    require(row['boot']==basis['boot'] and row['driver']==basis['driver'],'profile/native boot or driver differs')
    for case in row['cases']:
        args=[case['mode']] if case['kind']=='safety' else [case['arm'],str(case['rows']),'512']
        target=[basis['binaries'][case['kind']]['path'],*args]
        expected=([row['tools_proof']['artifacts']['unitrace']['path'],'--device-timing',
            '--chrome-device-logging','--chrome-event-buffer-size','128',*target] if case['profiled'] else target)
        require(case['argv']==expected,'profile launch differs from exact guarded images/flags')
        env=case['environment'];require(env['ZES_ENABLE_SYSMAN']=='0' and env['EnableDirectSubmission']=='0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and env['NEOReadDebugKeys']=='1'
            and env['ZE_ENABLE_HANDLE_LIFETIME']=='0' and 'PMI_RANK' not in env and 'PMIX_RANK' not in env,
            'qualified immutable non-MPI safety environment required')
    return basis


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'existing profile plan;inspect,no retry')
    previous=ur_validation.committed(RESULT,('prefill_upload_overlap_profile_v1',))
    require(previous['stage']=='failed' and previous['ownership_closed'] and not previous['active']
        and previous['failure_observation']['native_exit_code']==0
        and previous['failure_observation']['host_red_result']['passed_expected_red'] is True,
        'preserve original failed host-flow assumption and observed-schema red control')
    basis,proof=native.verify();tools=collector_proof()
    project=component.original.coherent.build.common.source.pin(P,PREFIXES)
    cases=[]
    def add(label,kind,profiled,mode=None,arm=None,rows=None):
        root=ROOT/label
        reference=basis['cases'][0 if kind=='safety' else 11]
        env=dict(reference['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
        if kind=='safety':env['UR_LOG_TRACING']='level:info;flush:info;output:file,'+str(root/'ur.trace')
        argv=[basis['binaries'][kind]['path'],*([mode] if kind=='safety' else [arm,str(rows),'512'])]
        env=engine_once.native_environment(basis,env)
        if profiled:
            env['STRATA_UNITRACE_TARGET_PRELOAD']=env.pop('LD_PRELOAD')
            env.update(UNITRACE_SystemTime='1',UNITRACE_FollowChildProcess='1',UNITRACE_TraceOutputDir=str(root/'trace'))
            argv=[tools['artifacts']['unitrace']['path'],'--device-timing','--chrome-device-logging',
                  '--chrome-event-buffer-size','128',*argv]
        cases.append(dict(label=label,kind=kind,profiled=profiled,root=str(root),argv=argv,
                          environment=env,mode=mode,arm=arm,rows=rows))
    for mode in EXPECTED_SAFETY:add('interop-'+mode,'safety',True,mode=mode)
    for rows in [1024,4096]:
        for arm in ['serial','overlap']:
            for i,observed in enumerate([False,True,True,False]):
                add(f'{rows}-{arm}-{i}-'+('profiled' if observed else 'clean'),'benchmark',observed,arm=arm,rows=rows)
    ROOT.mkdir(mode=0o700);(ROOT/'owned').mkdir(mode=0o700)
    row=dict(stage='prepared',project=project,native_proof=proof,tools_proof=tools,
        interpreters={key:dict(path=str(Path(path).resolve()),sha256=sha(path))
            for key,path in [('controller',sys.executable),('host','/usr/bin/python3')]},
        boot=basis['boot'],driver=basis['driver'],cases=cases,results=[],commands=[],
        active=False,ownership_closed=True,passed=False,complete=False,
        prior_failure=dict(key='prefill_upload_overlap_profile_v1',outcome='Native forward normal0/no fault;reader incorrectly required host API flow source in device-only capture. Original failed cohort and red test remain failed/expected-red,respectively.',
            new_contract='GPUside H2Dt/D2Hs markers plus native-PID suffix and HOSTmetadata,under absent PMI/PMIXrank;no host API endpoints inferred. Same original native images/collector/bytes/math and E2speed gates.'),
        interoperability_qualified=False,observer_overhead_qualified=False,model_trial_allowed=False,
        host_command=['/usr/bin/python3','-B','-m','unittest','discover','-s','tests/core','-p','test_upload_overlap*.py','-v'],
        environment_recipe=dict(native_evidence=native.EVIDENCE,source=native.NATIVE,
            safety_case=0,benchmark_case=11,collector_flags=['--device-timing','--chrome-device-logging','--chrome-event-buffer-size','128'],
            safety_UR_full_trace=True,benchmark_UR_full_trace=False),
        plan=dict(question='P1:does the qualified device-only PTI collector preserve the actual Runtime event bridge? P2:which device intervals explain E2wait-order wall gain and variability?',
            opportunity='E2M4096ratios0.575224/0.585516/0.991001pass its component screen;M1024ratios0.668329/0.565964/1.003774fail. These outcomes remain immutable. Native/provider/recipe reuse is historical admission,not a new model/overlap claim.',
            units='No native,math,placement,copy primitive or readiness-order code changes. P1collector/event interoperation is a separate safety prerequisite. P2changes only device collector on/off for fixed serial or overlap arms and fixed shape. Compare each shape/arm independently;no combination or model claim.',
            order='Commit source and plan,host historical/trace controls,seven fresh profiled E1bprocesses metadata/forward/reverse/permanent/cancel/cancel-throw/refuse,then C/T/T/Cper serial and overlap arm at1024and4096rows. One child per next,commit each closure;no reordering,retries or exclusions.',
            safety='Original six host admission methods and E2seven-case safety already qualified exact6231/824eimages. Ten new host controls include the observed device-only schema,missing/foreign/duplicate marker refusal,historical eligibility and trace count/dependency interpretation. P1full UR/L0validation traces;required before/after provider+actual non-MPIprocess environment/argv+private first-init proof;50msheld/manual release and original100mssubmit assertions. Every owner normal0/closed,no warning/error,new fault or changed boot/driver.',
            counts='P1URenqueue counts0/10/12/16/7/7/6;reader kernels0/2/4/6/1/1/1;raw256Bcopies0/1/2/2/1/1/1. P2each trace exactly513payload1971200Bcopies and514DenseKernelcalls;exclude first copy and first two dense warmups. Pinned PTIdevice-only format associates each unique GPUwork ID with one GPUside H2Dt and one D2Hs marker,matching lane,id,timestamp and native-PID category suffix with HOSTmetadata. This is not a host API/enqueue endpoint join. All device intervals/markers must fit actual owned CLOCK_MONOTONIC_RAWlifetime.',
            timing='Native512epoch wall/100msfloor/clock-pair<=0.001batch unchanged. Report all clean/profiled samples. Per shape/arm,match T1/C0andT2/C3;each native wall ratio must lie0.9..1.1 for qualified quantitative attribution. A failed observer gate retains perturbed observations and never qualifies speed or causality. Finish planned controls despite slow samples;stop on safety/identity/trace/coverage failures.',
            dependencies='Order copies and dense kernels by their actual single-stream device starts. Every next copy follows both previous copy and previous dense;every next dense follows previous dense;serial dense follows same-epoch copy. Predeclared timestamp edge tolerance2000ns;report minimum margin and fail larger violations. Intersections/unions remain different from native wall and arithmetic saturation.',
            budgets='120s wall/240sCPU/4GiB RSS per native;32MiB text and individual file;512MiB total campaign across controller-owned root. Each trace<=32MiB/50000events. No metrics,counters,Sysman,host API trace,pause/shared session,model or new binaries.',
            stop='First source/build/typed-plan/provider/env/argv,numerical/guard/validation,ownership,deadline,new-fault,clock or executed-count/dependency failure closes cohort failed;no reset,foreign process action or refreshed hashes.',
            provenance='Main-authored controller/reader/tests under repository LICENSE;exact independent MIT PTI4b6d742collector/source and owned launcher,plus original owned14508math. No XeStrata implementation/design source.',
            retention='Same two current E2images and925compiler25link7objectcensus;no new generation. Keep useful samples/counts/union/overlap/dependency margins and qualification decisions,not normal trace/maps/logs. Retire normal output at closure. Review E2generation at this qualification/attribution decision;preserve26phase reference images and separate legacy holds.',
            limits='No revised1024speed gate,model PP/TG/TTFT,ordinary RAMsource views,production policy,quality,decode handoff,physical262144/RESTORE,bus bytes/VRAMtraffic,saturation or inference adoption.'))
    save(row);print('Prepared separate collector/event safety and resource-attribution plan;commit before host/native stages.',flush=True)


def maps(pid,row,case,basis):
    allowed=json.loads(Path(basis['runtime_manifest']['path']).read_text())
    images=[basis['binaries'][case['kind']],basis['validation_provider'],*basis['native_validation_probe'].values()]
    if case['profiled']:images.append(row['tools_proof']['artifacts']['libunitrace_tool.so'])
    allowed.update({str(Path(v['path']).resolve()):v['sha256'] for v in images})
    allowed={str(Path(p).resolve()):h for p,h in allowed.items()};found={}
    for line in Path('/proc',str(pid),'maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and 'x' in fields[1] and fields[5].startswith('/'):
            path=str(Path(fields[5]).resolve());require(allowed.get(path)==sha(path),'unknown/changed actual provider:'+path);found[path]=allowed[path]
    required=[basis['binaries'][case['kind']]['path'],*map(lambda x:x['path'],basis['native_validation_probe'].values())]
    if case['profiled']:required.append(row['tools_proof']['artifacts']['libunitrace_tool.so']['path'])
    require(all(str(Path(path).resolve()) in found for path in required),'required image/preload/collector absent')
    return dict(files=len(found),sha256=signature(found),paths=found)


def expected_environment(case,row,basis):
    env=dict(case['environment'])
    if case['profiled']:
        preload=basis['native_validation_probe']['library']['path']
        env.update(LD_PRELOAD=row['tools_proof']['artifacts']['libunitrace_tool.so']['path']+':'+preload,
            ZE_ENABLE_TRACING_LAYER='1',INTEL_LIBITTNOTIFY64='libunitrace_tool.so',
            UNITRACE_VERSION='2.4.1 (4b6d74294cdc72766c00a9211fe2e4261ee94244)',
            UNITRACE_LD_PRELOAD_OLD=preload,UNITRACE_DeviceTiming='1',
            UNITRACE_ChromeDeviceLogging='1',UNITRACE_ChromeEventBufferSize='128')
    return env


def execute(action):
    row=json.loads((ROOT/'record.json').read_text());basis=check(row)
    require(row['stage'] in (['prepared'] if action=='host' else ['host-qualified','testing']),'fresh ordered profiling stage required')
    supervisor=component.owner_api.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',supervisor.commands));cursor=None
    try:
        if action=='host':
            env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
            entry,out,err=supervisor.run('host',row['host_command'],env,P,**HOST)
            require(component.owner_api.closed(entry) and entry['exit_code']==0 and 'Ran 10 tests' in err.read_text()
                and '\nOK\n' in err.read_text(),'all ten historical/trace host controls required:'+str(err))
            row.update(stage='host-qualified',passed=True,host_controls_passed=True)
        else:
            index=len(row['results']);require(index<len(row['cases']),'profile campaign already closed')
            case=row['cases'][index];root=Path(case['root']);require(not root.exists(),'existing profile child;inspect receipt/handle,no retry')
            root.mkdir(mode=0o700)
            for name in ['tmp','cache','trace']:(root/name).mkdir(mode=0o700)
            journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
            cursor=journal.split('-- cursor: ',1)[1].strip();row.update(stage='testing',active=True,ownership_closed=False);save(row)
            gates=[];records=[];provider_counts=[]
            expected=['before-runtime','runtime-ready']+(['buffers-ready'] if case['mode']!='metadata' else [])+['before-exit'] if case['kind']=='safety' else ['before-runtime','runtime-ready','prewarm-ready','before-timing','before-exit']
            def interact(channel,_entry):
                while True:
                    item=json.loads(channel.line(45))
                    if item['kind']=='gate':
                        name=item['name'];require(name==expected[len(gates)],'profile gate sequence differs')
                        actual=dict(v.split('=',1) for v in Path('/proc',str(channel.proc.pid),'environ').read_text().strip('\0').split('\0'))
                        wanted=expected_environment(case,row,basis)
                        require(all(actual.get(k)==v for k,v in wanted.items()) and set(actual)-set(wanted)<= {'PWD','OLDPWD','SHLVL','_'},'actual immutable native environment differs')
                        target=[basis['binaries'][case['kind']]['path'],*([case['mode']] if case['kind']=='safety' else [case['arm'],str(case['rows']),'512'])]
                        require(Path('/proc',str(channel.proc.pid),'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')==target,'actual native argv/identity differs')
                        mapped=maps(channel.proc.pid,row,case,basis)
                        if name!='before-runtime':
                            needed=[component.event.LOADER,case['environment']['ZE_ENABLE_ALT_DRIVERS'],case['environment']['UR_ADAPTERS_FORCE_LOAD'],'/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so']
                            if case['kind']=='safety':needed.append(component.event.VALIDATION)
                            require(all(str(Path(p).resolve()) in mapped['paths'] for p in needed),'required initialized provider absent')
                        provider_counts.append(dict(gate=name,files=mapped['files'],sha256=mapped['sha256']));gates.append(name);channel.send(b'continue\n')
                    else:
                        records.append(item)
                        if item['kind']=='done':channel.exit(30);break
            begin=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            entry,out,err=supervisor.run(case['label'],case['argv'],case['environment'],P,interaction=interact,**BUDGET)
            end=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            require(component.owner_api.closed(entry) and entry['exit_code']==0,'profiled native child failed:'+str(err))
            first_init=ur_validation.preload_evidence(err.read_text(),entry['pid'])
            require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size,'UR validation output')
            require(not re.search(r'\[(?:error|critical)\]|upload (?:lane|overlap) failure:|Strata SYCL cleanup:',err.read_text(),re.I),'collector/ZEvalidation/cleanup error')
            require(gates==expected,'incomplete profile gates')
            result=dict(label=case['label'],kind=case['kind'],profiled=case['profiled'],passed=True,native_exit_code=0,ownership_closed=True,
                providers=provider_counts,first_init=first_init,raw_lifetime=dict(begin=begin,end=end))
            observed=[]
            if case['profiled']:
                files=list((root/'trace').glob('*.json'))
                if case['mode']=='metadata' and not files:
                    result.update(no_application_trace_emitted=True,GPU_events=0)
                else:
                    require(len(files)==1 and files[0].stat().st_size<=32<<20,'one bounded trace required')
                    observed=trace_events(trace_reader.load_json(files[0]),entry['pid'],begin,end)
                    result.update(trace_sha256=sha(files[0]),GPU_events=len(observed))
            if case['kind']=='safety':
                require(records==[dict(kind='done',passed=True,case=case['mode'])],'incomplete collector/event safety result')
                enqueues=re.findall(r'---> (urEnqueue\w+)',(root/'ur.trace').read_text())
                kernels=[e for e in observed if 'Memory' in e['name'] and 'reader' in e['name']]
                copies=[e for e in observed if e['name'].startswith('zeCommandListAppendMemoryCopy') and e['name'].endswith('[256]')]
                require((len(enqueues),len(kernels),len(copies))==EXPECTED_SAFETY[case['mode']],'exact safety application/kernel/copy counts differ')
                result.update(mode=case['mode'],application_enqueues=len(enqueues),reader_kernels=len(kernels),raw_copies=len(copies))
            else:
                require(len(records)==2 and records[0]['kind']=='batch' and records[1]==dict(kind='done',passed=True),'complete unchanged benchmark result required')
                batch=records[0];require(batch['arm']==case['arm'] and batch['gemm_rows']==case['rows'] and batch['iterations']==512
                    and batch['gemm_n']==1280 and batch['gemm_k']==2560 and batch['requested_H2D_bytes']==1009254400
                    and batch['lookahead']==1 and batch['oracle_and_guards'] is True and batch['wall_s']>=.1
                    and 0<=batch['timer_pair_max_s']<=batch['wall_s']*.001,'actual unchanged batch differs or is ineligible')
                result.update(rows=case['rows'],arm=case['arm'],batch=batch)
                if case['profiled']:result['attribution']=benchmark_attribution(observed,case['arm'])
            check(row,False);row['results'].append(result);row.update(passed=True,active=False,ownership_closed=True)
            if len(row['results'])==7:row['interoperability_qualified']=True
            if len(row['results'])==len(row['cases']):
                decision={}
                for rows in [1024,4096]:
                    for arm in ['serial','overlap']:
                        samples=[x for x in row['results'] if x.get('rows')==rows and x.get('arm')==arm]
                        ratios=[samples[1]['batch']['wall_s']/samples[0]['batch']['wall_s'],samples[2]['batch']['wall_s']/samples[3]['batch']['wall_s']]
                        decision[f'{rows}-{arm}']=dict(profile_over_clean_wall_ratios=ratios,observer_overhead_qualified=all(.9<=x<=1.1 for x in ratios))
                row.update(stage='closed',complete=True,decision=decision,
                    observer_overhead_qualified=all(x['observer_overhead_qualified'] for x in decision.values()))
    except BaseException as error:row.update(stage='failed',passed=False,complete=False,error=repr(error));raise
    finally:
        row['commands'].extend(supervisor.commands);row['active']=supervisor.active is not None
        row['ownership_closed']=supervisor.active is None and all(component.owner_api.closed(c) for c in row['commands'])
        if cursor:
            try:
                after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                row['fresh_faults']=[s for s in after.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)',s,re.I)]
                row['boot_unchanged']=row['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip();row['driver_after']=component.guards.driver_guard()
                if row['fresh_faults'] or not row['boot_unchanged'] or row['driver_after']!=row['driver']:row.update(stage='failed',passed=False,complete=False)
            except BaseException as error:row.update(stage='failed',passed=False,complete=False,final_health_error=repr(error))
        if not row['passed'] and len(row['results'])<=7:row['interoperability_qualified']=False
        save(row)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','complete','ownership_closed','interoperability_qualified','observer_overhead_qualified','model_trial_allowed']},sort_keys=True),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['host'],['next']],'prepare|host|next')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);component.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:raise SystemExit(execute(sys.argv[1]))
