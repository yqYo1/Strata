#!/usr/bin/env python3
"""Own one official long-input request with a qualified external GPU recorder."""
import collections
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode=True
import profile_window as window
import prefill_status as status

tool=window.tool
P,B,RESULT=window.P,window.B,window.RESULT
c=status.c
gen,assets=c.gen,c.assets
ROOT=B/'unitrace-production-32k-v1'


def sha(path):return window.file_sha(path)


def inputs(revision):
    return dict(gen.benchmark_pins(revision),**window.inputs(revision))


def qualification(container):
    record=container['unitrace_recording_window']
    tool.guards.require(record['passed'] and record['ownership_closed'] and not record['active']
        and record['request_window_qualified'] and record['fresh_faults']==[] and record['boot_unchanged'],
        'qualified small recording window required')
    window.admission(record,container)
    build,proof,preload=status.qualification()
    tool.guards.require(build['binary']==record['assets']['native_binary'],'unchanged qualified native required')
    return build,proof,preload,record['assets']


def current(record,container,full_models=False):
    build,proof,preload,pinned=qualification(container)
    tool.guards.require(record['binary']==build['binary'] and record['assets']==pinned
        and record['native_validation_probe']==preload and record['qualification']==proof,'profile native/tool evidence changed')
    pins=inputs(record['source_commit'])
    tool.guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['source_fingerprint'],
                        'profile benchmark/controller source changed')
    original,ref=status.original_plan()
    tool.guards.require(ref==record['original_configuration'],'original model/configuration reference changed')
    gen.runtime_assets_unchanged(dict(mode='engine-compare',runtime_asset_delta=original['runtime_asset_delta']))
    assets.model_pins(record['config'],original['model_assets'],full=full_models)
    for path,expected in record['generated_inputs'].items():tool.guards.require(sha(path)==expected,'generated profile input changed: '+path)
    for path,expected in original['tokenizer_assets'].items():tool.guards.require(sha(path)==expected,'tokenizer changed')
    tool.guards.require(assets.machine_identity()==record['machine_identity'],'model platform changed')


def profile(pinned,root,session):
    return dict(launcher=pinned['launcher'],collector=pinned['collector'],session=session,
        environment=dict(UNITRACE_SystemTime='1',UNITRACE_FollowChildProcess='1',UNITRACE_TraceOutputDir=str(root/'trace'),LD_DEBUG='libs'))


def prepare(container):
    tool.guards.require('unitrace_production_request' not in container and not ROOT.exists(),'inspect existing request; never blind retry')
    build,proof,preload,pinned=qualification(container)
    original,ref=status.original_plan();revision=tool.git(P,'rev-parse','HEAD').decode().strip();pins=inputs(revision)
    config=copy.deepcopy(original['config']);root_old=original['root']
    config['args'][config['args'].index('--expert-profile')+1]=str(ROOT/'expert-profile.bin')
    config.update(log=str(ROOT/'engine.stderr'),env={})
    env={k:v.replace(root_old,str(ROOT)) for k,v in original['environment'].items()}
    env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',UR_LOG_TRACING='level:warning;flush:warning;output:stderr',
        CUDA_MODULE_LOADING='EAGER',STRATA_MEASUREMENT_ROOT=str(ROOT),STRATA_TRACE='1')
    tool.guards.require('LD_PRELOAD' not in env and env['ZES_ENABLE_SYSMAN']=='0','target-only validation/Sysman-off required')
    settings=json.loads((P/'tools/sycl/measurement_environment.json').read_text())
    for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        tool.guards.require(env[name]==settings['environment'][name],'qualified safety setting changed')
    assets.model_pins(config,original['model_assets'],full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','trace','preflight'):(ROOT/name).mkdir()
    for name in ('tmp','cache','trace'):(ROOT/'preflight'/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',ROOT/'expert-profile.bin')
    tool.guards.require(sha(ROOT/'expert-profile.bin')==original['profile_sha256'],'expert profile differs')
    session='StrataModel'+secrets.token_hex(12);pre_session='StrataModelCPU'+secrets.token_hex(12)
    native=dict(binary=build['binary'],native_validation_probe=preload,native_profile=profile(pinned,ROOT,session))
    preflight=dict(binary=pinned['canary'],native_validation_probe=preload,native_profile=profile(pinned,ROOT/'preflight',pre_session))
    generated={}
    for path,value in ((ROOT/'config.json',config),(ROOT/'plan.json',native),(ROOT/'preflight/plan.json',preflight)):
        path.write_text(json.dumps(value,indent=2)+'\n');generated[str(path)]=sha(path)
    generated[str(ROOT/'expert-profile.bin')]=sha(ROOT/'expert-profile.bin')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
        '--prefill-only','--prefill-tokens','32768','--think','--runs','1','--workload-id','unitrace_32k_20261021',
        '--profile-request-gates','--fail-fast-engine-error','--ready-timeout','600','--out',str(ROOT/'bench')]
    record=dict(schema='strata-unitrace-production-request-v1',stage='prepared',source_commit=revision,
        build_source_commit=build['source_commit'],source_count=len(pins),source_fingerprint=assets.fingerprint(pins),
        binary=build['binary'],qualification=proof,native_validation_probe=preload,assets=pinned,
        original_configuration=ref,machine_identity=assets.machine_identity(),config=config,native_plan=native,
        preflight_plan=preflight,environment=env,command=command,generated_inputs=generated,
        gates='First CPU-only canary through updated engine_once with same private validation/collector prefix: constructor/first-init order, native/tool bytes, owned closure, exact-session cleanup and no fault. Commit that result before one bench.sh process. Start recorder paused through initialization and one short thinking warmup. Benchmark long-request gates synchronously resume before HTTP and pause after complete stream. Exactly one >=32768 fresh/read input,24-or-fewer thinking tokens, successful official/journal sample, same-native identity/namespace/PCI device, finite positive GPU services within controlled RAW window, unique IDs/host sources, source/model/runtime checks and normal target/control/session closure. Stop on first failure; no retry.',
        limits='Observed whole long request includes prefill, restoration, final prompt forward and up to24 generated tokens. First HTTP stream output bounds a prefill-dominated prefix, not exact native prefill end. Service sums/unions are not arithmetic, bus/GDDR saturation or a complete graph critical path. Observer timing is not clean speed or quality/decode/full-context adoption evidence.',
        retention='Main owns one complete request timeline only until family/service/copy/submission attribution and next matched-reference selection;256MiB timeline budget for tens of thousands of production commands. Keep compact useful samples/conservation/limits; retire routine raw/transactions. No new engine generation.',
        budgets=dict(wall=1800,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=256<<20,total_cap=512<<20),
        active=False,passed=False,ownership_closed=True,model_qualified=False,hardware_limiting_resource_known=False)
    container['unitrace_production_request']=record;RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(stage='prepared',minimum_input=32768,source_files=len(pins),new_model_generation=False)))


def session_path(session):return Path('/dev/shm/uctrl'+str(os.getuid())+'_'+session)


def clock_pair():
    before=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW);mono=time.monotonic_ns();after=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return dict(raw_before_ns=before,monotonic_ns=mono,raw_after_ns=after)


def capture_summary(record):
    files=list((ROOT/'trace').glob('*.json'));tool.guards.require(len(files)==1,'one complete production timeline required')
    path=files[0];value=json.loads(path.read_text());events=value['traceEvents']
    gpu=[e for e in events if e.get('ph')=='X' and e.get('cat')=='gpu_op'];host=[e for e in events if e.get('ph')=='X' and e.get('cat')=='cpu_op']
    tool.guards.require(gpu and host,'production GPU/host timing missing')
    native=record['native_identity']['pid'];sources=collections.Counter(str(e.get('id','')) for e in events
        if e.get('pid')==native and e.get('ph')=='s' and e.get('cat','').startswith('Flow_H2D_'))
    ids=[];names={};kinds=collections.Counter();copies=collections.Counter()
    begin,end=record['request_raw_begin_ns']/1000,record['request_raw_end_ns']/1000
    for event in gpu:
        tool.guards.require(all(isinstance(event.get(k),(int,float)) and math.isfinite(event[k]) for k in ('ts','dur'))
            and event['dur']>=0 and begin-1000<=event['ts']<=event['ts']+event['dur']<=end+1000,'invalid/out-of-window GPU interval')
        identifier=str(event.get('args',{}).get('id',''));ids.append(identifier)
        tool.guards.require(identifier and sources[identifier]==1,'missing production host source')
        name=event['name'];kind='copy' if name.startswith('zeCommandListAppendMemoryCopy') else 'fill' if name.startswith('zeCommandListAppendMemoryFill') else 'barrier' if name.startswith('zeCommandListAppendBarrier') else 'other' if name.startswith('zeCommandList') else 'kernel'
        kinds[kind]+=1
        if kind=='kernel':tool.guards.require(event['dur']>0,'nonpositive kernel service')
        if kind in ('copy','fill'):
            size=re.search(r'\[(\d+)\]$',name);tool.guards.require(size is not None,'unclassified requested bytes')
            copies[name.rsplit('[',1)[0]]+=int(size[1])
        row=names.setdefault(name,dict(count=0,sum_us=0.0,kind=kind));row['count']+=1;row['sum_us']+=event['dur']
    tool.guards.require(len(set(ids))==len(ids),'duplicate production GPU IDs')
    labels={e['pid']:e.get('args',{}).get('name','') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    lanes={str(p):labels.get(p,'') for p in {e['pid'] for e in gpu}}
    tool.guards.require(all(name.startswith('DEVICE<') and '] 0:5:0:0 #' in name for name in lanes.values()),'unexpected production PCI lane')
    dominant=sorted(names.items(),key=lambda pair:pair[1]['sum_us'],reverse=True)[:20]
    return dict(timeline_sha256=sha(path),timeline_bytes=path.stat().st_size,gpu_events=len(gpu),host_events=len(host),
        operation_counts=dict(kinds),requested_copy_fill_bytes=dict(copies),device_lanes=lanes,
        gpu_union_us=window.production.union_us(gpu),host_api_union_us=window.production.union_us(host),
        controlled_request_wall_us=end-begin,
        largest_service_names=[dict(name=name,**row) for name,row in dominant],
        raw_window_containment=True,unique_gpu_ids=True,all_host_sources=True,
        limits='Name service sums can overlap. Whole controlled request includes short decode. No hardware saturation, exact prefill-only extent or full graph dependency coverage claim.')


def execute(container,record,probe_only):
    current(record,container,full_models=not probe_only)
    tool.guards.require(record['stage']==('prepared' if probe_only else 'preflight-passed'),'ordered fresh preflight/model actions required')
    if not probe_only:tool.guards.require(record['preflight']['passed'],'CPU wrapper preflight must pass')
    root=ROOT/'preflight' if probe_only else ROOT;plan=record['preflight_plan'] if probe_only else record['native_plan']
    session=plan['native_profile']['session'];tool.guards.require(not session_path(session).exists(),'preexisting exact session; never overwrite')
    env=dict(record['environment'],STRATA_MEASUREMENT_ROOT=str(root),TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();driver=tool.guards.driver_guard()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(line.split(': ',1)[1] for line in journal.splitlines() if line.startswith('-- cursor: '));owner=None;controls=None;entry=None
    observation=dict(passed=False,session_removed=False,controls=[])
    def persist():
        record['active']=any(o and o.active is not None for o in (owner,controls))
        (root/'record.json').write_text(json.dumps(dict(record,observation=observation,
            owned_commands=owner.commands if owner else [],control_commands=controls.commands if controls else []),indent=2)+'\n')
    owner=tool.supervision.Owner(root/'owned',persist);controls=tool.supervision.Owner(root/'controls',persist)
    def control(action,expected=None):
        before=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        result,_,_=controls.run(action,[record['assets']['launcher']['path'],'--'+action,session],env,root,
            wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=256<<20,total_cap=512<<20)
        tool.guards.require(tool.supervision.closed(result) and result.get('exit_code')==0,'session control failed')
        if expected is not None:window.state(session_path(session),expected)
        observation['controls'].append(dict(action=action,raw_before_ns=before,raw_after_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
    def interact(io,leader):
        for expected in ('begin','end'):
            while True:
                line=io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '):break
            gate=json.loads(line.removeprefix('STRATA_PROFILE_GATE '))
            tool.guards.require(gate==dict(stage=expected,phase='prefill-only',run=0),'unexpected long-request gate')
            start=json.loads((ROOT/'engine-start.json').read_text());native=start['native'];identity=tool.supervision.identity(native['pid'])
            tool.guards.require(identity and identity['start_ticks']==native['start_ticks'] and identity['sid']==leader['pid']
                and Path('/proc',str(native['pid']),'exe').resolve()==Path(record['binary']['path']).resolve(),'native owned exec identity differs')
            namespace=os.readlink('/proc/'+str(native['pid'])+'/ns/time')
            tool.guards.require(namespace==os.readlink('/proc/self/ns/time'),'native/caller time namespaces differ')
            if expected=='begin':
                window.state(session_path(session),1);control('resume',0)
                record.update(native_identity=native,time_namespace=namespace,clock_begin=clock_pair(),
                              request_raw_begin_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
            else:
                record.update(request_raw_end_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),clock_end=clock_pair())
                control('pause',1)
            io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();tool.guards.require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    record['stage']='preflight-running' if probe_only else 'running';persist()
    try:
        command=[sys.executable,'-B',str(P/'tools/sycl/engine_once.py')] if probe_only else record['command']
        budgets=dict(record['budgets'],wall=90,cpu=90,rss_cap=2<<30) if probe_only else record['budgets']
        entry,out,err=owner.run('cpu-wrapper' if probe_only else 'bench',command,env,P,
            interaction=None if probe_only else interact,**budgets)
        tool.guards.require(tool.supervision.closed(entry) and entry.get('exit_code')==0,'owned wrapper/benchmark failed')
        native=json.loads((root/'engine-exit.json').read_text());tool.guards.require(native['exit_code']==0 and native['native_profile']==plan['native_profile'],
            'native exit/profile claim mismatch')
        stderr=err.read_text() if probe_only else (ROOT/'engine.stderr').read_text()
        activation=window.probe.validation.preload_evidence(stderr,native['native']['pid'])
        tool.guards.require(activation['loader']==record['native_validation_probe']['loader']['path'],'native validation provider changed')
        observation['activation']=activation
        if probe_only:
            body=json.loads(out.read_text());tool.guards.require(body==dict(pid=native['native']['pid'],initialized=0,checked=46,expected=46,version=4294967295,closed=0),'CPU canary failed')
            def marker(phase,path):
                found=[m.start() for m in re.finditer(r'calling '+phase+r':\s+([^\s]+)',stderr) if Path(m[1]).resolve()==Path(path).resolve()]
                tool.guards.require(len(found)==1,'ambiguous CPU wrapper loader marker');return found[0]
            tool.guards.require(marker('init',record['native_validation_probe']['library']['path'])<stderr.index('phase=before-main')
                <marker('init',record['assets']['collector']['path'])<stderr.index('strata UR canary begin')
                <stderr.index('strata UR canary end')<marker('fini',record['assets']['collector']['path']),'CPU wrapper constructor/retention order differs')
            observation['canary']=body
        else:
            results=list((ROOT/'bench').glob('*/results.json'));tool.guards.require(len(results)==1,'one official result required')
            result=json.loads(results[0].read_text());samples=result['results']['prefill_only']['requests']
            tool.guards.require(result['complete'] and result['params']['prefill_tokens']==32768 and result['params']['think'] is True
                and len(samples)==1 and not result['results']['failures'],'official32K suite failed')
            sample=samples[0]
            tool.guards.require(all(sample.get(k) is True for k in ('ok','eligible','stream_complete','thinking_requested','reasoning_observed'))
                and sample['actual_prompt_tokens']==sample['actual_read_tokens']>=32768 and sample['actual_reused_tokens']==0
                and 1<=sample['generated_tokens']<=24,'long request extent/completion differs')
            journal_evidence,_=gen.journal_evidence(ROOT);tool.guards.require(journal_evidence['valid'] and journal_evidence['complete'],'request journal failed')
            tool.guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',stderr),'native runtime error')
            assets.verify_pool(stderr,5);record.update(request=sample,journal=journal_evidence,capture=capture_summary(record))
        current(record,container,full_models=not probe_only);observation['passed']=True
    except BaseException as error:record.update(error=repr(error));observation['error']=repr(error)
    finally:
        try:
            if window.ended(entry) and owner.active is None and controls.active is None and all(window.ended(e) for e in controls.commands):
                if session_path(session).exists():control('stop')
                observation['session_removed']=not session_path(session).exists()
            else:observation['cleanup_pending']='Process closure unproven; leave exact session untouched.'
        except Exception as error:observation['cleanup_error']=repr(error)
        record['ownership_closed']=all(o.active is None and all(window.ended(e) for e in o.commands) for o in (owner,controls))
        try:observation.update(window.production.fault_evidence(cursor,boot))
        except Exception as error:observation['final_probe_error']=repr(error)
        observation['passed']=bool(observation['passed'] and record['ownership_closed'] and observation['session_removed']
            and observation.get('fresh_faults')==[] and observation.get('boot_unchanged') and observation.get('driver_after')==driver
            and 'final_probe_error' not in observation)
        record['preflight' if probe_only else 'model_observation']=observation
        record.update(stage='preflight-passed' if probe_only and observation['passed'] else 'closed' if observation['passed'] else 'failed',
                      passed=bool(not probe_only and observation['passed']))
        persist();container['unitrace_production_request']=record;RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(stage=record['stage'],passed=observation['passed'],ownership_closed=record['ownership_closed'],error=record.get('error'))))
    return 0 if observation['passed'] else 1


def main():
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','probe','run'),'use profile_model.py prepare|probe|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tool.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container);return 0
        tool.guards.require(RESULT.read_bytes()==tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),'commit exact model profile plan/result before action')
        return execute(container,container['unitrace_production_request'],sys.argv[1]=='probe')


if __name__=='__main__':raise SystemExit(main())
