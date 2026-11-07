#!/usr/bin/env python3
"""Main-owned coverage admission for the current production profiling tool."""
import collections
import fcntl
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import unitrace_probe as probe

tool = probe.tool
P, B, RESULT = tool.P, tool.B, tool.RESULT
ROOT = B/'unitrace-known-work-v2'


def inputs(revision):
    pins = probe.inputs(revision)
    name = str(Path(__file__).resolve().relative_to(P))
    data = tool.git(P,'show',revision+':'+name)
    tool.guards.require(data==(P/name).read_bytes(),'commit production controller first')
    pins[name]=tool.sha(data)
    return pins


def assets(container):
    loader=container['unitrace_loader_probe']
    tool.guards.require(loader['passed'] and loader['complete'] and not loader['active'] and loader['ownership_closed']
        and loader['fresh_faults']==[] and loader['session_removed'] and loader['loader_order_verified']
        and loader['collector_retained_to_exit'],'closed same-native loader qualification required')
    pinned=probe.pin_assets(container)
    tool.guards.require(pinned==loader['assets'],'loader/current assets differ')
    full=json.loads((B/'main-native-gdn-engine/record.json').read_text())
    fixture=full['fixtures']['sycl_submission_test']
    tool.guards.require(tool.sha(Path(fixture['path']).read_bytes())==fixture['sha256'],'fixture bytes differ')
    return dict(pinned,fixture=fixture,loader_source_commit=loader['source_commit'],
        loader_proof_sha256=tool.sha(json.dumps(loader,sort_keys=True,separators=(',',':')).encode()))


def fault_evidence(cursor,boot):
    text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
    return dict(fresh_faults=[line for line in text.splitlines() if re.search(
        r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',line)],
        boot_unchanged=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),driver_after=tool.guards.driver_guard())


def loader_order(text,pid,pinned):
    proof=probe.validation.preload_evidence(text,pid)
    positions={}
    for label,phase,name in (('validation','init',pinned['preload']['library']['path']),
                             ('collector','init',pinned['collector']['path']),('fini','fini',pinned['collector']['path'])):
        matches=[m for m in re.finditer(r'calling '+phase+r':\s+([^\s]+)',text)
                 if Path(m[1]).resolve()==Path(name).resolve()]
        tool.guards.require(len(matches)==1,'ambiguous loader marker: '+label)
        positions[label]=matches[0].start()
    first=text.index('compat: create streams and USM owners')
    tool.guards.require(positions['validation']<text.index('phase=before-main')<positions['collector']<first<positions['fini'],
                        'fixture loader/first-init order differs')
    return proof


def union_us(events):
    end=None;total=0.0
    for start,finish in sorted((float(e['ts']),float(e['ts'])+float(e['dur'])) for e in events):
        if end is None or start>end: total+=finish-start
        elif finish>end: total+=finish-end
        end=finish if end is None else max(end,finish)
    return total


def coverage(cell,wall_begin,wall_end,native_pid):
    files=sorted((ROOT/cell['name']/'trace').rglob('*.json'))
    tool.guards.require(len(files)==1,'one complete native JSON timeline required')
    raw=files[0].read_bytes();value=json.loads(raw)
    events=value['traceEvents'] if isinstance(value,dict) else value
    gpu=[e for e in events if e.get('ph')=='X' and e.get('cat')=='gpu_op']
    host=[e for e in events if e.get('ph')=='X' and e.get('cat')=='cpu_op']
    kernels=[e for e in gpu if 'main' in e.get('name','')]
    tool.guards.require(gpu and host,'GPU and host API timing required')
    tool.guards.require(all(isinstance(e.get(k),(int,float)) and math.isfinite(e[k]) for e in gpu+host for k in ('ts','dur'))
        and all(e['dur']>=0 for e in gpu+host),'invalid trace timing')
    counts=collections.Counter(e['name'] for e in gpu)
    identifiers=[str(e.get('args',{}).get('id','')) for e in gpu]
    sources=collections.Counter(str(e.get('id','')) for e in events if e.get('ph')=='s'
        and e.get('pid')==native_pid and e.get('cat','').startswith('Flow_H2D_'))
    sinks=collections.Counter(str(e.get('id','')) for e in events if e.get('ph')=='t'
        and e.get('pid')==native_pid and e.get('cat','').startswith('Flow_D2H_'))
    kernel_timing=all(e['dur']>0 for e in kernels)
    clock_bounds=all(wall_begin/1000-1000<=e['ts']<=e['ts']+e['dur']<=wall_end/1000+1000 for e in gpu)
    unique=all(identifiers) and len(set(identifiers))==len(identifiers)
    source_complete=all(sources[i]==1 for i in identifiers)
    required_sinks=[str(e['args']['id']) for e in gpu if e.get('name')!='zeCommandListAppendBarrier']
    correlated=source_complete and all(sinks[i]==1 for i in required_sinks)
    return dict(timeline_sha256=tool.sha(raw),timeline_bytes=len(raw),gpu_events=len(gpu),host_events=len(host),
        program_kernels=len(kernels),expected_program_kernels=cell['expected_kernels'],gpu_names=dict(counts),
        program_kernel_service_positive=kernel_timing,zero_gpu_durations=sum(e['dur']==0 for e in gpu),
        gpu_raw_clock_bounds=clock_bounds,gpu_identifiers_unique=unique,required_gpu_flow_pairs=correlated,
        all_gpu_host_sources=source_complete,unsunk_barrier_ids=[str(e['args']['id']) for e in gpu
            if e.get('name')=='zeCommandListAppendBarrier' and sinks[str(e['args']['id'])]!=1],
        gpu_union_us=union_us(gpu),program_kernel_union_us=union_us(kernels),
        known_work_coverage_passed=len(kernels)==cell['expected_kernels'],
        timing_qualified=kernel_timing and clock_bounds,correlation_qualified=unique and correlated,
        limits='Whole bounded fixture in source-supported SystemTime1/CLOCK_MONOTONIC_RAW domain; all-device union is not physical saturation or a production critical path. Containment alone does not qualify sub-millisecond queue-delay attribution. Every GPU event needs its actual host source; non-barrier work also needs its host sink. Unsunk barriers are retained explicitly, not called lost commands. Independent raw-clock and UR/native reconciliation precede production.')


def ur_inventory(text):
    """Expand captured operations by actual graph replays, rather than counting capture twice."""
    captures={};graphs={};executables={};direct=collections.Counter();replayed=collections.Counter();calls=collections.Counter()
    graph_replays=0
    for line in text.splitlines():
        match=re.search(r'<--- (ur\w+)\((.*)\) -> UR_RESULT_SUCCESS;',line)
        if not match:continue
        name,args=match.groups();calls[name]+=1
        def handle(field):
            found=re.search(r'\.'+field+r' = (0x[0-9a-f]+)',args)
            tool.guards.require(found is not None,'missing UR handle '+field)
            return found[1]
        if name=='urQueueBeginCaptureIntoGraphExp':
            queue,graph=handle('hQueue'),handle('hGraph')
            tool.guards.require(queue not in captures,'nested capture unexpected')
            captures[queue]=graph;graphs[graph]=collections.Counter()
        elif name=='urQueueEndGraphCaptureExp':
            tool.guards.require(captures.pop(handle('hQueue'),None) is not None,'unknown capture end')
        elif name=='urGraphInstantiateGraphExp':
            output=re.search(r'\.phExecGraph = 0x[0-9a-f]+ \((0x[0-9a-f]+)\)',args)
            tool.guards.require(output is not None and handle('hGraph') in graphs,'unknown graph instantiate')
            executables[output[1]]=graphs[handle('hGraph')].copy()
        elif name=='urEnqueueGraphExp':
            executable=handle('hGraph');tool.guards.require(executable in executables,'unknown graph replay')
            replayed.update(executables[executable]);graph_replays+=1
        elif name.startswith('urEnqueueKernelLaunch') or name in ('urEnqueueUSMMemcpy','urEnqueueUSMFill',
                'urEnqueueEventsWaitWithBarrier','urEnqueueEventsWaitWithBarrierExt'):
            kind='kernel' if name.startswith('urEnqueueKernelLaunch') else 'copy' if name=='urEnqueueUSMMemcpy' else 'fill' if name=='urEnqueueUSMFill' else 'barrier'
            size=re.search(r'\.size = (\d+)',args) if kind in ('copy','fill') else None
            tool.guards.require(kind not in ('copy','fill') or size is not None,'missing UR requested byte count')
            values=collections.Counter({kind:1})
            if size:values[kind+'_bytes']=int(size[1])
            queue=handle('hQueue')
            (graphs[captures[queue]] if queue in captures else direct).update(values)
    tool.guards.require(not captures,'unterminated capture')
    return dict(direct=dict(direct),replayed=dict(replayed),expanded=dict(direct+replayed),
        templates=[dict(v) for v in graphs.values()],graph_replays=graph_replays,
        capture_regions=calls['urQueueBeginCaptureIntoGraphExp'])


def analyse(container,record,pinned):
    """Separate closed-data contract analysis; original failed run is never relabeled."""
    tool.guards.require(record['stage']=='failed' and record['ownership_closed'] and not record['active']
        and record['fresh_faults']==[] and record['boot_unchanged'] and len(record['observations'])==2,
        'closed two-process raw-domain evidence required')
    tool.guards.require('unitrace_coverage_analysis' not in container,'inspect existing analysis; no overwrite')
    revision=tool.git(P,'rev-parse','HEAD').decode().strip();pins=inputs(revision)
    tool.guards.require(RESULT.read_bytes()==tool.git(P,'show',revision+':'+str(RESULT.relative_to(P))),
        'commit original failed run before closed-data analysis')
    for name,expected in record['input_sha256'].items():
        original=tool.git(P,'show',record['source_commit']+':'+name)
        tool.guards.require(tool.sha(original)==expected,'original measurement source unrecoverable')
    tool.guards.require(pinned==record['assets'],'measurement artifacts changed')
    rows=[]
    for cell,observation in zip(record['cells'],record['observations'],strict=True):
        tool.guards.require(observation['execution_passed'] and observation['session_removed'] and
            observation['fresh_faults']==[],'fixture execution not qualified')
        check=coverage(cell,observation['native_raw_begin_ns'],observation['native_raw_end_ns'],observation['pid'])
        tool.guards.require(check['timeline_sha256']==observation['coverage']['timeline_sha256'] and
            check['known_work_coverage_passed'] and check['timing_qualified'] and check['gpu_identifiers_unique']
            and check['all_gpu_host_sources'],'frozen timing/source/count gate failed')
        path=ROOT/cell['name']/'ur.log';raw=path.read_bytes();inventory=ur_inventory(raw.decode())
        actual=collections.Counter()
        for name,count in check['gpu_names'].items():
            kind='kernel' if 'main' in name else 'copy' if name.startswith('zeCommandListAppendMemoryCopy') else 'fill' if name.startswith('zeCommandListAppendMemoryFill') else 'barrier' if name=='zeCommandListAppendBarrier' else None
            tool.guards.require(kind is not None,'unclassified device command')
            actual[kind]+=count
            if kind in ('copy','fill'):
                size=re.search(r'\[(\d+)\]$',name);tool.guards.require(size is not None,'unclassified requested bytes')
                actual[kind+'_bytes']+=int(size[1])*count
        tool.guards.require(dict(actual)==inventory['expanded'],'UR expanded commands/bytes differ from device records')
        timeline=next((ROOT/cell['name']/'trace').glob('*.json'));events=json.loads(timeline.read_text())['traceEvents']
        gpu=[e for e in events if e.get('cat')=='gpu_op' and e.get('ph')=='X']
        sources={str(e['id']):e for e in events if e.get('pid')==observation['pid'] and e.get('ph')=='s' and e.get('cat','').startswith('Flow_H2D_')}
        sinks={str(e['id']):e for e in events if e.get('pid')==observation['pid'] and e.get('ph')=='t' and e.get('cat','').startswith('Flow_D2H_')}
        graph_api=[e for e in events if e.get('cat')=='cpu_op' and e.get('ph')=='X' and e.get('name') in
                   ('zeCommandListAppendGraphExt','zeCommandListAppendGraphExp')]
        missing=[e for e in gpu if e['name']!='zeCommandListAppendBarrier' and str(e['args']['id']) not in sinks]
        associations=collections.Counter()
        for event in missing:
            source=sources[str(event['args']['id'])]['ts']
            matches=[i for i,g in enumerate(graph_api) if g['ts']<=source<=g['ts']+g['dur']]
            tool.guards.require(len(matches)==1,'unsunk non-barrier work not uniquely associated with a graph API')
            associations[matches[0]]+=1
        tool.guards.require(len(graph_api)==inventory['graph_replays'] and
            (not graph_api or len(associations)==len(graph_api) and set(associations.values())=={3}),
            'known three-work-node graph association failed')
        rows.append(dict(name=cell['name'],timeline_sha256=check['timeline_sha256'],UR_sha256=tool.sha(raw),
            inventory=inventory,device_inventory=dict(actual),program_kernels=check['program_kernels'],
            raw_clock_containment=True,positive_kernel_service=True,unique_device_ids=True,all_host_sources=True,
            unsunk_nonbarrier_records=len(missing),graph_api_groups=len(graph_api),
            graph_nonbarrier_sources_per_group=3 if graph_api else 0,unsunk_barrier_records=len(check['unsunk_barrier_ids'])))
    analysis=dict(schema='strata-unitrace-closed-coverage-analysis-v1',source_commit=revision,input_sha256=pins,
        original_measurement_source=record['source_commit'],original_run_status=record['stage'],original_run_passed=record['passed'],
        original_result=dict(commit=revision,path=str(RESULT.relative_to(P)),sha256=tool.sha(RESULT.read_bytes())),
        native_executed=False,analysis_passed=True,known_fixture_service_coverage_qualified=True,rows=rows,
        production_trace_ready=False,request_window_qualified=False,full_critical_path_qualified=False,
        decision='Both fixed fixtures conserve every program kernel/copy/fill/barrier and requested byte after expanding graph captures. All62/606 main kernels have positive service within caller raw bounds. The61 graph APIs source exactly183 work nodes without individual host sinks; complete fixture numerical checks establish aggregate completion, not per-node host waits. Original strict sink-gate failures remain failed. Independent clock/source review and bounded recording-window qualification still precede32K+ production attribution.',
        retention='Closed raw traces remain consumed by the active independent research/native review. Retire after that reconciliation and request-window plan;64MiB original budget, Main owner. Do not retain raw timelines merely because this analysis passed.')
    container['unitrace_coverage_analysis']=analysis;RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(analysis_passed=True,native_executed=False,production_trace_ready=False,
        inventory=[r['device_inventory'] for r in rows])))
    return 0


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tool.guards.no_model_owners([])
        container=json.loads(RESULT.read_text());record=container.get('unitrace_gpu_coverage')
        pinned=assets(container)
        if action=='analyse':return analyse(container,record,pinned)
        if action=='prepare':
            tool.guards.require(record and record['stage']=='failed' and record['ownership_closed'] and not record['active']
                and record['source_commit'].startswith('eb9aa2547') and record['error']=="RuntimeError('known-work timing/count/correlation gate failed')"
                and not ROOT.exists(),'only the diagnosed closed v1 clock/flow failure permits this new v2 plan')
            revision=tool.git(P,'rev-parse','HEAD').decode().strip();pins=inputs(revision)
            previous=RESULT.read_bytes()
            tool.guards.require(previous==tool.git(P,'show',revision+':'+str(RESULT.relative_to(P))),'commit original failed coverage first')
            ROOT.mkdir(mode=0o700);cells=[]
            for name,mode,count in (('submission','--submission',606),('native-graphs','--native',62)):
                root=ROOT/name;root.mkdir()
                for folder in ('tmp','cache','trace'):(root/folder).mkdir()
                session='StrataKnown'+secrets.token_hex(12)
                env=dict(container['unitrace_loader_probe']['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),UNITRACE_SystemTime='1',
                    UNITRACE_TraceOutputDir=str(root/'trace'),UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.log'))
                command=[pinned['launcher']['path'],'--session',session,'--host-timing','--device-timing',
                    '--chrome-kernel-logging','--chrome-call-logging','--chrome-event-buffer-size','128',pinned['fixture']['path'],mode]
                cells.append(dict(name=name,mode=mode,expected_kernels=count,session=session,environment=env,command=command,
                                  stop_command=[pinned['launcher']['path'],'--stop',session]))
            record=dict(schema='strata-unitrace-known-work-v1',stage='prepared',source_commit=revision,input_sha256=pins,assets=pinned,cells=cells,
                previous_failed_plan=dict(commit=revision,path=str(RESULT.relative_to(P)),sha256=tool.sha(previous),
                    outcome='V1 failed UTC containment/all-barrier-host-sink gates despite normal606-kernel execution, complete host sources and no fresh fault. New source-supported raw-clock domain and non-barrier sink criteria are prospective; old failure unchanged and graph still unrun.'),
                clock_domain='UNITRACE_SystemTime=1 before target exec skips epoch conversion; compare against paired caller CLOCK_MONOTONIC_RAW nanoseconds. No epoch-offset correction, new collector/model binary or mutable environment.',
                gates='Two fixed processes, submission then native graphs; stop on source/owner/fault/fixture/count/timing/correlation failure. Unchanged same-build3b fixture and9836 recovered inputs. Exact main-kernel counts606/62; numerical/guard/footer checks, positive finite kernel service and all GPU records within native caller CLOCK_MONOTONIC_RAW interval+/-1ms. Unique GPU IDs and exactly one host H2D source for every device record; one D2H host sink for each non-barrier record. Retain unsunk barriers explicitly, not as missing device work. Independently reconcile copy/fill/graph inventories and raw/device clock conversion before32K+ production. First-UR-init, loader order and retention still required. Named sessions start resumed and are removed only after proven closure, including failures. No warmed request-window, production observer or speed qualification follows.',
                expected_source_work=dict(submission='1 initial + (1+119+120+121+241) queued +3 ordinary-host kernels =606.',
                    native_graphs='1 initial +12 graph +24 segment +24 cross-queue +1 post-event-destroy replay =62; graph upload finalizes only.'),
                budgets=dict(wall_s_per_target=90,cpu_s=90,rss_bytes=4<<30,text_bytes_per_command=8<<20,file_bytes=32<<20,total_bytes=64<<20),
                retention='Keep raw timelines/UR traces only until independent known-work, copy/fill/graph and correlation reconciliation closes;64MiB total. Commit compact counts/identities/gates and relevant timing summaries. No new model or fixture binary generation.',
                active=False,complete=False,passed=False,ownership_closed=False,gpu_timing_qualified=False,model_qualified=False,observations=[])
            container['unitrace_gpu_coverage']=record;RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
            print('Prepared fixed two-process GPU coverage plan; commit before run.');return 0
        tool.guards.require(action=='run' and record and record['stage']=='prepared','committed prepared coverage required')
        tool.guards.require(RESULT.read_bytes()==tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),'commit coverage plan first')
        tool.guards.require(inputs(record['source_commit'])==record['input_sha256'] and pinned==record['assets'],'coverage input changed')
        tool.guards.driver_guard();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
        cursor=next(line.split(': ',1)[1] for line in journal.splitlines() if line.startswith('-- cursor: '));owner=None

        def persist():
            record['active']=bool(owner and owner.active is not None)
            (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')

        owner=tool.supervision.Owner(ROOT/'owned',persist);record['stage']='running';persist()
        try:
            for cell in record['cells']:
                session=Path('/dev/shm/uctrl'+str(os.getuid())+'_'+cell['session']);entry=None
                tool.guards.require(not session.exists(),'preexisting exact session; never overwrite')
                observation=dict(name=cell['name'],execution_passed=False,coverage_passed=False,session_removed=False)
                record['observations'].append(observation)
                try:
                    begin=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    entry,out,err=owner.run(cell['name'],cell['command'],cell['environment'],ROOT/cell['name'],wall=90,cpu=90,
                        rss_cap=4<<30,text_cap=8<<20,file_cap=32<<20,total_cap=64<<20)
                    end=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    tool.guards.require(tool.supervision.closed(entry) and entry.get('exit_code')==0,'fixture exit/ownership failed')
                    proof=loader_order(err.read_text(),entry['pid'],pinned)
                    stdout=out.read_text();footer='fixture_status=PASS mode=submission' if cell['mode']=='--submission' else 'SYCL host adapter: events, profiling, graph replay, copies and lifecycle passed'
                    tool.guards.require(footer in stdout,'fixture correctness footer missing')
                    observation.update(execution_passed=True,pid=entry['pid'],preload_evidence=proof,native_raw_begin_ns=begin,
                        native_raw_end_ns=end,native_raw_wall_ns=end-begin,
                        fixture_footer=footer,coverage=coverage(cell,begin,end,entry['pid']))
                    checks=observation['coverage'];observation['coverage_passed']=all(checks[k] for k in ('known_work_coverage_passed','timing_qualified','correlation_qualified'))
                    tool.guards.require(observation['coverage_passed'],'known-work timing/count/correlation gate failed')
                finally:
                    if entry and tool.supervision.closed(entry) and owner.active is None:
                        if session.exists():
                            stop,_,_=owner.run(cell['name']+'-stop',cell['stop_command'],cell['environment'],ROOT/cell['name'],
                                wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=32<<20,total_cap=64<<20)
                            tool.guards.require(tool.supervision.closed(stop) and stop.get('exit_code')==0,'exact session stop failed')
                        observation['session_removed']=not session.exists()
                    observation.update(fault_evidence(cursor,boot));persist()
                tool.guards.require(observation['session_removed'] and observation['fresh_faults']==[] and observation['boot_unchanged'],
                    'session/fault/boot gate failed; stop before next process')
            tool.guards.require(inputs(record['source_commit'])==record['input_sha256'] and assets(container)==record['assets'],'post-run inputs/assets changed')
            record.update(stage='closed',complete=True,passed=True,gpu_timing_qualified=True)
        except BaseException as error:record.update(stage='failed',error=repr(error),passed=False)
        finally:
            record['ownership_closed']=owner.active is None and all(tool.supervision.closed(c) for c in owner.commands)
            try:record.update(fault_evidence(cursor,boot))
            except Exception as error:record['final_probe_error']=repr(error)
            record['passed']=bool(record['passed'] and record['ownership_closed'] and record.get('fresh_faults')==[]
                and record.get('boot_unchanged') and 'final_probe_error' not in record)
            persist();container['unitrace_gpu_coverage']=record;RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','gpu_timing_qualified','error')}))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run','analyse'),'use production_profile.py prepare|run|analyse')
    raise SystemExit(main(sys.argv[1]))
