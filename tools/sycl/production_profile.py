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
ROOT = B/'unitrace-known-work-v1'


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
    correlated=all(sources[i]==sinks[i]==1 for i in identifiers)
    return dict(timeline_sha256=tool.sha(raw),timeline_bytes=len(raw),gpu_events=len(gpu),host_events=len(host),
        program_kernels=len(kernels),expected_program_kernels=cell['expected_kernels'],gpu_names=dict(counts),
        program_kernel_service_positive=kernel_timing,zero_gpu_durations=sum(e['dur']==0 for e in gpu),
        gpu_wall_clock_bounds=clock_bounds,gpu_identifiers_unique=unique,all_gpu_flow_pairs=correlated,
        gpu_union_us=union_us(gpu),program_kernel_union_us=union_us(kernels),
        known_work_coverage_passed=len(kernels)==cell['expected_kernels'],
        timing_qualified=kernel_timing and clock_bounds,correlation_qualified=unique and correlated,
        limits='Whole bounded fixture; all-device union is not physical saturation or a production critical path. Host/GPU clock containment alone does not qualify sub-millisecond queue-delay attribution. Copy/fill inventory and graph replay require the independent UR/native record reconciliation before production.')


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tool.guards.no_model_owners([])
        container=json.loads(RESULT.read_text());record=container.get('unitrace_gpu_coverage')
        pinned=assets(container)
        if action=='prepare':
            tool.guards.require(record is None and not ROOT.exists(),'inspect existing coverage job; no retry')
            revision=tool.git(P,'rev-parse','HEAD').decode().strip();pins=inputs(revision)
            ROOT.mkdir(mode=0o700);cells=[]
            for name,mode,count in (('submission','--submission',606),('native-graphs','--native',62)):
                root=ROOT/name;root.mkdir()
                for folder in ('tmp','cache','trace'):(root/folder).mkdir()
                session='StrataKnown'+secrets.token_hex(12)
                env=dict(container['unitrace_loader_probe']['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
                    UNITRACE_TraceOutputDir=str(root/'trace'),UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.log'))
                command=[pinned['launcher']['path'],'--session',session,'--host-timing','--device-timing',
                    '--chrome-kernel-logging','--chrome-call-logging','--chrome-event-buffer-size','128',pinned['fixture']['path'],mode]
                cells.append(dict(name=name,mode=mode,expected_kernels=count,session=session,environment=env,command=command,
                                  stop_command=[pinned['launcher']['path'],'--stop',session]))
            record=dict(schema='strata-unitrace-known-work-v1',stage='prepared',source_commit=revision,input_sha256=pins,assets=pinned,cells=cells,
                gates='Two fixed processes, submission then native graphs; stop on any source/owner/fault/fixture/count/timing/correlation failure. Use unchanged same-build3b fixture and9836 recovered inputs. Exact main-kernel counts606/62; numerical/guard/footer checks plus positive finite kernel service, GPU records inside native Unix-wall interval+/-1ms, unique device IDs and source/sink flow pair for each GPU event. Collect UR trace independently for later copy/fill/graph reconciliation. Single-thread first-UR-init, loader order and retention must still pass. Named sessions start resumed for whole-fixture coverage and are removed only after proven closure, including failures. No warm request-window, production observer or performance qualification follows from these checks.',
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
                    begin=time.time_ns()
                    entry,out,err=owner.run(cell['name'],cell['command'],cell['environment'],ROOT/cell['name'],wall=90,cpu=90,
                        rss_cap=4<<30,text_cap=8<<20,file_cap=32<<20,total_cap=64<<20)
                    end=time.time_ns()
                    tool.guards.require(tool.supervision.closed(entry) and entry.get('exit_code')==0,'fixture exit/ownership failed')
                    proof=loader_order(err.read_text(),entry['pid'],pinned)
                    stdout=out.read_text();footer='fixture_status=PASS mode=submission' if cell['mode']=='--submission' else 'SYCL host adapter: events, profiling, graph replay, copies and lifecycle passed'
                    tool.guards.require(footer in stdout,'fixture correctness footer missing')
                    observation.update(execution_passed=True,pid=entry['pid'],preload_evidence=proof,native_unix_wall_ns=end-begin,
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
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use production_profile.py prepare|run')
    raise SystemExit(main(sys.argv[1]))
