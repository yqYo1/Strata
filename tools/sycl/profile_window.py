#!/usr/bin/env python3
"""Build and qualify a host-controlled recording window around known GPU work."""
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode=True
import production_profile as production

probe=production.probe
tool=probe.tool
P,B,RESULT=tool.P,tool.B,tool.RESULT
ROOT=B/'unitrace-recording-window-v1'
SOURCE='bench/sycl/profile_window.cpp'


def inputs(revision):
    pins=production.inputs(revision)
    for name in ('tools/sycl/profile_window.py',SOURCE):
        data=tool.git(P,'show',revision+':'+name)
        tool.guards.require(data==(P/name).read_bytes(),'commit exact window input: '+name)
        pins[name]=tool.sha(data)
    return pins


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def compiler_manifest(compiler):
    install=Path(compiler).parents[1]
    paths=set()
    for root in (install/'bin',install/'include',install/'lib',Path('/usr/include'),Path('/usr/lib/gcc/x86_64-linux-gnu/13/include')):
        paths.update(p.resolve() for p in root.rglob('*') if p.is_file())
    paths.update(Path(p).resolve() for p in ('/usr/bin/ld','/usr/bin/as','/usr/lib/x86_64-linux-gnu/libc.so.6',
        '/usr/lib/x86_64-linux-gnu/libm.so.6','/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
        '/usr/lib/x86_64-linux-gnu/libgcc_s.so.1','/usr/lib/x86_64-linux-gnu/crt1.o',
        '/usr/lib/x86_64-linux-gnu/crti.o','/usr/lib/x86_64-linux-gnu/crtn.o',
        '/usr/lib/x86_64-linux-gnu/libze_loader.so'))
    return [dict(path=str(p),sha256=file_sha(p)) for p in sorted(paths)]


def admission(record,container):
    tool.guards.require(inputs(record['source_commit'])==record['input_sha256'],'window source changed')
    tool.guards.require(probe.pin_assets(container)==record['assets'],'window runtime/native/tool assets changed')
    path=ROOT/'compiler-inputs.json'
    tool.guards.require(file_sha(path)==record['compiler_inputs']['sha256'],'compiler manifest changed')
    rows=json.loads(path.read_text());tool.guards.require(len(rows)==record['compiler_inputs']['files'],'compiler input count changed')
    for row in rows:tool.guards.require(file_sha(row['path'])==row['sha256'],'compiler/header asset changed: '+row['path'])
    if 'binary' in record:tool.guards.require(file_sha(record['binary']['path'])==record['binary']['sha256'],'window executable changed')


def state(path,expected):
    raw=path.read_bytes()
    tool.guards.require(len(raw)==1024 and struct.unpack_from('=i',raw)[0]==expected,'exact session state differs')


def clock_evidence(stdout):
    timers=[json.loads(line.removeprefix('profile_timer ')) for line in stdout.splitlines() if line.startswith('profile_timer ')]
    samples=[json.loads(line.removeprefix('profile_clock ')) for line in stdout.splitlines() if line.startswith('profile_clock ')]
    tool.guards.require(len(timers)==1 and len(samples)==8,'complete timer/clock samples required')
    timer=timers[0]
    tool.guards.require(timer['frequency_hz']>0 and all(0<timer[k]<=64 for k in ('timestamp_bits','kernel_timestamp_bits')),
                        'invalid timer properties')
    phases={}
    for phase in ('ready','drained'):
        rows=[s for s in samples if s['phase']==phase]
        tool.guards.require(len(rows)==4 and all(s['before_ns']<=s['host_ns']<=s['after_ns'] for s in rows),
                            'paired device host clock outside native raw bracket')
        widths=[s['after_ns']-s['before_ns'] for s in rows]
        tool.guards.require(min(widths)<=100000,'best clock bracket exceeds predeclared100us budget')
        phases[phase]=dict(samples=rows,best_bracket_ns=min(widths),worst_bracket_ns=max(widths))
    return dict(properties=timer,tick_quantum_ns=1e9/timer['frequency_hz'],phases=phases,
        limits='Caller brackets bound the paired-host samples, not every collector conversion or graph edge. No sub-millisecond queue-delay or full critical-path accuracy claim.')


def trace_evidence(root,expected,begin,end,pid):
    files=sorted((root/'trace').rglob('*.json'))
    tool.guards.require(len(files)<=1,'unexpected multiple trace files')
    events=[];identity=None
    if files:
        raw=files[0].read_bytes();identity=dict(sha256=tool.sha(raw),bytes=len(raw))
        value=json.loads(raw);events=value['traceEvents'] if isinstance(value,dict) else value
    gpu=[e for e in events if e.get('ph')=='X' and e.get('cat')=='gpu_op']
    kernels=[e for e in gpu if 'main' in e.get('name','')]
    tool.guards.require(len(kernels)==expected,'recording state leaked or lost known kernels')
    actual=collections.Counter();sources=collections.Counter(str(e.get('id','')) for e in events
        if e.get('pid')==pid and e.get('ph')=='s' and e.get('cat','').startswith('Flow_H2D_'))
    ids=[]
    for event in gpu:
        name=event['name'];kind='kernel' if 'main' in name else 'copy' if name.startswith('zeCommandListAppendMemoryCopy') else 'fill' if name.startswith('zeCommandListAppendMemoryFill') else 'barrier' if name=='zeCommandListAppendBarrier' else None
        tool.guards.require(kind is not None,'unknown device operation')
        actual[kind]+=1
        if kind in ('copy','fill'):
            size=re.search(r'\[(\d+)\]$',name);tool.guards.require(size is not None,'missing requested bytes')
            actual[kind+'_bytes']+=int(size[1])
        tool.guards.require(all(isinstance(event.get(k),(int,float)) and math.isfinite(event[k]) for k in ('ts','dur'))
            and event['dur']>=0 and (kind!='kernel' or event['dur']>0),'invalid device service')
        tool.guards.require(begin/1000-1000<=event['ts']<=event['ts']+event['dur']<=end/1000+1000,
                            'GPU service outside controlled raw window+/-1ms')
        identifier=str(event.get('args',{}).get('id',''));ids.append(identifier)
        tool.guards.require(identifier and sources[identifier]==1,'missing unique host source')
    tool.guards.require(len(set(ids))==len(ids),'duplicate GPU identities')
    expected_inventory=dict(kernel=606,copy=53,copy_bytes=55844,fill=1,fill_bytes=1028,barrier=20) if expected else {}
    tool.guards.require(dict(actual)==expected_inventory,'recorded work/bytes do not match controlled fixture')
    return dict(timeline=identity,program_kernels=len(kernels),device_inventory=dict(actual),
        gpu_union_us=production.union_us(gpu),program_kernel_union_us=production.union_us(kernels),
        host_sources_complete=True,raw_window_contained=True)


def check(container,record):
    plan=record['check'];target_owner=None;control_owner=None
    tool.guards.require(record['stage']=='check-prepared' and not plan.get('observations'),'fresh committed window check required')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(line.split(': ',1)[1] for line in journal.splitlines() if line.startswith('-- cursor: '))
    record['driver_before']=tool.guards.driver_guard()
    def persist():
        record['active']=any(o and o.active is not None for o in (target_owner,control_owner))
        (ROOT/'check-record.json').write_text(json.dumps(dict(record,
            target_commands=target_owner.commands if target_owner else [],control_commands=control_owner.commands if control_owner else []),indent=2)+'\n')
    target_owner=tool.supervision.Owner(ROOT/'target-owned',persist)
    control_owner=tool.supervision.Owner(ROOT/'control-owned',persist)
    record.update(stage='checking',passed=False);plan['observations']=[];persist()
    try:
        for cell in plan['cells']:
            root=ROOT/cell['name'];session=Path('/dev/shm/uctrl'+str(os.getuid())+'_'+cell['session'])
            tool.guards.require(not session.exists(),'preexisting unique session; never overwrite')
            entry=None;observation=dict(name=cell['name'],passed=False,session_removed=False,controls=[])
            plan['observations'].append(observation)
            def control(action,expected=None):
                before=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                command=[record['assets']['launcher']['path'],'--'+action,cell['session']]
                result,_,_=control_owner.run(cell['name']+'-'+action,command,cell['environment'],root,
                    wall=15,cpu=15,rss_cap=2<<30,text_cap=1<<20,file_cap=32<<20,total_cap=128<<20)
                after=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                tool.guards.require(tool.supervision.closed(result) and result.get('exit_code')==0,'owned session control failed')
                if expected is not None:state(session,expected)
                observation['controls'].append(dict(action=action,raw_before_ns=before,raw_after_ns=after,expected_state=expected))
            def interact(io,child):
                ready=None
                for _ in range(20):
                    line=io.line(20)
                    if line.startswith('profile_window phase=ready '):ready=line;break
                    tool.guards.require(line.startswith(('profile_timer ','profile_clock ')),'unexpected pre-ready protocol')
                match=re.fullmatch(r'profile_window phase=ready pid=(\d+) raw_ns=(\d+)',ready or '')
                tool.guards.require(match and int(match[1])==child['pid'],'owned native ready identity differs')
                main_ns=os.readlink('/proc/self/ns/time');native_ns=os.readlink('/proc/'+str(child['pid'])+'/ns/time')
                tool.guards.require(main_ns==native_ns,'caller/native time namespaces differ')
                observation.update(pid=child['pid'],time_namespace=main_ns,
                    time_offsets=Path('/proc/'+str(child['pid'])+'/timens_offsets').read_text(),ready_raw_ns=int(match[2]))
                state(session,1)
                if cell['name']=='window':control('resume',0)
                observation['release_raw_ns']=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                io.send(b'continue\n')
                drained=None
                for _ in range(32):
                    line=io.line(30)
                    if line.startswith('profile_window phase=drained '):drained=line;break
                    tool.guards.require(line.startswith(('case ','fixture_status=','profile_window fixture_status=','profile_clock ')),
                                        'unexpected fixture protocol')
                match=re.fullmatch(r'profile_window phase=drained pid=(\d+) raw_ns=(\d+)',drained or '')
                tool.guards.require(match and int(match[1])==child['pid'],'owned native drain identity differs')
                observation['drained_raw_ns']=int(match[2])
                if cell['name']=='window':control('pause',1)
                else:state(session,1)
                io.send(b'continue\n');io.proc.stdin.close();io.exit(20)
            try:
                entry,out,err=target_owner.run(cell['name'],cell['command'],cell['environment'],root,interaction=interact,
                    wall=90,cpu=90,rss_cap=4<<30,text_cap=8<<20,file_cap=32<<20,total_cap=128<<20)
                tool.guards.require(tool.supervision.closed(entry) and entry.get('exit_code')==0,'window native exit/closure failed')
                stdout=out.read_text();stderr=err.read_text()
                tool.guards.require('fixture_status=PASS mode=submission' in stdout and 'profile_window fixture_status=0' in stdout,
                                    'complete numerical/guard footer missing')
                observation['preload_evidence']=production.loader_order(stderr,entry['pid'],record['assets'])
                inventory=production.ur_inventory((root/'ur.log').read_text())
                expected=dict(kernel=606,copy=53,copy_bytes=55844,fill=1,fill_bytes=1028,barrier=20)
                tool.guards.require(inventory['expanded']==expected and inventory['graph_replays']==0,'actual GPU command inventory differs')
                observation.update(clock=clock_evidence(stdout),UR_sha256=file_sha(root/'ur.log'),
                    executed_inventory=inventory['expanded'],coverage=trace_evidence(root,cell['expected_kernels'],
                        observation['release_raw_ns'],observation['drained_raw_ns'],entry['pid']),execution_passed=True)
                observation['passed']=True
            finally:
                if entry and tool.supervision.closed(entry) and target_owner.active is None and control_owner.active is None:
                    if session.exists():control('stop')
                    observation['session_removed']=not session.exists()
                else:observation['cleanup_pending']='Target/control closure unproven; leave exact session untouched.'
                observation.update(production.fault_evidence(cursor,boot));persist()
            tool.guards.require(observation['passed'] and observation['session_removed'] and observation['fresh_faults']==[]
                and observation['boot_unchanged'],'window/session/fault gate failed; stop before next process')
        admission(record,container)
        record.update(stage='closed',passed=True,request_window_qualified=True)
    except BaseException as error:record.update(stage='failed',passed=False,error=repr(error))
    finally:
        record['ownership_closed']=all(o.active is None and all(tool.supervision.closed(e) for e in o.commands)
            for o in (target_owner,control_owner))
        try:record.update(production.fault_evidence(cursor,boot))
        except Exception as error:record['final_probe_error']=repr(error)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record.get('fresh_faults')==[]
            and record.get('boot_unchanged') and 'final_probe_error' not in record)
        record['request_window_qualified']=record['passed']
        persist();container['unitrace_recording_window']=record
        RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','request_window_qualified','error')}))
    return 0 if record['passed'] else 1


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tool.guards.no_model_owners([])
        container=json.loads(RESULT.read_text());record=container.get('unitrace_recording_window')
        if action=='prepare':
            tool.guards.require(record is None and not ROOT.exists(),'inspect existing window; no blind retry')
            analysis=container['unitrace_coverage_analysis']
            tool.guards.require(analysis['analysis_passed'] and analysis['known_fixture_service_coverage_qualified']
                and not analysis['native_executed'],'closed known-work service reconciliation required')
            revision=tool.git(P,'rev-parse','HEAD').decode().strip();pins=inputs(revision)
            pinned=probe.pin_assets(container);full=json.loads((B/'main-native-gdn-engine/record.json').read_text())
            compiler=full['compiler'];tool.guards.require(file_sha(compiler['path'])==compiler['sha256'],'compiler differs')
            ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
            rows=compiler_manifest(compiler['path']);manifest=ROOT/'compiler-inputs.json'
            manifest.write_text(json.dumps(rows,indent=2)+'\n')
            command=[compiler['path'],'-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-id-queries-range=size_t',
                '-DSTRATA_ENABLE_SYCL=1','-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),str(P/SOURCE),
                *[str(P/'src/sycl'/(name+'.cpp')) for name in ('runtime','compat','command_profile')],
                '/usr/lib/x86_64-linux-gnu/libze_loader.so','-o',str(ROOT/'profile_window')]
            record=dict(schema='strata-profile-recording-window-v1',stage='prepared',source_commit=revision,input_sha256=pins,
                assets=pinned,compiler=compiler,compiler_inputs=dict(path=str(manifest),files=len(rows),sha256=file_sha(manifest)),
                command=command,build_environment=dict(full['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
                gates='One new standalone fixture, not a model engine. Wrapper includes original maintained compat test and adds bounded stdin gates before/after its complete main body, plus LevelZero device timer properties and four paired-clock brackets at each boundary. Native get-properties/global-timestamps calls add no kernel/copy/fill work. Build exact committed recipe, then commit executable identity before preparing two fresh processes: paused-throughout then resume-after-ready/pause-after-drained. Both execute606 kernels/numerical guards; expected recorded kernels0/606. Separate owned CPU control commands mutate only the unique session shared memory; target environment remains immutable. All control/target processes, constructor proof and fault/boot checks must close; no model yet.',
                retention='Current standalone recording-window qualification; retire executable and build outputs after the model control path is qualified, retaining recoverable source/recipe and compact control/coverage evidence.',
                budgets=dict(build_wall_s=900,build_cpu_s=900,rss_bytes=8<<30,text_bytes=8<<20,file_bytes=128<<20,total_bytes=128<<20),
                active=False,passed=False,ownership_closed=True,request_window_qualified=False,model_qualified=False)
            container['unitrace_recording_window']=record;RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
            print(json.dumps(dict(stage='prepared',compiler_inputs=len(rows),new_model_generation=False)));return 0
        tool.guards.require(record is not None and RESULT.read_bytes()==tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),
                            'commit exact window plan/result before action')
        admission(record,container)
        if action=='check-prepare':
            tool.guards.require(record['stage']=='built' and record['passed'] and record['ownership_closed'] and 'check' not in record,
                                'closed qualified build required')
            cells=[]
            for name,count in (('paused',0),('window',606)):
                root=ROOT/name;root.mkdir()
                for folder in ('tmp','cache','trace'):(root/folder).mkdir()
                session='StrataWindow'+secrets.token_hex(12)
                env=dict(container['unitrace_loader_probe']['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
                    UNITRACE_SystemTime='1',UNITRACE_TraceOutputDir=str(root/'trace'),
                    UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.log'))
                command=[record['assets']['launcher']['path'],'--session',session,'--start-paused','--host-timing','--device-timing',
                    '--chrome-kernel-logging','--chrome-call-logging','--chrome-event-buffer-size','128',record['binary']['path'],'--submission']
                cells.append(dict(name=name,expected_kernels=count,session=session,environment=env,command=command))
            record.update(stage='check-prepared',passed=False,check=dict(cells=cells,
                gates='Two fixed fresh native processes execute unchanged606-kernel/53-copy/1-fill/20-barrier fixture and numerical guards. Paused-throughout records zero GPU work. Window resumes only at native READY, pauses only after native DRAINED, records exact inventory and requested bytes with positive main-kernel service. Every event must lie between release/drained raw clocks+/-1ms with unique ID/host source. Native and caller must share observed time namespace. Four paired-clock brackets before and after; all host samples inside native raw bracket and minimum width per phase<=100us; record frequency/valid bits and tick quantum. These bounds do not qualify sub-millisecond queue delays or graph critical paths. Stop exact sessions only after native/control closure. Fault, boot, constructor/retention and post-input checks required. Stop after first failure; no model execution.',
                wall_s_per_target=90,wall_s_per_control=15,raw_text_budget_bytes=64<<20,total_budget_bytes=128<<20))
            container['unitrace_recording_window']=record;RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
            print('Prepared fixed paused/window GPU check; commit before run.');return 0
        if action=='check':return check(container,record)
        if action=='build':
            tool.guards.require(record['stage']=='prepared' and not (ROOT/'record.json').exists(),'prepared unused window build required')
            owner=None
            def persist():
                record['active']=bool(owner and owner.active is not None)
                (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
            owner=tool.supervision.Owner(ROOT/'build-owned',persist);record['stage']='building';persist()
            try:
                entry,_,err=owner.run('build',record['command'],record['build_environment'],P,wall=900,cpu=900,
                    rss_cap=8<<30,text_cap=8<<20,file_cap=128<<20,total_cap=128<<20)
                tool.guards.require(tool.supervision.closed(entry) and entry.get('exit_code')==0,'window build failed: '+str(err))
                admission(record,container)
                record.update(stage='built',passed=True,binary=dict(path=str(ROOT/'profile_window'),sha256=file_sha(ROOT/'profile_window'),
                    bytes=(ROOT/'profile_window').stat().st_size))
            except BaseException as error:record.update(stage='failed',passed=False,error=repr(error))
            finally:
                record['ownership_closed']=owner.active is None and all(tool.supervision.closed(e) for e in owner.commands)
                record['passed']=bool(record['passed'] and record['ownership_closed']);persist()
                container['unitrace_recording_window']=record;RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
            print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','error')}));return 0 if record['passed'] else 1
        raise RuntimeError('unsupported window action')


if __name__=='__main__':
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build','check-prepare','check'),
                        'use profile_window.py prepare|build|check-prepare|check')
    raise SystemExit(main(sys.argv[1]))
