#!/usr/bin/env python3
"""Owned collector admission on the retained eager-prefill native generation.

Main-authored,repository LICENSE. Reuses owned current_pti_probe.py at
12e74253f and prefill_window_device.py at113fdee4c. No XeStrata input,
new native build,model request,performance comparison or saturation claim.
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
import prefill_workspace_admission as workspace
import request_profile_reader as reader

capacity,core,device=workspace.capacity,workspace.core,workspace.device
P,B,RESULT=capacity.P,capacity.B,capacity.RESULT
ROOT,KEY=B/'eager-pti-probe-v1','eager_pti_probe_v1'
THIS,TEST='tools/sycl/eager_pti_probe.py','tools/test_eager_pti_probe.py'
require,sha,signature=capacity.require,capacity.sha,capacity.signature
LABELS=('control','trace')
BUDGET=dict(wall=180,cpu=360,rss_cap=4<<30,text_cap=32<<20,file_cap=64<<20,total_cap=128<<20)


def compact(row):
    value={k:v for k,v in row.items() if k not in ('sources','environment','commands')}
    value['source_inputs']=dict(files=len(row['sources']),sha256=signature(row['sources']))
    value['environment_sha256']=signature(row['environment'])
    return value


def save(row):
    core.write(ROOT/'record.json',row)
    values=json.loads(RESULT.read_text());values[KEY]=compact(row);core.write(RESULT,values)


def basis():
    native,component=workspace.device_gate()
    cpu=pti.committed_json(pti.RESULT,P);tools=pti.verify_tools(cpu)
    prior=cpu['unitrace_current_runtime_probe_v2']
    require(prior['stage']=='closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active']
        and [r['label'] for r in prior['results']]==list(LABELS)
        and all(r['passed'] and r['native_exit_code']==0 and not r['fresh_faults'] for r in prior['results'])
        and prior['results'][1]['raw_clock_mapping_checked']
        and prior['admission']['tools']==tools and prior['preload']==native['native_validation_probe']
        and prior['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        and prior['driver']==native['driver'],'exact closed collector/provider/clock foundation required')
    return native,component,tools,signature(prior)


def environment(row,label):
    require(label in LABELS,'unplanned collector arm')
    root=ROOT/label;env=dict(row['environment'])
    require(not env.get('LD_PRELOAD') and not any(k.startswith('UNITRACE_') for k in env)
        and env.get('ZET_ENABLE_METRICS','0')!='1'
        and all(env.get(k)==v for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
            NEOReadDebugKeys='1',EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',
            ZES_ENABLE_SYSMAN='0').items()),'immutable qualified safety and collector-free parent required')
    env.update(device.FIXTURE_ENV)
    env.update(TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
        STRATA_MEASUREMENT_ROOT=str(root),STRATA_SYCL_PREFILL_FULLFIT_UPLOAD='eager',STRATA_TRACE='1',
        LD_DEBUG='libs',STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    shim=row['native_validation_probe']['library']['path']
    if label=='control':env['LD_PRELOAD']=shim
    else:env.update(STRATA_UNITRACE_TARGET_PRELOAD=shim,UNITRACE_SystemTime='1',
        UNITRACE_FollowChildProcess='1',UNITRACE_TraceOutputDir=str(root/'trace'))
    require(not any('xestrata' in str(v).lower() for v in env.values()),'foreign provider origin refused')
    return env


def command(row,label):
    require(label in LABELS,'unplanned collector arm')
    target=[row['fixture']['path'],'--fullfit-eager-smoke']
    if label=='control':return target
    return [row['tools']['artifacts']['unitrace']['path'],'--device-timing','--chrome-device-logging',
        '--chrome-event-buffer-size','128',*target]


def source_pins(revision):
    sources=core.pins(revision)
    require({THIS,TEST}<=sources.keys(),'commit actual collector controller and tests')
    recovered=dict(core.source.device.common.source.blobs(P,revision,sources))
    require(set(recovered)==set(sources) and all(sha(P/name)==expected
        and pti.pti.sha(recovered[name])==expected for name,expected in sources.items()),
        'uncommitted or unrecoverable collector inputs')
    return sources


def current(row):
    require(device.build.ur_validation.committed(RESULT,(KEY,))==compact(row)
        and source_pins(row['source_commit'])==row['sources'],'commit frozen probe stage/source first')
    native,component,tools,prior=basis()
    require(component==row['component_gate'] and tools==row['tools'] and prior==row['prior_clock_gate_sha256']
        and native['binaries']['sycl_expert_bank_test']==row['fixture']
        and native['native_validation_probe']==row['native_validation_probe']
        and native['recipe']['environment']==row['environment'] and native['driver']==row['driver']
        and sha(device.build.ROOT/'record.json')==row['native_build_record_sha256'],
        'native composition,fixture,collector or platform changed')
    require(core.comparison.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes'],
        'finite probe envelope exceeded')
    for label in LABELS:
        require(row['arms'][label]==dict(command=command(row,label),environment_sha256=signature(environment(row,label))),
            'declared pre-exec arm changed')


def trace_summary(trace,begin,end):
    require(type(begin) is int and type(end) is int and 0<begin<end,'positive enclosing RAW bounds required')
    require(type(trace) is dict and type(trace.get('traceEvents')) is list,'complete trace event array required')
    events=[]
    for e in trace['traceEvents']:
        if e.get('cat')!='gpu_op':continue
        require(e.get('ph')=='X','complete GPU intervals required')
        start=reader.nanoseconds(e.get('ts'));stop=start+reader.nanoseconds(e.get('dur'))
        require(begin<=start<=stop<=end,'GPU time outside actual CLOCK_MONOTONIC_RAW lifetime')
        events.append((start,stop))
    require(events and any(b>a for a,b in events),'nonempty positive GPU timing required')
    return dict(GPU_events=len(events),GPU_service_ns=sum(b-a for a,b in events),
        observed_GPU_union_ns=reader.union_ns(events),raw_clock_mapping_checked=True,
        complete_service_count_coverage_qualified=False)


def providers(errors,row,label,pid):
    allowed=json.loads(Path(row['runtime_manifest']['path']).read_text())
    allowed.update({v['path']:v['sha256'] for v in row['native_validation_probe'].values()})
    if label=='trace':
        item=row['tools']['artifacts']['libunitrace_tool.so'];allowed[item['path']]=item['sha256']
    allowed={str(Path(p).resolve()):h for p,h in allowed.items()}
    loaded={str(Path(p.strip()).resolve()) for owner,p in re.findall(
        r'^\s*(\d+):\s+calling init:\s*(/[^\n]+)$',errors,re.M) if int(owner)==pid}
    require(loaded and all(p in allowed and sha(p)==allowed[p] for p in loaded),
        'unknown or changed same-PID initialized ELF')
    required={str(Path(v['path']).resolve()) for v in row['native_validation_probe'].values()}
    required.add(str(Path(row['environment']['UR_ADAPTERS_FORCE_LOAD']).resolve()))
    if label=='trace':required.add(str(Path(row['tools']['artifacts']['libunitrace_tool.so']['path']).resolve()))
    require(required<=loaded and any('/libigc.so' in p for p in loaded)
        and any('libsycl.so' in p for p in loaded) and any('libze_intel_gpu.so' in p for p in loaded),
        'required actual GPU/probe/collector constructors absent')
    return dict(initialized_ELFs=len(loaded),sha256=signature({p:allowed[p] for p in sorted(loaded)}),
        source='same-PID glibc constructors;not complete live-map or saturation coverage')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh probe only;no retry')
    native,component,tools,prior=basis()
    revision=device.build.source.git(P,'rev-parse','HEAD').decode().strip();sources=source_pins(revision)
    row=dict(stage='prepared',source_commit=revision,sources=sources,component_gate=component,tools=tools,
        prior_clock_gate_sha256=prior,native_source_commit=native['source_commit'],
        native_base=native['native_base_commit'],native_overlays=native['overlays'],
        native_build_record_sha256=sha(device.build.ROOT/'record.json'),
        fixture=native['binaries']['sycl_expert_bank_test'],native_validation_probe=native['native_validation_probe'],
        environment=native['recipe']['environment'],runtime_manifest=native['foundation']['runtime_manifest'],
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),driver=native['driver'],
        arms={},results=[],commands=[],passed=False,complete=False,active=False,ownership_closed=True,
        budgets=BUDGET,whole_bytes=288<<20,
        plan=dict(question='Does the exact retained eager fixture close normally with the qualified collector and RAW clock?',
            unit='Only collector injection differs;same same-build fixture,4experts/256and512stride/2layers,'
                'normal and cancellation cases,immutable pre-exec safety,quiet validation and complete API trace.',
            order='Control then trace in separate owned processes;commit each closure;no concurrent jobs.',
            acceptance='Exact committed controller/helpers/provider/build/fixture/collector identity;'
                'both original numerical cases,checked cancellations/refusals,first-init proof and same-PID provider constructors;'
                'normal0/reapedempty,quiet validation,unchangedboot/driver,no fresh fault. Trace must have one target-PID JSON,'
                'nonempty positive GPU intervals inside actual enclosing CLOCK_MONOTONIC_RAW bounds.',
            stopping='Stop first mismatch,warning,fault,source/owner/clock failure or finite budget;no retry or refreshed hashes.',
            limits='Small collector admission only;no model,observer-overhead,complete service-count coverage,'
                'prefill drain,physical bus/saturation,ordinary teardown,full context or inference adoption.',
            retention='Reuse the current8targets and exact existing collector;no fourth native generation. '
                'Keep compact outcomes/identities/clock/provider evidence for immediate64Kresource attribution;'
                'retire successful logs,trace,cache and private lifecycle after verified closure.'))
    for label in LABELS:row['arms'][label]=dict(command=command(row,label),environment_sha256=signature(environment(row,label)))
    ROOT.mkdir(mode=0o700)
    for label in LABELS:
        for name in ('tmp','cache','trace'):(ROOT/label/name).mkdir(parents=True,mode=0o700)
    save(row);print('Prepared existing-image control/trace probe;commit before native execution.',flush=True)


def run(label):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(label in LABELS and [r['label'] for r in row['results']]==list(LABELS[:LABELS.index(label)])
        and all(r['passed'] for r in row['results']),'ordered one-shot collector stage required')
    root=ROOT/label
    with (root/'attempt.json').open('x') as stream:json.dump(dict(label=label),stream)
    cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in cursor.splitlines() if s.startswith('-- cursor: '))
    owner=None;result=dict(label=label,passed=False,complete=False,host_probe_accepted=False,
        native_execution_attempted=False,GPU_executed=None)
    def persist():core.write(root/'execution.json',dict(active=bool(owner and owner.active),commands=owner.commands if owner else []))
    owner=device.build.ownership.Owner(root/'owned',persist)
    try:
        begin=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        result['native_execution_attempted']=True
        entry,out,err=owner.run(label,command(row,label),environment(row,label),P,**BUDGET)
        end=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW);errors=err.read_text()
        result['observed_native_exit_code']=entry['exit_code']
        require(device.build.ownership.closed(entry) and entry['exit_code']==0,'normal owned native closure required')
        cases=device.gpu_cases('bank-smoke',out.read_text());validation=device.build.ur_validation.preload_evidence(errors,entry['pid'])
        device.diagnostics.diagnostics(errors,entry['pid'])
        require(validation['loader']==row['native_validation_probe']['loader']['path'] and 'urEnqueue' in errors,
            'actual qualified loader/submission evidence absent')
        result['GPU_executed']=True
        require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size,
            'parameter validation warning/error')
        result.update(cases=cases,validation=validation,providers=providers(errors,row,label,entry['pid']),
            native_identity=next(v for v in entry['owners'] if v['pid']==entry['pid']),
            native_exit_code=0,enclosing_raw_ns=dict(begin=begin,end=end))
        if label=='trace':
            files=list((root/'trace').glob('*.json'))
            require(len(files)==1 and files[0].name.endswith('.'+str(entry['pid'])+'.json'),'one exact target trace required')
            result.update(trace_summary(reader.load_json(files[0]),begin,end),trace_sha256=sha(files[0]))
        current(row);result.update(host_probe_accepted=True,complete=True,GPU_executed=True)
    except BaseException as error:result['error']=repr(error)
    finally:
        device.build.ur_validation.finish(result,owner,cursor,row['boot'])
        result['passed']=bool(result['passed'] and result.get('driver_after')==row['driver'] and result['complete'])
        row['results'].append(result);row['commands']+=owner.commands
        row.update(stage='closed' if label=='trace' and result['passed'] else 'ready' if result['passed'] else 'failed',
            passed=label=='trace' and result['passed'],complete=label=='trace' and result['passed'],
            active=bool(owner.active),ownership_closed=result['ownership_closed']);save(row)
    require(result['passed'],result.get('error','collector fixture admission failed'))
    print(label+' exact eager fixture/collector closed0;model/observer-cost/saturation remain pending.',flush=True)


if __name__=='__main__':
    require(sys.executable==capacity.reference.model.PYTHON and len(sys.argv)==2
        and sys.argv[1] in ('prepare',*LABELS),'pinned Python:prepare|control|trace')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device.build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:run(sys.argv[1])
