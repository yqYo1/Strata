#!/usr/bin/env python3
"""Bounded current-runtime PTI safety/clock probe on an existing fixture.

Main-authored; repository LICENSE. No rebuild, retired graph fixture, metrics,
model speed, complete kernel-coverage or prefill-boundary qualification.
"""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import current_pti_admission as admission
import request_profile_reader as reader
import ur_validation as validation

P, B = admission.CHECKOUT, admission.pti.B
RESULT = admission.RESULT
ROOT, KEY = B / 'current-pti-probe-v1', 'unitrace_current_runtime_probe_v1'
THIS = 'tools/sycl/current_pti_probe.py'
require, digest = admission.require, admission.digest
HOST_CHECK = dict(command=['/home/yayoi/.local/share/strata-sycl/venv/bin/python3',
    '-B', '-m', 'unittest', 'tools.test_current_pti_probe'], wall_seconds=90,
    CPU_seconds=60, RSS_bytes=1 << 30, text_bytes=128 << 10, total_bytes=4 << 20,
    GPU_executed=False)


def source_pin():
    revision = admission.pti.git(P, 'rev-parse', 'HEAD').decode().strip()
    names = admission.pti.git(P, 'ls-tree', '-r', '--name-only', revision, 'tools/sycl',
        'tools/test_current_pti_probe.py', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md').decode().splitlines()
    pins = {}
    for name in names:
        raw = admission.blob(P, revision, name)
        require(raw == (P / name).read_bytes(), 'commit every controller/helper before this probe')
        pins[name] = admission.pti.sha(raw)
    require(THIS in pins and 'tools/test_current_pti_probe.py' in pins,
            'the actual runner and its controls must be committed')
    return dict(revision=revision, files=len(pins), inputs=pins)


def basis():
    container = admission.committed_json(RESULT, P)
    gate = container['unitrace_current_admission_host_v3']
    require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and gate['unit_tests'] == 4,
            'closed current direct host admission required')
    for name, expected in gate['source_sha256'].items():
        require(digest(P / name) == expected, 'qualified admission/reader source changed')
    now = admission.admit()
    require({k:v for k,v in now.items() if k != 'controller_commit'} ==
            {k:v for k,v in gate['admission'].items() if k != 'controller_commit'},
            'only an uncompiled probe may extend the closed current admission')
    native, _ = admission.native.verify_original()
    return now, native['binaries']['sycl_submission_test'], native['native_validation_probe']


def environment(row, label):
    root = ROOT / label
    env = dict(row['base_environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
        STRATA_MEASUREMENT_ROOT=str(root), STRATA_TEST_MAIN_GATES='1',
        STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'],
        LD_DEBUG='libs', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.log'))
    require(env.get('ZES_ENABLE_SYSMAN') == '0' and env.get('EnableDirectSubmission') == '0'
            and env.get('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD') == '1'
            and 'LD_PRELOAD' not in env and env.get('ZET_ENABLE_METRICS', '0') != '1',
            'preserve qualified safety and target-only preload')
    if label == 'control': env['LD_PRELOAD'] = row['preload']['library']['path']
    else:
        require(label == 'trace', 'declared probe arm required')
        env.update(STRATA_UNITRACE_TARGET_PRELOAD=row['preload']['library']['path'],
            UNITRACE_SystemTime='1', UNITRACE_FollowChildProcess='1', UNITRACE_TraceOutputDir=str(root / 'trace'))
    return env


def command(row, label):
    target = [row['fixture']['path'], '--submission']
    if label == 'control': return target
    require(label == 'trace', 'declared probe arm required')
    return [row['admission']['tools']['artifacts']['unitrace']['path'], '--device-timing',
        '--chrome-device-logging', '--chrome-event-buffer-size', '128', *target]


def compact(row):
    result = dict(row)
    result['controller'] = {k:v for k,v in row['controller'].items() if k != 'inputs'}
    return result


def save(row):
    (ROOT / 'record.json').write_text(json.dumps(row, indent=2)+'\n')
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2)+'\n')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'one fresh probe plan')
    admitted, fixture, preload = basis()
    require(set(preload)=={'loader','library'}, 'exact qualified validation preload schema')
    row = dict(stage='prepared', controller=source_pin(), admission=admitted, fixture=fixture,
        preload=preload, base_environment=json.loads((P/'tools/sycl/measurement_environment.json').read_text())['environment'],
        boot=admitted['boot'], driver=admitted['driver'], results=[], active=False, passed=False, complete=False,
        ownership_closed=True, GPU_executed=False,
        plan='Commit source/plan;control then trace in separate owned processes,commit each closure. '
             'At before-runtime,runtime-ready,before-exit require actual target/preload/current mapped ELF '
             'hashes,first-init validation proof,ten submission cases and normal0. Trace needs nonempty '
             'complete GPUevents inside actual enclosing CLOCK_MONOTONIC_RAW process bounds with '
             'SystemTime1. Stop first identity,source,provider,owner,validation,fault,budget or clock failure;no retry.',
        limits='Small fixture safety/clock gate only. No complete service/count coverage,physical bus '
               'traffic,prefill critical path,model request,observer-overhead or throughput qualification. '
               'Retained validation reference does not qualify ordinary provider teardown. Old failures remain.',
        retention='Use the existing fast14508same-build fixture and surviving collector;no binary generation. '
                  'Retain compact identity/map/clock/correctness/fault outcomes,retire normal trace/cache '
                  'after verification. Next consumer is a separately committed32Krequest/overhead plan.')
    ROOT.mkdir(mode=0o700)
    for label in ('control', 'trace'):
        root=ROOT/label;root.mkdir()
        for name in ('tmp','cache','trace'): (root/name).mkdir()
        environment(row,label);command(row,label)
    save(row)
    print('Prepared current PTI fixture probe;commit before control/trace.')


def mapped(pid, row, label):
    allowed = json.loads(Path(row['admission']['runtime_manifest']['path']).read_text())
    allowed.update({v['path']:v['sha256'] for v in row['preload'].values()})
    allowed[row['fixture']['path']] = row['fixture']['sha256']
    if label == 'trace':
        item=row['admission']['tools']['artifacts']['libunitrace_tool.so'];allowed[item['path']]=item['sha256']
    allowed={str(Path(p).resolve()):h for p,h in allowed.items()}
    paths=set()
    for line in Path('/proc',str(pid),'maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or not fields[5].startswith('/'): continue
        path=Path(fields[5]);require(not str(path).endswith(' (deleted)'), 'deleted owned mapped file')
        with path.open('rb') as stream:
            if stream.read(4)!=b'\x7fELF':continue
        real=str(path.resolve());require(real in allowed and digest(real)==allowed[real], 'unknown/changed mapped ELF: '+real)
        paths.add(real)
    required={str(Path(v['path']).resolve()) for v in row['preload'].values()}
    required.add(str(Path(row['fixture']['path']).resolve()))
    if label=='trace':required.add(str(Path(row['admission']['tools']['artifacts']['libunitrace_tool.so']['path']).resolve()))
    require(required <= paths, 'actual target/preload/collector mapping incomplete')
    return dict(mapped_ELFs=len(paths),unknown=0,missing=0,paths_sha256=admission.pti.sha(json.dumps(sorted(paths)).encode()))


def run(label):
    row=json.loads((ROOT/'record.json').read_text())
    require(admission.committed_json(RESULT,P)[KEY]==compact(row), 'commit exact prior probe stage')
    require([x['label'] for x in row['results']]==([] if label=='control' else ['control'])
            and all(x['passed'] for x in row['results']), 'one planned stage without retry')
    require(all(digest(P/n)==h for n,h in row['controller']['inputs'].items()), 'frozen source changed')
    admitted,fixture,preload=basis()
    require({k:v for k,v in admitted.items() if k!='controller_commit'} ==
            {k:v for k,v in row['admission'].items() if k!='controller_commit'}
            and fixture==row['fixture'] and preload==row['preload'], 'frozen assets changed')
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
    result=dict(label=label,passed=False,host_probe_accepted=False,clock_domain=reader.CLOCK,maps=[],
        native_execution_attempted=False,GPU_work_admitted=False,GPU_executed=None)
    owner=None
    def persist():
        (ROOT/label/'execution.json').write_text(json.dumps(dict(
            active=bool(owner and owner.active),commands=owner.commands if owner else []),indent=2)+'\n')
    owner=admission.pti.supervision.Owner(ROOT/label/'owned',persist)
    start=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    def interact(io, entry):
        for phase in ('before-runtime','runtime-ready','before-exit'):
            while True:
                line=io.line(90)
                if line=='COMPONENT_READY '+phase:break
                require(line.startswith('case '),'unexpected fixture protocol: '+line)
            result['maps'].append(dict(phase=phase,**mapped(entry['pid'],row,label)))
            if phase=='runtime-ready':result['GPU_work_admitted']=True
            io.send(b'X')
        require(io.line(30)=='case teardown=PASS' and io.line(30)=='fixture_status=PASS mode=submission', 'final numeric fixture status')
        io.exit(90)
    try:
        result['native_execution_attempted']=True
        entry,out,err=owner.run(label,command(row,label),environment(row,label),P,interaction=interact,
            wall=120,cpu=120,rss_cap=2<<30,text_cap=16<<20,file_cap=64<<20,total_cap=128<<20)
        end=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        text=out.read_text();errors=err.read_text()
        require(admission.pti.supervision.closed(entry) and entry['exit_code']==0
                and len([s for s in text.splitlines() if s.startswith('case ')])==10, 'ten numerical cases and normal closure required')
        result['GPU_executed']=True
        proof=validation.preload_evidence(errors,entry['pid'])
        require(proof['loader']==preload['loader']['path'] and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called',errors), 'first-init/validation proof failed')
        if label=='trace':
            events=[];files=sorted((ROOT/label/'trace').glob('*.json'))
            require(len(files)==1,'one bounded current trace required')
            for event in reader.load_json(files[0])['traceEvents']:
                if event.get('cat')!='gpu_op':continue
                require(event.get('ph')=='X','complete GPUinterval required')
                low,duration=reader.nanoseconds(event.get('ts')),reader.nanoseconds(event.get('dur'))
                require(start<=low<=low+duration<=end,'trace clock differs from actual enclosing RAW bounds')
                events.append((low,low+duration))
            require(events and any(b>a for a,b in events),'nonempty timed GPUwork required')
            result.update(trace_sha256=digest(files[0]),GPU_events=len(events),GPU_service_ns=sum(b-a for a,b in events),
                observed_GPU_union_ns=reader.union_ns(events),raw_clock_mapping_checked=True)
        require(all(digest(P/n)==h for n,h in row['controller']['inputs'].items()), 'source changed during native probe')
        after,after_fixture,after_preload=basis()
        require({k:v for k,v in after.items() if k!='controller_commit'} ==
                {k:v for k,v in row['admission'].items() if k!='controller_commit'}
                and after_fixture==fixture and after_preload==preload, 'assets changed during native probe')
        result.update(passed=True,host_probe_accepted=True,native_exit_code=0,GPU_executed=True,
            enclosing_raw_ns=dict(begin=start,end=end),preload_evidence=proof)
    except BaseException as error:result['error']=repr(error)
    finally:
        validation.finish(result,owner,cursor,row['boot'])
        result['passed']=bool(result['passed'] and result.get('driver_after')==row['driver'])
        row['results'].append(result);row.update(stage='closed' if label=='trace' and result['passed'] else 'ready' if result['passed'] else 'failed',
            passed=result['passed'],complete=label=='trace' and result['passed'],active=False,
            ownership_closed=result['ownership_closed'],GPU_executed=result['GPU_executed'])
        save(row)
    require(result['passed'],result.get('error','current native probe failed'))
    print(label+' current fixture/collector stage closed;model/overhead remain pending.')


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','control','trace'),'prepare|control|trace')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);admission.pti.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:run(sys.argv[1])
