#!/usr/bin/env python3
"""Serial traced qualification of actual pair-model production objects."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode=True
import build_qsa_pair_model as build
import measurement_guards as guards
from measure_down_scatter import postchecks

P,B,ROOT,RESULT,KEY=build.P,build.B,build.ROOT,build.RESULT,build.KEY
runtime=build.runtime
require,digest,write=build.require,build.digest,build.write
MODES=['properties','union','small4','small16','correctness4','correctness16']


def inputs(revision):
    pending=[Path(__file__).resolve()];seen=set();pins={}
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);name=str(path.relative_to(P));raw=build.model.git('show',revision+':'+name)
        require(path.read_bytes()==raw,'commit exact controller:'+name);pins[name]=build.model.sha(raw)
        for node in ast.walk(ast.parse(raw)):
            mods=[v.name for v in node.names] if isinstance(node,ast.Import) else \
                [node.module] if isinstance(node,ast.ImportFrom) else []
            for mod in mods:
                local=path.parent/((mod or '').split('.')[0]+'.py')
                if local.is_file():pending.append(local)
    return pins


def admitted(record):
    row=record[KEY]['build'];require(row['stage']=='built-not-executed' and row['passed'] and
        row['model_built'] and row['ownership_closed'] and not row['active'],'closed successful model build required')
    full=json.loads((ROOT/'record.json').read_text())
    require(build.compact(full)==row,'model build receipt differs');build.admit(full)
    for item in [*row['binaries'].values(),row['fixture'],*row['common_production_objects'].values()]:
        require(digest(item['path'])==item['sha256'],'production artifact changed')
    manifest=row['actual_compiler_inputs'];require(digest(manifest['path'])==manifest['sha256'],'dependency evidence changed')
    for path,sha in json.loads(Path(manifest['path']).read_text()).items():require(digest(path)==sha,'compiled input changed:'+path)
    rt=runtime.validation.committed(runtime.RESULT);runtime.current(rt)
    return row,rt


def prepare(mode):
    record=runtime.validation.committed(RESULT);native,rt=admitted(record)
    q=record[KEY].setdefault('qualification',{});require(mode not in q,'no retry or overwrite of a gate')
    position=MODES.index(mode)
    if position:
        prior=q[MODES[position-1]]['outcome'];require(prior['passed'] and prior['complete'] and prior['ownership_closed'],'prior gate failed/open')
    folder=ROOT/mode;require(not folder.exists(),'fresh gate folder required')
    native_mode='properties' if mode=='properties' else 'union' if mode=='union' else re.sub(r'\d+$','',mode)
    repeats=1 if mode in ['properties','union'] else 3
    cases=[]
    for i in range(repeats):
        run=folder/f'process-{i}';env=dict(rt['environment'],TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),
            ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
            UR_LOG_TRACING='level:info;flush:info;output:file,'+str(run/'api.trace'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(run/'validation.log'))
        arg=next(p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')) if mode=='properties' else '16' if mode.endswith('16') else '4'
        cases.append(dict(label=f'process-{i}',command=[native['fixture']['path'],native_mode,arg],environment=env))
    revision=build.model.git('rev-parse','HEAD').decode().strip()
    plan=dict(stage='prepared',source_commit=revision,controller_inputs=inputs(revision),binary=native['fixture'],
        runtime_files=rt['runtime_files'],preload=rt['preload'],cases=cases,native_mode=native_mode,
        expected_cases={'properties':4,'union':38,'small':8,'correctness':19}[native_mode],
        budgets=dict(wall=240,cpu=900,rss_cap=10<<30,text_cap=24<<20,total_cap=256<<20,file_cap=64<<20),
        criteria='Bounded property query then38set cases,three8case small processes at eachPage4/16,three19case extended processes at eachPage4/16. Actual model production objects;independent host set/BF16oracle,malformed metadata/NaNisolation,full-output pack and wrapper/core equivalence,guards,input immutability,actual ABI/local arguments and launch counts. Pre-main parameter validation,immutable qualified runtime/environment,B570DRM,closed owners and no fresh fault. Stop first failure. Full-address components are not model full-context fill or RESTORE;no speed/adoption from these tests.',
        retention='Keep compact numerical/property/actual-dispatch outcomes and source identities. Delete successful traces and logs after verification;no full-history consumer.')
    folder.mkdir(mode=0o700)
    for case in cases:
        run=folder/case['label'];run.mkdir();(run/'tmp').mkdir();(run/'cache').mkdir()
    write(folder/'plan.json',plan);q[mode]=plan;write(RESULT,record);print('Prepared '+mode+';commit before run.',flush=True)


def live_native(pid,environment,rt,binary):
    proc=Path('/proc',str(pid));actual=dict(v.split(b'=',1) for v in (proc/'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(actual.get(k.encode())==v.encode() for k,v in environment.items()),'native environment mutated')
    mapped=runtime.executable_mappings((proc/'maps').read_text())
    require(mapped<=set(rt['runtime_files'])|{binary['path']},'unknown executable provider:'+str(mapped-set(rt['runtime_files'])-{binary['path']}))
    neo=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p)
    require(neo in mapped and str(Path(environment['UR_ADAPTERS_FORCE_LOAD']).resolve()) in mapped and rt['preload']['loader'] in mapped,'native provider absent')
    drm=[]
    for fd in (proc/'fdinfo').iterdir():
        fields=re.findall(r'^(drm-driver|drm-pdev):\s*(\S+)\s*$',fd.read_text(),re.M)
        if fields:
            require(len(fields)==2 and dict(fields)=={'drm-driver':'xe','drm-pdev':'0000:05:00.0'},'wrong native DRM device');drm.append(fd.name)
    require(drm,'B570descriptor absent')
    return dict(mapped_qualified_runtime=True,observed_NEO=neo,DRM_descriptors=len(drm),environment_immutable=True)


def launches(trace,properties,mode):
    identities={p['kernel_name_sha256']:p for p in properties};created={};counts=dict.fromkeys(['pack','union','selected','attention'],0)
    local={'pack':[],'union':[],'selected':[16384,1024,64,64,64,256,64,2112],
        'attention':[16384,1536,3168,96,96,96,256,128,96]}
    for line in trace.splitlines():
        if '<--- urKernelCreate(' in line and 'UR_RESULT_SUCCESS' in line:
            match=re.search(r'\((_ZTS\S+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)',line)
            if match:created[match[2]]=match[1]
        if '<--- urEnqueueKernelLaunchWithArgsExp(' not in line or 'UR_RESULT_SUCCESS' not in line:continue
        handle=re.search(r'\.hKernel = (0x[0-9a-f]+)',line);require(handle and handle[1] in created,'kernel launch without creation')
        prop=identities.get(hashlib.sha256(created[handle[1]].encode()).hexdigest())
        if prop is None:continue # Original pack and explicit dependency guard.
        kind=prop['kind'];counts[kind]+=1
        require('.workDim = 1,' in line and '.numArgs = '+prop['args']+',' in line and
            re.search(r'\.pLocalWorkSize = 0x[0-9a-f]+ \('+prop['wg']+r'\)',line),'actual launch ABI/geometry differs')
        sizes=[int(v) for v in re.findall(r'UR_EXP_KERNEL_ARG_TYPE_LOCAL, \.index = \d+, \.size = (\d+)',line)]
        require(sorted(sizes)==sorted(local[kind]),'actual local arguments differ')
    expected=dict(zip(counts,[0,38,0,0] if mode=='union' else [6,12,8,14] if mode=='small' else [13,32,19,32]))
    require(counts==expected,'actual production dispatch counts differ:'+str(counts));return counts


def run(mode):
    record=runtime.validation.committed(RESULT);plan=record[KEY]['qualification'][mode];folder=ROOT/mode
    require(plan['stage']=='prepared' and plan==json.loads((folder/'plan.json').read_text()) and not (folder/'owned.json').exists(),'fresh frozen gate required')
    native,rt=admitted(record);require(inputs(plan['source_commit'])==plan['controller_inputs'] and plan['binary']==native['fixture'] and plan['runtime_files']==rt['runtime_files'],'gate input changed')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=re.findall(r'^-- cursor: (.+)$',text,re.M);require(len(cursor)==1,'kernel fault cursor unavailable')
    outcome=dict(passed=False,complete=False,active=False,ownership_closed=False,processes=[],model_qualified=False);owner=None
    def persist():
        outcome['active']=bool(owner and owner.active is not None)
        write(folder/'owned.json',dict(outcome,commands=owner.commands if owner else []))
    owner=runtime.supervision.Owner(folder/'owned',persist)
    try:
        for case in plan['cases']:
            observed={}
            def interaction(io,entry):
                require(io.line(30)=='PAIR_GATE before-runtime','native entry differs');io.send(b'X')
                require(io.line(90)=='PAIR_GATE runtime-ready','runtime gate failed')
                observed['start']=live_native(io.proc.pid,case['environment'],rt,plan['binary']);io.send(b'X')
                while io.line(210)!='PAIR_GATE before-exit':pass
                observed['end']=live_native(io.proc.pid,case['environment'],rt,plan['binary']);io.send(b'X');io.proc.stdin.close();io.exit(25)
            entry,out,err=owner.run(case['label'],case['command'],case['environment'],P,interaction=interaction,**plan['budgets'])
            require(runtime.supervision.closed(entry) and entry['exit_code']==0,'native failed or owner open')
            stdout,stderr=out.read_text(),err.read_text();proof=runtime.validation.preload_evidence(stderr,entry['pid'])
            require(proof['loader']==rt['preload']['loader'],'pre-main provider mismatch')
            body=stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|PAIR_ERROR',body),'native runtime error')
            runfolder=folder/case['label'];warnings=runfolder/'validation.log'
            require(not warnings.exists() or warnings.stat().st_size==0,'validation warning/error')
            trace=(runfolder/'api.trace').read_text()
            if mode=='properties':require('urKernelGet' in trace and not re.search(r'urEnqueue\w*Kernel',trace),'property query/enqueue scope differs')
            else:require(re.search(r'urEnqueue\w*Kernel',trace),'actual kernel enqueue absent')
            prefix='PAIR_PROPERTY ' if mode=='properties' else 'PAIR_CASE '
            lines=[line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
            require(len(lines)==plan['expected_cases'] and stdout.count('PAIR_DONE mode='+plan['native_mode']+' passed=1')==1,'case coverage incomplete')
            rows=[]
            for line in lines:
                row=dict(part.split('=',1) for part in line.split())
                if 'name' in row:row['kernel_name_sha256']=hashlib.sha256(row.pop('name').encode()).hexdigest()
                rows.append(row)
            dispatch=None if mode=='properties' else launches(trace,record[KEY]['qualification']['properties']['outcome']['processes'][0]['cases'],plan['native_mode'])
            outcome['processes'].append(dict(label=case['label'],passed=True,cases=rows,dispatch=dispatch,validation=proof,observed=observed))
            post=dict(passed=True,ownership_closed=True);postchecks(post,boot,cursor[0]);require(post['passed'],'fresh fault/boot check failed')
            admitted(record);persist();print('Native '+mode+' closed '+case['label'],flush=True)
        outcome.update(passed=True,complete=True)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=owner.active is None and all(runtime.supervision.closed(c) for c in owner.commands)
        outcome['active']=False;postchecks(outcome,boot,cursor[0]);persist()
        plan['stage']='closed' if outcome['passed'] else 'failed';plan['outcome']=outcome;write(RESULT,record)
    print(json.dumps({k:v for k,v in outcome.items() if k!='processes'}),flush=True)
    return int(not outcome['passed'])


if __name__=='__main__':
    arg=sys.argv[1];mode=arg.removeprefix('prepare-');require(mode in MODES,'unknown native qualification mode')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        if arg.startswith('prepare-'):prepare(mode)
        else:raise SystemExit(run(mode))
