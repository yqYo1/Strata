#!/usr/bin/env python3
"""Main-owned official-SDK prefill Top-k public-entry qualification.

Repository LICENSE. Reuses the closed private controller's source/runtime/owner
checks. Prepare -> commit -> build -> commit -> reviewed bounded run. No legacy
compilation,implicit retry,model performance claim or automatic adoption.
"""
import ast
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode=True
import build_fused_profile as closure
import measurement_guards as guards
import measurement_owner as supervision
import topk_early_sources as generator
import host_link_inventory as host_link
import relink_qsa_k_model as linking

r,P,B=closure.r,closure.P,closure.B
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY='qsa_topk_prefill_entry_v1'
ROOT=B/'topk-prefill-entry-v1'
OFFICIAL=Path('/opt/intel/oneapi/compiler/2026.2')
NATIVE=['src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp','bench/sycl/topk_early.cpp',
    'src/kernels/sycl/qsa_select.cpp','src/kernels/sycl/qsa_topk_prefill_early.cpp']
FLAGS=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
    '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1','-pthread',
    '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat')]
BUILD_LIMITS=dict(wall=300,cpu=900,rss_cap=6<<30,text_cap=16<<20,file_cap=256<<20,total_cap=512<<20)
RUN_LIMITS=dict(wall=180,cpu=300,rss_cap=4<<30,text_cap=16<<20,file_cap=64<<20,total_cap=256<<20)
require,digest,write=guards.require,supervision.sha,closure.write



def git(*args):
    return r.build.git(P, *args)


def public():
    data=json.loads(RESULT.read_text())
    committed=json.loads(git('show','HEAD:'+str(RESULT.relative_to(P))))
    require(data.get(KEY)==committed.get(KEY),'commit exact Top-k plan/result')
    return data


def source_pins(revision):
    names=set(NATIVE) | set(git('ls-files','include').decode().splitlines()) | {
        'tools/sycl/test_topk_early.py','tools/sycl/test_topk_plan.py',generator.SOURCE,
        'src/prefill/prefill.cpp','cmake/sycl_backend.cmake'}
    pending=[Path(__file__).resolve()];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);names.add(str(path.relative_to(P)))
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[n.name for n in node.names] if isinstance(node,ast.Import) else \
                    [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                candidate=path.parent/((module or '').split('.')[0]+'.py')
                if candidate.is_file():pending.append(candidate)
    names.add('tools/sycl/measurement_environment.json')
    pins={}
    for name,raw in r.build.blobs(P,revision,sorted(names)):
        require((P/name).read_bytes()==raw,'commit Top-k input:'+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
    return pins


def packages():
    return subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n',
        'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2','intel-oneapi-dpcpp-cpp-2026.2',
        'intel-oneapi-umf-1.2'],text=True,timeout=15).splitlines()


def runtime():
    rt=r.validation.committed(r.RESULT)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and not rt['fresh_faults'],
            'closed qualified runtime required')
    return rt,r.current(rt)


def commands():
    compiler=str(OFFICIAL/'bin/icpx')
    units=[P/name for name in NATIVE]
    rows=[[compiler,*FLAGS,'-MD','-MF',str(ROOT/(str(i)+'.d')),'-c',str(path),'-o',str(ROOT/(str(i)+'.o'))]
          for i,path in enumerate(units)]
    rows.append([compiler,'-fsycl','-fsycl-id-queries-range=size_t','-pthread','-save-temps=obj','-v','-Wl,-t',
                 *[str(ROOT/(str(i)+'.o')) for i in range(len(units))],'-ldl','-o',str(ROOT/'selector_probe')])
    return rows


def verify(row):
    rt,full=runtime()
    require(digest(r.RESULT)==row['runtime_sha256'] and full['binary']==row['selected_binary'],
            'Top-k runtime/fallback changed')
    require(guards.driver_guard()==row['driver'] and rt['boot']==row['boot'],'Top-k driver/boot changed')
    require(packages()==row['sdk']['packages'] and digest(row['sdk']['path'])==row['sdk']['sha256'],
            'original official SDK manifest/packages changed')
    sdk=json.loads(Path(row['sdk']['path']).read_text())
    require(len(sdk)==row['sdk']['files'] and all(digest(p)==h for p,h in sdk.items()),'frozen official SDK/host bytes changed')
    require(all(digest(p)==h for p,h in row['runtime_files'].items()),'official component provider changed')
    require(host_link.inventory(str(OFFICIAL/'bin/icpx'),row['build_environment'])==row['host_link_inventory'],
            'prospective host linker input inventory changed')
    require(digest(row['inputs']['path'])==row['inputs']['sha256'],'Top-k input manifest changed')
    inputs=json.loads(Path(row['inputs']['path']).read_text())
    require(inputs['project']==source_pins(row['source_commit']),'Top-k source/controller changed')
    require(generator.prefill_proof()==row['prefill_proof'],'prefill body bridge differs')
    for name,h in inputs['external'].items():require(digest(name)==h,'frozen external input changed:'+name)
    require(commands()==row['commands'],'Top-k recipe changed')
    loader=next(p for p in row['runtime_files'] if Path(p).name.startswith('libze_loader.so.'))
    require(row['cases']==planned_cases(loader),'bounded public-entry cases changed')
    for name,value in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',
            EnableDirectSubmission='0',STRATA_SYCL_SELECT_RADIX4='1',STRATA_SYCL_SELECT_CYCLIC='0').items():
        require(row['native_environment'].get(name)==value,'public-entry safety/policy key changed:'+name)
    require(ENV_FLAG not in row['native_environment'],'set the route only in each pre-exec case')
    allowed={str((P/name).resolve()):h for name,h in inputs['project'].items()}
    allowed.update(inputs['external']);allowed.update(sdk);allowed.update(row['runtime_files'])
    return allowed,rt


def build(data):
    row=data[KEY];require(row['stage']=='prepared' and not (ROOT/'build').exists(),'fresh committed Top-k build')
    allowed,_=verify(row);owner=None
    owner=supervision.Owner(ROOT/'build',lambda:write(ROOT/'build-owner.json',owner.commands))
    try:
        row.update(stage='building',active=True,ownership_closed=False);write(RESULT,data)
        for i,argv in enumerate(row['commands']):
            entry,out,err=owner.run('compile-'+str(i),argv,row['build_environment'],ROOT,**BUILD_LIMITS)
            require(supervision.closed(entry) and entry['exit_code']==0,'Top-k compile/link failed:'+str(i))
            if i==len(row['commands'])-1:
                link_allowed=dict(allowed)
                link_allowed.update({str(ROOT/(str(i)+'.o')):digest(ROOT/(str(i)+'.o')) for i in range(len(row['commands'])-1)})
                links=linking.traced_inputs(out.read_text(),err.read_text(),link_allowed,ROOT,ROOT)
                write(ROOT/'link-inputs.json',links)
        observed={}
        for i in range(len(row['commands'])-1):
            text=(ROOT/(str(i)+'.d')).read_text().replace('\\\n',' ').split(':',1)[1]
            require('\\' not in text,'unexpected dependency escaping')
            for name in text.split():
                path=str(Path(name).resolve());h=digest(path)
                require(allowed.get(path)==h,'unfrozen compiler input:'+path);observed[path]=h
        require(all(str(P/name) in observed for name in NATIVE),'incomplete native source closure')
        write(ROOT/'compiled-inputs.json',observed);verify(row)
        binary=ROOT/'selector_probe'
        row.update(stage='built',build_passed=True,binary=dict(path=str(binary),bytes=binary.stat().st_size,sha256=digest(binary)),
                   compiled_inputs=dict(path=str(ROOT/'compiled-inputs.json'),sha256=digest(ROOT/'compiled-inputs.json'),files=len(observed),uncovered=0),
                   link_inputs=dict(path=str(ROOT/'link-inputs.json'),sha256=digest(ROOT/'link-inputs.json'),frozen=len(links['frozen_inputs']),uncovered=0))
    except BaseException as e:row.update(stage='build-failed',build_passed=False,error=repr(e))
    finally:
        row.update(active=owner.active is not None,ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands));write(RESULT,data)
    print(json.dumps({k:row.get(k) for k in ('stage','build_passed','ownership_closed','error')}),flush=True)


def fields(line):
    return dict(part.split('=',1) for part in line.split()[1:] if '=' in part)


ENV_FLAG='STRATA_SYCL_PREFILL_TOPK_EARLY'
ENDPOINTS=(1,3,4,5,31,32,33,2047,2048,2051,2052,32768,65536,117488,262141,262142,262143,262144)


def case_environment(base,case):
    env=dict(base)
    for name,value in case.get('environment',{}).items():
        if value is None:env.pop(name,None)
        else:env[name]=value
    return env


def planned_cases(loader):
    cases=[dict(kind='host-canary'),dict(kind='properties',argv=['--properties',loader])]
    policies=[('unset',None,'0',{}),('off','0','0',{}),('on','1','1',{}),
        ('empty','','invalid',{}),('text','true','invalid',{}),('partial','01','invalid',{}),
        ('missing-radix','1','invalid',{'STRATA_SYCL_SELECT_RADIX4':None}),
        ('disabled-radix','1','invalid',{'STRATA_SYCL_SELECT_RADIX4':'0'}),
        ('cyclic-conflict','1','invalid',{'STRATA_SYCL_SELECT_CYCLIC':'1'})]
    cases.extend(dict(kind='policy',label=name,argv=['--policy',expected],
        environment={ENV_FLAG:flag,**extra}) for name,flag,expected,extra in policies)
    cases.extend(dict(kind='smoke',label=name,argv=['--smoke'],
        environment={ENV_FLAG:flag},early=flag=='1') for name,flag in [('unset',None),('off','0'),('on','1')])
    cases.extend(dict(kind='numerics',label=name,argv=['--numerics'],
        environment={ENV_FLAG:flag},early=flag=='1') for name,flag in [('off','0'),('on','1')])
    return cases


def prepare(data):
    require(KEY not in data and not ROOT.exists(),'fresh public-entry scope only')
    proposal=data['qsa_topk_prefill_route'];component=data['qsa_topk_early_component_v2']
    require(proposal['stage']=='source-bridge-passed' and proposal['host_attempts'][-1]['passed'] and
        generator.prefill_proof()==proposal['source_proof'],'closed exact production source bridge required')
    require(component['passed'] and component['complete'] and component['ownership_closed'] and
        not component['active'] and not component['fresh_faults'] and component['retirement']['complete'] and
        component['decision']['mechanism_screen_passed'],'closed successful private component required')
    # Recover original preparation from reachable Git,never from retired manifests.
    prior=json.loads(git('show','519cd056a:'+str(RESULT.relative_to(P))))['qsa_topk_early_component_v2']
    rt,full=runtime();revision=git('rev-parse','HEAD').decode().strip();pins=source_pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    sdk=copy.deepcopy(prior['sdk'])
    require(sdk['path']==sdk['original_manifest']['path'] and sdk['sha256']==sdk['original_manifest']['sha256'],
            'original official SDK manifest required')
    old_lib=Path(next(p for p in rt['runtime_files'] if '/libsycl.so.' in p)).parent
    external={p:h for p,h in closure.external_inputs(rt).items() if not Path(p).is_relative_to(old_lib)}
    build_env=copy.deepcopy(prior['build_environment']);native_env=copy.deepcopy(prior['native_environment'])
    for env in (build_env,native_env):env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    inventory=host_link.inventory(str(OFFICIAL/'bin/icpx'),build_env);external.update(inventory['files'])
    runtime_files=dict(prior['runtime_files'])
    for name,h in inventory['files'].items():
        with Path(name).open('rb') as stream:elf=stream.read(4)==b'\x7fELF'
        if elf and '.so' in Path(name).name:runtime_files[name]=h
    write(ROOT/'inputs.json',dict(project=pins,external=external))
    native_env.pop('UR_LOG_TRACING',None);native_env.pop(ENV_FLAG,None)
    native_env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        STRATA_SYCL_SELECT_RADIX4='1',STRATA_SYCL_SELECT_CYCLIC='0')
    for name,value in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0').items():
        require(native_env.get(name)==value,'mandatory runtime safety key differs:'+name)
    loader=next(p for p in runtime_files if Path(p).name.startswith('libze_loader.so.'))
    row=dict(stage='prepared',source_commit=revision,inputs=dict(path=str(ROOT/'inputs.json'),sha256=digest(ROOT/'inputs.json')),
        prefill_proof=generator.prefill_proof(),sdk=sdk,runtime_sha256=digest(r.RESULT),selected_binary=full['binary'],
        driver=guards.driver_guard(),boot=rt['boot'],runtime_files=runtime_files,preload=prior['preload'],
        host_link_inventory=inventory,commands=commands(),build_workdir=str(ROOT),build_environment=build_env,
        native_environment=native_env,cases=planned_cases(loader),build_limits=BUILD_LIMITS,run_limits=RUN_LIMITS,
        active=False,ownership_closed=True,passed=False,
        contract='Exact numeric per-cell oracle and ascending IDs;public prefill selection,unchanged decode/short API and ref API,twice each with explicit/default active capacity. Persistent host DMA owners and last-member fail-stop drain.',
        numerical_plan='One compiled dedicated-kernel property;9startup-policy cases in separate pre-exec environments;traced17cell/3query/cap5smoke for unset/off/on;145cases in EACH off/on process over18context endpoints x8patterns plus14mixed-invalid/zero rows.60host admission refusals across3public APIs per numerical process. Full262144extent is selector-only,not physical model qualification.',
        acceptance='Complete all16ordered cases with zero exit/no new fault/no validation warning and exact expected environment/provider/ELF at3gates per native process. Require actual tiny-trace kernel creation to match early on/off and numerical output/guards/inputs for all3APIs. No retries or omission. This is public-entry correctness/source admission only;no timing or whole-model/decode throughput claim.',
        provenance='Main-authored from this fork\'s qualified44636e26c private controller/fixture and independent public APIs;repository LICENSE. Original build/environment basis519cd056a with immutable official SDK snapshot. No external implementation.',
        retention='One imminent small public-entry fixture executable,not a third engine. Preserve selected6e40/candidate9a79 for their named qualification. Keep compact per-process outcomes,case census/source/binary/faults;retire successful raw traces,objects and private fixture build after exact Git recovery at bridge closure. Ordinary UR/provider destruction remains unqualified under the retained-reference preload.')
    verify(row);data[KEY]=row;write(RESULT,data)
    print('Public-entry recipe and immutable inputs prepared;commit before build.',flush=True)


def case_checks(cases,smoke=False):
    expected=[(17,32,3,5,0,0)] if smoke else [
        (n,n,min(3,n),2051,p,0) for n in ENDPOINTS for p in range(8)]+[(65,128,14,13,6,1)]
    actual=[]
    for c in cases:
        require(all(c.get(k)=='PASS' for k in ('oracle','inputs','guards')) and
            c.get('repeats')=='2' and c.get('apis')=='3','public numerical/guard/API gate')
        actual.append(tuple(int(c[k]) for k in ('cells','capacity','queries','cap','pattern','invalid')))
    require(actual==expected,'public numerical geometry census differs')


def check_result(case,parsed,trace=''):
    kind=case['kind']
    require(not parsed['SAMPLE'],'public-entry correctness must not masquerade as a speed test')
    if kind=='properties':require(len(parsed['PROPERTY'])==1 and parsed['PROPERTY'][0]['PASS']=='1','compiled property gate')
    if kind=='policy':
        expected=case['argv'][-1];wanted=dict(expected=expected,invalid=str(int(expected=='invalid')),
            enabled=str(int(expected=='1')),PASS='1')
        require(parsed['POLICY']==[wanted],'actual startup policy differs')
    if kind in ('smoke','numerics'):
        require(parsed['ROUTE']==[dict(early=str(int(case['early'])),
            entry='qsa_block_topk_prefill_early' if case['early'] else 'qsa_block_topk_prefill')],
            'actual prefill dispatch differs')
        case_checks(parsed['CASE'],smoke=kind=='smoke')
        if kind=='smoke':
            require('urEnqueueKernelLaunch' in trace,'tiny public-entry trace missing')
            creations=[s for s in trace.splitlines() if 'urKernelCreate' in s]
            require(any('submit_topk_prefill_early' in s for s in creations)==case['early'],
                'actual early kernel creation differs from selected public route')
        else:require(parsed['ADMISSION']==[dict(refusals='60',apis='3',output='PASS')],'public host admission gate')

def kernel_faults(cursor):
    raw=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'],timeout=15)
    require(len(raw)<2<<20,'kernel fault evidence budget')
    return [s for s in [json.loads(line).get('MESSAGE','') for line in raw.decode().splitlines() if line]
        if re.search(r'(?:\bxe\b|drm|GPU).*(?:fault|hang|wedg|reset|timed out)',s,re.I)]


def run(data):
    row=data[KEY];require(row['stage']=='built' and row['build_passed'] and row['ownership_closed'],
        'commit successful Top-k build')
    require(row.get('source_review',{}).get('bounded_execution_accepted') is True,'review new fixture before GPU')
    verify(row);require(digest(row['binary']['path'])==row['binary']['sha256'] and not (ROOT/'native').exists(),
        'fresh verified native run only')
    raw=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in raw.splitlines() if s.startswith('-- cursor: '))
    owner=None;owner=supervision.Owner(ROOT/'native',lambda:write(ROOT/'native-owner.json',owner.commands))
    measurements=row['measurements']=[]
    try:
        row.update(stage='running',active=True,ownership_closed=False,complete=False);write(RESULT,data)
        for i,case in enumerate(row['cases']):
            env=case_environment(row['native_environment'],case);warning=ROOT/'native'/('case-'+str(i)+'-validation.log')
            trace=ROOT/'native'/('case-'+str(i)+'-trace.log')
            env['UR_LOG_VALIDATION']='level:warning;flush:warning;output:file,'+str(warning)
            if case['kind']=='smoke':env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(trace))
            if case['kind']=='host-canary':
                asset=row['preload']['canary'];require(digest(asset['path'])==asset['sha256'],'canary changed')
                entry,out,err=owner.run('case-'+str(i),[asset['path']],env,P,**RUN_LIMITS)
                require(supervision.closed(entry) and entry['exit_code']==0 and json.loads(out.read_text())==
                    dict(pid=entry['pid'],initialized=0,checked=46,expected=46,version=(1<<32)-1,closed=0),'host canary failed')
                require(r.validation.preload_evidence(err.read_text(),entry['pid'])['loader']==row['preload']['loader'],
                    'host loader differs')
                require(not warning.exists() or not warning.stat().st_size,'host canary warning')
                require(not kernel_faults(cursor),'fresh kernel fault');guards.driver_guard()
                measurements.append(dict(kind=case['kind'],passed=True,ownership_closed=True));write(RESULT,data);continue
            lines=[];gates=[]
            def observe(io,entry):
                while True:
                    line=io.line(seconds=60);lines.append(line)
                    if not line.startswith('GATE stage='):continue
                    stage=line.split('=',1)[1];require(len(gates)<3 and
                        stage==('before-runtime','runtime-ready','before-exit')[len(gates)],'native gate order')
                    proc=Path('/proc',str(entry['pid']))
                    require(str((proc/'exe').resolve())==row['binary']['path'] and digest(proc/'exe')==row['binary']['sha256'],
                        'actual native ELF differs')
                    actual=dict(v.split(b'=',1) for v in (proc/'environ').read_bytes().split(b'\0') if v)
                    require(actual=={k.encode():v.encode() for k,v in env.items()},'native environment differs')
                    mapped=r.executable_mappings((proc/'maps').read_text());allowed=set(row['runtime_files'])|{row['binary']['path']}
                    require(mapped<=allowed,'unfrozen native providers:'+repr(mapped-allowed))
                    for name in mapped:require(digest(name)==(row['binary']['sha256'] if name==row['binary']['path']
                        else row['runtime_files'][name]),'mapped provider changed')
                    if stage!='before-runtime':
                        require(any(s.startswith('DEVICE name=') and 'B570' in s for s in lines),'actual B570 missing')
                        require(any('libze_intel_gpu.so.1.17.39758' in p for p in mapped) and
                            any('libur_adapter_level_zero_v2.so' in p for p in mapped),'native driver/adapter missing')
                    gates.append(stage);io.send(b'continue\n')
                    if stage=='before-exit':io.exit(10);break
            entry,out,err=owner.run('case-'+str(i),[row['binary']['path'],*case['argv']],env,P,
                interaction=observe,**RUN_LIMITS)
            require(supervision.closed(entry) and entry['exit_code']==0 and len(gates)==3,'native owner/exit failed')
            require(any(s.startswith('DONE ') for s in lines) and not any('FAIL' in s for s in lines),'native completion failed')
            require(r.validation.preload_evidence(err.read_text(),entry['pid'])['loader']==row['preload']['loader'],
                'actual official UR loader differs')
            require(not warning.exists() or not warning.stat().st_size,'native validation warning')
            require(lines.count('COMPARATOR refusals=3 identity=PASS')==1,'comparator controls absent')
            parsed={prefix:[fields(s) for s in lines if s.startswith(prefix+' ')]
                for prefix in ('PROPERTY','CASE','ADMISSION','ROUTE','POLICY','SAMPLE')}
            check_result(case,parsed,trace.read_text() if trace.exists() else '')
            measurements.append(dict(kind=case['kind'],label=case.get('label'),argv=case['argv'],passed=True,
                process={k:entry.get(k) for k in ('pid','exit_code','elapsed_seconds','peak_session_rss_bytes',
                    'observation_complete','direct_child_reaped','session_empty')},
                cases=parsed['CASE'],properties=parsed['PROPERTY'],admission=parsed['ADMISSION'],
                route=parsed['ROUTE'],policy=parsed['POLICY']))
            write(RESULT,data);require(not kernel_faults(cursor),'fresh kernel fault');guards.driver_guard()
        verify(row);require(digest(row['binary']['path'])==row['binary']['sha256'],'native ELF changed')
        require(len(measurements)==16,'all planned public-entry processes required')
        row['decision']=dict(public_entry_correctness_passed=True,inference_adopted=False,
            full_model_context_qualified=False,decode_throughput_qualified=False)
        row.update(stage='closed',complete=True,passed=True)
    except BaseException as e:row.update(stage='failed',passed=False,error=repr(e))
    finally:
        try:row.update(fresh_faults=kernel_faults(cursor),driver_after=guards.driver_guard())
        except BaseException as e:row.update(passed=False,fault_evidence_error=repr(e))
        row.update(active=owner.active is not None,ownership_closed=owner.active is None and
            all(supervision.closed(c) for c in owner.commands))
        row['passed']=bool(row.get('passed') and row['ownership_closed'] and not row.get('fresh_faults'));write(RESULT,data)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','error')}),flush=True)


def main():
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build','run'),'choose prepare,build,run')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        data=public();{'prepare':prepare,'build':build,'run':run}[sys.argv[1]](data)


if __name__=='__main__':main()
