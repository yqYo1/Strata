#!/usr/bin/env python3
"""Main-owned official-SDK exact-set Top-k component;repository LICENSE.

Prepare -> commit -> build -> commit -> reviewed staged run. The common build,
source/runtime/owner checks derive from e10997bdd's closed normalizer controller.
No production selection,legacy compilation,implicit retry or automatic adoption.
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
KEY='qsa_topk_early_component_v2'
ROOT=B/'topk-early-component-v2'
OFFICIAL=Path('/opt/intel/oneapi/compiler/2026.2')
NATIVE=['src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp','bench/sycl/topk_early.cpp']
FLAGS=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
    '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1','-pthread',
    '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-I'+str(ROOT/'generated')]
BUILD_LIMITS=dict(wall=300,cpu=900,rss_cap=6<<30,text_cap=16<<20,file_cap=256<<20,total_cap=512<<20)
RUN_LIMITS=dict(wall=600,cpu=900,rss_cap=5<<30,text_cap=32<<20,file_cap=128<<20,total_cap=512<<20)
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
        'tools/sycl/test_topk_early.py','tools/sycl/test_topk_plan.py',generator.SOURCE}
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
    units=[P/name for name in NATIVE]+[ROOT/'generated'/name for name in ('original.cpp','candidate.cpp')]
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
    require(generator.proof()==row['generator_proof'],'Top-k generator differs')
    generated=generator.generated()
    require({name:hashlib.sha256(raw).hexdigest() for name,raw in generated.items()}==inputs['generated'],
            'prospective generated source changed')
    for name,raw in generated.items():require((ROOT/'generated'/name).read_bytes()==raw,'actual generated source changed:'+name)
    for name,h in inputs['external'].items():require(digest(name)==h,'frozen external input changed:'+name)
    require(commands()==row['commands'],'Top-k recipe changed')
    allowed={str((P/name).resolve()):h for name,h in inputs['project'].items()}
    allowed.update({str(ROOT/'generated'/name):h for name,h in inputs['generated'].items()})
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
        require(all(str(ROOT/'generated'/name) in observed for name in ('original.cpp','candidate.cpp','topk-early.hpp')),
                'incomplete generated compiler input closure')
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


def prepare(data):
    require(KEY not in data and not ROOT.exists(),'fresh Top-k scope only')
    proposal=data['qsa_topk_early_completion']
    require(proposal['stage']=='host-model-only' and proposal['host_controls']['passed'] and
        not proposal['GPU_executed'] and generator.proof()==proposal['generator_proof'],
        'original host model and unchanged selector constructor required')
    prior=json.loads(git('show','0353cb803:'+str(RESULT.relative_to(P))))['qsa_parallel_normalizer_v3']
    closed=data['qsa_parallel_normalizer_v3']
    require(closed['passed'] and closed['ownership_closed'] and not closed['active'] and
        not closed['fresh_faults'] and closed['retirement']['complete'],'closed official component basis required')
    rt,full=runtime();revision=git('rev-parse','HEAD').decode().strip();pins=source_pins(revision)
    predecessor=data['qsa_topk_early_component_v1']
    require(predecessor['build_passed'] and predecessor['ownership_closed'] and not predecessor['active'] and
        predecessor['source_review']['source_mechanism_supported'] and
        not predecessor['source_review']['bounded_execution_accepted'],
        'preserve unexecuted V1 source-review condition')
    require(digest(predecessor['inputs']['path'])==predecessor['inputs']['sha256'],'original V1 input manifest changed')
    previous=json.loads(Path(predecessor['inputs']['path']).read_text())
    require(set(pins)==set(previous['project']),'V2 project input set differs')
    changed=sorted(n for n,h in pins.items() if h!=previous['project'][n])
    require(changed==['tools/sycl/test_topk_plan.py','tools/sycl/topk_early.py'],
        'V2 may change only controller and host plan tests')
    require(generator.proof()==predecessor['generator_proof'],'V2 GPU constructor differs')
    require({n:hashlib.sha256(v).hexdigest() for n,v in generator.generated().items()}==previous['generated'],
        'V2 generated native inputs differ')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','generated'):(ROOT/name).mkdir(mode=0o700)
    generated=generator.generated()
    for name,raw in generated.items():(ROOT/'generated'/name).write_bytes(raw)
    sdk=copy.deepcopy(prior['sdk'])
    require(sdk['path']==sdk['original_manifest']['path'] and
        sdk['sha256']==sdk['original_manifest']['sha256'],'use original official SDK manifest')
    old_lib=Path(next(p for p in rt['runtime_files'] if '/libsycl.so.' in p)).parent
    external={p:h for p,h in closure.external_inputs(rt).items() if not Path(p).is_relative_to(old_lib)}
    build_env=copy.deepcopy(prior['build_environment']);native_env=copy.deepcopy(prior['native_environment'])
    for env in (build_env,native_env):env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    inventory=host_link.inventory(str(OFFICIAL/'bin/icpx'),build_env);external.update(inventory['files'])
    runtime_files=dict(prior['runtime_files'])
    for name,h in inventory['files'].items():
        with Path(name).open('rb') as stream:elf=stream.read(4)==b'\x7fELF'
        if elf and '.so' in Path(name).name:runtime_files[name]=h
    write(ROOT/'inputs.json',dict(project=pins,generated={n:hashlib.sha256(v).hexdigest() for n,v in generated.items()},external=external))
    native_env.pop('UR_LOG_TRACING',None);native_env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'
    for name,value in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0').items():
        require(native_env.get(name)==value,'mandatory runtime safety key differs:'+name)
    cases=[dict(kind='host-canary'),dict(kind='properties',argv=['--properties',
        next(p for p in runtime_files if Path(p).name.startswith('libze_loader.so.'))]),
        dict(kind='smoke',argv=['--smoke']),dict(kind='numerics',argv=['--numerics'])]
    cases += [dict(kind='compare',argv=['--compare',str(i)]) for i in range(3)]
    row=dict(stage='prepared',source_commit=revision,inputs=dict(path=str(ROOT/'inputs.json'),sha256=digest(ROOT/'inputs.json')),
        generator_proof=generator.proof(),sdk=sdk,runtime_sha256=digest(r.RESULT),selected_binary=full['binary'],
        driver=guards.driver_guard(),boot=rt['boot'],runtime_files=runtime_files,preload=prior['preload'],
        host_link_inventory=inventory,commands=commands(),build_workdir=str(ROOT),build_environment=build_env,
        native_environment=native_env,cases=cases,build_limits=BUILD_LIMITS,run_limits=RUN_LIMITS,
        active=False,ownership_closed=True,passed=False,
        predecessor=dict(scope='qsa_topk_early_component_v1',source_commit=predecessor['source_commit'],
            source_build_record='ed5f7680f',original_inputs=copy.deepcopy(predecessor['inputs']),
            original_binary=copy.deepcopy(predecessor['binary']),changed_project_inputs=changed,
            unchanged_project_inputs=len(pins)-len(changed),generated_inputs_unchanged=True,
            scope_limit='Same native fixture/headers/GPU constructor bytes for source-audit applicability only. New official build required;no native execution qualification is inherited.'),
        contract='Exact selected integer IDs,including ascending ties,NaNs,zeros,partial4-cell budgets and zero-weight spare. Independent numeric per-cell ranking;capture/normal output correspondence and exact pass counts. No floating arithmetic change or production/decode hook.',
        numerical_plan='Host canary;four normal/counter compiled properties;traced17cell/3query/cap5smoke;26host admission refusals and145device cases over18context endpoints x8patterns plus mixed invalid/zero steps. Actual final262144addresses are selector component coverage,not full-model context qualification.',
        performance_plan='Three fresh processes;6scenarios(context32768/65536/117488 x positive varied/quantized-tie scores),256causal queries,capacity131072,cap2051,1024calls/batch,checked drain every64calls. Qualification warms each normal arm twice plus one capture. Retain first timing pair(-1) separately,then3rotating matched pairs(0..2).144batches total. AllCPU references,uploads,allocations and output checks outside timing. Direct actual kernel events and enclosing wall;both>=20ms,1000-clock p99<=10us,subsequent5empty-control median<=5percent minimum wall.',
        acceptance='Each comparison process must show positive actual saved radix passes in at least one of its six complete timed scenarios. Mixed positive/zero cases stay eligible;keep tie cases and all144samples. For EACH enclosing wall and direct GPU service:all18process/scenario paired-median ratios<=1.01 AND geometric mean<=0.99. This is a prospectively bounded component screen,not model adoption. Complete all144batches unless a source/runtime/review/owner/fault/budget/correctness/mechanism/timer gate fails. No retry,replacement,exclusion or speed-based stop. Performance miss prohibits model trial.',
        resource_hypothesis='Top-k accounts4.428919159s in the source-bound9a79profile. Early completion removes global score scans and WG reductions;singleton-key recovery adds one scan/reduction only when needed for partial budgets. This tests GPU service/issue cost,not physical VRAM/PCIe saturation. No H2D within timing.',
        provenance='Main-authored fixture/controller;build/owner/environment checks derive from e10997bdd. Private Mapping1source constructor derives from selected6e40,repository LICENSE;no XeStrata implementation input.',
        retention='One imminent integer-selector component executable;no third engine. Selected6e40/candidate9a79 and their required fixtures remain for current qualification;legacy provenance holds unchanged. Keep compact correctness/fault/branch counts and all144individual timing samples. Retire component objects,traces,generated snapshots and consumed audit at decision after exact Git recovery.')
    verify(row);data[KEY]=row;write(RESULT,data)
    print('Top-k recipe/inputs/criteria prepared;commit before build.',flush=True)


def case_checks(parsed,expected):
    require(len(parsed['CASE'])==expected,'incomplete Top-k numerical case count')
    for c in parsed['CASE']:
        require(all(c.get(k)=='PASS' for k in ('oracle','counters','inputs','guards','repeatability')),
                'Top-k output/counter/input/guard gate')
        require(int(c['saved_passes'])>=0 and 0<=int(c['stopped_queries'])<=int(c['queries']),
                'invalid actual pass counters')


def timed_mechanism(cases):
    case_checks({'CASE':cases},6)
    observed={}
    for c in cases:
        pair=(int(c['cells']),int(c['pattern']))
        require(pair not in observed and int(c['queries'])==256 and int(c['capacity'])==131072 and
            int(c['cap'])==2051 and c['invalid']=='0','timed mechanism shape/count differs')
        saved,stopped=int(c['saved_passes']),int(c['stopped_queries'])
        require(0<=saved<=7*256 and (saved>0)==(stopped>0),'inconsistent radix savings')
        observed[pair]=dict(saved_passes=saved,stopped_queries=stopped)
    require(set(observed)=={(n,p) for n in (32768,65536,117488) for p in (0,1)},
        'incomplete timed mechanism census')
    require(any(v['saved_passes']>0 for v in observed.values()),'comparison process saved no radix passes')
    return observed


def decision(measurements):
    cases=[m for m in measurements if m['kind']=='compare']
    require(len(cases)==3 and [m['argv'][-1] for m in cases]==['0','1','2'] and
        len({m['process']['pid'] for m in cases})==3,'three distinct ordered processes required')
    ratios={k:[] for k in ('wall_s','service_s')};groups=[]
    for process,m in enumerate(cases):
        savings=timed_mechanism(m['cases'])
        samples=m['samples'];require(len(samples)==48,'48timing samples per process required')
        observed=set()
        for s in samples:
            stamp=(int(s['scenario']),int(s['repetition']),int(s['arm']))
            require(stamp not in observed,'duplicate timing sample');observed.add(stamp)
            require(int(s['process'])==process and int(s['queries'])==256 and int(s['capacity'])==131072 and
                int(s['cap'])==2051 and int(s['batch'])==1024 and int(s['checkpoint'])==64 and s['output']=='PASS',
                'timed shape/dispatch/correspondence changed')
            scenario=stamp[0];require(0<=scenario<6 and int(s['cells'])==(32768,65536,117488)[scenario//2] and
                int(s['pattern'])==scenario%2,'timed scenario changed')
            w=float(s['wall_s']);g=float(s['service_s'])
            require(math.isfinite(w) and math.isfinite(g) and w>=.02 and .02<=g<=1.01*w,'ineligible timing batch')
        require(observed=={(s,rep,arm) for s in range(6) for rep in (-1,0,1,2) for arm in (0,1)},'missing timing pair')
        require(len(m['timers'])==1 and int(m['timers'][0]['samples'])==1000 and
            int(m['timers'][0]['p99_ns'])<=10000 and int(m['timers'][0]['resolution_ns'])<=1000,'timer qualification')
        require(len(m['controls'])==6 and [int(v['repetition']) for v in m['controls']]==[-1,0,1,2,3,4],
                'empty controls incomplete')
        controls=[float(v['wall_s']) for v in m['controls'][1:]]
        require(all(math.isfinite(v) and v>0 for v in controls) and
            statistics.median(controls)<=.05*min(float(s['wall_s']) for s in samples),'empty-control overhead')
        for scenario in range(6):
            by={(int(s['repetition']),int(s['arm'])):s for s in samples if int(s['scenario'])==scenario}
            group=dict(process=process,scenario=scenario,
                radix_savings=savings[((32768,65536,117488)[scenario//2],scenario%2)],metrics={})
            for key in ratios:
                paired=[float(by[(rep,1)][key])/float(by[(rep,0)][key]) for rep in range(3)]
                median=statistics.median(paired);ratios[key].append(median)
                group['metrics'][key]=dict(paired_ratios=paired,median_ratio=median)
            groups.append(group)
    metrics={k:dict(geometric_mean_ratio=math.exp(statistics.mean(math.log(v) for v in a)),
        all18_within_noninferiority=all(v<=1.01 for v in a)) for k,a in ratios.items()}
    for v in metrics.values():v['passed']=v['all18_within_noninferiority'] and v['geometric_mean_ratio']<=.99
    return dict(process_scenario=groups,metrics=metrics,mechanism_screen_passed=all(v['passed'] for v in metrics.values()),
        inference_adopted=False,physical_saturation_qualified=False)


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
            env=dict(row['native_environment']);warning=ROOT/'native'/('case-'+str(i)+'-validation.log')
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
                for prefix in ('PROPERTY','CASE','ADMISSION','SAMPLE','SCENARIO','TIMER','CONTROL')}
            kind=case['kind']
            if kind=='properties':require(len(parsed['PROPERTY'])==4 and all(p['PASS']=='1' for p in parsed['PROPERTY']),
                'compiled property gate')
            if kind=='smoke':
                case_checks(parsed,1);c=parsed['CASE'][0]
                require((c['cells'],c['capacity'],c['queries'],c['cap'],c['pattern'])==('17','32','3','5','0') and
                    int(c['saved_passes'])>0 and trace.exists() and 'urEnqueue' in trace.read_text(),'tiny trace/mechanism absent')
            if kind=='numerics':
                case_checks(parsed,145);require(len(parsed['ADMISSION'])==1 and parsed['ADMISSION'][0]==
                    dict(refusals='26',PASS='1'),'host admission controls absent')
            if kind=='compare':
                case_checks(parsed,6);require(len(parsed['SCENARIO'])==6 and all(s['inputs']==s['guards']=='PASS'
                    for s in parsed['SCENARIO']),'scenario closure failed')
                timed_mechanism(parsed['CASE'])
            measurements.append(dict(kind=kind,argv=case['argv'],passed=True,
                process={k:entry.get(k) for k in ('pid','exit_code','elapsed_seconds','peak_session_rss_bytes',
                    'observation_complete','direct_child_reaped','session_empty')},
                cases=parsed['CASE'],properties=parsed['PROPERTY'],admission=parsed['ADMISSION'],
                samples=parsed['SAMPLE'],timers=parsed['TIMER'],controls=parsed['CONTROL']))
            write(RESULT,data);require(not kernel_faults(cursor),'fresh kernel fault');guards.driver_guard()
        verify(row);require(digest(row['binary']['path'])==row['binary']['sha256'],'native ELF changed')
        row['decision']=decision(measurements);row.update(stage='closed',complete=True,passed=True)
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
