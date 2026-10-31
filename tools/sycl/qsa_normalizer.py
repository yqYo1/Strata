#!/usr/bin/env python3
"""Own an official-SDK, prefill-only normalizer comparison. Repository LICENSE.

prepare -> commit -> build -> commit -> run. Source review precedes GPU execution.
The old exact-output deferral stays unchanged; no automatic retry or model adoption.
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

sys.dont_write_bytecode = True
import build_fused_profile as closure
import measurement_guards as guards
import measurement_owner as supervision
import qsa_normalizer_sources as generator
import qsa_normalizer_link_inventory as host_link
import relink_qsa_k_model as linking

r, P, B = closure.r, closure.P, closure.B
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'qsa_parallel_normalizer_v3'
ROOT = B / 'qsa-normalizer-official-v3'
OFFICIAL = Path('/opt/intel/oneapi/compiler/2026.2')
NATIVE = ['src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'src/kernels/sycl/prompt_attention.cpp', 'bench/sycl/qsa_normalizer.cpp']
FLAGS = ['-std=c++20', '-O3', '-DNDEBUG', '-g0', '-fsycl', '-fsycl-id-queries-range=size_t',
         '-fno-fast-math', '-ffp-contract=off', '-DSTRATA_ENABLE_SYCL=1', '-pthread',
         '-I'+str(P / 'include'), '-I'+str(P / 'include/strata/sycl/compat'),
         '-I'+str(ROOT / 'generated')]
BUILD_LIMITS = dict(wall=300,cpu=900,rss_cap=6 << 30,text_cap=16 << 20,file_cap=256 << 20,total_cap=512 << 20)
RUN_LIMITS = dict(wall=600,cpu=900,rss_cap=5 << 30,text_cap=32 << 20,file_cap=128 << 20,total_cap=512 << 20)
require, digest, write = guards.require, supervision.sha, closure.write


def git(*args):
    return r.build.git(P, *args)


def public():
    data=json.loads(RESULT.read_text())
    committed=json.loads(git('show','HEAD:'+str(RESULT.relative_to(P))))
    require(data.get(KEY)==committed.get(KEY),'commit exact normalizer plan/result')
    return data


def source_pins(revision):
    names=set(NATIVE) | set(git('ls-files','include').decode().splitlines()) | {'tools/sycl/test_qsa_normalizer.py'}
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
        require((P/name).read_bytes()==raw,'commit normalizer input:'+name)
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
                 *[str(ROOT/(str(i)+'.o')) for i in range(len(units))],'-ldl','-o',str(ROOT/'normalizer_probe')])
    return rows


def verify(row):
    rt,full=runtime()
    require(digest(r.RESULT)==row['runtime_sha256'] and full['binary']==row['selected_binary'],
            'normalizer runtime/fallback changed')
    require(guards.driver_guard()==row['driver'] and rt['boot']==row['boot'],'normalizer driver/boot changed')
    require(packages()==row['sdk']['packages'] and digest(row['sdk']['path'])==row['sdk']['sha256'],
            'original official SDK manifest/packages changed')
    sdk=json.loads(Path(row['sdk']['path']).read_text())
    require(len(sdk)==row['sdk']['files'] and all(digest(p)==h for p,h in sdk.items()),'frozen official SDK/host bytes changed')
    require(all(digest(p)==h for p,h in row['runtime_files'].items()),'official component provider changed')
    require(host_link.inventory(str(OFFICIAL/'bin/icpx'),row['build_environment'])==row['host_link_inventory'],
            'prospective host linker input inventory changed')
    require(digest(row['inputs']['path'])==row['inputs']['sha256'],'normalizer input manifest changed')
    inputs=json.loads(Path(row['inputs']['path']).read_text())
    require(inputs['project']==source_pins(row['source_commit']),'normalizer source/controller changed')
    require(generator.proof()==row['generator_proof'],'normalizer generator differs')
    generated=generator.generated()
    require({name:hashlib.sha256(raw).hexdigest() for name,raw in generated.items()}==inputs['generated'],
            'prospective generated source changed')
    for name,raw in generated.items():require((ROOT/'generated'/name).read_bytes()==raw,'actual generated source changed:'+name)
    for name,h in inputs['external'].items():require(digest(name)==h,'frozen external input changed:'+name)
    require(commands()==row['commands'],'normalizer recipe changed')
    allowed={str((P/name).resolve()):h for name,h in inputs['project'].items()}
    allowed.update({str(ROOT/'generated'/name):h for name,h in inputs['generated'].items()})
    allowed.update(inputs['external']);allowed.update(sdk);allowed.update(row['runtime_files'])
    return allowed,rt


def prepare(data):
    require(KEY not in data and not ROOT.exists(),'fresh normalizer scope only')
    prior=json.loads(git('show','59d35fdf2:'+str(RESULT.relative_to(P))))['prefill_copy_batch_v2']
    require(prior['passed'] and prior['ownership_closed'] and not prior['active'] and not prior['fresh_faults'],
            'closed official component basis required')
    contract=data['qsa_parallel_normalizer']['prospective_framework_review']
    require(contract['revision']=='877be2974aa151a984e52822c028276c2b1e9303' and
            contract['newly_inspected_primary_test']['sha256']=='99277da69b1a2505015a8bb393875e4bcff823defd44009e4641c10e53d01098',
            'pinned actual-model framework contract required')
    rt,full=runtime();revision=git('rev-parse','HEAD').decode().strip();pins=source_pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','generated'):(ROOT/name).mkdir(mode=0o700)
    generated=generator.generated()
    for name,raw in generated.items():(ROOT/'generated'/name).write_bytes(raw)
    # Reuse immutable, independently qualified official bytes,not a deleted build
    # or the unproven legacy SDK. This is a NEW XMX component qualification.
    sdk=copy.deepcopy(prior['sdk']);sdk['path']=sdk['original_manifest']['path']
    sdk['sha256']=sdk['original_manifest']['sha256']
    require(sdk['sha256']==prior['sdk']['sha256'],'original/duplicate SDK manifests differ')
    old_lib=Path(next(p for p in rt['runtime_files'] if '/libsycl.so.' in p)).parent
    external={p:h for p,h in closure.external_inputs(rt).items() if not Path(p).is_relative_to(old_lib)}
    build_env=copy.deepcopy(prior['build_environment']);native_env=copy.deepcopy(prior['native_environment'])
    for env in (build_env,native_env):env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    host_inventory=host_link.inventory(str(OFFICIAL/'bin/icpx'),build_env)
    external.update(host_inventory['files'])
    runtime_files=dict(prior['runtime_files'])
    for name,h in host_inventory['files'].items():
        with Path(name).open('rb') as stream:elf=stream.read(4)==b'\x7fELF'
        if elf and '.so' in Path(name).name:runtime_files[name]=h
    write(ROOT/'inputs.json',dict(project=pins,generated={n:hashlib.sha256(v).hexdigest() for n,v in generated.items()},external=external))
    native_env.pop('STRATA_TRANSFER_PROBE_GATES',None)
    native_env.pop('UR_LOG_TRACING',None);native_env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'
    for name,value in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0').items():
        require(native_env.get(name)==value,'mandatory runtime safety key differs:'+name)
    cases=[dict(kind='host-canary'),dict(kind='properties',argv=['--properties',next(p for p in prior['runtime_files'] if Path(p).name.startswith('libze_loader.so.'))]),
           dict(kind='smoke',argv=['--smoke']),dict(kind='numerics',argv=['--numerics'])]
    cases += [dict(kind='compare',argv=['--compare',str(i)]) for i in range(3)]
    row=dict(stage='prepared',source_commit=revision,inputs=dict(path=str(ROOT/'inputs.json'),sha256=digest(ROOT/'inputs.json')),
        generator_proof=generator.proof(),sdk=sdk,runtime_sha256=digest(r.RESULT),selected_binary=full['binary'],
        driver=guards.driver_guard(),boot=rt['boot'],runtime_files=runtime_files,preload=prior['preload'],
        host_link_inventory=host_inventory,
        predecessor='qsa_parallel_normalizer_v2 remains build-failed:all8commands exited0,but host linker inventory lacked the GNU linker/13system link inputs. No GPU execution or qualification inherited. This new scope prospectively queries and freezes those host files under the actual build environment.',
        commands=commands(),build_workdir=str(ROOT),build_environment=build_env,native_environment=native_env,cases=cases,
        build_limits=BUILD_LIMITS,run_limits=RUN_LIMITS,active=False,ownership_closed=True,passed=False,
        contract=dict(framework='SGLang',revision=contract['revision'],test=contract['newly_inspected_primary_test'],
            comparator='Both arms versus independent FP64 dense attention over the same BF16 Q/K/V operands,and candidate versus original:every finite output abs(a-b)<=0.02+0.02*abs(b). This adopts the pinned BF16 attention-output test criterion,not an intermediate-state tolerance or observed local SGLang run.',
            state='For4real/padded coordinates each numerical case,exact probability/BF16-PV,maximum/alpha and final FP32PVaccumulator bits. Only denominator/sum may differ;record their maximum delta,require finite nonnegative. Captured output equals its own uninstrumented arm. Repeat each normal arm with identical output bits. All inputs and allocation guards unchanged.',
            repeatability='Per-arm exact repeats and3deliberate comparator refusals;no guessed epsilon for FP32state. Independent host reference and cross-arm output checks are not a task-quality result.',
            task='A component pass permits only a new long-prefill model trial. Before adoption require matched bench.sh long+agent session and thinking>=1K output in observed MTP-friendly/weak workloads;no newly failing previously passing objective task. Repeated decode preservation and physical262144 fill/tails/handoff/RESTORE remain mandatory. Generated text need not match.'),
        numerical_plan='Official host canary;4compiled normal/state kernel properties before enqueue;traced17cell/17width/1query/page4smoke;then23numerical cases:page4/16,widths1/31/32/33/65/2051,3queries with masked rows;ones,dominant,subnormal/underflow-region scores and both online max-transition orders over65positions;final physical262144-KV-address component. Source/review/owner/preinit/provider/env/normalexit/no-fault gates throughout. This is not full-model262144 qualification.',
        performance_plan='3fresh processes;4scenarios(context32768/117488 x contiguous/gapped),8192queries,cap2051,page4,batch8. Both arms warm once untimed;retain first timing pair(-1) separately,then3rotating matched pairs(0..2).96timing batches total. Each wall>=100ms,timer1000p99<=10us,5empty-control median<=5percent smallest batch. Record pack+attention events and enclosing wall;output checks outside timer.',
        acceptance='All execution/numerical gates pass. For EACH wall,attention,and(pack+attention),all12process/scenario median candidate/original ratios<=1 and geometric mean<=0.99. Stop first source/runtime/owner/fault/budget/correctness failure;no retry,replacement,exclusion or speed-based stop. A performance miss prohibits model trial;component success is not inference adoption.',
        resource_hypothesis='Current9a79QSA-prefix27.081s of134.659s request. Replace32uniform subgroup exchanges and lane0serial additions with one FP32group reduction of p0+p1,leaving exp,DPAS,storage/memory traffic,geometry and queues fixed. This tests dependency/issue cost,not arithmetic or VRAM saturation;matched physical ceilings remain unknown.',
        provenance='Independent Main generator/source recovered from73414ebc2 against exact selected6e40prefix. New fixture/controller are Main-authored under repository LICENSE. SGLang is an Apache2algorithm/criterion reference;no source/test copied. XeStrata excluded.',
        retention='One imminent component executable;selected6e40and9a79engines/required fixtures unchanged,no third engine. Keep compact numerical/state/refusal outcomes,individual96timings,exact source/recipe/SDK/executable identities. Retire component objects,normal traces,generated snapshots and consumed review at decision after exact Git recovery. Independent legacy holds remain.')
    verify(row);data[KEY]=row;write(RESULT,data);print('Normalizer source/recipe/criteria prepared;commit before build.',flush=True)


def build(data):
    row=data[KEY];require(row['stage']=='prepared' and not (ROOT/'build').exists(),'fresh committed normalizer build')
    allowed,_=verify(row);owner=None
    owner=supervision.Owner(ROOT/'build',lambda:write(ROOT/'build-owner.json',owner.commands))
    try:
        row.update(stage='building',active=True,ownership_closed=False);write(RESULT,data)
        for i,argv in enumerate(row['commands']):
            entry,out,err=owner.run('compile-'+str(i),argv,row['build_environment'],ROOT,**BUILD_LIMITS)
            require(supervision.closed(entry) and entry['exit_code']==0,'normalizer compile/link failed:'+str(i))
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
        require(all(str(ROOT/'generated'/name) in observed for name in ('original.cpp','candidate.cpp','qsa-head.hpp')),
                'incomplete generated compiler input closure')
        write(ROOT/'compiled-inputs.json',observed);verify(row)
        binary=ROOT/'normalizer_probe'
        row.update(stage='built',build_passed=True,binary=dict(path=str(binary),bytes=binary.stat().st_size,sha256=digest(binary)),
                   compiled_inputs=dict(path=str(ROOT/'compiled-inputs.json'),sha256=digest(ROOT/'compiled-inputs.json'),files=len(observed),uncovered=0),
                   link_inputs=dict(path=str(ROOT/'link-inputs.json'),sha256=digest(ROOT/'link-inputs.json'),frozen=len(links['frozen_inputs']),uncovered=0))
    except BaseException as e:row.update(stage='build-failed',build_passed=False,error=repr(e))
    finally:
        row.update(active=owner.active is not None,ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands));write(RESULT,data)
    print(json.dumps({k:row.get(k) for k in ('stage','build_passed','ownership_closed','error')}),flush=True)


def fields(line):
    return dict(part.split('=',1) for part in line.split()[1:] if '=' in part)


def numerical_check(parsed,smoke):
    expected={(17,17,1,4,0,0)} if smoke else \
        {(2113,cap,3,page,0,1) for page in (4,16) for cap in (1,31,32,33,65,2051)} | \
        {(97,65,1,page,kind,0) for page in (4,16) for kind in (1,2,3,4,5)} | {(262144,2051,1,16,0,0)}
    rows=parsed['NUMERICAL'];states=parsed['STATE'];oracles=parsed['ORACLE']
    require(len(rows)==len(states)==len(oracles)==len(expected),'numerical/state/oracle counts')
    require({tuple(int(s[k]) for k in ('cells','cap','queries','page','kind','masked')) for s in rows}==expected,
            'actual numerical cases differ')
    for s,state,oracle in zip(rows,states,oracles):
        require(s['inputs']==s['guards']==s['repeatability']=='PASS' and
                state['PV_bits']==state['probability_bits']==state['maximum_alpha_bits']==state['accumulator_bits']==state['guards']=='PASS' and state['coordinates']=='4' and
                oracle['PASS']=='1' and int(oracle['compared'])==2*int(s['queries'])*24*256 and
                float(oracle['rtol'])==float(oracle['atol'])==.02,'numerical/state contract failed')
        require(all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in
                    (s['output_budget_ratio'],oracle['maximum_budget_ratio'])) and
                math.isfinite(float(state['normalizer_max_abs_delta'])) and float(state['normalizer_max_abs_delta'])>=0,
                'nonfinite or failed numerical comparison')


def decision(measurements):
    runs=[m for m in measurements if m['kind']=='compare'];require(len(runs)==3,'three comparison processes required')
    require({m['argv'][-1] for m in runs}=={'0','1','2'},'three DISTINCT processes required')
    summaries=[]
    for m in runs:
        rows=m['samples'];process=m['argv'][-1]
        require(m['passed'] and len(rows)==32 and len(m['case_checks'])==4 and
                {s['scenario'] for s in m['case_checks']}=={'0','1','2','3'} and
                all(s['inputs']==s['guards']=='PASS' for s in m['case_checks']),'all comparison scenarios required')
        require({(s['scenario'],s['repetition'],s['arm']) for s in rows}==
            {(str(s),str(rep),str(a)) for s in range(4) for rep in range(-1,3) for a in range(2)},'matched comparison coverage')
        warm=[s for s in rows if int(s['repetition'])>=0]
        require(all(float(s['wall_s'])>=.1 for s in warm),'ineligible short timing batch')
        empty=[float(s['wall_s']) for s in m['controls'] if int(s['repetition'])>=0]
        require(len(empty)==5 and statistics.median(empty)<=.05*min(float(s['wall_s']) for s in warm),'empty control cost')
        require(len(m['timers'])==1 and int(m['timers'][0]['samples'])==1000 and int(m['timers'][0]['p99_ns'])<=10000,'clock control')
        for s in rows:
            require(s['process']==process and s['queries']=='8192' and s['cap']=='2051' and s['page']=='4' and s['batch']=='8' and s['output']=='PASS',
                    'actual timed shape/output differs')
            scenario=int(s['scenario'])
            require(s['cells']==str((32768,117488)[scenario//2]) and s['gapped']==str(scenario%2),
                    'actual first/subsequent scenario geometry differs')
            require(all(math.isfinite(float(s[k])) and float(s[k])>0 for k in ('wall_s','pack_s','attention_s')) and
                    float(s['pack_s'])+float(s['attention_s'])<=1.01*float(s['wall_s']),'invalid service/wall clocks')
        for scenario in range(4):
            selected=[s for s in warm if int(s['scenario'])==scenario]
            require(all(s['cells']==str((32768,117488)[scenario//2]) and s['gapped']==str(scenario%2) for s in selected),'scenario geometry differs')
            metrics={}
            for metric in ('wall_s','attention_s','combined_s'):
                def value(s):return float(s['pack_s'])+float(s['attention_s']) if metric=='combined_s' else float(s[metric])
                ratios=[value(next(s for s in selected if s['arm']=='1' and int(s['repetition'])==rep))/
                        value(next(s for s in selected if s['arm']=='0' and int(s['repetition'])==rep)) for rep in range(3)]
                metrics[metric]=dict(paired_ratios=ratios,median_ratio=statistics.median(ratios))
            summaries.append(dict(process=process,scenario=scenario,metrics=metrics))
    by_metric={}
    for metric in ('wall_s','attention_s','combined_s'):
        ratios=[s['metrics'][metric]['median_ratio'] for s in summaries]
        gm=math.exp(statistics.mean(math.log(x) for x in ratios))
        by_metric[metric]=dict(geometric_mean_ratio=gm,all12_no_slower=all(x<=1 for x in ratios),passed=all(x<=1 for x in ratios) and gm<=.99)
    return dict(process_scenario=summaries,metrics=by_metric,mechanism_screen_passed=all(v['passed'] for v in by_metric.values()),
                inference_adopted=False,physical_saturation_qualified=False)


def run(data):
    row=data[KEY];require(row['stage']=='built' and row['build_passed'] and row['ownership_closed'],'commit successful normalizer build')
    require(row.get('source_review',{}).get('bounded_execution_accepted') is True,'review new fixture/contract before GPU')
    verify(row);require(digest(row['binary']['path'])==row['binary']['sha256'] and not (ROOT/'native').exists(),'fresh verified native run only')
    raw=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in raw.splitlines() if s.startswith('-- cursor: '))
    owner=None;owner=supervision.Owner(ROOT/'native',lambda:write(ROOT/'native-owner.json',owner.commands))
    measurements=row['measurements']=[]
    try:
        row.update(stage='running',active=True,ownership_closed=False,complete=False);write(RESULT,data)
        for i,case in enumerate(row['cases']):
            env=dict(row['native_environment']);warning=ROOT/'native'/('case-'+str(i)+'-validation.log');trace=ROOT/'native'/('case-'+str(i)+'-trace.log')
            env['UR_LOG_VALIDATION']='level:warning;flush:warning;output:file,'+str(warning)
            if case['kind']=='smoke':env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',UR_LOG_TRACING='level:info;flush:info;output:file,'+str(trace))
            if case['kind']=='host-canary':
                asset=row['preload']['canary'];require(digest(asset['path'])==asset['sha256'],'canary changed')
                entry,out,err=owner.run('case-'+str(i),[asset['path']],env,P,**RUN_LIMITS)
                require(supervision.closed(entry) and entry['exit_code']==0 and json.loads(out.read_text())==dict(pid=entry['pid'],initialized=0,checked=46,expected=46,version=(1 << 32)-1,closed=0),'host canary failed')
                require(r.validation.preload_evidence(err.read_text(),entry['pid'])['loader']==row['preload']['loader'],'host loader differs')
                require(not warning.exists() or not warning.stat().st_size,'host canary warning')
                measurements.append(dict(kind=case['kind'],passed=True,ownership_closed=True));write(RESULT,data);continue
            lines=[];gates=[]
            def observe(io,entry):
                while True:
                    line=io.line(seconds=60);lines.append(line)
                    if not line.startswith('GATE stage='):continue
                    stage=line.split('=',1)[1];require(len(gates)<3 and stage==('before-runtime','runtime-ready','before-exit')[len(gates)],'native gate order')
                    proc=Path('/proc',str(entry['pid']));require(str((proc/'exe').resolve())==row['binary']['path'] and digest(proc/'exe')==row['binary']['sha256'],'actual native ELF differs')
                    actual=dict(v.split(b'=',1) for v in (proc/'environ').read_bytes().split(b'\0') if v)
                    require(actual=={k.encode():v.encode() for k,v in env.items()},'native environment differs')
                    mapped=r.executable_mappings((proc/'maps').read_text());allowed=set(row['runtime_files'])|{row['binary']['path']}
                    require(mapped<=allowed,'unfrozen native providers:'+repr(mapped-allowed))
                    for name in mapped:require(digest(name)==(row['binary']['sha256'] if name==row['binary']['path'] else row['runtime_files'][name]),'mapped provider changed')
                    if stage!='before-runtime':
                        require(any(s.startswith('DEVICE name=') and 'B570' in s for s in lines),'actual B570 missing')
                        require(any('libze_intel_gpu.so.1.17.39758' in p for p in mapped) and any('libur_adapter_level_zero_v2.so' in p for p in mapped),'native driver/adapter missing')
                    gates.append(stage);io.send(b'continue\n')
                    if stage=='before-exit':io.exit(10);break
            entry,out,err=owner.run('case-'+str(i),[row['binary']['path'],*case['argv']],env,P,interaction=observe,**RUN_LIMITS)
            require(supervision.closed(entry) and entry['exit_code']==0 and len(gates)==3,'native owner/exit failed')
            require(any(s.startswith('DONE ') for s in lines) and not any('FAIL' in s for s in lines),'native completion failed')
            require(r.validation.preload_evidence(err.read_text(),entry['pid'])['loader']==row['preload']['loader'],'actual official UR loader differs')
            require(not warning.exists() or not warning.stat().st_size,'native validation warning')
            require(len([s for s in lines if s.startswith('COMPARATOR refusals=3 identity=PASS')])==1,'comparator controls absent')
            parsed={prefix:[fields(s) for s in lines if s.startswith(prefix+' ')] for prefix in ('PROPERTY','NUMERICAL','STATE','ORACLE','SAMPLE','CASE','TIMER','CONTROL')}
            kind=case['kind']
            if kind=='properties':require(len(parsed['PROPERTY'])==4 and all(s['PASS']=='1' for s in parsed['PROPERTY']),'compiled property gate')
            if kind in ('smoke','numerics'):
                numerical_check(parsed,kind=='smoke')
                if kind=='smoke':require(trace.exists() and 'urEnqueue' in trace.read_text(),'small full trace absent')
            m=dict(kind=kind,argv=case['argv'],passed=True,process={k:entry.get(k) for k in ('pid','exit_code','elapsed_seconds','peak_session_rss_bytes','observation_complete','direct_child_reaped','session_empty')},
                   numerical=parsed['NUMERICAL'],states=parsed['STATE'],oracles=parsed['ORACLE'],properties=parsed['PROPERTY'],samples=parsed['SAMPLE'],case_checks=parsed['CASE'],timers=parsed['TIMER'],controls=parsed['CONTROL'])
            measurements.append(m);write(RESULT,data);guards.driver_guard()
        verify(row);require(digest(row['binary']['path'])==row['binary']['sha256'],'native ELF changed')
        row['decision']=decision(measurements);row.update(stage='closed',complete=True,passed=True)
    except BaseException as e:row.update(stage='failed',passed=False,error=repr(e))
    finally:
        try:
            raw=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'],timeout=15)
            require(len(raw)<2 << 20,'kernel fault evidence budget')
            row['fresh_faults']=[s for s in [json.loads(line).get('MESSAGE','') for line in raw.decode().splitlines() if line] if re.search(r'(?:\bxe\b|drm|GPU).*(?:fault|hang|wedg|reset|timed out)',s,re.I)]
            row['driver_after']=guards.driver_guard()
        except BaseException as e:row.update(passed=False,fault_evidence_error=repr(e))
        row.update(active=owner.active is not None,ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands))
        row['passed']=bool(row.get('passed') and row['ownership_closed'] and not row.get('fresh_faults'));write(RESULT,data)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','error')}),flush=True)


def main():
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build','run'),'choose prepare,build,run')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        data=public();{'prepare':prepare,'build':build,'run':run}[sys.argv[1]](data)


if __name__=='__main__':main()
