#!/usr/bin/env python3
"""Qualify canonical Q6 C4 separately for the actual decode shapes; Main only.

Main-authored from our owned compiler census and bounded fixture supervision.
Repository LICENSE applies. No XeStrata source or implementation is used.
The frozen model cohort must close before this independent component starts.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import select
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import benchmark_runtime as python_runtime
import build_compat_profile as base
import measurement_guards as guards
import measurement_owner as owner
import q6_dpas_component as reader
import ur_validation as validation

P, B = base.P, base.B
ROOT, KEY = B / 'q6-dense-c4-v1', 'q6_dense_c4_component_v2'
PREVIOUS_KEY = 'q6_dense_c4_component_v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
COHORT, MODEL = 'q6_packed_head_generation_speed_v1', 'q6_head_packed_model_entry_v1'
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
KERNEL = '_ZTSN6strata7kernels9component12_GLOBAL__N_117Q6DpasC4CanonicalE'
SHAPES = ((2560,512),(6144,2560),(2560,6144),(2560,10240),(2560,12288),(2560,248320))
SMALL = ((256,1),(256,17),(512,33),(6144,33))
UNITS = ('src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp',
    'src/kernels/sycl/native_mmvq.cpp','src/kernels/sycl/native_q5_head.cpp',
    'src/kernels/sycl/native_q3_esimd.cpp','src/kernels/sycl/native_q4_esimd.cpp',
    'src/kernels/sycl/native_q2_esimd.cpp','src/kernels/sycl/native_iq4xs_esimd.cpp',
    'src/kernels/sycl/native_iq_esimd.cpp','src/kernels/sycl/native_mmq_xmx.cpp',
    'src/kernels/sycl/iq.cpp','src/kernels/sycl/q6_dpas_component.cpp','bench/sycl/q6_canonical.cpp')
require, sha, write, signature = guards.require, base.digest, base.write, base.signature


def basis():
    # The primary builder imports this module's image reader. Resolve the
    # builder only after module initialization, for either import order.
    import build_q6_head_model as primary
    cohort = validation.committed(RESULT,(COHORT,))
    require(cohort['stage']=='closed' and cohort['complete'] and cohort['passed'] and
        cohort['ownership_closed'] and not cohort['active'] and len(cohort['processes'])==6 and
        all(v['passed'] and v['complete'] and v['ownership_closed'] and
            v['native_exit_code']==0 and v['fresh_faults']==[] for v in cohort['processes']),
        'close all six frozen model processes before this separate component')
    native = json.loads((primary.ROOT/'record.json').read_text())
    require(validation.committed(RESULT,(primary.KEY,))==primary.compact(native) and
        native['stage']=='built' and native['passed'] and native['ownership_closed'] and
        native['actual_inputs']['uncovered']==0 and cohort['binary']==native['binaries']['strata'],
        'same qualified current native reference required')
    original = native['project']
    recovered = {p:hashlib.sha256(raw).hexdigest() for p,raw in
        base.source.blobs(P,original['revision'],original['source_sha256'])}
    require(recovered==original['source_sha256'],'original native inputs must remain recoverable')
    runtime = {p:h for p,h in recovered.items() if p.startswith(('src/sycl/','include/strata/sycl/'))}
    shared = set(UNITS[:-1]) | set(runtime) | {'src/kernels/sycl/q6_dpas_component.hpp',
        'include/strata/core/verify_window_policy.hpp'}
    require(runtime and all(p in recovered and sha(P/p)==recovered[p] for p in shared),
        'component must retain the exact qualified runtime,kernel and reference-policy source')
    for binary in native['binaries'].values():
        require(sha(binary['path'])==binary['sha256'] and Path(binary['path']).stat().st_size==binary['bytes'],
            'qualified engine/required fixture bytes changed')
    return native,dict(record=dict(path=str(primary.ROOT/'record.json'),sha256=sha(primary.ROOT/'record.json')),
        source_commit=original['revision'],engine=native['binaries']['strata'],
        runtime_sources=runtime,shared_source_sha256=signature({p:recovered[p] for p in sorted(shared)}))


def image(blob):
    spec = importlib.util.spec_from_file_location('dense_q6_image',reader.__file__)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.KERNEL_ENTRY=KERNEL
    return module.embedded_dpas(blob,allow_shared_image=True)


def context():
    spec=importlib.util.spec_from_file_location('dense_q6_build',base.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    ctx.ROOT,ctx.KEY,ctx.RESULT=ROOT,KEY,RESULT
    ctx.TARGET,ctx.CPP,ctx.UNITS='sycl_q6_canonical_component',UNITS[-1],UNITS
    ctx.PREFIXES=('CMakeLists.txt','cmake','src','include','third_party','bench/sycl','tools/sycl')
    ctx.BUDGET=dict(wall=1200,cpu=3600,rss_cap=8<<30,text_cap=16<<20,file_cap=128<<20,total_cap=768<<20)
    ctx.basis=basis
    original_compact=ctx.compact
    def compact(row):
        result=original_compact(row)
        if 'admission_project' in row:
            result['admission_project']={k:v for k,v in row['admission_project'].items() if k!='source_sha256'}
        return result
    ctx.compact=compact
    def current(row,committed=True):
        if committed:require(validation.committed(RESULT,(KEY,))==ctx.compact(row),'commit the exact component stage first')
        _,identity=basis();require(identity==row['selected_native'],'reference provenance changed')
        check=dict(row)
        if 'admission_project' in row:
            original=row['project']
            require(row['admission_source_commit']==row['admission_project']['revision'] and
                original['count']==len(original['source_sha256']) and
                signature(original['source_sha256'])==original['aggregate_sha256'] and
                row['binary']['source_commit']==row['actual_inputs']['source_commit']==original['revision'],
                'controller admission must preserve the exact original fixture source')
            check['project']=row['admission_project']
        ctx.build.check_inputs(check)
        require(all(p in row['project']['source_sha256'] for p in UNITS) and
            'tools/sycl/q6_canonical_component.py' in row['project']['source_sha256'],'missing fixture/controller input')
        require(row['recipe']['targets']==row['recipe']['build_targets']==[ctx.TARGET] and
            not row['recipe']['environment'].get('LD_PRELOAD') and sha(PYTHON)==row['python_sha256'],
            'fixture-only recipe or Python changed')
        manifest=row['runtime_manifest']
        require(sha(manifest['path'])==manifest['sha256'] and
            all(sha(p)==h for p,h in row['runtime_pins'].items()),'runtime bytes changed')
        require(guards.driver_guard()==row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'driver/boot changed')
        if 'binary' in row:
            require(sha(row['binary']['path'])==row['binary']['sha256'] and
                Path(row['binary']['path']).stat().st_size==row['binary']['bytes'],'component ELF changed')
        if 'actual_inputs' in row:
            actual=row['actual_inputs'];require(sha(actual['path'])==actual['sha256'],'compiler census changed')
            census=json.loads(Path(actual['path']).read_text())
            require(all(sha(p)==h for field in ('compiled','generated','links') for p,h in census[field].items()),
                'actual compiler/generated/link input changed')
        if 'codegen' in row:require(image(Path(row['binary']['path']).read_bytes())==row['codegen'],'canonical DPAS image changed')
    ctx.current=current
    return ctx


def fields(line,prefix):
    require(line.startswith(prefix),'expected native record')
    pairs=[p.split('=',1) for p in line[len(prefix):].split()]
    require(all(len(p)==2 for p in pairs) and len({p[0] for p in pairs})==len(pairs),'malformed/duplicate fields')
    return dict(pairs)


def number(f,key):
    value=f.get(key,'');require(value.isdecimal(),'missing/noninteger '+key)
    return int(value)


def validated_trace(text,pid):
    """Attest the constructor canary before checking subsequent native work."""
    proof=validation.preload_evidence(text,pid)
    marker=re.search(r'^strata UR validation: pid='+str(pid)+r' phase=before-main [^\n]*\n',text,re.M)
    require(marker is not None,'complete validated constructor boundary required')
    errors=lambda value:re.findall(r'(?:UR|ZE)_RESULT_ERROR_[A-Z_]+',value)
    # The pinned constructor deliberately tests preinit37 and null-handle46.
    # Parameter-only runs need not print enum names, but the receipt is mandatory.
    require(set(errors(text[:marker.end()]))<=
        {'UR_RESULT_ERROR_UNINITIALIZED','UR_RESULT_ERROR_INVALID_NULL_HANDLE'},
        'unexpected error before the validated constructor boundary')
    body=text[marker.end():]
    require(not errors(body),'native UR/ZE error after the validated constructor boundary')
    return proof,body


def batch_size(n):
    return 4096 if n<=1024 else 256 if n==248320 else 1024


def admissions(text,shapes):
    rows=[fields(v,'ADMISSION ') for v in text.splitlines() if v.startswith('ADMISSION ')]
    require([(number(v,'k'),number(v,'n')) for v in rows]==list(shapes),'all actual matrix admissions required')
    for v in rows:
        k,n=number(v,'k'),number(v,'n');wb=n*(k//256)*210
        reference='columns' if (k,n) in SHAPES and (k,n)!=(2560,512) else 'generic'
        require(number(v,'canonical_bytes')==wb and
            number(v,'packed_bytes')==number(v,'additional_candidate_weight_bytes')==0 and
            number(v,'device_payload_bytes')==wb+(k//32)*36*4+n*32 and
            number(v,'max_alloc_bytes')>=wb and number(v,'graphs')==2 and number(v,'nodes_each')==1 and
            v.get('reference')==reference and v.get('scope')=='prequantized-canonical-C4',
            'actual allocation/reference/graph admission differs')


def numerical(text,shapes,*,timing=False):
    admissions(text,shapes)
    lines=text.splitlines();rows=[fields(v,'NUMERIC ') for v in lines if v.startswith('NUMERIC ')]
    require([(number(v,'k'),number(v,'n')) for v in rows]==[s for s in shapes for _ in range(2)],
        'complete ordered per-shape numerical checks required')
    by_shape={}
    for v in rows:
        k,n=number(v,'k'),number(v,'n');value=float(v.get('max_abs_error','nan'));oracle=number(v,'oracle_rows')
        require(v.get('passed')==v.get('guards')=='1' and number(v,'columns')==4 and
            number(v,'outputs')==4*n and math.isfinite(value) and value>=0 and
            min(n,512)<=oracle<=n and (n>1024 or oracle==n),'numerical/oracle/guard contract')
        cell=by_shape.setdefault(f'{k}x{n}',dict(cases=0,max_abs_error=0.,minimum_oracle_rows=n))
        cell['cases']+=1;cell['max_abs_error']=max(cell['max_abs_error'],value)
        cell['minimum_oracle_rows']=min(cell['minimum_oracle_rows'],oracle)
    require(lines.count('C1_CONTROL passed=1 dpas_launched=0')==(0 if timing else len(shapes)) and
        sum(v.startswith('C1_CONTROL ') for v in lines)==(0 if timing else len(shapes)),'C1 refusal controls')
    return dict(cases=len(rows),shapes=by_shape,finite_and_guards=True,inputs_immutable=True,
        C1_negative_controls=0 if timing else len(shapes))


def timing(text,index):
    require(type(index) is int and index in (0,1,2),'fresh timing process0..2')
    lines=text.splitlines();clocks=[fields(v,'CLOCK ') for v in lines if v.startswith('CLOCK ')]
    require(len(clocks)==1 and number(clocks[0],'process')==index and number(clocks[0],'floor_ns')==10000000,
        'one actual timer control')
    timer=number(clocks[0],'median_stamp_ns');require(timer>0,'positive timer overhead')
    rows=[]
    for line in lines:
        if not line.startswith('SAMPLE '):continue
        f=fields(line,'SAMPLE ');k,n=number(f,'k'),number(f,'n');batch=batch_size(n)
        v={a:number(f,a) for a in ('process','arm','sample','k','n','columns','launches','graph_launches',
            'nodes_per_graph','wall_ns','cpu_ns','submit_wall_ns','submit_cpu_ns','wait_wall_ns','wait_cpu_ns')}
        require(v['process']==index and v['columns']==4 and v['launches']==v['graph_launches']==batch and
            v['nodes_per_graph']==1 and f.get('route')=='compat_graph' and
            f.get('reference')==('generic' if (k,n)==(2560,512) else 'columns'),'actual timed route/extent')
        require(v['wall_ns']>=max(10000000,300*timer) and
            v['wall_ns']==v['submit_wall_ns']+v['wait_wall_ns'] and
            v['cpu_ns']==v['submit_cpu_ns']+v['wait_cpu_ns'] and
            v['cpu_ns']<=v['wall_ns']+10*timer,'timer floor/disjoint accounting')
        v['ns_per_replay']=v['wall_ns']/batch;rows.append(v)
    order=(1,0) if index==1 else (0,1)
    require([(v['k'],v['n'],v['arm'],v['sample']) for v in rows]==
        [(k,n,a,s) for k,n in SHAPES for a in order for s in range(5)],'complete timing order;no duplicate or excluded sample')
    require(lines.count('TIMING passed=1 samples=60 shapes=6 model_speed=0 physical_traffic=0 cache=one-repeated-matrix')==1 and
        sum(v.startswith('TIMING ') for v in lines)==1,
        'complete timing footer/cache scope')
    numeric=numerical(text,SHAPES,timing=True)
    medians={f'{k}x{n}':{str(a):statistics.median(v['ns_per_replay'] for v in rows
        if (v['k'],v['n'],v['arm'])==(k,n,a)) for a in (0,1)} for k,n in SHAPES}
    return dict(index=index,samples=rows,median_ns=medians,numerical=numeric,timer_overhead_ns=timer,
        scope='One repeated prequantized canonical matrix;graph submission/completion and parameter validation charged. Not model cache,critical wall,physical traffic or model throughput.')


def decision(runs):
    require([v['index'] for v in runs]==[0,1,2],'three independent timing processes required')
    cells={}
    for k,n in SHAPES:
        name=f'{k}x{n}';ratios=[v['median_ns'][name]['1']/v['median_ns'][name]['0'] for v in runs]
        cells[name]=dict(candidate_over_reference_process_ratios=ratios,
            geometric_mean_time_ratio=math.exp(statistics.mean(math.log(v) for v in ratios)),
            component_gain_qualified=all(v<1 for v in ratios))
    return dict(cells=cells,eligible_shapes=[s for s,v in cells.items() if v['component_gain_qualified']],
        inference_adopted=False,model_speed_qualified=False,physical_saturation_qualified=False,
        next='Select only independently winning decode C4 shapes for a fresh production trial;preserve prompt/C1 and every losing shape. Matched model,agent/task,prefill preservation and physical262144/RESTORE remain separate.')


def controls():
    def sample(index):
        out=[f'CLOCK median_stamp_ns=100 floor_ns=10000000 process={index}']
        for k,n in SHAPES:
            wb=n*(k//256)*210
            out.append(f'ADMISSION k={k} n={n} canonical_bytes={wb} packed_bytes=0 additional_candidate_weight_bytes=0 device_payload_bytes={wb+(k//32)*36*4+n*32} max_alloc_bytes=1073741824 graphs=2 nodes_each=1 reference='+('generic' if n==512 else 'columns')+' scope=prequantized-canonical-C4')
            numeric=f'NUMERIC passed=1 k={k} n={n} columns=4 outputs={n*4} oracle_rows={min(n,512)} max_abs_error=0.00001 guards=1'
            out.append(numeric)
            for a in ((1,0) if index==1 else (0,1)):
                for s in range(5):
                    batch=batch_size(n)
                    out.append(f'SAMPLE process={index} arm={a} k={k} n={n} columns=4 sample={s} launches={batch} graph_launches={batch} nodes_per_graph=1 route=compat_graph wall_ns=20000000 cpu_ns=10000000 submit_wall_ns=5000000 submit_cpu_ns=4000000 wait_wall_ns=15000000 wait_cpu_ns=6000000 reference='+('generic' if n==512 else 'columns'))
            out.append(numeric)
        out.append('TIMING passed=1 samples=60 shapes=6 model_speed=0 physical_traffic=0 cache=one-repeated-matrix')
        return '\n'.join(out)
    for i in (0,1,2):timing(sample(i),i)
    good=sample(0)
    invalid=[good.replace('wall_ns=20000000','wall_ns=9999999',1),
        good.replace('graph_launches=1024','graph_launches=1023',1),
        good.replace('reference=generic','reference=columns',1),
        good.replace('guards=1','guards=0',1),good.replace('oracle_rows=512','oracle_rows=511',1),
        good.replace('wait_cpu_ns=6000000','wait_cpu_ns=6000001',1),
        good.replace('SAMPLE process=0 arm=0 k=6144','SAMPLE process=0 arm=0 k=6143',1),
        good.replace('samples=60 shapes=6','samples=59 shapes=6'),
        good.replace('additional_candidate_weight_bytes=0','additional_candidate_weight_bytes=1',1),
        '\n'.join(v for v in good.splitlines() if not v.startswith('SAMPLE process=0 arm=0 k=2560 n=512 columns=4 sample=0 '))]
    for text in invalid:
        try:timing(text,0)
        except RuntimeError:continue
        raise RuntimeError('corrupted component timing/numerical record accepted')
    pid=31
    begin=f'strata UR validation: pid={pid} phase=begin loader=/qualified/libur_loader.so\n'
    receipt=f'strata UR validation: pid={pid} phase=before-main threads_before=1 preinit=37 config_create=0 layers_query=0 enable=0 init=0 config_release=0 checked=46 untouched=1 retained_ref=1\n'
    constructor='UR_RESULT_ERROR_UNINITIALIZED\nUR_RESULT_ERROR_INVALID_NULL_HANDLE\n'
    for text in (begin+receipt+'native body\n',begin+constructor+receipt+'native body\n'):
        proof,body=validated_trace(text,pid)
        require(proof['dispatch_result']==46 and body=='native body\n','positive constructor/body separation')
    good=begin+constructor+receipt+'native body\n'
    bad_trace=[good+'UR_RESULT_ERROR_INVALID_NULL_HANDLE\n',good+'ZE_RESULT_ERROR_DEVICE_LOST\n',
        begin+'UR_RESULT_ERROR_OUT_OF_RESOURCES\n'+receipt,good.replace('checked=46','checked=47'),
        good+receipt,good.replace('phase=begin loader','phase=beginX loader'),
        begin+receipt.rstrip('\n'),good+'UR_RESULT_ERROR_UNINITIALIZED\n']
    for text in bad_trace:
        try:validated_trace(text,pid)
        except RuntimeError:continue
        raise RuntimeError('invalid constructor/body diagnostic accepted')
    return dict(passed=True,positive_process_orders=3,corrupt_variants=len(invalid),
        trace_controls=dict(positive=2,negative=len(bad_trace),passed=True),GPU_executed=False)


def admit(ctx):
    """Change only the controller before any GPU stage; reuse exact build bytes."""
    require(KEY not in validation.committed(RESULT),'fresh controller admission required')
    original=json.loads((ROOT/'record.json').read_text())
    previous=validation.committed(RESULT,(PREVIOUS_KEY,))
    require(ctx.compact(original)==previous and original['stage']=='built' and
        original['compilation_passed'] and not original['GPU_executed'] and
        not original['model_executed'] and not original['active'] and original['ownership_closed'] and
        len(original['results'])==1 and original['results'][0]==dict(mode='host',passed=True,
            native_exit_code=0,GPU_executed=False,ownership_closed=True),
        'preserve the compiled and31-case CPU-qualified pre-GPU result')
    project=ctx.source.pin(P,ctx.PREFIXES)
    before,after=original['project']['source_sha256'],project['source_sha256']
    require(before=={p:hashlib.sha256(raw).hexdigest() for p,raw in
        ctx.source.blobs(P,original['project']['revision'],before)} and set(before)==set(after) and
        {p for p in before if before[p]!=after[p]}=={'tools/sycl/q6_canonical_component.py'},
        'only the declared controller may change;all native/fixture sources stay exact')
    row=copy.deepcopy(original)
    row.pop('reader_controls');row.pop('module_import_controls');row.pop('pending_before_gpu')
    row.update(stage='admitted',passed=False,complete=False,admission_project=project,
        admission_source_commit=project['revision'],qualification_reuse=dict(
            original_result_key=PREVIOUS_KEY,original_evidence_commit=ctx.source.git(P,'rev-parse','HEAD').decode().strip(),
            native_source_commit=original['project']['revision'],fixture_sha256=original['binary']['sha256'],
            native_recompiled=False,native_CPU_host_cases_reused=31,GPU_qualification_inherited=False,
            scope='Same14TUs,801compiler inputs,24external links,recipe,census,fixture and selected image. Original build and CPU checks stay unchanged in reachable Git. Recheck constructor/body trace predicates before any GPU stage.'))
    row['plan']['trace_validation']='Unique qualified pre-main receipt must attest expected37/46. Only those named errors may precede it;every UR/ZE error after it remains a failure. Require2positive/8negative CPU trace controls,then fresh small full-trace GPU evidence. No prior GPU result or model gain inherited.'
    ctx.current(row,committed=False);ctx.save(row)
    data=json.loads(RESULT.read_text());require(data[PREVIOUS_KEY]==previous,'original result changed during admission')
    del data[PREVIOUS_KEY];write(RESULT,data)
    print('Admitted unchanged fixture for fresh trace controls;original result archived in Git. Commit first.',flush=True)


def prepare(ctx):
    import build_q6_head_model as primary
    require(not ROOT.exists() and KEY not in validation.committed(RESULT),'fresh component plan only')
    native,identity=basis();model=validation.committed(RESULT,(MODEL,))
    require(model['passed'] and model['complete'] and model['ownership_closed'],'bounded current model/runtime proof required')
    recipe=copy.deepcopy(native['recipe']);old_root=str(primary.ROOT)
    recipe['configure']=[v.replace(native['project']['root'],str(P)).replace(old_root,str(ROOT)) for v in recipe['configure']]
    recipe['environment']={k:v.replace(old_root,str(ROOT)) for k,v in recipe['environment'].items()}
    recipe['targets']=recipe['build_targets']=[ctx.TARGET];recipe.pop('host_modes',None)
    require(recipe['configure'][recipe['configure'].index('-S')+1]==str(P) and
        recipe['configure'][recipe['configure'].index('-B')+1]==str(ROOT/'build'),'isolated build paths')
    env=dict(model['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),
        PYTHONPYCACHEPREFIX=str(ROOT/'cache/python'),STRATA_MEASUREMENT_ROOT=str(ROOT),STRATA_SYCL_DECODE_Q6_DPAS_HEAD='0',
        STRATA_Q6_DPAS_GATE='1',UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/'validation.log'))
    for k in ('LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER'):env.pop(k,None)
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and env['EnableDirectSubmission']=='0' and
        env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE']==env['ZES_ENABLE_SYSMAN']=='0','qualified runtime safety required')
    manifest=model['runtime_manifest']
    row=dict(stage='prepared',project=ctx.source.pin(P,ctx.PREFIXES),dependencies=copy.deepcopy(native['dependencies']),
        recipe=recipe,toolchain=native['toolchain'],external_inputs=native['external_inputs'],
        fallbacks=dict(selected_engine=identity['engine']),selected_native=identity,environment=env,
        runtime_manifest=manifest,runtime_pins=json.loads(Path(manifest['path']).read_text()),
        native_validation_probe=model['probe'],python_sha256=sha(PYTHON),boot=model['boot'],driver=model['driver'],
        commands=[],results=[],timing=[],passed=False,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,model_executed=False,budgets=ctx.BUDGET,
        host_budget=dict(wall=120,cpu=120,rss_cap=1<<30,text_cap=1<<20,file_cap=2<<20,total_cap=64<<20),
        native_budget=dict(wall=900,cpu=1800,rss_cap=8<<30,text_cap=16<<20,file_cap=32<<20,total_cap=768<<20),
        plan=dict(question='Which actual decode C4 Q6 shapes benefit from direct canonical DPAS without another weight allocation?',
            source='4ac5c8af8:current model request5 static shape table;qualified shared-head cohort must close first.',
            reference='Actual production columns entry for five shapes;generic native_mmvq for2560x512. Same canonical/Q8_1 allocation and one-node graphs in both arms.',
            shapes=SHAPES,small=SMALL,order='CPU parser controls,fresh14TU build/IR census,31native host cases,small4shape/2seed traced checks,full6shape/2seed checks,three independent timing processesR/C,C/R,R/C. Commit every closed stage;stop first failure,no retry.',
            timing='Five samples/arm/shape/process;two warmup batches.4096replays for512rows,1024for other dense shapes,256for the full head. Require>=10ms and>=300stamp units;submission/wait wall and threadCPU accounting stays disjoint. Retain all180samples.',
            acceptance='Per-shape all3process median time ratios<1;no minimum improvement floor. Keep losing shapes and prompt/C1 on incumbents. No global dispatch/model adoption from a component.',
            limits='Synthetic byte-valid Q6/Q8 and independent scalar/envelope oracle. One repeatedly accessed matrix;cache behavior differs from the complete model. No physical traffic,saturation,task,PP/TG,agent,full262144 or RESTORE claim.'),
        retention='One temporary standalone fixture for this named comparison;no new engine generation. Keep useful per-shape numbers/all timing samples/source/recipe/fault results. Retire normal traces,JIT and fixture build after closure. Preserve current engine/phase references and unresolved legacy holds.')
    ctx.current(row,committed=False);ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    ctx.save(row);print('Prepared six-shape component;commit before CPU controls/build.',flush=True)


def run(ctx,mode):
    row=json.loads((ROOT/'record.json').read_text());ctx.current(row)
    if mode=='controls':
        require(row['stage'] in ('prepared','admitted') and 'reader_controls' not in row,'fresh parser controls only')
        try:row['reader_controls']=controls()
        except BaseException as error:
            row.update(stage='failed',passed=False,complete=False,error=repr(error),reader_controls=dict(passed=False,GPU_executed=False))
            ctx.save(row);raise
        if row['stage']=='admitted':row['stage']='built'
        ctx.save(row);print('CPU parser controls passed;commit before next native stage.',flush=True);return
    if mode=='build':
        require(row['reader_controls']['passed'],'CPU parser controls first');ctx.compile_fixture()
        row=json.loads((ROOT/'record.json').read_text())
        try:row['codegen']=image(Path(row['binary']['path']).read_bytes())
        except BaseException as error:
            row.update(stage='failed',compilation_passed=True,passed=False,complete=False,error=repr(error))
            ctx.save(row);raise
        row.update(compilation_passed=True,passed=False,complete=False);ctx.save(row);return
    stages=['host','check-small','check','time0','time1','time2']
    require(mode in stages and row['stage']=='built' and len(row['results'])==stages.index(mode) and
        all(v['passed'] for v in row['results']),'fresh ordered fixture stage only')
    scope=None;outcome=dict(mode=mode,passed=False);cursor=None
    def persist():write(ROOT/(mode+'-execution.json'),dict(active=bool(scope and scope.active),commands=scope.commands if scope else []))
    try:
        scope=owner.Owner(ROOT/('owned-'+mode),persist)
        env=dict(row['environment']);log=ROOT/(mode+'-validation.log')
        if mode=='host':
            env.pop('STRATA_Q6_DPAS_GATE',None)
            event,out,err=scope.run('host',[row['binary']['path'],'--host'],env,P,**row['host_budget'])
            outcome['native_exit_code']=event.get('exit_code')
            require(owner.closed(event) and event['exit_code']==0 and
                out.read_text().splitlines().count('HOST passed=1 cases=31 GPU=0')==1,'31 actual native CPU controls required')
            outcome.update(passed=True,native_exit_code=0,GPU_executed=False)
        else:
            if mode!='check-small':env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
            env.update(LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(log))
            cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
            def interact(io,command):
                for phase,tag in enumerate(('STRATA_Q6_DPAS_GATE ','STRATA_Q6_DPAS_DEVICE_GATE ')):
                    while True:
                        line=io.line(180)
                        if line.startswith(tag):break
                    require(line==tag+'pid='+str(command['pid']) and
                        str(Path('/proc',str(command['pid']),'exe').resolve())==row['binary']['path'],'actual fixture process/gate identity')
                    identity=owner.identity(command['pid'])
                    require(identity and identity['pid']==identity['sid']==command['pid'] and
                        all(identity[k]==command['direct_identity'][k] for k in ('pid','start_ticks','sid')),
                        'actual owned fixture lifetime at both gates')
                    outcome['native_identity']=identity
                    actual=dict(v.split('=',1) for v in Path('/proc',str(command['pid']),'environ').read_text().split('\0') if '=' in v)
                    mapped=set(python_runtime.code_maps(command['pid']))
                    names=('LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER','UR_ADAPTERS_FORCE_LOAD','ZE_ENABLE_ALT_DRIVERS')
                    required={str(Path(env[k]).resolve()) for k in names[:2 if phase==0 else 4]}
                    require(actual==env and required<=mapped and
                        not mapped-(set(row['runtime_pins'])|{row['binary']['path']}),'immutable environment and phase-appropriate provider mappings')
                    outcome.setdefault('provider_admission',[]).append(dict(phase=phase,mapped_files=len(mapped),
                        mapped_path_sha256=hashlib.sha256('\0'.join(sorted(mapped)).encode()).hexdigest(),required_present=True,environment_immutable=True))
                    if phase:outcome['GPU_initialized']=True
                    io.send(b'continue\n')
                io.proc.stdin.close()
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
                io.exit(1)
            args=['--time',mode[-1]] if mode.startswith('time') else ['--'+mode]
            outcome['native_execution_attempted']=True
            event,out,err=scope.run(mode,[row['binary']['path'],*args],env,P,interaction=interact,**row['native_budget'])
            outcome['native_exit_code']=event.get('exit_code')
            require(owner.closed(event) and event['exit_code']==0,'owned native component failed')
            text,diagnostic=out.read_text(),err.read_text();outcome['native_exit_code']=0
            outcome['validation'],body=validated_trace(diagnostic,event['pid'])
            require(outcome['validation']['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'] and
                (not log.exists() or log.stat().st_size==0),'actual parameter-validation/UR result failed')
            if mode.startswith('time'):
                parsed=timing(text,int(mode[-1]));row['timing'].append(parsed)
                outcome['timing_index']=parsed['index']
            else:
                small=mode=='check-small';shapes=SMALL if small else SHAPES
                outcome['numerical']=numerical(text,shapes)
                require(text.splitlines().count(f'CHECK passed=1 small={int(small)} cases={2*len(shapes)} route=compat_graph columns=4')==1,'complete native check footer')
                if small:require('Q6DpasC4Canonical' in body and 'urEnqueueCommandBufferExp' in body,'actual traced canonical graph path')
            row['GPU_executed']=True;outcome.update(passed=True,host_probe_accepted=True,GPU_executed=True)
    except BaseException as error:outcome.update(passed=False,host_probe_accepted=False,error=repr(error))
    finally:
        if cursor is not None and scope:
            validation.finish(outcome,scope,cursor,row['boot'])
            if outcome.get('driver_after')!=row['driver']:outcome.update(passed=False,driver_mismatch=True)
        outcome['ownership_closed']=not scope or scope.active is None and all(owner.closed(v) for v in scope.commands)
        if scope and scope.commands and 'native_exit_code' not in outcome:
            outcome['native_exit_code']=scope.commands[-1].get('exit_code')
        outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed'])
        row['results'].append(outcome);row['active']=bool(scope and scope.active);row['ownership_closed']=outcome['ownership_closed']
        if not outcome['passed']:row.update(stage='failed',passed=False,complete=False,error=outcome.get('error','fixture qualification failed'))
        elif mode=='time2':row.update(stage='closed',passed=True,complete=True,decision=decision(row['timing']))
        ctx.save(row)
    require(outcome['passed'],outcome.get('error','fixture stage failed'))
    print('Closed '+mode+';component/model adoption remain separate.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','admit','controls','build','host','check-small','check','time0','time1','time2'),'six-shape component stage')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        ctx=context()
        if sys.argv[1]=='prepare':prepare(ctx)
        elif sys.argv[1]=='admit':admit(ctx)
        else:run(ctx,sys.argv[1])
