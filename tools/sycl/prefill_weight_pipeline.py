#!/usr/bin/env python3
"""Independent capacity and nonrecycling prefill comparisons; Main owns all jobs.

Main-authored from own05eceacd/535911a2b controls; repository LICENSE.
Closed nine-arm campaign restores from05eceacd/1496a4902, never this v2 entrypoint.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode=True
import prefill_upload_overlap as base
import kernel_trace

P,B,RESULT=base.P,base.B,base.RESULT
ROOT=B/'prefill-unique-slot-math-v1'
VIEW=ROOT/'source'
RECORD=ROOT/'record.json'
KEY='prefill_unique_slot_math_v1'
OLD_KEY='prefill_sycl_copy_groups_v1'
SOURCE='bench/sycl/prefill_sycl_copy_math.cpp'
LANE='include/strata/prefill/sycl_copy_groups.hpp'
UNIQUE='include/strata/prefill/sycl_unique_slot_copies.hpp'
TEST='tests/core/test_weight_pipeline_plan.py'
PREFIXES=[*base.PREFIXES,SOURCE,LANE,UNIQUE,TEST]
SEED_EVIDENCE='07735df239679929be29c859561ae0501a9a3ab7'
SEED_SOURCE='2187bfa5ddff6d8331187b8cfc22157852c2d461'
BUILD_EVIDENCE='ee8a41fd6'
SAFETY_EVIDENCE='4eb8276e0'
ROWS=(128,1024,8192)
ARMS=('base128','base512','unique512')
# (single-write ownership, physical slots); all other scheduling stays fixed.
POLICIES={'base128':(False,128),'base512':(False,512),'unique512':(True,512)}
UNITS={'S':('base128','base512'),'U':('base512','unique512'),
       'netUnique':('base128','unique512')}
BLOCKS=(ARMS,tuple(reversed(ARMS)),('base512','base128','unique512'))
PAYLOAD=1971200
EXPERTS=512
BUILD=dict(base.BUILD,file_cap=1280<<20,total_cap=3<<30)
RUN=dict(base.RUN)
TRACED_RUN=dict(RUN,text_cap=64<<20,file_cap=96<<20,total_cap=160<<20)
RAW_CAP=512<<20
TOTAL_CAP=3<<30
BACKEND=dict(STRATA_SYCL_PARALLEL_IQ_DEQUANT='1',STRATA_SYCL_PREFILL_NATIVE16='bf16',
    STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
    STRATA_SYCL_PREFILL_GU_SWIGLU='large',STRATA_SYCL_PREFILL_DOWN_SCATTER='large',
    STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original')
SAFETY=dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',
    EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',ZES_ENABLE_SYSMAN='0')
require,sha,fingerprint=base.require,base.sha,base.fingerprint
git,write=base.git,base.write


def metadata_gate(row):
    for name in ['native_validation_probe','runtime_manifest','validation_provider','driver','boot','weights']:
        require(name in row and bool(row[name]),'missing transitive native input:'+name)


def seed_recipe():
    # Closed preparation needs the committed recipe,not a disposable checkpoint
    # or the retired native executable. Fresh source/dependency checks follow.
    seed=json.loads(git(P,'show',SEED_EVIDENCE+':'+str(RESULT.relative_to(P))))['prefill_weight_pipeline_v2']
    require(seed['stage']=='closed' and seed['passed'] is True and seed['complete'] is True
        and seed['active'] is False and seed['ownership_closed'] is True
        and seed['native_source_commit']==seed['project']['revision']==SEED_SOURCE
        and seed['original_math_runtime_source']==base.original.SOURCE,
        'exact committed closed seed recipe required')
    metadata_gate(seed)
    return seed


def extract_weights():
    # Stream all512experts instead of retaining a1GB Python concatenation.
    # Original full-model identity is separate from newly hashed role slices.
    model=json.loads((B/'coherent-prefill-model-v3/record.json').read_text())['models']
    layout=next(Path(p) for p in model if p.endswith('/native_experts.txt'))
    require(sha(layout)==model[str(layout)]['sha256'],'original native layout changed')
    line=next(v for v in layout.read_text().splitlines() if v.split()[:1]==['2'])
    values=[int(x) for x in line.split()]
    require(len(values)==8 and values[:3]==[2,22,20] and values[4]==1971200,
            'fixture requires actual layer2IQ2_S/IQ4_NL geometry')
    shard=next(Path(p) for p in model if p.endswith('-00001-of-00002.gguf'))
    st=shard.stat();identity=model[str(shard)]['identity']
    require(all(getattr(st,'st_'+key)==identity[name] for key,name in
        [('dev','dev'),('ino','ino'),('size','bytes'),('mtime_ns','mtime_ns'),('ctime_ns','ctime_ns')]),
        'original model file identity changed;do not infer a new full hash')
    sizes=[524800,524800,921600]
    hashes={role:hashlib.sha256() for role in ['gate','up','down']}
    path=ROOT/'weights.bin'
    with shard.open('rb') as f,path.open('xb') as out:
        for expert in range(EXPERTS):
            for role,at,size in zip(['gate','up','down'],values[5:],sizes):
                offset=at+expert*size;f.seek(offset);raw=f.read(size)
                require(len(raw)==size,'actual model role is out of bounds')
                out.write(raw);hashes[role].update(raw)
    after=shard.stat()
    require(all(getattr(after,'st_'+key)==getattr(st,'st_'+key) for key in
        ['dev','ino','size','mtime_ns','ctime_ns']),'model file changed while extracting role slices')
    path.chmod(0o444);require(path.stat().st_size==EXPERTS*PAYLOAD,'complete real layer corpus required')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),layer=2,experts=EXPERTS,
        layout=dict(path=str(layout),sha256=sha(layout)),source_file=str(shard),
        original_source_file_identity=identity,original_full_file_sha256=model[str(shard)]['sha256'],
        full_file_rehashed=False,
        roles=[dict(role=role,base=at,expert_stride=size,bytes=size,experts=EXPERTS,
                    ordered_concatenated_role_sha256=hashes[role].hexdigest())
               for role,at,size in zip(['gate','up','down'],values[5:],sizes)],
        order='expert0..511,each gate/up/down;source offset=role base+expert*role stride')



def admission(row):
    # Outcomes are outputs, not inputs to later independent arms. Only this
    # complete immutable subtree may admit a launch. A build changes it once
    # (new ELF/census), requiring a commit before the first native case.
    keys=['project','native_source_commit','source_view','foundation','allowed',
          'recipe','recipe_tools','runtime_manifest','native_validation_probe',
          'validation_provider','build_environment','driver','boot','weights',
          'phase_references','historical_safety','cases','plan']
    value={k:row[k] for k in keys}
    value['project']={k:v for k,v in row['project'].items() if k!='source_sha256'}
    value['allowed']=dict(files=len(row['allowed']),sha256=fingerprint(row['allowed']))
    if 'binary' in row:
        value['binary']=row['binary']
        value['actual_inputs']={k:dict(files=len(v),sha256=fingerprint(v))
                                for k,v in row['actual_inputs'].items()}
    return value


def compact(row):
    return dict(admission=admission(row),**{k:row[k] for k in
        ['stage','passed','complete','active','ownership_closed','results',
         'model_trial_allowed','performance_qualified','analysis','error',
         'host_admission_controls','fresh_faults','boot_unchanged','driver_after'] if k in row})


def save(row):
    write(RECORD,row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def safety_prerequisite():
    key='prefill_unique_slot_lifetime_v1'
    current=base.ur_validation.committed(RESULT,(key,))
    archived=json.loads(git(P,'show',SAFETY_EVIDENCE+':'+str(RESULT.relative_to(P))))[key]
    require(current['stage']=='closed' and all(current[k] is True for k in
            ['passed','complete','ownership_closed']) and current['active'] is False
            and len(current['results'])==len(archived['results'])==15
            and not current['fresh_faults'] and current['model_trial_allowed'] is False
            and current['performance_qualified'] is False
            and current['retirement']['stage']=='complete', 'closed retired15case prerequisite required')
    require(all(not Path(v['path']).exists() for v in current['retirement']['paths']),
            'ended small generation/captures must actually be retired')
    for now,then in zip(current['results'],archived['results']):
        for name in ['label','passed','native_exit_code','ownership_closed','kernels',
                     'payload_copies','readiness_barriers','providers_sha256']:
            require(fingerprint(now[name])==fingerprint(then[name]),'historical safety outcome changed')
    for name in [LANE,UNIQUE]:
        require(sha(P/name)==hashlib.sha256(subprocess.check_output(
            ['git','show',archived['native_source_commit']+':'+name],cwd=P)).hexdigest(),
            'copy-owner source differs from the tested safety contract')
    return dict(evidence=SAFETY_EVIDENCE,source=archived['native_source_commit'],
        result_sha256=fingerprint(archived),phase_references=archived['phase_references'],
        scope='Historical identical owner source only;retired fixture ELF not reused. Fresh full-math image independently validated;no model/fault qualification.')


def case_gate(case,weights):
    require(case['family']=='math' and case['arm'] in ARMS and case['rows'] in ROWS
            and type(case['rows']) is int and case['mode'] in ['metadata','batch']
            and type(case['trace']) is bool
            and case['argv']==[str(ROOT/'benchmark'),case['mode'],case['arm'],str(case['rows']),weights]
            and case['root']==str(ROOT/case['label']), 'frozen native case differs')
    require(all(case['environment'].get(k)==v for k,v in (BACKEND|SAFETY).items()),
            'immutable backend/runtime safety settings differ')


def expected_cases():
    values=[('metadata-'+a,'metadata',a,ROWS[0],True,None) for a in ARMS]
    values += [(f'validated-{m}-{a}','batch',a,m,True,None) for m in ROWS for a in ARMS]
    values += [(f'clean-{m}-block{i}-{a}','batch',a,m,False,i)
               for m in ROWS for i,block in enumerate(BLOCKS) for a in block]
    return values


def plan_gate(row):
    require([(c['label'],c['mode'],c['arm'],c['rows'],c['trace'],c['pair'])
             for c in row['cases']]==expected_cases(), 'finite39case order required')
    for case in row['cases']:case_gate(case,row['weights']['path'])
    require(row['model_trial_allowed'] is False and row['performance_qualified'] is False,
            'this component cohort does not admit model trials')


def batch_gate(batch,arm,rows):
    require(arm in ARMS and rows in ROWS,'finite native batch')
    expected=dict(rows=rows,epochs=512,requested_H2D_bytes=EXPERTS*PAYLOAD,copies=512,
        gu_fused=512 if rows>=2334 else 0,down_fused=512 if rows>=2334 else 0,
        split_primary_rows=rows//2,actual_source_experts=EXPERTS,source_footprint_bytes=EXPERTS*PAYLOAD,
        copy_adapter_calls=512,ready_barrier_calls=8,reader_barrier_calls=0 if arm=='unique512' else 8,
        single_write_copy_calls=512 if arm=='unique512' else 0,timed_initial_slot=0)
    require(batch.get('kind')=='batch' and batch.get('arm')==arm
        and all(type(batch.get(k)) is int and batch[k]==v for k,v in expected.items())
        and all(batch.get(k) is True for k in ['production_gemm_dispatch','split_tail_checked',
            'full_bitwise_reference_and_guards','warm_owner_retired']), 'actual timed work differs')
    require(all(type(batch.get(k)) in [int,float] and math.isfinite(batch[k]) for k in
        ['wall_s','timer_pair_max_s']) and batch['wall_s']>=.1
        and 0<=batch['timer_pair_max_s']<=.001*batch['wall_s'], 'ineligible timing batch/clock overhead')


def source_proof_gate(proof,arm):
    unique,slots=POLICIES[arm]
    expected=dict(kind='source-proof',ordinary_readonly_mapping=True,host_slots=0,policy=arm,
        dependency_bridge=True,separate_queue=True,single_write_destinations=unique,
        physical_slots=slots,prefetch_window=128,ready_width=64,refill_width=64,
        reader_marker='none-single-write' if unique else 'after-math',
        admitted_bank_allocation_bytes=slots*PAYLOAD+512,runtime_check_and_observer_retained=True)
    require(fingerprint(proof)==fingerprint(expected),'actual policy/allocation admission differs')


def analyze(results):
    analysis={}
    for rows in ROWS:
        require(len([v for v in results if v['clean'] and v['rows']==rows])==9,
                'retain exactly nine original clean samples per shape')
        groups=[{v['arm']:v for v in results if v['clean'] and v['rows']==rows and v['pair']==i}
                for i in range(3)]
        require(all(set(g)==set(ARMS) for g in groups),'three complete matched blocks required')
        analysis[str(rows)]={}
        for name,(before,after) in UNITS.items():
            ratios=[g[after]['batch']['wall_s']/g[before]['batch']['wall_s'] for g in groups]
            life=[g[after]['lifecycle_plus_batch_s']/g[before]['lifecycle_plus_batch_s'] for g in groups]
            analysis[str(rows)][name]=dict(reference=before,candidate=after,paired_time_ratios=ratios,
                geometric_mean_ratio=math.prod(ratios)**(1/3),paired_lifecycle_ratios=life,
                repeated_component_gain=all(r<1 for r in ratios),model_adoption_qualified=False)
    return analysis


def admission_gate(row):
    require(fingerprint(admission(row))==fingerprint(base.ur_validation.committed(RESULT,(KEY,'admission'))),
            'commit exact immutable plan/build evidence first')


def output_gate(row):
    require(fingerprint(compact(row))==fingerprint(json.loads(RESULT.read_text())[KEY]),
            'private/public outcome mismatch;inspect existing operation before advancing')


def check(row,committed=True):
    if committed:admission_gate(row)
    plan_gate(row)
    current=base.original.coherent.build.common.source.pin(P,PREFIXES)
    require(all(current[k]==row['project'][k] for k in
            ['root','count','aggregate_sha256','source_sha256']),'frozen source/controller changed')
    require(base.guards.driver_guard()==row['driver'] and row['boot']==
            Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'boot/driver changed')
    require(all(sha(p)==h for p,h in row['recipe_tools'].items()),'compiler tool changed')
    require(sha(row['weights']['path'])==row['weights']['sha256'],'actual corpus changed')
    require(sha(row['runtime_manifest']['path'])==row['runtime_manifest']['sha256'],'runtime manifest changed')
    require(all(sha(p)==h for p,h in json.loads(Path(row['runtime_manifest']['path']).read_text()).items()),
            'qualified runtime provider changed')
    require(safety_prerequisite()==row['historical_safety'],'copy-owner qualification source changed')
    require(all(sha(v['path'])==v['sha256'] for v in row['phase_references'].values()),
            'selected phase reference changed')
    required=row['allowed'] if 'binary' not in row else {p:h for g in row['actual_inputs'].values() for p,h in g.items()}
    require(all(sha(p)==h for p,h in required.items()),'actual source/SDK/link/object input changed')
    if 'binary' in row:require(sha(row['binary']['path'])==row['binary']['sha256'],'actual executable changed')


def artifact_gate():
    files=[p for p in ROOT.rglob('*') if p.is_file()]
    require(sum(p.stat().st_size for p in files)<=TOTAL_CAP,'whole cohort artifact budget')
    text=[p for p in files if p.suffix in ['.stdout','.stderr','.trace','.log']]
    require(sum(p.stat().st_size for p in text)<=RAW_CAP,'whole cohort raw text budget')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh math cohort;no retry')
    prior=base.ur_validation.committed(RESULT,(OLD_KEY,))
    require(prior['complete'] and prior['ownership_closed'] and not prior['active']
            and prior['retention']['stage']=='complete'
            and not Path(prior['retention']['root']).exists(),'retire old component generation first')
    safety=safety_prerequisite();native,foundation=base.original.verify_original();seed=seed_recipe()
    original=json.loads(git(P,'show',BUILD_EVIDENCE+':'+str(RESULT.relative_to(P))))[OLD_KEY]
    project=base.original.coherent.build.common.source.pin(P,PREFIXES)
    ROOT.mkdir(mode=0o700)
    for name in ['tmp','cache','owned']:(ROOT/name).mkdir(mode=0o700)
    generated={}
    names=[p for p in native['project']['source_sha256'] if p.startswith(('include/','src/'))]
    for name,raw in base.original.coherent.build.common.source.blobs(P,base.original.SOURCE,names):
        path=VIEW/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);generated[str(path)]=sha(path)
    allowed=generated|json.loads(Path(native['external_inputs']['path']).read_text())
    allowed.update(json.loads(Path(seed['runtime_manifest']['path']).read_text()))
    for dependency in native['dependencies']:
        if 'sycl-tla' in dependency['root']:allowed.update(base.original.coherent.recover(dependency))
    for name in [SOURCE,LANE,UNIQUE]:allowed[str(P/name)]=sha(P/name)
    for image in [seed['validation_provider'],*seed['native_validation_probe'].values()]:
        allowed[str(Path(image['path']).resolve())]=image['sha256']
    before=str(Path(original['binary']['path']).parent)
    recipe=[]
    for old in original['recipe']:
        if old['label'] in ['lifetime','link-lifetime']:continue
        entry={k:([x.replace(before,str(ROOT)) for x in v] if isinstance(v,list) else
                  v.replace(before,str(ROOT)) if isinstance(v,str) else v) for k,v in old.items()}
        recipe.append(entry)
    weights=extract_weights();cases=[]
    for label,mode,arm,rows,trace,pair in expected_cases():
        root=ROOT/label
        template=next(v for v in seed['cases'] if v['source']=='usm' and v['mode']=='serial' and v['trace'] is trace)
        env={k:v.replace(template['root'],str(root)) for k,v in template['environment'].items()};env.update(BACKEND)
        cases.append(dict(label=label,root=str(root),family='math',mode=mode,arm=arm,rows=rows,trace=trace,pair=pair,
            argv=[str(ROOT/'benchmark'),mode,arm,str(rows),weights['path']],environment=env))
    row=dict(stage='prepared',passed=False,complete=False,active=False,ownership_closed=True,project=project,
        native_source_commit=project['revision'],source_view=dict(original_commit=base.original.SOURCE,
        files=len(generated),sha256=fingerprint(generated)),foundation=foundation,allowed=allowed,
        recipe=recipe,recipe_tools=original['recipe_tools'],weights=weights,
        runtime_manifest=seed['runtime_manifest'],native_validation_probe=seed['native_validation_probe'],
        validation_provider=seed['validation_provider'],historical_safety=safety,
        phase_references=safety['phase_references'],driver=base.guards.driver_guard(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        build_environment={k:v.replace(before,str(ROOT)) for k,v in original['build_environment'].items()},
        cases=cases,results=[],commands=[],model_trial_allowed=False,performance_qualified=False,
        plan=dict(units=UNITS,policies=POLICIES,rows=ROWS,blocks=BLOCKS,
            question='At identical separate queue/window128/ready64/refill64/math/bytes,what do512instead of128physical slots and enforced no-reuse without overwrite-reader markers cost?',
            coherent='S changes only physical slot count. U compares identical512slot layouts;single-write enforcement and omitting reader markers are inseparable. Uat128would be invalid for512writes. netUnique checks the combined policy;do not infer constituent success from it. Queue/readiness/window tuning stays a later independent plan.',
            timing='Each independent process builds a512real-expert HostUSM numerical reference,warms2jobs,drains/destroys that owner,and creates a fresh owner for the512timed ordinary-RAMcopies. Same reset in every arm;timed slot ordinal0. Reference/reset/validation outside batch;terminal drain inside. Keep preparation/reset/retirement separately,including their sum with the batch. Full bitwise/split/tail/status/guard/source checks and all final128or512payloads. Batch>=100ms;128clock-pair max<=.001batch.',
            order='3metadata,9full-math traced validations,then27clean independent processes over128/1024/8192rows in3declared3-arm blocks. Shape rows are not model input tokens. No model trial from this plan.',
            decision='Each S/U/netUnique at each shape requires all3paired batch ratios<1,no minimum gain. Keep lifecycle-inclusive ratios and all individual samples. Valid losses continue the frozen cohort;other units remain eligible. No blanket shape adoption. Hardware saturation and model speed remain unknown.',
            source='Main-authored own05eceacd math plus own535911a2b identical copy owners;fresh original14508Runtime/IQ/native16/Gemm/TLA,dequant formats/order/storage/rounding unchanged. No XeStrata code or design.',
            build='Fresh9TUs/11objects and one benchmark,13owned commands including host controls. No object/archive reuse. Compiler/dependency/recipe/providers/ELF/corpus/actual dispatch checked before launch. Admission subtree immutable;outcome updates need no per-case commit because they are not comparison inputs. Commit build before native cases;commit useful results before any later qualification consumer.',
            retention='Bounded third component generation for39cases only,preserve both selected phase references. Recover exact final source/recipe/corpus from pushed Git,retain useful27samples/9decisions,then retire this build/corpus/raw/source view before another candidate. Total artifact3GiB;raw text512MiB for9full-math API traces and build/control logs,plus per-child caps.',
            stop='First source/provider/count/validation/numeric/dispatch/closure/deadline/fault/timing/budget failure is terminal;do not retry or refresh hashes. Finite owned children. No GPU reset or foreign process/service action.',
            limits='Synthetic fixed-M/two-job pattern,not model routing/PP/TG/TTFT. No async/submission/drain-fault or active-observer qualification,model/task/agent/decode/full262144/RESTORE or ordinary provider teardown. Preserve all prior failures.'))
    plan_gate(row);artifact_gate();save(row)
    print('Prepared39case comparison;commit immutable admission before build.',flush=True)


def execute(action):
    row=json.loads(RECORD.read_text())
    require(row['stage'] in (['prepared'] if action=='build' else ['built','testing']),'terminal/wrong-stage cohort')
    owner,cursor=None,None
    try:
        output_gate(row);check(row);artifact_gate()
        label='build' if action=='build' else 'case-'+str(len(row['results']))
        with (ROOT/(label+'.started')).open('x') as f:json.dump(dict(action=action,source=row['native_source_commit']),f)
        owner=base.owner_api.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',owner.commands))
        row.update(active=True,ownership_closed=False,stage='building' if action=='build' else 'testing');save(row)
        if action=='build':
            actual=dict(compiled={},links={},objects={});allowed=dict(row['allowed'])
            for recipe in row['recipe']:
                command,out,err=owner.run(recipe['label'],recipe['argv'],row['build_environment'],ROOT,**BUILD)
                require(base.owner_api.closed(command) and command['exit_code']==0,'native build/control failure:'+str(err))
                if recipe['label']=='host-controls':
                    require('\nOK\n' in err.read_text(),'host negative-admission controls');row['host_admission_controls']=True
                if 'dep' in recipe:
                    for name in shlex.split(Path(recipe['dep']).read_text().replace('\\\n',' ').split(':',1)[1]):
                        path=str(Path(name).resolve());h=sha(path);require(allowed.get(path)==h,'unfrozen compiler input:'+path);actual['compiled'][path]=h
                if recipe.get('link'):
                    for line in out.read_text().splitlines():
                        p=Path(line.strip())
                        if p.is_absolute() and p.is_file():
                            path=str(p.resolve());h=sha(path);require(allowed.get(path)==h,'unfrozen link input:'+path);actual['links'][path]=h
                for path in recipe['outputs']:
                    allowed[str(Path(path).resolve())]=sha(path)
                    if path.endswith('.o'):actual['objects'][path]=sha(path)
            require(len(actual['compiled'])>900 and len(actual['links'])>10 and len(actual['objects'])==11,
                    'fresh9TU/11object dependency census required')
            row.update(stage='built',passed=True,actual_inputs=actual,
                binary=dict(path=str(ROOT/'benchmark'),sha256=sha(ROOT/'benchmark')))
        else:
            case=row['cases'][len(row['results'])];root=Path(case['root']);require(not root.exists(),'no native-case retry')
            root.mkdir(mode=0o700)
            for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
            owner=base.owner_api.Owner(root/'owned',lambda:write(ROOT/'ownership.json',owner.commands))
            cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).split('-- cursor: ',1)[1].strip()
            expected=['before-runtime','runtime-ready']+(['prewarm-ready','before-timing'] if case['mode']=='batch' else [])+['before-exit']
            gates,records,maps=[],[],{}
            env=base.engine_once.native_environment(row,case['environment'])
            def interact(channel,_entry):
                while True:
                    item=json.loads(channel.line(45))
                    if item['kind']=='gate':
                        name=item['name'];require(len(gates)<len(expected) and name==expected[len(gates)],'native gate order')
                        actual=dict(v.split('=',1) for v in Path('/proc',str(channel.proc.pid),'environ').read_text().strip('\0').split('\0'))
                        require(all(actual.get(k)==v for k,v in env.items()) and set(actual)-set(env)<= {'PWD','OLDPWD','SHLVL','_'},'actual startup environment differs')
                        require(Path('/proc',str(channel.proc.pid),'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')==case['argv'],'actual native argv differs')
                        maps[name]=base.event.mappings(channel.proc.pid,row)
                        if name!='before-runtime':
                            needed=[base.event.LOADER,env['ZE_ENABLE_ALT_DRIVERS'],env['UR_ADAPTERS_FORCE_LOAD'],'/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so']
                            if case['trace']:needed.append(base.event.VALIDATION)
                            require(all(str(Path(p).resolve()) in maps[name] for p in needed),'initialized native provider missing')
                        gates.append(name);channel.send(b'continue\n')
                    else:
                        records.append(item)
                        if item['kind']=='done':channel.exit(30);break
            command,out,err=owner.run(case['label'],case['argv'],env,P,interaction=interact,
                                     **(TRACED_RUN if case['trace'] else RUN))
            require(base.owner_api.closed(command) and command['exit_code']==0,'native math case failed:'+str(err))
            validation=base.ur_validation.preload_evidence(err.read_text(),command['pid'])
            require(gates==expected and records[-1]==dict(kind='done',passed=True),'native completion missing')
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,'UR validation output')
            require(not re.search(r'\[(?:error|critical)\]|prefill fixed-slot math fixture failure:|Strata SYCL cleanup:',err.read_text(),re.I),'native validation/cleanup failure')
            require(len(records)==(4 if case['mode']=='batch' else 3),'complete native source/math result required')
            source_proof_gate(records[0],case['arm']);life=records[-2]
            require(life['kind']=='source' and all(type(life[k]) in [int,float] and math.isfinite(life[k]) and life[k]>=0 for k in
                ['preparation_s','reset_s','retirement_s']),'invalid owner lifecycle')
            result={k:case[k] for k in ['label','family','arm','rows','mode','pair']}
            result.update(clean=not case['trace'],passed=True,native_exit_code=0,ownership_closed=True,
                validation=validation,providers_sha256=fingerprint(maps),source_proof=records[0],source_lifecycle=life)
            if case['mode']=='batch':
                batch_gate(records[1],case['arm'],case['rows']);result['batch']=records[1]
                result['lifecycle_plus_batch_s']=sum(life[k] for k in ['preparation_s','reset_s','retirement_s'])+records[1]['wall_s']
            if case['trace']:
                trace=(root/'ur.trace').read_text();kernels=len(re.findall(r'---> urEnqueueKernelLaunch\w*',trace))
                copies=sum('SUCCESS (ZE_RESULT_SUCCESS) in zeCommandListAppendMemoryCopy(' in line and
                    ('size='+str(PAYLOAD)+',') in line for line in err.read_text().splitlines())
                barriers=trace.count('---> urEnqueueEventsWaitWithBarrierExt')
                expected_math=0 if case['mode']=='metadata' else (4104 if case['rows']>=2334 else 6156)
                expected_copies=0 if case['mode']=='metadata' else 1026
                expected_barriers=0 if case['mode']=='metadata' else (9 if case['arm']=='unique512' else 18)
                require((kernels,copies,barriers)==(expected_math,expected_copies,expected_barriers),'actual math/copy/readiness/reader census differs')
                result['validated_counts']=dict(math_launches=kernels,payload_sized_native_copies=copies,
                    marker_barriers=barriers,payload_bytes=PAYLOAD)
                if case['mode']=='batch':
                    named=kernel_trace.launch_census(trace)
                    parallel={key:sum(v['launches'] for name,v in named['per_kernel'].items() if key in name)
                        for key in ['parallel_iq_launchILi22ELb1','parallel_iq_launchILi20ELb0']}
                    require(named['launches']==expected_math and all(v==1026 for v in parallel.values()),'actual production IQ selection differs')
                    result['validated_counts']['parallel_IQ_launches']=parallel
            check(row,False);row['results'].append(result);row['passed']=True
            if len(row['results'])==len(row['cases']):row.update(stage='closed',complete=True,analysis=analyze(row['results']))
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,error=repr(error));raise
    finally:
        if owner is not None:row['commands'].extend(owner.commands)
        row['active']=owner is not None and owner.active is not None
        row['ownership_closed']=not row['active'] and all(base.owner_api.closed(c) for c in row['commands'])
        try:
            row['boot_unchanged']=row['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            row['driver_after']=base.guards.driver_guard()
            if cursor:
                lines=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                row['fresh_faults']=[v for v in lines.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)',v,re.I)]
            if not row['boot_unchanged'] or row['driver_after']!=row['driver'] or row.get('fresh_faults'):
                row.update(stage='failed',passed=False,complete=False)
            artifact_gate()
        except BaseException as error:row.update(stage='failed',passed=False,complete=False,health_error=repr(error))
        save(row)
    print(json.dumps(dict(stage=row['stage'],passed=row['passed'],completed_cases=len(row['results']),
        ownership_closed=row['ownership_closed'])),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['build'],['next'],['run']],'prepare|build|next|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='run':
            while True:
                code=execute('next')
                if code or json.loads(RECORD.read_text())['stage']=='closed':raise SystemExit(code)
        else:raise SystemExit(execute(sys.argv[1]))
