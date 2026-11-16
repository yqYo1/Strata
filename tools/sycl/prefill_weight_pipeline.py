#!/usr/bin/env python3
"""Production-Gemm source adapter comparison; Main owns native work.

Main-authored from this project's owned fixture controllers; repository LICENSE.
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
ROOT=B/'prefill-weight-source-v1'
VIEW=ROOT/'source'
KEY='prefill_weight_source_v1'
SOURCE='bench/sycl/prefill_weight_pipeline.cpp'
TEST='tests/core/test_weight_pipeline_plan.py'
PREFIXES=[*base.PREFIXES,SOURCE,TEST]
SEED_EVIDENCE='07735df239679929be29c859561ae0501a9a3ab7'
SEED_SOURCE='2187bfa5ddff6d8331187b8cfc22157852c2d461'
TRACED_RUN=dict(base.RUN,text_cap=64<<20,file_cap=96<<20,total_cap=160<<20)
# Includes one immutable1GB model corpus. It is not raw diagnostic text.
BUILD=dict(base.BUILD,file_cap=1280<<20,total_cap=2<<30)
ROWS=(16,128,1024,4096,8192)
PAYLOAD=1971200
EXPERTS=512
BACKEND=dict(STRATA_SYCL_PARALLEL_IQ_DEQUANT='1',STRATA_SYCL_PREFILL_NATIVE16='bf16',
    STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
    STRATA_SYCL_PREFILL_GU_SWIGLU='large',STRATA_SYCL_PREFILL_DOWN_SCATTER='large',
    STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original')
require,sha,fingerprint=base.require,base.sha,base.fingerprint
git,write=base.git,base.write


def compact(row):
    out={k:v for k,v in row.items() if k not in ['project','foundation','allowed','actual_inputs','commands']}
    out['project']={k:v for k,v in row['project'].items() if k!='source_sha256'}
    out['foundation_sha256']=fingerprint(row['foundation'])
    out['allowed']=dict(files=len(row['allowed']),sha256=fingerprint(row['allowed']))
    if 'actual_inputs' in row:
        out['actual_inputs']={k:dict(files=len(v),sha256=fingerprint(v)) for k,v in row['actual_inputs'].items()}
    return out


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


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


def check(row,committed=True):
    if committed:
        require(fingerprint(compact(row))==fingerprint(base.ur_validation.committed(RESULT,(KEY,))),
                'commit exact typed actual-weight plan/outcome first')
    metadata_gate(row)
    current=base.original.coherent.build.common.source.pin(P,PREFIXES)
    require(current['source_sha256']==row['project']['source_sha256'],'frozen actual-weight source changed')
    base.original.verify_original(expected=row['foundation'])
    for path,h in row['allowed'].items():require(sha(path)==h,'actual source/SDK input changed:'+path)
    for path,h in row['recipe_tools'].items():require(sha(path)==h,'actual compiler tool changed')
    require(sha(row['weights']['path'])==row['weights']['sha256'],'actual-weight bundle changed')
    require(base.guards.driver_guard()==row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'boot/driver changed')
    if 'binary' in row:
        require(sha(row['binary']['path'])==row['binary']['sha256'],'actual reader ELF changed')
        require(all(sha(p)==h for group in row['actual_inputs'].values() for p,h in group.items()),
                'actual compiler/link/object census changed')
        for case in row['cases']:
            require(case['arm'] in ['direct','staged'] and case['rows'] in ROWS and
                    case['argv']==[row['binary']['path'],case['mode'],case['arm'],str(case['rows']),row['weights']['path']]
                    and all(case['environment'].get(k)==v for k,v in BACKEND.items()),
                    'frozen production source argv/backend differs')


def batch_gate(batch,arm,rows):
    expected=dict(rows=rows,epochs=512,requested_H2D_bytes=EXPERTS*PAYLOAD,copies=512,
        gu_fused=512 if rows>=2334 else 0,down_fused=512 if rows>=2334 else 0,
        split_primary_rows=rows//2,actual_source_experts=EXPERTS,source_footprint_bytes=EXPERTS*PAYLOAD)
    require(arm in ['direct','staged'] and rows in ROWS and batch.get('kind')=='batch'
        and batch.get('arm')==arm and all(type(batch.get(k)) is int and batch[k]==v for k,v in expected.items())
        and all(batch.get(k) is True for k in ['production_gemm_dispatch','split_tail_checked','full_bitwise_reference_and_guards']),
        'actual production source/dispatch workload differs')
    require(all(type(batch.get(k)) in [int,float] and math.isfinite(batch[k]) for k in ['wall_s','timer_pair_max_s'])
        and batch['wall_s']>=.1 and 0<=batch['timer_pair_max_s']<=.001*batch['wall_s'],
        'ineligible production source batch/clock overhead')


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


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'existing actual-reader attempt;inspect owner,no retry')
    prior=base.ur_validation.committed(RESULT,('prefill_weight_pipeline_v2',))
    require(prior['stage']=='closed' and prior['source_cost_eligible'] is False
        and prior['model_trial_allowed'] is False
        and prior['retention']['stage']=='complete' and prior['retention']['all_selected_paths_absent'] is True,
        'preserve failed imported-source gate;use verified host-USMreference independently')
    lookahead=base.ur_validation.committed(RESULT,('prefill_weight_lookahead_v1',))
    require(lookahead['stage']=='closed' and lookahead['analysis']['scheduling_component_eligible'] is True
        and lookahead['retention']['stage']=='complete','closed immutable-USMscheduling evidence required')
    production=base.ur_validation.committed(base.original.RESULT,('coherent_prefill_model_v3',))
    require(production['environment']['STRATA_SYCL_PARALLEL_IQ_DEQUANT']=='1',
            'qualified model parallel-IQpolicy required')
    native,proof=base.original.verify_original()
    seed=seed_recipe();seed_root=str(Path(seed['binary']['path']).parent)
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
    for image in [seed['validation_provider'],*seed['native_validation_probe'].values()]:allowed[str(Path(image['path']).resolve())]=image['sha256']
    allowed[str(P/SOURCE)]=sha(P/SOURCE)
    weights=extract_weights()
    cc='/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    compilation=json.loads((base.original.ROOT/'build/compile_commands.json').read_text())
    recipes=[dict(label='host-controls',argv=['/usr/bin/python3','-B','-m','unittest','discover',
        '-s',str(P/'tests/core'),'-p','test_weight_pipeline_plan.py','-v'],outputs=[])]
    def compile_one(label,source,template,extra=()):
        entry=next(v for v in compilation if v['file']==str(P/template));flags=shlex.split(entry['command'])[1:]
        del flags[flags.index('-o'):flags.index('-o')+2];del flags[flags.index('-c'):flags.index('-c')+2]
        flags=[v.replace(str(P),str(VIEW)) for v in flags]
        dep,obj=ROOT/(label+'.d'),ROOT/(label+'.o')
        recipes.append(dict(label=label,argv=[cc,*flags,*extra,'-I'+str(VIEW/'src'),
            '-I'+str(VIEW/'tests/sycl'),'-MD','-MF',str(dep),'-c',str(source),'-o',str(obj)],
            dep=str(dep),outputs=[str(obj)]))
    for label,name in [('runtime','src/sycl/runtime.cpp'),('commands','src/sycl/command_profile.cpp'),
                       ('compat','src/sycl/compat.cpp'),
                       ('iq','src/kernels/sycl/iq.cpp'),('ops','src/prefill/sycl/native16.cpp'),
                       ('gemm','src/prefill/sycl/gemm.cpp'),('gemm16','src/prefill/sycl/gemm16.cpp'),
                       ('dense','src/prefill/sycl/vllm_xe2/native16.cpp')]:
        compile_one(label,VIEW/name,name,['-ffunction-sections','-fdata-sections'] if label!='dense' else [])
    compile_one('fixture',P/SOURCE,'src/sycl/runtime.cpp',['-fno-fast-math','-ffp-contract=off'])
    for label in ['dense-image','dense-host']:
        old=next(v for v in seed['recipe'] if v['label']==label)
        recipes.append({k:([x.replace(seed_root,str(ROOT)) for x in v] if isinstance(v,list) else v)
                        for k,v in old.items()})
    recipes.append(dict(label='link',argv=[cc,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-fsycl',*[str(ROOT/(x+'.o')) for x in ['runtime','commands','compat','iq','ops','gemm','gemm16','dense-image','dense-host','fixture']],
        base.event.LOADER,*['/opt/intel/oneapi/mkl/2026.1/lib/'+x for x in
            ['libmkl_sycl_blas.so','libmkl_intel_ilp64.so','libmkl_sequential.so','libmkl_core.so']],
        '-Wl,-t','-Wl,--gc-sections','-o',str(ROOT/'benchmark')],
        outputs=[str(ROOT/'benchmark')],link=True))
    cases=[]
    def case(label,mode,arm,rows,trace,pair=None):
        root=ROOT/label
        old=next(v for v in seed['cases'] if v['source']=='usm' and v['mode']=='serial' and v['trace'] is trace)
        env={k:v.replace(old['root'],str(root)) for k,v in old['environment'].items()}
        env.update(BACKEND)
        cases.append(dict(label=label,root=str(root),mode=mode,arm=arm,rows=rows,trace=trace,pair=pair,
            argv=[str(ROOT/'benchmark'),mode,arm,str(rows),weights['path']],environment=env))
    for arm in ['direct','staged']:case('metadata-'+arm,'metadata',arm,16,True)
    for rows in ROWS:
        for arm in ['direct','staged']:case(f'validated-{rows}-{arm}','batch',arm,rows,True)
    for rows in ROWS:
        for pair in range(3):
            for arm in (['direct','staged'] if pair%2==0 else ['staged','direct']):
                case(f'clean-{rows}-pair{pair}-{arm}','batch',arm,rows,False,pair)
    row=dict(stage='prepared',passed=True,complete=False,active=False,ownership_closed=True,
        project=project,native_source_commit=project['revision'],original_math_runtime_source=base.original.SOURCE,
        foundation=proof,seed_recipe=dict(source=SEED_SOURCE,evidence=SEED_EVIDENCE,
            sha256=fingerprint(seed),external_receipt_needed=False,old_ELFs_not_inputs=True),allowed=allowed,
        source_view=dict(original_commit=base.original.SOURCE,files=len(generated),sha256=fingerprint(generated)),
        weights=weights,
        production_iq_policy=dict(native_source=base.original.SOURCE,evidence_key='coherent_prefill_model_v3.environment.STRATA_SYCL_PARALLEL_IQ_DEQUANT',value='1'),
        recipe=recipes,recipe_tools={p:sha(p) for p in seed['recipe_tools']},runtime_manifest=seed['runtime_manifest'],
        validation_provider=seed['validation_provider'],native_validation_probe=seed['native_validation_probe'],
        build_environment={k:v.replace(seed_root,str(ROOT)) for k,v in seed['build_environment'].items()},
        driver=base.guards.driver_guard(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cases=cases,results=[],commands=[],model_trial_allowed=False,performance_qualified=False,
        plan=dict(
            unit='Source adapter S only:ordinary immutable read-only mapped RAM through the complete production cudaMemcpyAsync wrapper versus existing four-slot ExpertUploadStaging(HostUSM CPU memcpy plus SYCLcopy). Hold one scratch blob,the in-order compute queue,production Gemm dispatch and all math/precision/order fixed. The wrapper and source-owner differences are one coherent adapter comparison,not a claim about CPU memcpy or device service alone. Qcopy primitive and Oreadiness-order changes are separate future units,then their combination/interactions and net original-direct-source comparison.',
            build='Fresh nine translation units from immutable original14508Runtime/compat/IQ/ops/Gemm/Gemm16/vllmXe2 plus the current owned fixture;two fresh dense image/host objects. Identical original class definitions/preprocessing in every includer. No reused objects or archives. One temporary third component generation beside unchanged independent prefill14508and decode9849references. Retire it when source-adapter question closes;no replacement of the references.',
            inputs='Stream all512real layer2experts(IQ2_SGU/IQ4_NLDown) from the existing stat-pinned model using committed role base+expert*stride extraction. Hash each ordered role concatenation and the1009254400Bcorpus;label the original fullGGUFhash not rehashed. Both arms read all experts once per512job batch,with two warmups. Immutable1GBmodel-fixture exception:build total2GiB/per-file1280MiB,text32MiB;native case budgets unchanged,4GiBRSS. Corpus/source snapshot retire after closure.',
            order='Two traced metadata controls before appwork. Then traced direct/staged batches at16,128,1024,4096,8192rows in that order. Then for each shape three independent clean pairs direct/staged,staged/direct,direct/staged.42native processes total,no retry/reorder/exclusion. Each process has512reference jobs and two same-arm warmups outside timing,then512uploads/math plus checked drain timed. Reference/reset/correctness checks outside timing. Trace captures64MiBtext/96MiBfile/160MiBtotal;all children120swall/240sCPU/4GiBRSS.',
            gates='Exact committed typed plan/source,original source/SDK/compiler/dependency/runtime,executable/object/link/asset identity,actual startup argv/environment/providers,quiet validation and positive closure. Pre-exec bf16/vllm/auto,GUlarge,Downlarge,split original,parallelIQ1;no environment mutation. Actual timed fusedGU/Down counts0below2334and512above. Both primary/tail planes match the same ordered independent HostUSM/SYCLcopy reference bitwise;finite/status0/all source+device guards/final expert511payload. Ordinary readonly RAMsource proof and staged0or4hostslots. Metadata0app launches/copies;traced math6156fallback or4104fused launches and1027payload-sized native copies,including1026parallelIQ22GUand1026IQ20Down. APIcounts are not bus traffic. Every batch>=100ms and128clockpair max<=.001batch.',
            decision='Characterize Ssource-adapter cost by shape:retain all three steady staged/direct ratios,their geometric mean,and adapter preparation+wall+retirement ratios. A slower Sunit remains slower;no source-cost threshold or global-stage adoption. Completing this control permits a separately committed QorOunit plan with its own criteria,not a model trial. Final production promotion requires repeated net gain over the original complete direct-source entrypoint and all model/phase/context gates. Preserve old global-stage and mapped-source failures and immutable-USMlookahead win unchanged.',
            stop='Stop first source/build/reference/guard/actual-dispatch/trace/validation/ownership/deadline/fault failure. Slower valid timing samples are retained and complete the declared order. A short timing batch is ineligible and stops this cohort;only a new committed plan may change repetitions. No implicit fallback/hash refresh/GPUreset/foreign action.',
            retention='Keep compact individual samples,actual dispatch/source/count/correctness/fault evidence and exact source/recipe/corpus identities. Retire routine traces,ended fresh component,immutable source view and extracted1GBcorpus after closure and exact original recovery. Installed agent SDK cache copies are dependencies,not model generations;235closed failed-worker files/2700788050Bwere retired before this build. Legacy closed native prototypes remain distinct provenance holds with bounded original-recipe recovery next;they are not qualified fallback generations.',
            limits='Actual production Gemm and source wrapper entrypoints at five representative shapes;fixed10route jobs and two route index patterns over512real expert weights are synthetic,not actual model routing. Common baseline-USMallocation/population,source mapping,Runtime/setup and reference work are outside adapter lifecycle;preparation/retirement are staging-specific incremental work. No gather/cache publication/modelPP/TG/TTFT/hardware saturation/decode/agent/262144RESTORE qualification. Current limiting hardware remains unknown.',
            provenance='Main-authored fixture/controller/extraction under repository LICENSE. Original owned14508source and previously checked Intel TLA dense implementation,with existing checked provenance. No XeStrata code/design or external copied proposal.'))
    metadata_gate(row);save(row);print('Prepared actual-weight scheduling plan;commit before build.',flush=True)


def execute(action):
    public=base.ur_validation.committed(RESULT,(KEY,))
    require(public['stage'] in (['prepared'] if action=='build' else ['built','testing']),
            'terminal or wrong-stage actual-reader cohort;preserve its original receipt')
    row=json.loads((ROOT/'record.json').read_text())
    require(row['stage'] in (['prepared'] if action=='build' else ['built','testing']),
            'terminal or wrong-stage actual-reader cohort;preserve its original receipt')
    try:check(row)
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,prelaunch_error=repr(error));save(row);raise
    supervisor=base.owner_api.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',supervisor.commands));cursor=None
    try:
        if action=='build':
            actual=dict(compiled={},links={},objects={});allowed=dict(row['allowed'])
            for recipe in row['recipe']:
                entry,out,err=supervisor.run(recipe['label'],recipe['argv'],row['build_environment'],ROOT,**BUILD)
                require(base.owner_api.closed(entry) and entry['exit_code']==0,'fresh actual-reader build failed:'+str(err))
                if recipe['label']=='host-controls':
                    require('Ran 8 tests' in err.read_text() and '\nOK\n' in err.read_text(),'source/metadata/scheduling controls required')
                    row['host_admission_controls']=dict(methods=8,passed=True,device_executed=False)
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
            require(len(actual['compiled'])>900 and len(actual['links'])>10 and len(actual['objects'])==11,'complete fresh nine-TU census required')
            row.update(stage='built',passed=True,actual_inputs=actual,binary=dict(path=str(ROOT/'benchmark'),sha256=sha(ROOT/'benchmark')))
        else:
            case=row['cases'][len(row['results'])];root=Path(case['root']);require(not root.exists(),'existing case:no retry')
            root.mkdir(mode=0o700)
            for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
            cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).split('-- cursor: ',1)[1].strip()
            row.update(stage='testing',active=True,ownership_closed=False);save(row)
            expected=['before-runtime','runtime-ready']+([] if case['mode']=='metadata' else ['prewarm-ready','before-timing'])+['before-exit']
            gates,records,providers=[],[],{}
            env=base.engine_once.native_environment(row,case['environment'])
            def interaction(channel,_entry):
                while True:
                    item=json.loads(channel.line(45))
                    if item['kind']=='gate':
                        name=item['name'];require(len(gates)<len(expected) and name==expected[len(gates)],'actual reader gate order')
                        actual=dict(v.split('=',1) for v in Path('/proc',str(channel.proc.pid),'environ').read_text().strip('\0').split('\0'))
                        require(all(actual.get(k)==v for k,v in env.items()) and set(actual)-set(env)<= {'PWD','OLDPWD','SHLVL','_'},'actual startup environment differs')
                        require(Path('/proc',str(channel.proc.pid),'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')==case['argv'],'actual source/reader argv differs')
                        providers[name]=base.event.mappings(channel.proc.pid,row)
                        if name!='before-runtime':
                            needed=[base.event.LOADER,env['ZE_ENABLE_ALT_DRIVERS'],env['UR_ADAPTERS_FORCE_LOAD'],'/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so']
                            if case['trace']:needed.append(base.event.VALIDATION)
                            require(all(str(Path(p).resolve()) in providers[name] for p in needed),'required actual reader provider missing')
                        require(sha(row['weights']['path'])==row['weights']['sha256'],'weight fixture changed during native work')
                        gates.append(name);channel.send(b'continue\n')
                    else:
                        records.append(item)
                        if item['kind']=='done':channel.exit(30);break
            entry,out,err=supervisor.run(case['label'],case['argv'],env,P,interaction=interaction,
                **(TRACED_RUN if case['trace'] else base.RUN))
            require(base.owner_api.closed(entry) and entry['exit_code']==0,'actual-reader native failed:'+str(err))
            validation=base.ur_validation.preload_evidence(err.read_text(),entry['pid'])
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,'UR validation output')
            require(not re.search(r'\[(?:error|critical)\]|prefill source adapter failure:|Strata SYCL cleanup:',err.read_text(),re.I),'native validation/cleanup error')
            require(gates==expected and records[-1]==dict(kind='done',passed=True),'incomplete actual reader result')
            proof=records[0];require(proof==dict(kind='source-proof',ordinary_readonly_mapping=True,
                staged=case['arm']=='staged',host_slots=4 if case['arm']=='staged' else 0),'actual source proof differs')
            result=dict(label=case['label'],arm=case['arm'],rows=case['rows'],mode=case['mode'],pair=case['pair'],clean=not case['trace'],passed=True,native_exit_code=0,ownership_closed=True,source_proof=proof,validation=validation,providers_sha256=fingerprint(providers))
            lifecycle=records[-2];require(lifecycle['kind']=='source' and all(type(lifecycle[k]) in [int,float]
                and math.isfinite(lifecycle[k]) and lifecycle[k]>=0 for k in ['preparation_s','retirement_s']),
                'invalid source lifecycle')
            result['source_lifecycle']=lifecycle
            if case['mode']!='metadata':
                require(len(records)==4,'actual reader batch required');batch=records[1]
                batch_gate(batch,case['arm'],case['rows'])
                result.update(batch=batch,lifecycle_plus_batch_s=lifecycle['preparation_s']+batch['wall_s']+lifecycle['retirement_s'])
            else:require(len(records)==3,'metadata contains unexpected work')
            if case['trace']:
                trace=(root/'ur.trace').read_text();kernels=len(re.findall(r'---> (urEnqueueKernelLaunch\w*)',trace))
                copies=sum('SUCCESS (ZE_RESULT_SUCCESS) in zeCommandListAppendMemoryCopy(' in v and 'size=1971200,' in v for v in err.read_text().splitlines())
                expected_math=0 if case['mode']=='metadata' else (4104 if case['rows']>=2334 else 6156)
                require(kernels==expected_math and copies==(0 if case['mode']=='metadata' else 1027),'actual reader math/copy count differs')
                result['validated_counts']=dict(math_launches=kernels,payload_sized_native_copy_calls=copies)
                if case['mode']!='metadata':
                    named=kernel_trace.launch_census(trace)
                    names=named['per_kernel']
                    parallel={key:sum(v['launches'] for name,v in names.items() if key in name)
                              for key in ['parallel_iq_launchILi22ELb1','parallel_iq_launchILi20ELb0']}
                    require(named['launches']==expected_math and all(v==1026 for v in parallel.values()),
                            'actual production parallel-IQlaunch selection differs')
                    result['validated_counts']['parallel_IQ_launches']=parallel
            check(row,False);row['results'].append(result);row.update(passed=True,active=False,ownership_closed=True)
            if len(row['results'])==len(row['cases']):
                analysis={}
                for rows in ROWS:
                    ratios=[];lifecycle_ratios=[]
                    for pair in range(3):
                        values={v['arm']:v for v in row['results'] if v['clean'] and v['pair']==pair and v['rows']==rows}
                        require(set(values)=={'direct','staged'},'complete actual source pair required')
                        ratios.append(values['staged']['batch']['wall_s']/values['direct']['batch']['wall_s'])
                        lifecycle_ratios.append(values['staged']['lifecycle_plus_batch_s']/values['direct']['lifecycle_plus_batch_s'])
                    analysis[str(rows)]=dict(paired_staged_over_direct=ratios,geometric_mean_ratio=math.prod(ratios)**(1/3),
                        paired_adapter_lifecycle_ratios=lifecycle_ratios,repeated_source_adapter_gain=all(x<1 for x in ratios),
                        source_cost_characterized=True,model_adoption_qualified=False)
                row.update(stage='closed',complete=True,analysis=analysis)
    except BaseException as error:row.update(stage='failed',passed=False,complete=False,error=repr(error));raise
    finally:
        row['commands'].extend(supervisor.commands);row['active']=supervisor.active is not None
        row['ownership_closed']=supervisor.active is None and all(base.owner_api.closed(v) for v in row['commands'])
        if cursor:
            try:
                after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                row['fresh_faults']=[v for v in after.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)',v,re.I)]
                row['boot_unchanged']=row['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip();row['driver_after']=base.guards.driver_guard()
                if row['fresh_faults'] or not row['boot_unchanged'] or row['driver_after']!=row['driver']:row.update(stage='failed',passed=False,complete=False)
            except BaseException as error:row.update(stage='failed',passed=False,complete=False,final_health_error=repr(error))
        save(row)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','complete','ownership_closed','analysis']},sort_keys=True),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['build'],['next']],'prepare|build|next')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:raise SystemExit(execute(sys.argv[1]))
