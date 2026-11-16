#!/usr/bin/env python3
"""Independent fixed-slot prefill copy/marker comparisons.

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
ROOT=B/'prefill-sycl-copy-groups-v1'
VIEW=ROOT/'source'
KEY='prefill_sycl_copy_groups_v1'
RECORD=ROOT/'record.json'
OWNERSHIP=ROOT/'ownership.json'
PREVIOUS_EVIDENCE='9888d6ad88f3cb9565cd64a327de74885109094f'
SOURCE='bench/sycl/prefill_sycl_copy_math.cpp'
LIFETIME='tests/sycl/prefill_sycl_copy_groups.cpp'
LANE='include/strata/prefill/sycl_copy_groups.hpp'
TEST='tests/core/test_weight_pipeline_plan.py'
PREFIXES=[*base.PREFIXES,SOURCE,LIFETIME,LANE,TEST]
SEED_EVIDENCE='07735df239679929be29c859561ae0501a9a3ab7'
SEED_SOURCE='2187bfa5ddff6d8331187b8cfc22157852c2d461'
TRACED_RUN=dict(base.RUN,text_cap=64<<20,file_cap=96<<20,total_cap=160<<20)
# Includes one immutable1GB model corpus. It is not raw diagnostic text.
BUILD=dict(base.BUILD,file_cap=1280<<20,total_cap=2<<30)
ROWS=(128,1024,8192)
ARMS=('direct','shortcut','layout','bridge','queue','window','ready64','readers64','both64')
UNITS={'S':('direct','shortcut'),'L':('shortcut','layout'),'R':('layout','bridge'),
    'Q':('bridge','queue'),'W':('queue','window'),'A':('window','ready64'),
    'B':('window','readers64'),'AB_after_A':('ready64','both64'),
    'AB_after_B':('readers64','both64'),'netWindow':('direct','window'),
    'netReady':('direct','ready64'),'netReaders':('direct','readers64'),
    'netBoth':('direct','both64')}
BLOCKS=(ARMS,tuple(reversed(ARMS)),
    ('window','bridge','readers64','direct','both64','queue','ready64','shortcut','layout'))
POLICIES={
    'direct':(False,False,1,1,1,1),'shortcut':(False,False,1,1,1,1),
    'layout':(False,False,128,1,1,1),'bridge':(True,False,128,1,1,1),
    'queue':(True,True,128,1,1,1),'window':(True,True,128,128,1,1),
    'ready64':(True,True,128,128,64,1),'readers64':(True,True,128,128,1,64),
    'both64':(True,True,128,128,64,64)}
PAYLOAD=1971200
EXPERTS=512
BACKEND=dict(STRATA_SYCL_PARALLEL_IQ_DEQUANT='1',STRATA_SYCL_PREFILL_NATIVE16='bf16',
    STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
    STRATA_SYCL_PREFILL_GU_SWIGLU='large',STRATA_SYCL_PREFILL_DOWN_SCATTER='large',
    STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original')
SAFETY=dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',
    EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',ZES_ENABLE_SYSMAN='0')
require,sha,fingerprint=base.require,base.sha,base.fingerprint
git,write=base.git,base.write


def compact(row):
    out={k:v for k,v in row.items() if k not in ['project','native_project','foundation','allowed','actual_inputs','commands']}
    out['project']={k:v for k,v in row['project'].items() if k!='source_sha256'}
    out['foundation_sha256']=fingerprint(row['foundation'])
    out['allowed']=dict(files=len(row['allowed']),sha256=fingerprint(row['allowed']))
    if 'native_project' in row:
        out['native_project']=dict(revision=row['native_project']['revision'],sha256=fingerprint(row['native_project']))
    if 'actual_inputs' in row:
        out['actual_inputs']={k:dict(files=len(v),sha256=fingerprint(v)) for k,v in row['actual_inputs'].items()}
    return out


def save(row):
    write(RECORD,row)
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
    for case in row['cases']:case_gate(case,row['weights']['path'])
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
        require(sha(row['lifetime_binary']['path'])==row['lifetime_binary']['sha256'],
                'actual lifetime fixture ELF changed')
        require(all(sha(p)==h for group in row['actual_inputs'].values() for p,h in group.items()),
                'actual compiler/link/object census changed')


def lifetime_done(mode,width):
    if mode in ['grain1','ready64','readers64','both64']:
        return dict(kind='done',passed=True,physical_slots=128,experts=256,
            ready_width=64 if mode in ['ready64','both64'] else 1,
            readers_width=64 if mode in ['readers64','both64'] else 1)
    return dict(kind='done',passed=True,group_width=width)


def lifetime_counts(mode,width):
    if mode=='metadata':return 0,0
    if mode=='ordered':return 3*(3*width+2),3*width+2
    if mode=='throw-copy':return 0,width+(width==1)
    if mode=='throw-reader':return 3*width,width+(width in [1,64])
    if mode in ['grain1','ready64','readers64','both64']:return 768,256
    raise RuntimeError('finite lifetime mode required')


def case_gate(case,weights):
    require(all(case['environment'].get(k)==v for k,v in (BACKEND|SAFETY).items()),
            'frozen pre-exec math backend or qualified safety settings differ')
    if case['family']=='lifetime':
        require(case['mode'] in ['metadata','ordered','throw-copy','throw-reader','grain1','ready64','readers64','both64']
            and case['placement'] in ['compute','separate'] and type(case['width']) is int and case['width'] in [1,2,64]
            and (case['mode'] not in ['grain1','ready64','readers64','both64'] or case['width']==1)
            and case['trace'] is True and case['rows']==0 and case['arm']=='lifetime'
            and case['argv']==[str(ROOT/'lifetime'),case['mode'],case['placement'],str(case['width'])],
            'frozen lifetime protocol differs')
    else:
        require(case['family']=='math' and case['arm'] in ARMS and type(case['rows']) is int
            and case['rows'] in ROWS and case['mode'] in ['metadata','batch']
            and case['argv']==[str(ROOT/'benchmark'),case['mode'],case['arm'],str(case['rows']),weights],
            'frozen production math argv differs')


def case_mappings(pid,row,case):
    require(case['family'] in ['math','lifetime'],'unknown fixture provider scope')
    image=row['lifetime_binary'] if case['family']=='lifetime' else row['binary']
    require(case['argv'][0]==image['path'],'fixture argv does not name its pinned image')
    return base.event.mappings(pid,dict(row,binary=image))


def batch_gate(batch,arm,rows):
    require(arm in ARMS,'finite batch policy required')
    adapter,_,_,_,ready,readers=POLICIES[arm]
    expected=dict(rows=rows,epochs=512,requested_H2D_bytes=EXPERTS*PAYLOAD,copies=512,
        gu_fused=512 if rows>=2334 else 0,down_fused=512 if rows>=2334 else 0,
        split_primary_rows=rows//2,actual_source_experts=EXPERTS,source_footprint_bytes=EXPERTS*PAYLOAD,
        copy_compat_calls=512 if arm=='direct' else 0,
        copy_shortcut_calls=512 if arm in ['shortcut','layout'] else 0,
        copy_adapter_calls=512 if adapter else 0,
        ready_barrier_calls=512//ready if adapter else 0,
        reader_barrier_calls=512//readers if adapter else 0)
    require(arm in ARMS and rows in ROWS and batch.get('kind')=='batch'
        and batch.get('arm')==arm and all(type(batch.get(k)) is int and batch[k]==v for k,v in expected.items())
        and all(batch.get(k) is True for k in ['production_gemm_dispatch','split_tail_checked','full_bitwise_reference_and_guards']),
        'actual production source/dispatch workload differs')
    require(all(type(batch.get(k)) in [int,float] and math.isfinite(batch[k]) for k in ['wall_s','timer_pair_max_s'])
        and batch['wall_s']>=.1 and 0<=batch['timer_pair_max_s']<=.001*batch['wall_s'],
        'ineligible production source batch/clock overhead')


def source_proof_gate(proof,arm):
    require(arm in ARMS,'finite policy required')
    adapter,separate,slots,window,ready,readers=POLICIES[arm]
    expected=dict(kind='source-proof',ordinary_readonly_mapping=True,host_slots=0,policy=arm,
        dependency_bridge=adapter,separate_queue=separate,physical_slots=slots,prefetch_window=window,
        ready_width=ready,readers_width=readers,reader_marker='after-math' if adapter else 'none',
        runtime_check_and_observer_retained=True)
    require(fingerprint(proof)==fingerprint(expected),'actual independent marker/layout policy differs')


def analyze(results):
    analysis={}
    for rows in ROWS:
        clean=[v for v in results if v['family']=='math' and v['clean'] and v['rows']==rows]
        require(len(clean)==3*len(ARMS),'three complete nine-arm blocks required')
        groups=[]
        for pair in range(3):
            values={v['arm']:v for v in clean if v['pair']==pair}
            require(set(values)==set(ARMS),'complete independent-arm block required')
            groups.append(values)
        analysis[str(rows)]={}
        for name,(reference,candidate) in UNITS.items():
            ratios=[v[candidate]['batch']['wall_s']/v[reference]['batch']['wall_s'] for v in groups]
            lifecycle=[v[candidate]['lifecycle_plus_batch_s']/v[reference]['lifecycle_plus_batch_s'] for v in groups]
            analysis[str(rows)][name]=dict(reference=reference,candidate=candidate,
                paired_time_ratios=ratios,geometric_mean_ratio=math.prod(ratios)**(1/3),
                paired_lifecycle_ratios=lifecycle,repeated_component_gain=all(x<1 for x in ratios),
                model_adoption_qualified=False)
        interactions=[v['both64']['batch']['wall_s']*v['window']['batch']['wall_s']/
            (v['ready64']['batch']['wall_s']*v['readers64']['batch']['wall_s']) for v in groups]
        analysis[str(rows)]['interaction']=dict(paired_multiplicative_time_ratios=interactions,
            geometric_mean_ratio=math.prod(interactions)**(1/3),
            scope='2x2marker interaction at fixed128one-expert slots/window;not additive hardware attribution.')
    return analysis


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
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh bounded group cohort only;inspect owner,no retry')
    prior=base.ur_validation.committed(RESULT,('prefill_sycl_copy_lane_v2',))
    require(prior['stage']=='closed' and prior['complete'] and prior['passed'] and not prior['active']
        and prior['ownership_closed'] and prior['model_trial_allowed'] is False
        and prior['retention']['stage']=='complete' and prior['retention']['all_selected_paths_absent']
        and not Path(prior['retention']['root']).exists(),'close/recover/retire previous component before fresh generation')
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
    for name in [SOURCE,LIFETIME,LANE]:allowed[str(P/name)]=sha(P/name)
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
    for label,name in [('fixture',SOURCE),('lifetime',LIFETIME)]:
        compile_one(label,P/name,'src/sycl/runtime.cpp',
            ['-I'+str(P/'include'),'-fno-fast-math','-ffp-contract=off'])
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
    recipes.append(dict(label='link-lifetime',argv=[cc,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-fsycl',*[str(ROOT/(x+'.o')) for x in ['runtime','commands','compat','lifetime']],
        base.event.LOADER,'-Wl,-t','-Wl,--gc-sections','-o',str(ROOT/'lifetime')],
        outputs=[str(ROOT/'lifetime')],link=True))
    cases=[]
    def environment(root,trace):
        old=next(v for v in seed['cases'] if v['source']=='usm' and v['mode']=='serial' and v['trace'] is trace)
        env={k:v.replace(old['root'],str(root)) for k,v in old['environment'].items()}
        env.update(BACKEND);return env
    def lifetime(mode,placement,width):
        label=f'lifetime-{mode}-{placement}-w{width}';root=ROOT/label
        cases.append(dict(label=label,root=str(root),family='lifetime',mode=mode,arm='lifetime',rows=0,
            placement=placement,width=width,trace=True,pair=None,
            argv=[str(ROOT/'lifetime'),mode,placement,str(width)],environment=environment(root,True)))
    for placement in ['compute','separate']:lifetime('metadata',placement,1)
    for width in [1,2,64]:
        for placement in ['compute','separate']:lifetime('ordered',placement,width)
    for mode in ['throw-copy','throw-reader']:
        for width in [1,64]:
            for placement in ['compute','separate']:lifetime(mode,placement,width)
    for mode in ['grain1','ready64','readers64','both64']:
        for placement in ['compute','separate']:lifetime(mode,placement,1)
    def case(label,mode,arm,rows,trace,pair=None):
        root=ROOT/label;cases.append(dict(label=label,root=str(root),family='math',mode=mode,arm=arm,
            rows=rows,trace=trace,pair=pair,argv=[str(ROOT/'benchmark'),mode,arm,str(rows),weights['path']],
            environment=environment(root,trace)))
    for arm in ARMS:case('metadata-'+arm,'metadata',arm,ROWS[0],True)
    for rows in ROWS:
        for arm in ARMS:case(f'validated-{rows}-{arm}','batch',arm,rows,True)
    for rows in ROWS:
        for pair,order in enumerate(BLOCKS):
            for arm in order:case(f'clean-{rows}-block{pair}-{arm}','batch',arm,rows,False,pair)
    row=dict(stage='prepared',passed=False,complete=False,active=False,ownership_closed=True,
        prior_copy_lane=dict(evidence_commit=PREVIOUS_EVIDENCE,retired=True,net_gain_shapes=[],
            original_outcome='All five old net gates failed;preserve individual Q1/Ogains,do not relabel.'),
        project=project,native_source_commit=project['revision'],original_math_runtime_source=base.original.SOURCE,
        foundation=proof,allowed=allowed,recipe=recipes,recipe_tools={str(Path(x).resolve()):sha(x) for x in
            [cc,'/opt/intel/oneapi/compiler/2026.2/bin/compiler/clang-offload-bundler']},
        source_view=dict(original_commit=base.original.SOURCE,files=len(generated),sha256=fingerprint(generated)),
        weights=weights,production_iq_policy=BACKEND,runtime_manifest=seed['runtime_manifest'],
        native_validation_probe=seed['native_validation_probe'],validation_provider=seed['validation_provider'],
        build_environment=dict(seed['build_environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),driver=base.guards.driver_guard(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cases=cases,results=[],commands=[],
        model_trial_allowed=False,performance_qualified=False,
        plan=dict(question='Can independent marker batching remove the measured ownership-barrier penalty while preserving whole production-Gemm execution and actual source bytes?',
            units=UNITS,policies=POLICIES,rows=list(ROWS),blocks=BLOCKS,
            representative_shapes='128lower-row work,1024intermediate work,8192large fused path;bounded component screen,not all shapes/model lengths. Old16/4096evidence remains separate.',
            repetitions='Three complete nine-arm independent-process blocks per shape,forward/reverse/predeclared permutation. Keep all81clean samples;no exclusion/retry.',
            lifetime='24small traced processes:2zero-command metadata,6ordered widths1/2/64,8accepted-copy/reader host exceptions at1/64,and8fixed128-slot2x2marker/full-window/reuse cases.',
            native='Fresh original14508Runtime/compat/IQ/native16/Gemm/TLA math,ten TUs/twelve objects and both fixture ELFs;no archive reuse. Two known-good phase references retained;bounded third component generation retires at closure before another build.',
            isolated_math='Ordinary immutable1GBactual layer2IQ2_S/IQ4_NL corpus,all512experts,512individual copies and unchanged compute-order math. Slot phase starts after two warm copies in all arms. Reference512HostUSMjobs and reset/guards outside timing;terminal drain inside. Sources/allocations outlive both queues.',
            gates='Exact committed source/controller/recipe/SDK/runtime/provider/ELF/corpus and timed copy/marker dispatch counts;normal0/quiet/positive closure/no fresh fault. Actual allocation admission before residency claims. All bitwise output/split/tail/status/source/device guards pass;all128final slots checked. Lifetime counts from exact source. Timed batch>=100ms and128clock-pair max<=.001batch.',
            decision='Each unit/variant requires all three paired batch-time ratios below1;no minimum gain. Retain every plumbing loss. Compare each ready/readers/both/window variant with complete original separately;fastest credible eligible variant remains a possible later model development base even if another variant fails. Report2x2multiplicative interaction separately. No model trial authorized here.',
            stop='Stop on first source/provider/validation/count/numeric/dispatch/closure/deadline/fault/eligibility failure. Valid slow samples complete declared order. No retries,posthoc threshold changes,hash refresh,GPUreset or foreign process/service action.',
            retention='Keep useful individual81clean samples/39unit results and3interaction points,exact source/recipe/binary/corpus identities and fault/limits. At closure recover original inputs from pushed Git,select both phase references and retire this third component/corpus/view/raw. Retire unused prototype exports only after its named consumer ends.',
            limits='Synthetic row mappings/two job patterns,not model routing or input tokens. No PP/TG/TTFT,model/phase/task/agent/physical262144/RESTORE/ordinary-teardown qualification. Host exceptions do not qualify submission/async/drain failures or active observers. Physical CPU/GPU/bus bottleneck remains unknown;counts/requested bytes are not bandwidth or saturation.',
            provenance='Main-authored695565fbd header/lifetime and own e5cd358math/controller;standard SYCL APIsemantics only,owned14508math and previously checked IntelTLA source. No XeStrata implementation/design,external proposal or engine adoption. DG/DH and DI/DJsource corrections applied;execution is new.'))
    metadata_gate(row);save(row);print('Prepared actual-weight scheduling plan;commit before build.',flush=True)


def execute(action):
    public=base.ur_validation.committed(RESULT,(KEY,))
    require(public['stage'] in (['prepared'] if action=='build' else ['built','testing']),
            'terminal or wrong-stage actual-reader cohort;preserve its original receipt')
    row=json.loads(RECORD.read_text())
    require(row['stage'] in (['prepared'] if action=='build' else ['built','testing']),
            'terminal or wrong-stage actual-reader cohort;preserve its original receipt')
    try:check(row)
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,prelaunch_error=repr(error));save(row);raise
    supervisor=base.owner_api.Owner(ROOT/'owned',lambda:write(OWNERSHIP,supervisor.commands));cursor=None
    try:
        if action=='build':
            actual=dict(compiled={},links={},objects={});allowed=dict(row['allowed'])
            for recipe in row['recipe']:
                entry,out,err=supervisor.run(recipe['label'],recipe['argv'],row['build_environment'],ROOT,**BUILD)
                require(base.owner_api.closed(entry) and entry['exit_code']==0,'fresh actual-reader build failed:'+str(err))
                if recipe['label']=='host-controls':
                    require('Ran 13 tests' in err.read_text() and '\nOK\n' in err.read_text(),'source/metadata/scheduling controls required')
                    row['host_admission_controls']=dict(methods=13,passed=True,device_executed=False)
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
            require(len(actual['compiled'])>900 and len(actual['links'])>10 and len(actual['objects'])==12,'complete fresh ten-TU/twelve-object census required')
            row.update(stage='built',passed=True,actual_inputs=actual,
                binary=dict(path=str(ROOT/'benchmark'),sha256=sha(ROOT/'benchmark')),
                lifetime_binary=dict(path=str(ROOT/'lifetime'),sha256=sha(ROOT/'lifetime')))
        else:
            case=row['cases'][len(row['results'])];root=Path(case['root']);require(not root.exists(),'existing case:no retry')
            root.mkdir(mode=0o700)
            for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
            supervisor=base.owner_api.Owner(root/'owned',lambda:write(OWNERSHIP,supervisor.commands))
            cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).split('-- cursor: ',1)[1].strip()
            row.update(stage='testing',active=True,ownership_closed=False);save(row)
            expected=['before-runtime','runtime-ready']+(['prewarm-ready','before-timing']
                if case['family']=='math' and case['mode']=='batch' else [])+['before-exit']
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
                        providers[name]=case_mappings(channel.proc.pid,row,case)
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
            require(not re.search(r'\[(?:error|critical)\]|prefill (?:fixed-slot math fixture|grouped-copy fixture) failure:|Strata SYCL cleanup:',err.read_text(),re.I),'native validation/cleanup error')
            require(gates==expected and records[-1]==(lifetime_done(case['mode'],case['width']) if case['family']=='lifetime' else dict(kind='done',passed=True)),'incomplete actual reader result')
            result=dict(label=case['label'],family=case['family'],arm=case['arm'],rows=case['rows'],
                mode=case['mode'],pair=case['pair'],clean=not case['trace'],passed=True,native_exit_code=0,
                ownership_closed=True,validation=validation,providers_sha256=fingerprint(providers))
            if case['family']=='lifetime':
                require(records==[lifetime_done(case['mode'],case['width'])],'lifetime fixture result differs')
                result.update(placement=case['placement'],width=case['width'],
                    numerical_or_refusal_checks='Actual pinned fixture assertions for this explicit mode;not a model/fault-injection qualification.')
            else:
                proof=records[0];source_proof_gate(proof,case['arm']);result['source_proof']=proof
                lifecycle=records[-2];require(lifecycle['kind']=='source' and all(type(lifecycle[k]) in [int,float]
                    and math.isfinite(lifecycle[k]) and lifecycle[k]>=0 for k in ['preparation_s','retirement_s']),
                    'invalid source lifecycle')
                result['source_lifecycle']=lifecycle
                if case['mode']=='batch':
                    require(len(records)==4,'actual math batch required');batch=records[1]
                    batch_gate(batch,case['arm'],case['rows'])
                    result.update(batch=batch,lifecycle_plus_batch_s=lifecycle['preparation_s']+batch['wall_s']+lifecycle['retirement_s'])
                else:require(len(records)==3,'math metadata contains unexpected work')
            if case['trace']:
                trace=(root/'ur.trace').read_text();kernels=len(re.findall(r'---> (urEnqueueKernelLaunch\w*)',trace))
                size=256 if case['family']=='lifetime' else PAYLOAD
                copies=sum('SUCCESS (ZE_RESULT_SUCCESS) in zeCommandListAppendMemoryCopy(' in v
                    and ('size='+str(size)+',') in v for v in err.read_text().splitlines())
                if case['family']=='lifetime':
                    expected_math,expected_copies=lifetime_counts(case['mode'],case['width'])
                else:
                    expected_math=0 if case['mode']=='metadata' else (4104 if case['rows']>=2334 else 6156)
                    expected_copies=0 if case['mode']=='metadata' else 1026+(POLICIES[case['arm']][2]==1)
                require(kernels==expected_math and copies==expected_copies,'actual protocol math/copy count differs')
                result['validated_counts']=dict(math_launches=kernels,payload_sized_native_copy_calls=copies,
                    payload_bytes=size)
                if case['family']=='math' and case['mode']=='batch':
                    named=kernel_trace.launch_census(trace);names=named['per_kernel']
                    parallel={key:sum(v['launches'] for name,v in names.items() if key in name)
                              for key in ['parallel_iq_launchILi22ELb1','parallel_iq_launchILi20ELb0']}
                    require(named['launches']==expected_math and all(v==1026 for v in parallel.values()),
                            'actual production parallel-IQlaunch selection differs')
                    result['validated_counts']['parallel_IQ_launches']=parallel
            check(row,False);row['results'].append(result);row.update(passed=True,active=False,ownership_closed=True)
            if len(row['results'])==len(row['cases']):
                row.update(stage='closed',complete=True,analysis=analyze(row['results']))
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
