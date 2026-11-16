#!/usr/bin/env python3
"""Independent prefill copy dependency/queue/retirement comparisons.

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
ROOT=B/'prefill-sycl-copy-lane-v1'
VIEW=ROOT/'source'
KEY='prefill_sycl_copy_lane_v1'
SOURCE='bench/sycl/prefill_sycl_copy_math.cpp'
LIFETIME='tests/sycl/prefill_sycl_copy_lane.cpp'
LANE='include/strata/prefill/sycl_copy_lane.hpp'
TEST='tests/core/test_weight_pipeline_plan.py'
PREFIXES=[*base.PREFIXES,SOURCE,LIFETIME,LANE,TEST]
SEED_EVIDENCE='07735df239679929be29c859561ae0501a9a3ab7'
SEED_SOURCE='2187bfa5ddff6d8331187b8cfc22157852c2d461'
TRACED_RUN=dict(base.RUN,text_cap=64<<20,file_cap=96<<20,total_cap=160<<20)
# Includes one immutable1GB model corpus. It is not raw diagnostic text.
BUILD=dict(base.BUILD,file_cap=1280<<20,total_cap=2<<30)
ROWS=(16,128,1024,4096,8192)
ARMS=('direct','shortcut','bridge','queue','lookahead')
UNITS={'R0':('shortcut','bridge'),'Q1':('bridge','queue'),
       'O':('queue','lookahead'),'net':('direct','lookahead')}
BLOCKS=(ARMS,tuple(reversed(ARMS)),('bridge','lookahead','direct','queue','shortcut'))
PAYLOAD=1971200
EXPERTS=512
BACKEND=dict(STRATA_SYCL_PARALLEL_IQ_DEQUANT='1',STRATA_SYCL_PREFILL_NATIVE16='bf16',
    STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
    STRATA_SYCL_PREFILL_GU_SWIGLU='large',STRATA_SYCL_PREFILL_DOWN_SCATTER='large',
    STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original')
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


def case_gate(case,weights):
    require(all(case['environment'].get(k)==v for k,v in BACKEND.items()),
            'frozen pre-exec math backend differs')
    if case['family']=='lifetime':
        require(case['mode'] in ['metadata','ordered','throw-copy','throw-reader']
            and case['placement'] in ['compute','separate'] and case['retirement'] in ['math','iq']
            and case['trace'] is True and case['rows']==0 and case['arm']=='lifetime'
            and case['argv']==[str(ROOT/'lifetime'),case['mode'],case['placement'],case['retirement']],
            'frozen lifetime protocol case differs')
    else:
        require(case['family']=='math' and case['arm'] in ARMS and type(case['rows']) is int
            and case['rows'] in ROWS and case['mode'] in ['metadata','batch']
            and case['argv']==[str(ROOT/'benchmark'),case['mode'],case['arm'],str(case['rows']),weights],
            'frozen production math argv/source policy differs')


def batch_gate(batch,arm,rows):
    expected=dict(rows=rows,epochs=512,requested_H2D_bytes=EXPERTS*PAYLOAD,copies=512,
        gu_fused=512 if rows>=2334 else 0,down_fused=512 if rows>=2334 else 0,
        split_primary_rows=rows//2,actual_source_experts=EXPERTS,source_footprint_bytes=EXPERTS*PAYLOAD)
    require(arm in ARMS and rows in ROWS and batch.get('kind')=='batch'
        and batch.get('arm')==arm and all(type(batch.get(k)) is int and batch[k]==v for k,v in expected.items())
        and all(batch.get(k) is True for k in ['production_gemm_dispatch','split_tail_checked','full_bitwise_reference_and_guards']),
        'actual production source/dispatch workload differs')
    require(all(type(batch.get(k)) in [int,float] and math.isfinite(batch[k]) for k in ['wall_s','timer_pair_max_s'])
        and batch['wall_s']>=.1 and 0<=batch['timer_pair_max_s']<=.001*batch['wall_s'],
        'ineligible production source batch/clock overhead')


def source_proof_gate(proof,arm):
    bridge=arm in ['bridge','queue','lookahead']
    expected=dict(kind='source-proof',ordinary_readonly_mapping=True,host_slots=0,policy=arm,
        dependency_bridge=bridge,separate_queue=arm in ['queue','lookahead'],
        reader_marker=('after-both-IQ' if arm=='lookahead' else 'after-math') if bridge else 'none',
        runtime_check_and_observer_retained=True)
    require(arm in ARMS and fingerprint(proof)==fingerprint(expected),
            'actual dependency/queue/reader policy proof differs')


def analyze(results):
    analysis={}
    for rows in ROWS:
        clean=[v for v in results if v['family']=='math' and v['clean'] and v['rows']==rows]
        require(len(clean)==15,'three complete five-arm blocks required')
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
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'existing actual-reader attempt;inspect owner,no retry')
    wrapper=base.ur_validation.committed(RESULT,('prefill_weight_wrapper_v1',))
    require(wrapper['stage']=='closed' and wrapper['complete'] is True and wrapper['passed'] is True
        and wrapper['retention']['stage']=='complete' and wrapper['retention']['all_selected_paths_absent'] is True
        and wrapper['model_trial_allowed'] is False,
        'close/recover/retire Q0before the next component;do not relabel its per-shape decisions')
    source_control=base.ur_validation.committed(RESULT,('prefill_weight_source_v2',))
    require(source_control['stage']=='closed' and source_control['passed'] is True
        and source_control['complete'] is True and source_control['active'] is False
        and source_control['ownership_closed'] is True and source_control['model_trial_allowed'] is False
        and source_control['retention']['stage']=='complete',
        'close and retire the original ordinary-RAM/staging control first')
    legacy=base.ur_validation.committed(P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json',
        ('retained_builds_review_20261103','source_adapter_prebuild_review_20261116','closed_prototype_recovery_20261116'))
    require(legacy['stage']=='retired' and legacy['retirement']['all_six_roots_absent'] is True
        and all(not (B/v['root']).exists() for v in legacy['sources']),
        'retire the ended six prototypes after exact original input recovery before Q0')
    failed=base.ur_validation.committed(RESULT,('prefill_weight_source_v1',))
    require(failed['stage']=='failed' and failed['passed'] is False and failed['ownership_closed'] is True
        and failed['active'] is False and failed['partial_build_inputs_recovery']['final_executable_created'] is False
        and failed['retention']['stage']=='complete','preserve failed no-GPUfixture build and retire its ended inputs first')
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
        env.update(BACKEND)
        return env
    def lifetime(mode,placement,retirement='math'):
        label=f'lifetime-{mode}-{placement}-{retirement}';root=ROOT/label
        cases.append(dict(label=label,root=str(root),family='lifetime',mode=mode,arm='lifetime',rows=0,
            placement=placement,retirement=retirement,trace=True,pair=None,
            argv=[str(ROOT/'lifetime'),mode,placement,retirement],environment=environment(root,True)))
    for placement in ['compute','separate']:lifetime('metadata',placement)
    for placement in ['compute','separate']:
        for retirement in ['math','iq']:lifetime('ordered',placement,retirement)
    for mode in ['throw-copy','throw-reader']:
        for placement in ['compute','separate']:lifetime(mode,placement)
    def case(label,mode,arm,rows,trace,pair=None):
        root=ROOT/label
        cases.append(dict(label=label,root=str(root),family='math',mode=mode,arm=arm,rows=rows,
            trace=trace,pair=pair,argv=[str(ROOT/'benchmark'),mode,arm,str(rows),weights['path']],
            environment=environment(root,trace)))
    for arm in ARMS:case('metadata-'+arm,'metadata',arm,16,True)
    for rows in ROWS:
        for arm in ARMS:case(f'validated-{rows}-{arm}','batch',arm,rows,True)
    for rows in ROWS:
        for pair,order in enumerate(BLOCKS):
            for arm in order:case(f'clean-{rows}-block{pair}-{arm}','batch',arm,rows,False,pair)
    require(len(cases)==115,'finite ten-safety/five-metadata/25validated/75clean plan')
    for entry in cases:case_gate(entry,weights['path'])
    row=dict(stage='prepared',passed=True,complete=False,active=False,ownership_closed=True,
        prior_wrapper=dict(key='prefill_weight_wrapper_v1',sha256=fingerprint(wrapper),
            original_native=wrapper['native_source_commit'],
            summary='Q0all42closed. OnlyM1024passes its repeated wrapper-gain gate;others do not. Preserve all samples/decisions;no engine adoption.'),
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
            unit='Four separately evaluated contrasts sharing matched five-arm blocks:R0shortcut->bridge changes only required dependency/ownership token checks and barriers on compute;Q1bridge->queue changes only queue placement with matching in-order/profiling properties and full-math retirement;Oqueue->lookahead changes only marker placement after both IQreaders;net direct->lookahead compares the complete original compat source path. Ordinary immutable RAM,one quantized bank,separate dequantized GU/Down buffers,math/precision/row/order stay fixed. A five-arm aggregate never substitutes for each contrast.',
            build='Fresh ten TUs:original14508Runtime/commands/compat/IQ/ops/Gemm/Gemm16/vllmXe2 plus owned full-math/lifetime fixtures. Two fresh derived dense objects,twelve objects total,two executables,no old objects/archives. Both fixture TUs resolve original14508headers before the new owned lane header. One temporary third component generation alongside independent prefill14508/decode9849;retire it at this finite component closure before another engine generation.',
            inputs='All512real layer2IQ2_SGU/IQ4_NLDown experts streamed once per512job batch,1009254400Bimmutable ordinary-mapped corpus;independent HostUSMreference and two same-policy warmups outside timing. Original model stat identity and role hashes pinned;original fullGGUFhash is not rehashed. Build exception:2GiBtotal/1280MiBfile/32MiBtext;4GiBRSS/120swall/240sCPU native cases.',
            order='115independent native processes,no retry/reorder/exclusion:ten small traced lifetime cases(two metadata,four compute/separate x math/IQ ordered cases,two accepted-copy and two accepted-reader exception cases);five traced policy metadata cases;25traced production-math cases(16,128,1024,4096,8192 x five arms);75clean cases at each shape in three blocks:direct/shortcut/bridge/queue/lookahead,reverse,and bridge/lookahead/direct/queue/shortcut. Shared baseline samples are paired within each block for each independent contrast;keep order and every sample. Final checked lane drain is inside timing;reference/reset/validation and setup outside.',
            gates='Exact committed plan/source,compiler/dependency/runtime/ELF/census/corpus/provider/startup argv+environment,quiet validation,positive closure and no fresh fault. Safety URkernel counts metadata0/ordered6/throw-copy0/throw-reader1;256Bnative-copy counts0/4/2/3 respectively. All constructor/range/token/capacity/terminal-drain refusals,bank guards,two readers/separate-weight consumers and accepted-copy/reader unwind checks are native fixture assertions. Math exact policy proof,bitwise split/tail/status/guards/final511payload,compact/wide/fused dispatch,parallelIQ1. Traced math6156fallback/4104fused,1027payload-sized native copies,two IQroles1026each. Every timed batch>=100ms,timer-pair max<=.001batch.',
            decision='Keep all three candidate/reference wall and lifecycle ratios and geometric means for every contrast and shape. Each unit repeated-gain condition is all three wall ratios below1,no arbitrary minimum gain. R0and Q1are also enabling plumbing whose measured losses remain losses;their safety pass permits the prospectively separate next-unit test without claiming a speed gain. Final combined component eligibility requires all three net ratios below1and all numerical/lifetime gates at that shape. No model trial is authorized by this cohort;production needs a new model plan and net complete-entrypoint gain plus phase/task/262144/RESTORE gates.',
            stop='Stop first source/ELF/provider/guard/numeric/dispatch/trace/validation/closure/deadline/fault failure. Valid slower samples remain and complete the declared order. A batch below its declared floor stops as ineligible;new criteria/repetitions require a new committed plan,never hash refresh or retrospective exclusion. No GPUreset or foreign process/service action.',
            retention='Keep individual samples and per-unit outcomes,pinned source/recipe/corpus/dispatch/count/fault limits. Successful raw case output may be retired only after its useful committed result and explicit path/hash audit,with no live/private-reader consumer. At finite closure recover exact final inputs from pushed Git,then retire this component/corpus/source view/raw. Both phase references stay. The six obsolete prototypes and Q0generation are already retired;configured deployment/cache provenance holds stay separate.',
            limits='Synthetic route indices/two job patterns/all512actual weights,not model routing or PP/TG/TTFT. Queue separation and requested bytes do not prove physical independent engines,wire traffic,bandwidth or saturation. Current limiting hardware unknown;inspect traced queue/copy identity and use a separately qualified profiler when attribution remains unclear. Current small readers are integer surrogates;controlled append/marker/async/drain-fault and active-observer correlation remain broader gates. No physical262144/model/decode/agent/ordinary-teardown qualification.',
            provenance='Main-authored fixture/controller from owned Q0and standard SYCL APIs;repository LICENSE. Original owned14508math and checked Intel TLA provenance only. DA/DBsource review corrections are curated;terminal drain fix only syntax-checked before this fresh build. No XeStrata code/design or copied external proposal.'))
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
                    require('Ran 12 tests' in err.read_text() and '\nOK\n' in err.read_text(),'source/metadata/scheduling controls required')
                    row['host_admission_controls']=dict(methods=12,passed=True,device_executed=False)
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
            supervisor=base.owner_api.Owner(root/'owned',lambda:write(ROOT/'ownership.json',supervisor.commands))
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
            require(not re.search(r'\[(?:error|critical)\]|prefill (?:source adapter|SYCL copy lane fixture) failure:|Strata SYCL cleanup:',err.read_text(),re.I),'native validation/cleanup error')
            require(gates==expected and records[-1]==dict(kind='done',passed=True),'incomplete actual reader result')
            result=dict(label=case['label'],family=case['family'],arm=case['arm'],rows=case['rows'],
                mode=case['mode'],pair=case['pair'],clean=not case['trace'],passed=True,native_exit_code=0,
                ownership_closed=True,validation=validation,providers_sha256=fingerprint(providers))
            if case['family']=='lifetime':
                require(records==[dict(kind='done',passed=True)],'lifetime fixture result differs')
                result.update(placement=case['placement'],retirement=case['retirement'],
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
                    expected_math={'metadata':0,'ordered':6,'throw-copy':0,'throw-reader':1}[case['mode']]
                    expected_copies={'metadata':0,'ordered':4,'throw-copy':2,'throw-reader':3}[case['mode']]
                else:
                    expected_math=0 if case['mode']=='metadata' else (4104 if case['rows']>=2334 else 6156)
                    expected_copies=0 if case['mode']=='metadata' else 1027
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
