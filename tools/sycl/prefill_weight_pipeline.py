#!/usr/bin/env python3
"""Coherent actual-weight reader/source comparison; Main owns all native work.

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

P,B,RESULT=base.P,base.B,base.RESULT
ROOT=B/'prefill-weight-pipeline-v1'
VIEW=ROOT/'source'
KEY='prefill_weight_pipeline_v1'
SOURCE='bench/sycl/prefill_weight_pipeline.cpp'
TEST='tests/core/test_weight_pipeline_plan.py'
PREFIXES=[*base.PREFIXES,SOURCE,TEST]
SEED_EVIDENCE='9cc26a79bca65a4e2ca4327c7b79d4ae492843fe'
SEED_SOURCE='0a3295b89619cdc9cb4c6c4b2333043325fc8695'
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


def check(row,committed=True):
    if committed:
        require(fingerprint(compact(row))==fingerprint(base.ur_validation.committed(RESULT,(KEY,))),
                'commit exact typed actual-weight plan/outcome first')
    metadata_gate(row)
    current=base.original.coherent.build.common.source.pin(P,PREFIXES)
    require(current['source_sha256']==row['project']['source_sha256'],'frozen actual-weight source changed')
    base.original.verify_original(expected=row['foundation'])
    require(fingerprint(base.source_capability())==fingerprint(row['source_capability'])
        and fingerprint(base.property_evidence())==fingerprint(row['property_evidence']), 'mapping prerequisites changed')
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
            require(case['argv']==[row['binary']['path'],case['mode'],case['source'],row['weights']['path']],
                    'frozen actual reader/source argv differs')


def extract_weights():
    # The only model bytes needed here are layer2experts0/1. Keep the exact
    # original full-model identity separate from the newly hashed role slices.
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
    sizes=[524800,524800,921600];slices=[];parts=[]
    with shard.open('rb') as f:
        for expert in [0,1]:
            for role,at,size in zip(['gate','up','down'],values[5:],sizes):
                offset=at+expert*size;f.seek(offset);raw=f.read(size)
                require(len(raw)==size,'actual model role is out of bounds')
                parts.append(raw);slices.append(dict(expert=expert,role=role,offset=offset,
                    bytes=size,sha256=hashlib.sha256(raw).hexdigest()))
    after=shard.stat()
    require(all(getattr(after,'st_'+key)==getattr(st,'st_'+key) for key in
        ['dev','ino','size','mtime_ns','ctime_ns']),'model file changed while extracting role slices')
    path=ROOT/'weights.bin';path.write_bytes(b''.join(parts));path.chmod(0o444)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),layer=2,experts=[0,1],
        layout=dict(path=str(layout),sha256=sha(layout)),source_file=str(shard),
        original_source_file_identity=identity,original_full_file_sha256=model[str(shard)]['sha256'],
        full_file_rehashed=False,slices=slices,order='expert0gate/up/down,expert1gate/up/down')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'existing actual-reader attempt;inspect owner,no retry')
    prior=base.ur_validation.committed(RESULT,('prefill_mapped_wait_order_v2',))
    require(prior['stage']=='closed' and prior['analysis']['repeated_component_wall_gain'] is True
        and prior['retention']['stage']=='complete' and prior['retention']['all_selected_paths_absent'] is True,
        'preserve completed source/wait units and retire ended component before next generation')
    native,proof=base.original.verify_original()
    seed=json.loads((base.ROOT/'record.json').read_text())
    historical=json.loads(git(P,'show',SEED_EVIDENCE+':'+str(RESULT.relative_to(P))))[base.KEY]
    require(fingerprint(base.compact(seed))==fingerprint(historical)
        and seed['native_source_commit']==SEED_SOURCE,'exact original source/recipe receipt required')
    base.original.coherent.recover(seed['project'])
    project=base.original.coherent.build.common.source.pin(P,PREFIXES)
    ROOT.mkdir(mode=0o700)
    for name in ['tmp','cache','owned']:(ROOT/name).mkdir(mode=0o700)
    generated={}
    names=[p for p in native['project']['source_sha256'] if p.startswith(('include/','src/'))]
    for name,raw in base.original.coherent.build.common.source.blobs(P,base.original.SOURCE,names):
        path=VIEW/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);generated[str(path)]=sha(path)
    for name in [base.HEADER,base.MAPPED,base.PAGES]:
        path=VIEW/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((P/name).read_bytes());generated[str(path)]=sha(path)
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
                       ('iq','src/kernels/sycl/iq.cpp'),('ops','src/prefill/sycl/native16.cpp'),
                       ('dense','src/prefill/sycl/vllm_xe2/native16.cpp')]:
        compile_one(label,VIEW/name,name,['-ffunction-sections','-fdata-sections'] if label in ['iq','ops'] else [])
    compile_one('fixture',P/SOURCE,'src/sycl/runtime.cpp',['-fno-fast-math','-ffp-contract=off'])
    for label in ['dense-image','dense-host']:
        old=next(v for v in seed['recipe'] if v['label']==label)
        recipes.append({k:([x.replace(str(base.ROOT),str(ROOT)) for x in v] if isinstance(v,list) else v)
                        for k,v in old.items()})
    recipes.append(dict(label='link',argv=[cc,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-fsycl',*[str(ROOT/(x+'.o')) for x in ['runtime','commands','iq','ops','dense-image','dense-host','fixture']],
        base.event.LOADER,'-Wl,-t','-Wl,--gc-sections','-o',str(ROOT/'benchmark')],
        outputs=[str(ROOT/'benchmark')],link=True))
    cases=[]
    def case(label,mode,source,trace,pair=None):
        root=ROOT/label
        old=next(v for v in seed['cases'] if v.get('rows')==1024 and v['source']==source and v['trace'] is trace)
        env={k:v.replace(old['root'],str(root)) for k,v in old['environment'].items()}
        cases.append(dict(label=label,root=str(root),mode=mode,source=source,trace=trace,pair=pair,
            argv=[str(ROOT/'benchmark'),mode,source,weights['path']],environment=env))
    for source in ['usm','mapped-neo']:case('metadata-'+source,'metadata',source,True)
    for source in ['usm','mapped-neo']:case('validated-'+source,'serial',source,True)
    for pair in range(3):
        for source in (['usm','mapped-neo'] if pair%2==0 else ['mapped-neo','usm']):
            case(f'clean-pair{pair}-{source}','serial',source,False,pair)
    row=dict(stage='prepared',passed=True,complete=False,active=False,ownership_closed=True,
        project=project,native_source_commit=project['revision'],original_math_runtime_source=base.original.SOURCE,
        foundation=proof,seed_receipt=dict(path=str(base.ROOT/'record.json'),sha256=sha(base.ROOT/'record.json'),
            source=SEED_SOURCE,evidence=SEED_EVIDENCE,old_ELFs_not_inputs=True),allowed=allowed,
        source_view=dict(original_commit=base.original.SOURCE,files=len(generated),sha256=fingerprint(generated)),
        source_capability=base.source_capability(),property_evidence=base.property_evidence(),weights=weights,
        recipe=recipes,recipe_tools={p:sha(p) for p in seed['recipe_tools']},runtime_manifest=seed['runtime_manifest'],
        validation_provider=seed['validation_provider'],native_validation_probe=seed['native_validation_probe'],
        build_environment={k:v.replace(str(base.ROOT),str(ROOT)) for k,v in seed['build_environment'].items()},
        driver=base.guards.driver_guard(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cases=cases,results=[],commands=[],model_trial_allowed=False,performance_qualified=False,
        plan=dict(unit='Only source ownership differs:hostUSM versus page-aligned read-only imported heap. Exact layer2experts0/1bytes and original14508dequantGU22/Down20,FP16operands,BF16GU/Down,FP16hidden,FP32accumulation held fixed. Serial copy/read order,one private quantized bank,1024rows,128expert jobs,all buffers/queues fixed. Not a scheduling comparison.',
            build='Fresh six TUs with original Runtime/class/production math plus existing owned source headers. No archive/object reuse. IQ/ops host function sections and link garbage collection omit unrelated uncalled MMVQ APIs,identically in both source arms;dense AOTrecipe unchanged. One new temporary component generation beside the two retained independent phase references;ended prior generation physically retired.',
            inputs='Committed extraction recipe reads only actual layer2roles for experts0/1 from the existing pinned model. Hash each selected role and combined3942400Bbundle,with original file identity and prior full hash labelled not rehashed. Bundle is an external model fixture,not project source or synthetic packed data. No asset download or model edit.',
            order='Two traced metadata-only Runtime source admissions,then two traced serial actual-reader batches. Then three independent clean pairs USM/Neo,Neo/USM,USM/Neo. No retry,exclusion or reordering.128ordinary-USM/SYCL-copy reference jobs and two event-source warmup jobs outside timing.128actual source copies and identical six-kernel math jobs plus dual drain timed.',
            gates='Typed committed source/plan,original Git/SDK/compiler/link/object/ELF/weights,actual argv/startup environment/providers,source type1and positive imported IDs. Metadata zero app enqueues;serial full bitwise FP32output against identical ordered production math using independent hostUSM/SYCLcopies,finite outputs,sticky status0,all device/source/bank guards and final expert1payload. Traced1548math launches and259payload-sized native copies(reference128,lane130,finalreadback1);counts are APIwork,not bus traffic. Batch>=100ms;128timer-pair max<=.001batch. Quiet validation,normal0,closed owners,unchanged boot/driver,no fresh fault.',
            decision='All three mapped/USM steady ratios and all three preparation+wall+retirement ratios must<=1.10 to permit a newly committed actual-reader scheduling plan. Import is a subset of preparation. Keep all individual samples and ratios;not a speedup,production/model or target gate. Preserve old source4096failure and old E2USM1024failure.',
            stop='First source/build/reference/guard/validation/ownership/deadline/fault failure stops this cohort. Timing samples complete even if slower. No implicit fallback,hash refresh,GPUreset or foreign process action.',
            retention='Keep minimum useful samples,source/recipe/census,asset-slice identities and failures. Retire routine traces and ended component/raw/extracted weight fixture when its immediate consumer closes. Preserve both phase-reference engines;no fourth generation. Review the old source-only receipt hold during this recovery.',
            limits='Exact production fallback math kernels for M1024;explicit isolated entrypoint omits full dispatcher/gather/model costs. Only two real experts,not a whole layer/model,ordinary Arena direct import,physicalPCIe saturation,modelPP/TG/TTFT,agent,decode/handoff,262144/RESTORE or adoption. No current production prefill/decode selector changes.',
            provenance='Main-authored fixture/controller/extraction under repository LICENSE. Math/kernel/source-owner headers are exact owned14508or0a3295with previously checked Intel TLA provenance;no external implementation or XeStrata code/design.'))
    metadata_gate(row);save(row);print('Prepared actual-weight reader/source plan;commit before build.',flush=True)


def execute(action):
    row=json.loads((ROOT/'record.json').read_text())
    try:check(row)
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,prelaunch_error=repr(error));save(row);raise
    require(row['stage'] in (['prepared'] if action=='build' else ['built','testing']),'fresh ordered actual-reader stage required')
    supervisor=base.owner_api.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',supervisor.commands));cursor=None
    try:
        if action=='build':
            actual=dict(compiled={},links={},objects={});allowed=dict(row['allowed'])
            for recipe in row['recipe']:
                entry,out,err=supervisor.run(recipe['label'],recipe['argv'],row['build_environment'],ROOT,**base.BUILD)
                require(base.owner_api.closed(entry) and entry['exit_code']==0,'fresh actual-reader build failed:'+str(err))
                if recipe['label']=='host-controls':
                    require('Ran 3 tests' in err.read_text() and '\nOK\n' in err.read_text(),'source/metadata host controls required')
                    row['host_admission_controls']=dict(methods=3,passed=True,device_executed=False)
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
            require(len(actual['compiled'])>900 and len(actual['links'])>10 and len(actual['objects'])==8,'complete fresh six-TU census required')
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
            entry,out,err=supervisor.run(case['label'],case['argv'],env,P,interaction=interaction,**base.RUN)
            require(base.owner_api.closed(entry) and entry['exit_code']==0,'actual-reader native failed:'+str(err))
            validation=base.ur_validation.preload_evidence(err.read_text(),entry['pid'])
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,'UR validation output')
            require(not re.search(r'\[(?:error|critical)\]|prefill actual weight pipeline failure:|Strata SYCL cleanup:',err.read_text(),re.I),'native validation/cleanup error')
            require(gates==expected and records[-1]==dict(kind='done',passed=True),'incomplete actual reader result')
            proof=records[0];require(proof['kind']=='mapping-proof' and proof['source']==case['source']
                and type(proof['property_type']) is int and type(proof['allocation_ids']) is list and len(proof['allocation_ids'])==2
                and all(type(i) is int and (i>0 if case['source']=='mapped-neo' else i==0) for i in proof['allocation_ids'])
                and proof['property_type']==(1 if case['source']=='mapped-neo' else 0),'actual source proof differs')
            result=dict(label=case['label'],source=case['source'],mode=case['mode'],pair=case['pair'],clean=not case['trace'],passed=True,native_exit_code=0,ownership_closed=True,mapping_proof=proof,validation=validation,providers_sha256=fingerprint(providers))
            lifecycle=records[-2];require(lifecycle['kind']=='source' and all(type(lifecycle[k]) in [int,float]
                and math.isfinite(lifecycle[k]) and lifecycle[k]>=0 for k in ['preparation_s','import_s','retirement_s'])
                and lifecycle['preparation_s']>=lifecycle['import_s'],'invalid source lifecycle')
            result['source_lifecycle']=lifecycle
            if case['mode']=='serial':
                require(len(records)==4,'actual reader batch required');batch=records[1]
                require(batch['kind']=='batch' and batch['arm']=='serial' and batch['source']==case['source']
                    and batch['rows']==1024 and batch['epochs']==128 and batch['requested_H2D_bytes']==128*1971200
                    and batch['actual_weight_reader'] is True and batch['full_bitwise_reference_and_guards'] is True,'actual production-kernel workload differs')
                require(batch['wall_s']>=.1 and 0<=batch['timer_pair_max_s']<=.001*batch['wall_s'],'ineligible actual reader batch/clock overhead')
                result.update(batch=batch,lifecycle_plus_batch_s=lifecycle['preparation_s']+batch['wall_s']+lifecycle['retirement_s'])
            else:require(len(records)==3,'metadata contains unexpected work')
            if case['trace']:
                trace=(root/'ur.trace').read_text();kernels=len(re.findall(r'---> (urEnqueueKernelLaunch\w*)',trace))
                copies=sum('SUCCESS (ZE_RESULT_SUCCESS) in zeCommandListAppendMemoryCopy(' in v and 'size=1971200,' in v for v in err.read_text().splitlines())
                require(kernels==(0 if case['mode']=='metadata' else 1548) and copies==(0 if case['mode']=='metadata' else 259),'actual reader math/copy count differs')
                result['validated_counts']=dict(math_launches=kernels,payload_sized_native_copy_calls=copies)
            check(row,False);row['results'].append(result);row.update(passed=True,active=False,ownership_closed=True)
            if len(row['results'])==len(row['cases']):
                ratios=[];lifecycle_ratios=[]
                for pair in range(3):
                    values={v['source']:v for v in row['results'] if v['clean'] and v['pair']==pair}
                    require(set(values)=={'usm','mapped-neo'},'complete actual reader source pair required')
                    ratios.append(values['mapped-neo']['batch']['wall_s']/values['usm']['batch']['wall_s'])
                    lifecycle_ratios.append(values['mapped-neo']['lifecycle_plus_batch_s']/values['usm']['lifecycle_plus_batch_s'])
                row.update(stage='closed',complete=True,analysis=dict(paired_mapped_over_usm=ratios,
                    geometric_mean_ratio=math.prod(ratios)**(1/3),paired_lifecycle_ratios=lifecycle_ratios,
                    source_cost_eligible=all(x<=1.10 for x in ratios+lifecycle_ratios),speedup_or_model_adoption_qualified=False))
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
