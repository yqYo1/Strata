#!/usr/bin/env python3
"""Owned E2 component: fresh original Runtime/dense, only readiness-wait order.

Source -> prepare -> commit recipe -> build -> commit images -> next (one child).
No model trial is authorized by this component's timing result.
Main-authored; repository LICENSE.
"""
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import coherent_original_reuse as original
import engine_once
import measurement_guards as guards
import measurement_owner as owner_api
import upload_event_lane as event
import ur_validation

P, B = original.P, original.B
ROOT = B/'prefill-upload-overlap-v1'
VIEW = ROOT/'source'
RESULT = event.RESULT
KEY = 'prefill_upload_overlap_v1'
SOURCE = 'bench/sycl/prefill_upload_overlap.cpp'
SAFETY = event.SOURCE
HEADER = event.HEADER
PREFIXES = ['tools/sycl', SOURCE, SAFETY, HEADER, 'AGENTS.md',
            'docs/ARTIFACT_RETENTION.md', 'docs/BENCHMARKS.md', 'tests/core/test_upload_event_lane_plan.py']
BUILD = dict(wall=900, cpu=1800, rss_cap=8<<30, text_cap=32<<20,
             file_cap=256<<20, total_cap=512<<20)
RUN = dict(wall=120, cpu=240, rss_cap=4<<30, text_cap=32<<20,
           file_cap=32<<20, total_cap=128<<20)
require, sha, fingerprint = guards.require, owner_api.sha, original.fingerprint
git, write = event.git, event.write


def compact(row):
    value = {k:v for k,v in row.items() if k not in
             ('project', 'foundation', 'allowed', 'actual_inputs', 'commands')}
    value['project'] = {k:v for k,v in row['project'].items() if k!='source_sha256'}
    value['foundation_sha256'] = fingerprint(row['foundation'])
    value['allowed'] = dict(files=len(row['allowed']), sha256=fingerprint(row['allowed']))
    if 'actual_inputs' in row:
        value['actual_inputs'] = {k:dict(files=len(v), sha256=fingerprint(v))
                                  for k,v in row['actual_inputs'].items()}
    return value


def save(row):
    write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def parent_evidence():
    evidence = git(P, 'rev-parse', 'e0609f6fa').decode().strip()
    prior = json.loads(git(P, 'show', evidence+':'+str(RESULT.relative_to(P))))[event.KEY]
    full = json.loads((event.ROOT/'record.json').read_text())
    require(fingerprint(event.compact(full)) == fingerprint(prior)
            and prior['stage']=='closed' and prior['ownership_closed']
            and not prior['active'] and len(prior['results'])==7
            and all(v['passed'] and v['native_exit_code']==0 for v in prior['results']),
            'original closed seven-case E1 evidence required')
    recovered = original.coherent.recover(full['project'])
    allowed = recovered | json.loads(Path(full['external_inputs']['path']).read_text())
    allowed.update(json.loads(Path(full['runtime_manifest']['path']).read_text()))
    allowed.update(full['actual_inputs']['objects'])
    require(all(allowed.get(p)==h and (p in recovered or sha(p)==h)
                for group in full['actual_inputs'].values() for p,h in group.items())
            and sha(full['binary']['path'])==full['binary']['sha256'],
            'E1 original source/recipe/census and retained exact ELF required')
    require(recovered[str(P/HEADER)]==sha(P/HEADER), 'E1 native lane header changed')
    return dict(evidence_commit=evidence, native_source_commit=full['native_source_commit'],
                binary=full['binary'], recovered_project_files=len(recovered),
                recipe_sha256=fingerprint(full['recipe']),
                actual_inputs_sha256=fingerprint(full['actual_inputs']),
                execution_input=False, Runtime_context_qualified=False)


def check(row, committed=True):
    if committed:
        require(fingerprint(compact(row))==fingerprint(ur_validation.committed(RESULT,(KEY,))),
                'commit exact typed E2 plan/stage')
    current = original.coherent.build.common.source.pin(P, PREFIXES)
    require(current['source_sha256']==row['project']['source_sha256'], 'frozen E2 source changed')
    native,_ = original.verify_original(expected=row['foundation'])
    require(all(sha(p)==h for p,h in row['allowed'].items()), 'frozen source/SDK/dependency changed')
    require(all(sha(p)==h for p,h in row['recipe_tools'].items()), 'recipe tool changed')
    require(guards.driver_guard()==row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'], 'boot/driver changed')
    if 'binaries' in row:
        require(all(sha(v['path'])==v['sha256'] for v in row['binaries'].values()), 'E2 ELF changed')
        require(all(sha(p)==h for group in row['actual_inputs'].values() for p,h in group.items()),
                'fresh actual compiler/link inputs changed')
        for case in row['cases']:
            args = [case['mode']] if case['kind']=='safety' else [case['arm'],str(case['rows']),'512']
            require(case['argv']==[row['binaries'][case['kind']]['path'],*args], 'E2 launch identity differs')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'existing E2 root;inspect owner,no retry')
    native, proof = original.verify_original()
    parent = parent_evidence()
    project = original.coherent.build.common.source.pin(P, PREFIXES)
    # A fresh, immutable original view keeps ALL class definitions coherent.
    # No retained archive/object is linked. Current pending engine math is excluded.
    sources = {p:h for p,h in native['project']['source_sha256'].items()
               if p.startswith(('include/','src/'))}
    generated = {}
    ROOT.mkdir(mode=0o700)
    for name in ['tmp','cache','owned']: (ROOT/name).mkdir(mode=0o700)
    for name,raw in original.coherent.build.common.source.blobs(P, original.SOURCE, sources):
        path = VIEW/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
        require(sha(path)==sources[name], 'recovered original view differs')
        generated[str(path)] = sources[name]
    for name in [HEADER]:
        path = VIEW/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((P/name).read_bytes())
        generated[str(path)] = sha(P/name)
    allowed = generated | json.loads(Path(native['external_inputs']['path']).read_text())
    prior = ur_validation.committed(RESULT,('prefill_copy_engine_v6',))
    allowed.update(json.loads(Path(prior['runtime_manifest']['path']).read_text()))
    allowed[event.VALIDATION] = sha(event.VALIDATION)
    for dependency in native['dependencies']:
        if 'sycl-tla' in dependency['root']:
            allowed.update(original.coherent.recover(dependency))
    allowed.update({str(P/name):sha(P/name) for name in [SOURCE,SAFETY]})
    require(all(sha(p)==h for p,h in allowed.items()), 'original live dependency/SDK differs before build')
    cc = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    commands = json.loads((original.ROOT/'build/compile_commands.json').read_text())
    recipes = [dict(label='host-controls',argv=['/usr/bin/python3','-B','-m','unittest','discover',
        '-s',str(P/'tests/core'),'-p','test_upload_event_lane_plan.py','-v'],outputs=[])]
    def compile_one(label, source, template, extra=()):
        entry = next(v for v in commands if v['file']==str(P/template))
        flags = shlex.split(entry['command'])[1:]
        del flags[flags.index('-o'):flags.index('-o')+2]
        del flags[flags.index('-c'):flags.index('-c')+2]
        flags = [v.replace(str(P),str(VIEW)) for v in flags]
        dep, obj = ROOT/(label+'.d'), ROOT/(label+'.o')
        recipes.append(dict(label=label, argv=[cc,*flags,*extra,'-I'+str(VIEW/'src'),
            '-MD','-MF',str(dep),'-c',str(source),'-o',str(obj)],dep=str(dep),outputs=[str(obj)]))
    compile_one('runtime',VIEW/'src/sycl/runtime.cpp','src/sycl/runtime.cpp')
    compile_one('commands',VIEW/'src/sycl/command_profile.cpp','src/sycl/command_profile.cpp')
    compile_one('safety',P/SAFETY,'src/sycl/runtime.cpp', ['-DSTRATA_PREFILL_EVENT_RUNTIME_CONTEXT=1'])
    compile_one('dense',VIEW/'src/prefill/sycl/vllm_xe2/native16.cpp','src/prefill/sycl/vllm_xe2/native16.cpp')
    compile_one('benchmark',P/SOURCE,'src/sycl/runtime.cpp',['-fno-fast-math','-ffp-contract=off'])
    # Same isolated AOT-image + extracted host split as original CMake recipe.
    bundler = '/opt/intel/oneapi/compiler/2026.2/bin/compiler/clang-offload-bundler'
    recipes += [dict(label='dense-image',argv=[cc,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-O3','-fsycl','-fsycl-targets=spir64_gen','-fsycl-link','-fno-sycl-instrument-device-code',
        '-Xspirv-translator','-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate',
        '-Xsycl-target-backend=spir64_gen','-device bmg-g21-a0 -internal_options -cl-intel-256-GRF-per-thread',
        str(ROOT/'dense.o'),'-o',str(ROOT/'dense-image.o')],outputs=[str(ROOT/'dense-image.o')]),
        dict(label='dense-host',argv=[bundler,'-unbundle','-type=o','-targets=host-x86_64-unknown-linux-gnu',
             '-input='+str(ROOT/'dense.o'),'-output='+str(ROOT/'dense-host.o')],outputs=[str(ROOT/'dense-host.o')])]
    for kind, objects in [('safety',['runtime','safety']),
                          ('benchmark',['runtime','commands','dense-image','dense-host','benchmark'])]:
        recipes.append(dict(label='link-'+kind,argv=[cc,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
            '-fsycl',*[str(ROOT/(v+'.o')) for v in objects],event.LOADER,'-Wl,-t','-o',str(ROOT/kind)],
            outputs=[str(ROOT/kind)],link=True))
    cases = []
    def case(kind,label,args,trace=True,**fields):
        root = ROOT/label
        env = dict(prior['observer_environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'),
            ZEL_LOADER_LOGGING_LEVEL='trace' if trace else 'warn',
            ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1' if trace else '0',
            ZE_ENABLE_VALIDATION_LAYER='1' if trace else '0',
            ZEL_ENABLE_LOADER_LOGGING='1' if trace else '0')
        if trace: env['UR_LOG_TRACING']='level:info;flush:info;output:file,'+str(root/'ur.trace')
        cases.append(dict(kind=kind,label=label,root=str(root),argv=[str(ROOT/kind),*args],
                          environment=env,trace=trace,**fields))
    for mode in ['metadata','forward','reverse','permanent','cancel','cancel-throw','refuse']:
        case('safety','runtime-'+mode,[mode],mode=mode)
    for rows in [1024,4096]:
        for arm in ['serial','overlap']:
            case('benchmark',f'validated-{rows}-{arm}',[arm,str(rows),'512'],arm=arm,rows=rows)
    for rows in [1024,4096]:
        for pair in range(3):
            for arm in (['serial','overlap'] if pair%2==0 else ['overlap','serial']):
                case('benchmark',f'clean-{rows}-pair{pair}-{arm}',[arm,str(rows),'512'],
                     trace=False,arm=arm,rows=rows,pair=pair)
    row = dict(stage='prepared',project=project,foundation=proof,allowed=allowed,
        source_view=dict(original_commit=original.SOURCE,files=len(generated),sha256=fingerprint(generated)),
        parent_component=parent,native_source_commit=project['revision'],recipe=recipes,
        recipe_tools={v:sha(v) for v in [cc,bundler]},runtime_manifest=prior['runtime_manifest'],
        validation_provider=prior['validation_provider'],native_validation_probe=native['native_validation_probe'],
        build_environment=dict(native['recipe']['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
        driver=guards.driver_guard(),boot=prior['boot'],cases=cases,results=[],commands=[],
        passed=False,complete=False,active=False,ownership_closed=True,
        performance_qualified=False,model_trial_allowed=False,
        plan=dict(question='E2: does one-expert ordinal1 H2D overlap with existing original14508 dense compute lower synchronized component wall time?',
            opportunity='Original whole-request trace:copy union~10.31s,kernel union~28.66s,overlap~4ms. Copy-only replacement failed. No hardware-saturation claim;this studies scheduling,not copy throughput replacement.',
            units='U1 eager position is separately positive;U2copy aggregation and copy-only replacement retain their original failed gates. E1event ownership is a safety prerequisite. E1b tests ONLY production Runtime context. E2 changes ONLY readiness-wait position across independent dense compute;kernel,precision,bytes,allocation,copy primitive,lookahead,batch and device fixed. No U1/U2 or pending engine math is bundled.',
            build='Fresh original14508 include/src view plus unchanged E1lane header and two current fixtures. Five fresh TUs,all class definitions from the same view. Original isolated AOT-link/host-slice recipe. No archive/object reuse. Full actual compiler/link census against committed original source and SDK/dependency allowlist.',
            boundary='Explicitly isolated adapter scheduling discriminator,NOT whole production dispatcher. Current dense A uses resident fixed operands;copied next-bank B is independently byte-checked,not used by dense A. One reader fence per epoch prevents more than ONE lookahead. HostUSM only;ordinary RAM source view and actual production consumers remain separate.',
            order='Seven Runtime-context safety processes first,then serial/overlap validated batch at1024and4096rows,then three independent-process alternating pairs per shape:S/O,O/S,S/O. No retries/reordering/exclusions. One prewarmed dense and one event epoch excluded;512epochs and final dual drain timed;input setup,poison reset,full oracle/guards outside.',
            gates='Exact source/recipe/SDK/tool/ELF/typed plan;actual initialized providers before application work and at exit;all seven Runtime safety cases;both arm/shapes exact oracle,poison-to-payload and guards;quiet validation,normal0,closed owners,unchanged boot/driver/no fault. Each timed batch>=100ms;128empty clock-pair max<=0.001of batch. Missing/mismatched/ineligible data fails without hash refresh.',
            decision='Keep all individual process samples. Per shape independently,three overlap/serial time ratios must each<1 and geometric mean<1 for a reproducible component speed screen. No blanket minimum gain. Actual device-overlap attribution requires a separately committed bounded unitrace plan;until then no overlap-mechanism or model-trial qualification,even if wall time improves.',
            stop='First source/build/numerical/guard/validation/ownership/deadline/fault failure stops this cohort. Fixed timing samples complete even if slower;never change acceptance after results. No reset/unbind/foreign process action.',
            retention='One temporary two-fixture component besides unchanged two independent phase references. E1old executable is provenance-only during fresh build;retire at E2native qualification/closure after original input recovery. Retire E2at its scoped decision;no raw trace archive.',
            limits='No model PP/TG/TTFT,agent/quality,ordinary-RAM pinning,physical PCIe traffic/saturation,full262144/RESTORE,handoff or model adoption. Decoder and production selectors unchanged.',
            provenance='Main-authored fixtures/controller/event protocol,repository LICENSE. Dense is exact owned14508 implementation with its checked Intel TLA provenance. No XeStrata implementation reference.'))
    save(row);print('Prepared smallest-unit Runtime/E2 plan;commit recipe before build.',flush=True)


def execute(action):
    row = json.loads((ROOT/'record.json').read_text()); check(row)
    require(row['stage'] in (['prepared'] if action=='build' else ['built','testing']), 'fresh ordered E2 stage required')
    supervisor = owner_api.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',supervisor.commands))
    cursor = None
    try:
        if action=='build':
            actual = dict(compiled={},links={},objects={})
            allowed = dict(row['allowed'])
            for recipe in row['recipe']:
                entry,out,err = supervisor.run(recipe['label'],recipe['argv'],row['build_environment'],ROOT,**BUILD)
                require(owner_api.closed(entry) and entry['exit_code']==0,'fresh E2 build failed:'+str(err))
                if recipe['label']=='host-controls':
                    require('Ran 6 tests' in err.read_text() and '\nOK\n' in err.read_text(), 'all E1/E2 negative controls required')
                    row['host_admission_controls']=dict(methods=6,passed=True,device_executed=False)
                if 'dep' in recipe:
                    for name in shlex.split(Path(recipe['dep']).read_text().replace('\\\n',' ').split(':',1)[1]):
                        p = str(Path(name).resolve()); h = sha(p)
                        require(allowed.get(p)==h,'unfrozen actual compiler input:'+p);actual['compiled'][p]=h
                if recipe.get('link'):
                    for line in out.read_text().splitlines():
                        p = Path(line.strip())
                        if p.is_absolute() and p.is_file():
                            p = str(p.resolve());h = sha(p)
                            require(allowed.get(p)==h,'unfrozen actual link input:'+p);actual['links'][p]=h
                for p in recipe['outputs']:
                    allowed[str(Path(p).resolve())]=sha(p)
                    if p.endswith('.o'):actual['objects'][p]=sha(p)
            require(len(actual['compiled'])>600 and len(actual['links'])>10,'complete actual compiler/link census required')
            row.update(stage='built',passed=True,actual_inputs=actual,
                binaries={k:dict(path=str(ROOT/k),sha256=sha(ROOT/k)) for k in ['safety','benchmark']})
        else:
            index = len(row['results']);require(index<len(row['cases']),'all E2 cases already closed')
            case = row['cases'][index];root = Path(case['root']);require(not root.exists(),'inspect existing case,no retry')
            root.mkdir(mode=0o700)
            for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
            journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
            cursor = journal.split('-- cursor: ',1)[1].strip()
            row.update(stage='testing',active=True,ownership_closed=False,journal_cursor=cursor);save(row)
            if case['kind']=='safety':
                expected=['before-runtime','runtime-ready']+(['buffers-ready'] if case['mode']!='metadata' else [])+['before-exit']
            else:expected=['before-runtime','runtime-ready','prewarm-ready','before-timing','before-exit']
            gates,records,providers = [],[],{}
            def interaction(channel,_entry):
                while True:
                    item = json.loads(channel.line(45))
                    if item['kind']=='gate':
                        name=item['name'];require(name==expected[len(gates)],'E2 gate order')
                        mapping_row=dict(row,binary=row['binaries'][case['kind']])
                        providers[name]=event.mappings(channel.proc.pid,mapping_row)
                        if name!='before-runtime':
                            needed=[event.LOADER,case['environment']['ZE_ENABLE_ALT_DRIVERS'],case['environment']['UR_ADAPTERS_FORCE_LOAD'],
                                    '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so']
                            if case['trace']:needed.append(event.VALIDATION)
                            require(all(str(Path(p).resolve()) in providers[name] for p in needed),'required E2 initialized provider missing')
                        gates.append(name);channel.send(b'continue\n')
                    else:
                        records.append(item)
                        if item['kind']=='done':channel.exit(30);break
            entry,out,err=supervisor.run(case['label'],case['argv'],engine_once.native_environment(row,case['environment']),
                P,interaction=interaction,**RUN)
            require(owner_api.closed(entry) and entry['exit_code']==0,'E2 native failed:'+str(err))
            validation=ur_validation.preload_evidence(err.read_text(),entry['pid'])
            require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size,'UR validation output')
            require(not re.search(r'\[(?:error|critical)\]|upload (?:lane|overlap) failure:|Strata SYCL cleanup:',err.read_text(),re.I),
                    'ZE validation/cleanup error')
            require(gates==expected,'incomplete E2 gates')
            result=dict(label=case['label'],passed=True,native_exit_code=0,ownership_closed=True,validation=validation,gates=gates,
                        initialized_providers=providers['runtime-ready'],after_work_providers=providers['before-exit'])
            if case['kind']=='safety':
                require(records==[dict(kind='done',passed=True,case=case['mode'])],'incomplete Runtime safety result')
            else:
                require(len(records)==2 and records[0]['kind']=='batch' and records[1]==dict(kind='done',passed=True),'incomplete E2 batch')
                batch=records[0]
                require(batch['arm']==case['arm'] and batch['gemm_rows']==case['rows'] and batch['gemm_n']==1280
                    and batch['gemm_k']==2560 and batch['iterations']==512 and batch['requested_H2D_bytes']==512*1971200
                    and batch['lookahead']==1 and batch['oracle_and_guards'] is True,'actual E2 dispatch/workload differs')
                require(batch['wall_s']>=0.1 and 0<=batch['timer_pair_max_s']<=batch['wall_s']*.001,'ineligible timing batch/clock overhead')
                result.update(batch=batch,clean=not case['trace'],pair=case.get('pair'))
            if case['trace']:
                trace=(root/'ur.trace').read_text();enqueues=re.findall(r'---> (urEnqueue\w+)',trace)
                require(bool(enqueues)==(case.get('mode')!='metadata'),'actual E2 application work differs')
                result['application_enqueues']=len(enqueues)
            check(row,False);row['results'].append(result)
            row.update(passed=True,active=False,ownership_closed=True)
            if len(row['results'])==len(row['cases']):
                analysis={}
                for rows in [1024,4096]:
                    samples=[v for v in row['results'] if v.get('clean') and v['batch']['gemm_rows']==rows]
                    ratios=[]
                    for pair in range(3):
                        s={v['batch']['arm']:v['batch']['wall_s'] for v in samples if v['pair']==pair}
                        ratios.append(s['overlap']/s['serial'])
                    analysis[str(rows)]=dict(paired_time_ratios=ratios,geometric_mean=__import__('math').prod(ratios)**(1/3),
                        repeated_component_wall_gain=all(v<1 for v in ratios),actual_device_overlap_qualified=False)
                row.update(stage='closed',complete=True,analysis=analysis,
                    performance_qualified=any(v['repeated_component_wall_gain'] for v in analysis.values()))
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,error=repr(error));raise
    finally:
        row['commands'].extend(supervisor.commands);row['active']=supervisor.active is not None
        row['ownership_closed']=supervisor.active is None and all(owner_api.closed(c) for c in row['commands'])
        if cursor:
            try:
                after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                row['fresh_faults']=[v for v in after.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)',v,re.I)]
                row['boot_unchanged']=row['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                row['driver_after']=guards.driver_guard()
                if row['fresh_faults'] or not row['boot_unchanged']:row.update(stage='failed',passed=False,complete=False)
            except BaseException as error:row.update(stage='failed',passed=False,complete=False,final_health_error=repr(error))
        save(row)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','complete','ownership_closed','performance_qualified','model_trial_allowed']},sort_keys=True),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['build'],['next']], 'prepare|build|next')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:raise SystemExit(execute(sys.argv[1]))
