#!/usr/bin/env python3
"""Owned source-adapter component: fresh original Runtime/dense, fixed wait order.

Source -> prepare -> commit recipe -> build -> commit images -> next (one child).
No model trial is authorized by this component's timing result.
Main-authored; repository LICENSE.
"""
import fcntl
import json
import math
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
ROOT = B/'prefill-mapped-source-v2'
VIEW = ROOT/'source'
RESULT = event.RESULT
KEY = 'prefill_mapped_source_v2'
SOURCE = 'bench/sycl/prefill_upload_overlap.cpp'
SAFETY = event.SOURCE
HEADER = event.HEADER
MAPPED = 'include/strata/prefill/mapped_host_source.hpp'
PAGES = 'tests/sycl/prefill_host_pages.hpp'
PREFIXES = ['tools/sycl', SOURCE, SAFETY, HEADER, MAPPED, PAGES, 'AGENTS.md',
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
    evidence = '3ee710401f9cb8323757161dd53a17dce6689094'
    overlap = json.loads(git(P, 'show', evidence+':'+str(RESULT.relative_to(P))))['prefill_upload_overlap_v1']
    safety=['runtime-'+name for name in ['metadata','forward','reverse','permanent','cancel','cancel-throw','refuse']]
    require(overlap['stage']=='closed' and overlap['ownership_closed'] and not overlap['active']
            and len(overlap['results'])==23
            and [v['label'] for v in overlap['results'][:7]]==safety
            and all(v['passed'] is True and v['native_exit_code']==0 and v['ownership_closed'] is True
                    for v in overlap['results'])
            and overlap['analysis']['1024']['repeated_component_wall_gain'] is False
            and overlap['analysis']['4096']['repeated_component_wall_gain'] is True,
            'preserve original Runtime/E2qualification and both shape outcomes')
    return dict(evidence_commit=evidence, native_source_commit=overlap['native_source_commit'],
                original_result_sha256=fingerprint(overlap), execution_input=False,
                new_source_or_performance_qualification=False)


def capability_gate(value, archived):
    require(all(value.get(k) is True and archived.get(k) is True
                for k in ['passed','complete','ownership_closed'])
        and value.get('stage')==archived.get('stage')=='closed'
        and value.get('active') is False and archived.get('active') is False
        and fingerprint(value['capabilities'])==fingerprint(archived['capabilities']),
        'exact closed original capability evidence required')
    cap=value['capabilities']
    require(type(cap['api_version']) is int and cap['api_version']>=65552
        and type(cap['external_sysmem_mapping_version']) is int and cap['external_sysmem_mapping_version']>=65536
        and cap['read_only_mapping_reported_eligible'] is True
        and cap['mapping_or_transfer_attempted'] is False,
        'standard external mapping plus read-only API prerequisite required')
    return cap


def source_capability():
    key='prefill_host_source_capabilities_v2'
    value=ur_validation.committed(RESULT,(key,)); archive=value['archive']
    archived=json.loads(git(P,'show',archive['commit']+':'+archive['path']))[key]
    cap=capability_gate(value,archived)
    asset=value['queried_runtime_provider']
    require(sha(asset['path'])==asset['sha256'] and sha(value['header']['path'])==value['header']['sha256'],
            'queried driver/header changed before source-adapter qualification')
    return dict(evidence=archive,capabilities=cap,driver=asset,header=value['header'],
                mapping_or_transfer_qualified=False)


def property_gate(value, archived):
    require(all(value.get(k) is True and archived.get(k) is True
                for k in ['passed','complete','ownership_closed'])
        and value.get('stage')==archived.get('stage')=='closed'
        and value.get('active') is False and archived.get('active') is False
        and fingerprint(value['properties'])==fingerprint(archived['properties']),
        'exact closed original property observation required')
    prop=value['properties']
    require(prop['kind']=='mapping-properties' and type(prop['bytes']) is int and prop['bytes']==4096
        and prop['read_only'] is True and prop['same_address'] is True
        and prop['allocation_and_unimport_executed'] is True and prop['original_heap_unchanged'] is True
        and prop['explicit_native_context'] is True and prop['production_runtime_context_qualified'] is False
        and prop['source_admission_or_transfer_qualified'] is False
        and type(prop['application_commands']) is int and prop['application_commands']==0,
        'zero-command read-only heap lifecycle prerequisite required')
    for when in ['before','during','after']:
        require(all(type(prop[when][k]) is int for k in ['type','allocation_id','page_bytes']),
                'typed native allocation properties required')
    require(prop['before']==prop['after']==dict(type=0,allocation_id=0,page_bytes=0)
        and prop['during']['type']==1 and prop['during']['allocation_id']>0
        and prop['during']['page_bytes']==4096
        and type(prop['address_range']['result']) is int and prop['address_range']['result']==0
        and prop['address_range']['same_base'] is True
        and type(prop['address_range']['bytes']) is int and prop['address_range']['bytes']==4096,
        'pinned HOST property,positive ID,range and unimport removal required')
    return prop


def property_evidence():
    key='prefill_host_mapping_properties_v1'
    value=ur_validation.committed(RESULT,(key,));archive=value['archive']
    require(archive['commit']=='37017156e3d27a1f9a594a834bec320eeab350c5',
            'pin original property observation,not later metadata')
    archived=json.loads(git(P,'show',archive['commit']+':'+archive['path']))[key]
    prop=property_gate(value,archived);capability=source_capability()
    require(archived['project']['revision']==value['native_source_commit']=='b777f52f8ddfc60b71166cce9a68afb952c71dde'
        and fingerprint(value['queried_provider'])==fingerprint(capability['driver'])
        and archived['environment']['ZE_ENABLE_ALT_DRIVERS']==capability['driver']['path']
        and capability['capabilities']['api_version']==65553,
        'exact source,installed provider and API1.17 prerequisite required')
    return dict(evidence=archive,properties=prop,driver=capability['driver'],
                native_source_commit=value['native_source_commit'],new_source_admission_qualified=False)


def check(row, committed=True):
    if committed:
        require(fingerprint(compact(row))==fingerprint(ur_validation.committed(RESULT,(KEY,))),
                'commit exact typed E2 plan/stage')
    current = original.coherent.build.common.source.pin(P, PREFIXES)
    require(current['source_sha256']==row['project']['source_sha256'], 'frozen E2 source changed')
    native,_ = original.verify_original(expected=row['foundation'])
    require(fingerprint(source_capability())==fingerprint(row['source_capability']), 'frozen source capability changed')
    require(fingerprint(property_evidence())==fingerprint(row['property_evidence']), 'frozen property observation changed')
    require(all(sha(p)==h for p,h in row['allowed'].items()), 'frozen source/SDK/dependency changed')
    require(all(sha(p)==h for p,h in row['recipe_tools'].items()), 'recipe tool changed')
    require(guards.driver_guard()==row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'], 'boot/driver changed')
    if 'binaries' in row:
        require(all(sha(v['path'])==v['sha256'] for v in row['binaries'].values()), 'E2 ELF changed')
        require(all(sha(p)==h for group in row['actual_inputs'].values() for p,h in group.items()),
                'fresh actual compiler/link inputs changed')
        for case in row['cases']:
            args = [case['mode'],case['source']] if case['kind']=='safety' else [case['arm'],str(case['rows']),'512',case['source']]
            require(case['argv']==[row['binaries'][case['kind']]['path'],*args], 'E2 launch identity differs')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'existing E2 root;inspect owner,no retry')
    native, proof = original.verify_original()
    parent = parent_evidence(); capability=source_capability();properties=property_evidence()
    failed=ur_validation.committed(RESULT,('prefill_mapped_source_v1',))
    require(failed['stage']=='failed' and not failed['active']
        and failed['failure_closure']['physical_process_closed'] is True
        and failed['retirement']['stage']=='complete' and failed['retirement']['all_selected_paths_absent'] is True
        and not (B/'prefill-mapped-source-v1').exists(), 'preserve/retire failed strict source contract before V2')
    retired=ur_validation.committed(RESULT,('prefill_upload_overlap_v1',))['retirement']
    require(retired['stage']=='complete' and retired['all_selected_paths_absent']
            and not (B/'prefill-upload-overlap-v1').exists(), 'retire ended component generation before new build')
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
    for name in [HEADER,MAPPED,PAGES]:
        path = VIEW/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((P/name).read_bytes())
        generated[str(path)] = sha(P/name)
    allowed = generated | json.loads(Path(native['external_inputs']['path']).read_text())
    prior = ur_validation.committed(RESULT,('prefill_copy_engine_v6',))
    allowed.update(json.loads(Path(prior['runtime_manifest']['path']).read_text()))
    allowed[event.VALIDATION] = sha(event.VALIDATION)
    for dependency in native['dependencies']:
        if 'sycl-tla' in dependency['root']:
            allowed.update(original.coherent.recover(dependency))
    allowed.update({str(P/name):sha(P/name) for name in [SOURCE,SAFETY,PAGES]})
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
            '-I'+str(VIEW/'tests/sycl'),
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
    case('safety','runtime-metadata-usm',['metadata','usm'],mode='metadata',source='usm')
    case('safety','mapped-import',['import','mapped-neo'],mode='import',source='mapped-neo')
    for source in ['usm','mapped-neo']:
        for mode in ['forward','reverse','permanent','cancel','cancel-throw','refuse']:
            case('safety',f'{source}-{mode}',[mode,source],mode=mode,source=source)
    case('safety','mapped-retire',['retire','mapped-neo'],mode='retire',source='mapped-neo')
    for rows in [1024,4096]:
        for source in ['usm','mapped-neo']:
            case('benchmark',f'validated-{rows}-{source}',['serial',str(rows),'512',source],
                 arm='serial',source=source,rows=rows)
    for rows in [1024,4096]:
        for pair in range(3):
            for source in (['usm','mapped-neo'] if pair%2==0 else ['mapped-neo','usm']):
                case('benchmark',f'clean-{rows}-pair{pair}-{source}',['serial',str(rows),'512',source],
                     trace=False,arm='serial',source=source,rows=rows,pair=pair)
    row = dict(stage='prepared',project=project,foundation=proof,allowed=allowed,
        source_view=dict(original_commit=original.SOURCE,files=len(generated),sha256=fingerprint(generated)),
        parent_component=parent,source_capability=capability,property_evidence=properties,
        failed_standard_source=failed['archive'],native_source_commit=project['revision'],recipe=recipes,
        recipe_tools={v:sha(v) for v in [cc,bundler]},runtime_manifest=prior['runtime_manifest'],
        validation_provider=prior['validation_provider'],native_validation_probe=native['native_validation_probe'],
        build_environment=dict(native['recipe']['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
        driver=guards.driver_guard(),boot=prior['boot'],cases=cases,results=[],commands=[],
        passed=False,complete=False,active=False,ownership_closed=True,
        performance_qualified=False,model_trial_allowed=False,
        plan=dict(question='U3V2:does the explicit pinned-NEO HOST-property source contract satisfy the existing copy/reader protocol with acceptable total source-path cost?',
            opportunity='Original whole-request trace:copy union~10.31s,kernel union~28.66s,overlap~4ms. Copy-only replacement failed. No hardware-saturation claim;this studies scheduling,not copy throughput replacement.',
            units='One coherent source-admission unit:explicit pinned-NEO HOST-property opt-in,positive native allocation ID recorded in private lease and rechecked on every copy,and a shared active-import range registry that rejects proofless aliases through the plain-USM branch. These are inseparable:HOST is also the USMtype,so accepting its enum without the lease/registry would bypass imported ownership. Keep default standard HOST_IMPORTED admission strict. Existing host-USM is the reference under the same coherent wrapper. Preserve page-aligned read-only heap ownership,context/lane/range checks,tombstones,overlap rejection and checked drain-before-unimport. Hold serial readiness order,Runtime/queue ordinals,device destination,dense math/precision,bytes,lookahead,iterations and fixed operands constant. No readiness-order,U1eager,U2aggregation or engine-math change.',
            build='Fresh original14508 include/src view plus the committed lane/mapped-source headers and heap-page fixture. Five fresh TUs,all changed class includers rebuilt;no archive/object reuse. Preserve original isolated AOT-link/host split and full compiler/link census. The previous E2generation is physically retired;this is one temporary component generation beside two unchanged independent phase references.',
            boundary='Explicit source-adapter component,not the production model dispatcher. Resident dense A is fixed;copied immutable next-bank B is byte/guard-checked and is not its math operand. One reader fence per epoch bounds lookahead. The same source blob is repeatedly copied,so cache/residency behavior is not a production50GBsource or physical-bus claim. No PTIcollector or new runtime settings.',
            order='Metadata USM;zero-command actual-Runtime import that first rejects the standard contract,then admits explicit mapped-neo and reports positive HOST/id proof;all six USMcases;all six mapped-neo cases;held-copy retirement/stale/reimport refusal (15safety processes). Refusal cases include ordinary unknown heap and proofless mapped aliases. Four validated serial shape/source batches;then three independent-process pairs per1024/4096shape:USM/Neo,Neo/USM,USM/Neo. No retry,reordering,exclusion or criterion change. One exact dense and one event epoch prewarmed;512epochs plus dual drain timed.',
            gates='Exact committed source/recipe/SDK/tool/ELF/typed plan,original APIcapabilities and closed37017156e property observation;exact installed NEOimage/unchanged safety environment. Actual native argv/providers before work/at exit;positive HOST/id proof in actual-Runtime zero-command case and every mapped benchmark;all15safety cases;full exact dense oracle and poison-to-payload/source/edge guards. Full UR/LevelZero validation in safety/validated batches;clean parameter-validation timing processes. Validate actual513payload-copy/514dense URlaunch counts in traced batches. Quiet validation,normal0,closed owners,unchanged boot/driver/no fault. Every batch>=100ms;128clock-pair max<=0.001batch. A mapping refusal fails this new cohort;no raw unknown admission or further implicit compatibility fallback.',
            source_cost='Record allocation/touch/protection/import/fixture readback-poison preparation outside the timed512epochs;resident matrix fixture setup and Main gate pauses are excluded. Import is a subset of preparation,not an additive cost. Source retirement includes dual drain,unimport and original heap/host-USM release. Report steady wall and preparation+wall+retirement;no model TTFT or whole-process claim.',
            decision='Keep all individual samples and per-pair mapped/USM ratios separately per shape. Source-cost eligibility requires each of three steady-wall ratios<=1.10 AND each preparation+steady+retirement ratio<=1.10. This prospective10percent overhead budget only permits a later independently committed scheduling comparison;it is not a speedup/adoption/target gate. Report geometric means and spread. Old E2M1024failed/M4096passed outcomes remain unchanged. No model trial is allowed by this result.',
            stop='First source/build/numerical/guard/validation/ownership/deadline/fault failure stops this cohort. Fixed timing samples complete even if slower;never change acceptance after results. No reset/unbind/foreign process action.',
            retention='One temporary two-fixture component beside26unchanged phase-reference executables. Preserve source/recipe/census and useful samples;retire ordinary traces/build output at scoped closure. Retire failed source implementation if unsupported and no next consumer;keep independent safety fixes. A useful qualified source candidate may remain only for its named immediate scheduling comparison.',
            limits='No generic standards-conformance claim:official specification says HOST_IMPORTED while exact pinned driver was independently observed returning HOST. Registry covers this owned interface;other components must not independently map its heaps. Same-lane admission/copy/retirement is serialized. No model PP/TG/TTFT,agent/quality,ordinary-RAM pinning,physical PCIe saturation,full262144/RESTORE,handoff or model adoption. Decoder and production selectors unchanged.',
            provenance='Main-authored owner,private source proof,registry,fixtures/controller under repository LICENSE. API concepts from official oneapi-src/level-zero-spec EXT_ExternalMemMap/installed1.32headers;compatibility fact from Main-owned committed37017156e installed-provider observation,not an unproven driver-source checkout. No external implementation copied. Dense is exact owned14508with checked Intel TLA provenance. No XeStrata code/design.'))
    save(row);print('Prepared independent mapped-source plan;commit recipe before build.',flush=True)


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
                    require('Ran 14 tests' in err.read_text() and '\nOK\n' in err.read_text(), 'historical/capability/property controls required')
                    row['host_admission_controls']=dict(methods=14,passed=True,device_executed=False)
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
            native_environment=engine_once.native_environment(row,case['environment'])
            def interaction(channel,_entry):
                while True:
                    item = json.loads(channel.line(45))
                    if item['kind']=='gate':
                        name=item['name'];require(name==expected[len(gates)],'E2 gate order')
                        actual=dict(v.split('=',1) for v in Path('/proc',str(channel.proc.pid),'environ').read_text().strip('\0').split('\0'))
                        require(all(actual.get(k)==v for k,v in native_environment.items())
                            and set(actual)-set(native_environment)<= {'PWD','OLDPWD','SHLVL','_'}, 'actual native source environment differs')
                        require(Path('/proc',str(channel.proc.pid),'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')==case['argv'],
                                'actual source/shape argv differs')
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
            entry,out,err=supervisor.run(case['label'],case['argv'],native_environment,
                P,interaction=interaction,**RUN)
            require(owner_api.closed(entry) and entry['exit_code']==0,'E2 native failed:'+str(err))
            validation=ur_validation.preload_evidence(err.read_text(),entry['pid'])
            require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size,'UR validation output')
            require(not re.search(r'\[(?:error|critical)\]|upload (?:lane|overlap) failure:|Strata SYCL cleanup:',err.read_text(),re.I),
                    'ZE validation/cleanup error')
            require(gates==expected,'incomplete E2 gates')
            result=dict(label=case['label'],source=case['source'],passed=True,native_exit_code=0,ownership_closed=True,validation=validation,gates=gates,
                        initialized_providers=providers['runtime-ready'],after_work_providers=providers['before-exit'])
            if case['kind']=='safety':
                if case['mode']=='import':
                    require(len(records)==2 and records[0]['kind']=='mapping-proof'
                        and records[0]['contract']=='mapped-neo' and records[0]['strict_standard_rejected'] is True
                        and type(records[0]['property_type']) is int and records[0]['property_type']==1
                        and type(records[0]['allocation_id']) is int and records[0]['allocation_id']>0,
                        'actual Runtime pinned import/standard refusal proof required')
                    result['mapping_proof']=records.pop(0)
                require(records==[dict(kind='done',passed=True,case=case['mode'],source=case['source'])], 'incomplete source safety result')
                enqueues=re.findall(r'---> (urEnqueue\w+)',(root/'ur.trace').read_text())
                expected_count={'metadata':0,'import':0,'forward':10,'reverse':12,'permanent':16,
                                'cancel':7,'cancel-throw':7,'refuse':6,'retire':9}[case['mode']]
                require(len(enqueues)==expected_count, 'source safety application count differs')
                result.update(mode=case['mode'],application_enqueues=len(enqueues))
            else:
                require(len(records)==3 and records[0]['kind']=='batch' and records[1]['kind']=='source'
                    and records[1]['source']==case['source'] and records[2]==dict(kind='done',passed=True), 'incomplete source batch')
                batch=records[0]
                require(batch['arm']==case['arm'] and batch['gemm_rows']==case['rows'] and batch['gemm_n']==1280
                    and batch['gemm_k']==2560 and batch['iterations']==512 and batch['requested_H2D_bytes']==512*1971200
                    and batch['lookahead']==1 and batch['oracle_and_guards'] is True,'actual E2 dispatch/workload differs')
                require(batch['wall_s']>=0.1 and 0<=batch['timer_pair_max_s']<=batch['wall_s']*.001,'ineligible timing batch/clock overhead')
                source=records[1]
                require(type(source['import_property_type']) is int and type(source['import_allocation_id']) is int
                    and ((source['import_property_type']==1 and source['import_allocation_id']>0)
                         if case['source']=='mapped-neo' else
                         (source['import_property_type']==0 and source['import_allocation_id']==0)),
                    'actual benchmark source lease proof differs')
                require(all(type(source[k]) in (int,float) and math.isfinite(source[k]) and source[k]>=0
                    for k in ['preparation_s','import_s','retirement_s'])
                    and source['preparation_s']>=source['import_s'], 'invalid/double-counted source lifecycle interval')
                if case['trace']:
                    text=err.read_text();trace=(root/'ur.trace').read_text()
                    payloads=[v for v in text.splitlines() if 'SUCCESS (ZE_RESULT_SUCCESS) in zeCommandListAppendMemoryCopy(' in v and 'size=1971200,' in v]
                    kernels=re.findall(r'---> (urEnqueueKernelLaunch\w*)',trace)
                    require(len(payloads)==513 and len(kernels)==514, 'actual validated copy/dense call count differs')
                    result['validated_counts']=dict(payload_copies=len(payloads),dense_launches=len(kernels))
                result.update(batch=batch,clean=not case['trace'],pair=case.get('pair'),source_lifecycle=source,
                    source_lifecycle_plus_batch_s=source['preparation_s']+batch['wall_s']+source['retirement_s'])
            if case['trace']:
                trace=(root/'ur.trace').read_text();enqueues=re.findall(r'---> (urEnqueue\w+)',trace)
                require(bool(enqueues)==(case.get('mode') not in ['metadata','import']),'actual source application work differs')
                result['application_enqueues']=len(enqueues)
            check(row,False);row['results'].append(result)
            row.update(passed=True,active=False,ownership_closed=True)
            if len(row['results'])==len(row['cases']):
                analysis={}
                for rows in [1024,4096]:
                    samples=[v for v in row['results'] if v.get('clean') and v['batch']['gemm_rows']==rows]
                    ratios=[];lifecycle=[]
                    for pair in range(3):
                        s={v['source']:v for v in samples if v['pair']==pair}
                        require(set(s)=={'mapped-neo','usm'}, 'complete source pair required')
                        ratios.append(s['mapped-neo']['batch']['wall_s']/s['usm']['batch']['wall_s'])
                        lifecycle.append(s['mapped-neo']['source_lifecycle_plus_batch_s']/s['usm']['source_lifecycle_plus_batch_s'])
                    analysis[str(rows)]=dict(paired_mapped_over_usm=ratios,geometric_mean_ratio=math.prod(ratios)**(1/3),
                        paired_lifecycle_plus_batch_ratios=lifecycle,source_cost_eligible=all(v<=1.10 for v in ratios+lifecycle),
                        speedup_or_model_adoption_qualified=False)
                row.update(stage='closed',complete=True,analysis=analysis,
                    performance_qualified=False,source_cost_eligible=any(v['source_cost_eligible'] for v in analysis.values()))
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
