#!/usr/bin/env python3
"""Build/screen streamed QSA16 and its prospective engine; Main serializes work.

Repository LICENSE. Component evidence only, without a model/speed claim.
Use the owned native build/census readers in an isolated module context.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_q6_head_model as common
import measurement_guards as guards
import measurement_owner as ownership
import ur_validation as validation

P, B = common.P, common.B
ORIGIN = Path('/home/yayoi/orca/workspaces/Strata/sycl')
ROOT = B / 'qsa16-stream-components-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'qsa16_stream_components_v1'
TARGET = 'sycl_native16_qsa_test'
COMPONENT_ROOT, COMPONENT_KEY = ROOT, KEY
MODEL_BUILD = False
TARGETS = [TARGET]
SEED = B / 'q6-dense-head-backing-native-v3/record.json'
SEED_KEY = 'q6_dense_head_backing_build_v3'
UNITS = {'tests/sycl/qsa_native16.cpp', 'src/prefill/sycl/qsa_native16.cpp',
         'src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp'}
require, sha, write = guards.require, common.digest, common.write


def context():
    spec = importlib.util.spec_from_file_location('qsa16_stream_build_context', common.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.TARGETS = ROOT, KEY, TARGETS
    return ctx


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('project','dependencies','commands')}
    out['project'] = {k:v for k,v in row['project'].items() if k != 'source_sha256'}
    out['dependencies'] = [{k:v for k,v in d.items() if k != 'source_sha256'}
                           for d in row['dependencies']]
    out['commands'] = [dict(label=c['label'],exit_code=c.get('exit_code'),
        ownership_closed=ownership.closed(c),errors=c.get('errors',[]),error=c.get('error'))
        for c in row['commands']]
    if ('producer_component' in row or 'build_prerequisite' in row) and 'planned_compilation' in row:
        # The live private plan drives census verification. Its full parser
        # arrays need no second permanent copy in the public result.
        planned=out.pop('planned_compilation');out.pop('planned_translation_units')
        out['planned_compilation_summary']=dict(translation_units=len(planned),
            canonical_json_sha256=hashlib.sha256(json.dumps(planned,sort_keys=True,
                separators=(',',':')).encode()).hexdigest())
    return out


def save(row):
    write(ROOT/'record.json',row)
    public=json.loads(RESULT.read_text());public[KEY]=compact(row);write(RESULT,public)


def owned(root):
    root.mkdir(mode=0o700,exist_ok=True)
    scope=None
    def persist():
        write(root/'execution.json',dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    scope=ownership.Owner(root/'owned',persist)
    return scope


def probe_qualification():
    # The existing preload was built and qualified in ORIGIN. Keep that
    # provenance domain separate from this child/component's native source.
    spec=importlib.util.spec_from_file_location('qsa16_origin_validation',validation.__file__)
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    v.P=ORIGIN;v.INDEX=ORIGIN/validation.INDEX.relative_to(P)
    reference=json.loads((B/'decode-iq3-audit-speed-v2/record.json').read_text())
    return v.current_runtime_probe(reference)


def component_reference():
    """Recover the closed producer source, never rerun its retired executable."""
    row=json.loads((COMPONENT_ROOT/'record.json').read_text())
    public=json.loads(common.source.git(P,'show','HEAD:'+str(RESULT.relative_to(P))))
    require(public.get(COMPONENT_KEY)==compact(row) and row['stage']=='closed' and
        row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'],
        'committed closed producer qualification required')
    require(set(row['outcomes'])=={'small','full'} and
        all(v['passed'] and v['complete'] and v['ownership_closed'] and
            v['fresh_faults']==[] and v['boot_unchanged'] and v['host_probe_accepted']
            for v in row['outcomes'].values()) and
        row['outcomes']['full']['physical_cells']==262144,
        'small and full-capacity producer evidence required')
    retirement=row['raw_retirement']
    require(retirement['component_build_and_ELF_removed'] and
        sha(retirement['path'])==retirement['sha256'] and
        not Path(row['binary']['path']).exists() and
        sorted(p.name for p in COMPONENT_ROOT.iterdir())==['record.json','retirement.json'],
        'retire the temporary component generation before a fresh engine')
    original=row['project']
    recovered={n:hashlib.sha256(raw).hexdigest() for n,raw in
        common.source.blobs(P,original['revision'],original['source_sha256'])}
    require(recovered==original['source_sha256'],'original producer source recovery failed')
    common.source.git(P,'merge-base','--is-ancestor',original['revision'],'origin/feature/sycl')
    now=common.source.pin(P,common.PREFIXES)['source_sha256']
    require(set(now)==set(recovered) and {n for n in now if now[n]!=recovered[n]}=={
        'tools/sycl/check_qsa16_stream.py'},
        'only this prospective build controller may differ from the tested producer')
    return dict(key=COMPONENT_KEY,path=str(COMPONENT_ROOT/'record.json'),
        sha256=sha(COMPONENT_ROOT/'record.json'),native_source_commit=original['revision'],
        source_files=len(recovered),native_and_fixture_bytes_unchanged=True,
        source_recovered_from='pushed permanent feature/sycl',component_generation_retired=True,
        component_only=True,model_or_attention_qualification_inherited=False)


def current(row, *, committed=True):
    if committed:
        public=json.loads(common.source.git(P,'show','HEAD:'+str(RESULT.relative_to(P))))
        require(public.get(KEY)==compact(row),'commit this exact component stage first')
    project=common.source.pin(P,common.PREFIXES)
    require(project['source_sha256']==row['project']['source_sha256'],
            'component source/recipe changed; do not refresh frozen pins')
    for pin in (row['project'],*row['dependencies']):
        recovered={n:hashlib.sha256(raw).hexdigest() for n,raw in
            common.source.blobs(Path(pin['root']),pin['revision'],pin['source_sha256'])}
        require(recovered==pin['source_sha256'],'original source recovery failed')
        require(all(sha(Path(pin['root'])/n)==h for n,h in recovered.items()),'source bytes changed')
    require(sha(SEED)==row['recipe_seed']['sha256'] and
        sha(row['external_inputs']['path'])==row['external_inputs']['sha256'] and
        all(sha(p)==h for p,h in row['toolchain'].items()),'compiler/recipe/SDK manifest changed')
    require(all(sha(p)==h for p,h in json.loads(Path(row['external_inputs']['path']).read_text()).items()),
        'frozen external compiler/link input changed')
    for b in row['preserved_binaries']:
        require(Path(b['path']).stat().st_size==b['bytes'] and sha(b['path'])==b['sha256'],
                'existing native fallback changed')
    require(probe_qualification()==(row['native_validation_probe'],row['validation_qualification']),
            'qualified pre-main/runtime probe changed')
    if MODEL_BUILD:
        field = 'build_prerequisite' if 'build_prerequisite' in row else 'producer_component'
        require(component_reference()==row[field], 'recorded full-build prerequisite changed')
    for p,h in (row.get('configured_files',{})|row.get('generated_inputs',{})).items():
        require(sha(p)==h,'configured/generated component input changed')
    if 'binary' in row:
        require(sha(row['binary']['path'])==row['binary']['sha256'],'component executable changed')
        require(sha(row['actual_inputs']['path'])==row['actual_inputs']['sha256'],'compiler census changed')
    for binary in row.get('binaries',{}).values():
        require(Path(binary['path']).stat().st_size==binary['bytes'] and
            sha(binary['path'])==binary['sha256'],'fresh engine or same-build fixture changed')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh component attempt only')
    seed=json.loads(SEED.read_text())
    admitted=json.loads(common.source.git(P,'show','HEAD:'+str(RESULT.relative_to(P))))[SEED_KEY]
    require(admitted['recipe']==seed['recipe'] and admitted['native_source_commit']==seed['native_source_commit']
        and admitted['binaries']==seed['binaries'] and seed['passed'] and seed['ownership_closed'],
        'preserve the committed current native build and recipe')
    project=common.source.pin(P,common.PREFIXES)
    recipe=copy.deepcopy(seed['recipe'])
    old=SEED.parent
    recipe['configure']=[str(ROOT) + v[len(str(old)):] if v.startswith(str(old)) else v
                         for v in recipe['configure']]
    recipe['configure'][recipe['configure'].index('-S')+1]=str(P)
    recipe['environment']={k:v.replace(str(old),str(ROOT)) for k,v in recipe['environment'].items()}
    recipe['environment']['STRATA_SYCL_PREFILL_QSA_STREAM']='1'
    recipe['targets']=recipe['build_targets']=TARGETS
    require(recipe['configure'][recipe['configure'].index('-B')+1]==str(ROOT/'build'),
            'fresh component build root required')
    probe,proof=probe_qualification()
    fallback=Path(B/'cache-map-model-v1/build/strata')
    require(sha(fallback)=='552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a',
            'preserve validated prefill fallback')
    row=dict(stage='prepared',project=project,dependencies=copy.deepcopy(seed['dependencies']),
        recipe=recipe,recipe_seed=dict(path=str(SEED),sha256=sha(SEED)),
        toolchain=seed['toolchain'],external_inputs=seed['external_inputs'],
        native_source_commit=project['revision'],preserved_binaries=[*seed['binaries'].values(),
            dict(path=str(fallback),sha256=sha(fallback),bytes=fallback.stat().st_size,
                 origin_source_commit='479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54')],
        native_validation_probe=probe,validation_qualification=proof,
        plan='Commit every stage. Preserve existing five current native ELFs and552b fallback. Build only five actual QSA16/Runtime test TUs with the unchanged seed compiler/SDK recipe,then small traced/validated fixture,then full262144producer fixture with parameter validation. Normal0,closed owners,pre-main validation,quiet warnings,unchanged sources/runtime/boot and no fresh faults required. No retry,pin refresh,model/performance/adoption claim. Retire temporary component output after successor same-source model qualification;retain no extra engine.',
        retention='Two selected engine generations remain:a347 current decode and552b fastest qualified prefill fallback. This is a temporary third project-binary generation,component-only,needed to qualify the shared Host-allocation change without overwriting either. Review at full producer closure;retire its build/ELF before a fresh engine generation once exact source/recipe and compact evidence are recoverable,or retain only the specifically required fixture for an imminent same-source comparison. Existing legacy provenance holds are separate and unresolved.',
        budgets=dict(build=dict(wall=900,cpu=2400,rss_cap=8<<30,text_cap=16<<20,file_cap=512<<20,total_cap=1<<30),
                     small=dict(wall=120,cpu=600,rss_cap=4<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
                     full=dict(wall=180,cpu=900,rss_cap=4<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20)),
        commands=[],outcomes={},passed=False,complete=False,active=False,ownership_closed=True,
        model_qualified=False,physical262144_model_qualified=False,performance_qualified=False)
    if MODEL_BUILD:
        row['producer_component']=component_reference()
        row['GPU_executed']=False
        row['budgets']=dict(build=dict(wall=2400,cpu=7200,rss_cap=12<<30,
            text_cap=32<<20,file_cap=512<<20,total_cap=4<<30))
        row['plan']='Commit every stage. Build the full engine and four production-linked owner/submission/retirement/phase-policy fixtures with the unchanged current a347 compiler/SDK recipe. Require24compiler-census controls,at least120actual selected translation units,complete generated/compiler/link census and zero uncovered inputs. Native producer/Runtime/fixture bytes are unchanged from the closed small/262144component;no attention/model/runtime result is inherited. After build,commit a separate bounded native/actual-mode1-session plan before model execution. Preserve physical262144capacity and independent prefill/decode,RESTORE,numerical/task and matched PP/TG/TTFT gates;do not select a residency/cache budget from successful compilation.'
        row['retention']='Temporary third engine generation for an imminent streamed-prefill comparison requiring both the a347 current-decode reference and552b fastest-prefill reference. The earlier temporary producer generation is already retired. Preserve both references without overwrite;review at first mode1 model admission and matched phase closure,then retire the rejected/superseded generation after original source recovery. Four same-build fixtures belong to this candidate,not independent generations. Existing56legacy provenance holds remain separate.'
    ROOT.mkdir(mode=0o700)
    for n in ('tmp','cache'): (ROOT/n).mkdir(mode=0o700)
    current(row,committed=False);save(row)
    print('Prepared exact '+('engine' if MODEL_BUILD else 'component')+' build;commit before configure.',flush=True)


def build(stage):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=={'configure':'prepared','compile':'configured'}[stage],'wrong component stage')
    frozen=copy.deepcopy(row);scope=owned(ROOT/stage);outputs={}
    try:
        if stage=='configure':
            commands=[('host-controls',['/usr/bin/python3','-B','-m','unittest','tools.sycl.test_compile_census','-v']),
                ('configure',row['recipe']['configure']),
                ('planned',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','commands',*TARGETS])]
        else: commands=[('compile',['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',*TARGETS,'-j','2']),
                ('dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps'])]
        for label,command in commands:
            current(frozen)
            e,out,err=scope.run(label,command,row['recipe']['environment'],P,**row['budgets']['build'])
            outputs[label]=out
            require(ownership.closed(e) and e['exit_code']==0,'component build command failed: '+label)
            if label=='host-controls':
                text=out.read_text()+err.read_text();m=re.search(r'^Ran (\d+) tests? in ',text,re.M)
                require(m and int(m[1])>0 and '\nOK\n' in text and 'skipped=' not in text,
                        'compiler-census controls incomplete')
                row['host_controls']=dict(tests=int(m[1]),passed=True)
            print(label+' closed0',flush=True)
        ctx=context()
        if stage=='configure':
            planned=ctx.planned_compilation(outputs['planned'].read_text())
            units={v['file'] for v in planned}
            if MODEL_BUILD:
                required={'src/program/generate.cpp','src/prefill/prefill.cpp',
                          'src/prefill/sycl/qsa_native16.cpp','src/sycl/runtime.cpp'}
                require(len(planned)>=120 and {str(P/n) for n in required}<=units,
                        'complete selected engine/producer/Runtime compilation required')
            else:
                require(units=={str(P/n) for n in UNITS} and len(planned)==len(UNITS),
                        'exact declared component translation units required')
            row['planned_compilation']=planned;row['planned_translation_units']=sorted(v['file'] for v in planned)
            row['generated_inputs']={str(p.resolve()):sha(p) for p in (ROOT/'build').rglob('*')
                if p.is_file() and p.suffix in ('.h','.hpp','.c','.cpp','.cxx','.inc')}
            paths=[ROOT/'build'/n for n in ('build.ninja','CMakeCache.txt','compile_commands.json')]
            paths += sorted((ROOT/'build/CMakeFiles').glob('*.ninja'))+[outputs['planned']]
            row['configured_files']={str(p.resolve()):sha(p) for p in paths};row['stage']='configured'
        else:
            row['actual_inputs']=ctx.census(row,outputs['dependencies'],outputs['compile'],
                minimum_translation_units=120 if MODEL_BUILD else len(UNITS))
            if MODEL_BUILD or len(TARGETS)>1:
                row['binaries']={name:dict(path=str(ROOT/'build'/name),
                    sha256=sha(ROOT/'build'/name),bytes=(ROOT/'build'/name).stat().st_size,
                    origin_source_commit=row['native_source_commit']) for name in TARGETS}
            binary=ROOT/'build'/('strata' if MODEL_BUILD else TARGET)
            row['binary']=dict(path=str(binary),sha256=sha(binary),bytes=binary.stat().st_size,
                               origin_source_commit=row['native_source_commit'])
            row.update(stage='built',passed=True)
            if MODEL_BUILD:row['complete']=True
        current(frozen)
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row['commands']+=scope.commands;row['active']=bool(scope.active)
        row['ownership_closed']=not scope.active and all(ownership.closed(c) for c in scope.commands)
        save(row)
    require(row['stage']!='failed' and row['ownership_closed'],row.get('error','component build not closed'))


def screen(mode):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=={'small':'built','full':'small-passed'}[mode],'small validated screen precedes full extent')
    root=ROOT/mode;root.mkdir(mode=0o700)
    for n in ('tmp','cache'): (root/n).mkdir(mode=0o700)
    env=dict(row['recipe']['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
        LD_PRELOAD=row['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if mode=='small' else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    if mode=='full':env.pop('UR_LOG_TRACING',None)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();guards.driver_guard()
    cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor,'kernel journal cursor required')
    scope=owned(root);outcome=dict(passed=False,complete=False)
    try:
        command=[row['binary']['path']]+(['--streamed-full-extent'] if mode=='full' else [])
        e,out,err=scope.run(mode,command,env,P,**row['budgets'][mode])
        text=out.read_text();errors=err.read_text()
        marker=('QSA16 streamed KV producer physical262144 component PASS; attention/model/RESTORE/performance unqualified'
            if mode=='full' else 'QSA16 local component screen PASS; attention/model/physical262144/RESTORE/performance unqualified')
        require(ownership.closed(e) and e['exit_code']==0 and text.splitlines()==[marker],
                'native component screen failed/incomplete')
        proof=validation.preload_evidence(errors,e['pid'])
        require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,
                'runtime parameter validation warning')
        current(row)
        outcome.update(passed=True,complete=True,host_probe_accepted=True,native_exit_code=e['exit_code'],
            validation=proof,physical_cells=262144 if mode=='full' else 24,
            scope='Actual producer/Runtime component only;no attention/model/RESTORE/performance qualification')
    except BaseException as error:outcome.update(error=repr(error),host_probe_accepted=False)
    finally:
        validation.finish(outcome,scope,cursor,boot)
        row['outcomes'][mode]=outcome;row['commands']+=scope.commands
        row.update(stage=('closed' if mode=='full' else 'small-passed') if outcome['passed'] else 'failed',
            passed=outcome['passed'],complete=mode=='full' and outcome['passed'],
            active=bool(scope.active),ownership_closed=outcome['ownership_closed'])
        save(row)
    require(outcome['passed'] and outcome['complete'],outcome.get('error','component final probe failed'))
    print(mode+' producer component PASS;model and speed remain unqualified',flush=True)


if __name__=='__main__':
    actions=('prepare','configure','compile','small','full',
             'engine-prepare','engine-configure','engine-compile')
    require(sys.argv[1:] in ([s] for s in actions),'check_qsa16_stream.py '+ '|'.join(actions))
    action=sys.argv[1]
    if action.startswith('engine-'):
        MODEL_BUILD=True;action=action.removeprefix('engine-')
        # V1 closed after successful configure, before native compilation,
        # to avoid retaining its full per-TU parser arrays in Git.
        ROOT=B/'qsa16-stream-model-native-v2';KEY='qsa16_stream_model_native_v2'
        TARGETS=['strata','sycl_q6_head_owner_test','sycl_submission_test',
                 'sycl_verifier_retirement_test','sycl_verify_window_policy_test']
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if action=='prepare':prepare()
        elif action in ('configure','compile'):build(action)
        else:screen(action)
