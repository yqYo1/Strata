#!/usr/bin/env python3
"""Build and screen the actual streamed-QSA16 producer; Main serializes commands.

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
SEED = B / 'q6-dense-head-backing-native-v3/record.json'
SEED_KEY = 'q6_dense_head_backing_build_v3'
UNITS = {'tests/sycl/qsa_native16.cpp', 'src/prefill/sycl/qsa_native16.cpp',
         'src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp'}
require, sha, write = guards.require, common.digest, common.write


def context():
    spec = importlib.util.spec_from_file_location('qsa16_stream_build_context', common.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.TARGETS = ROOT, KEY, [TARGET]
    return ctx


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('project','dependencies','commands')}
    out['project'] = {k:v for k,v in row['project'].items() if k != 'source_sha256'}
    out['dependencies'] = [{k:v for k,v in d.items() if k != 'source_sha256'}
                           for d in row['dependencies']]
    out['commands'] = [dict(label=c['label'],exit_code=c.get('exit_code'),
        ownership_closed=ownership.closed(c),errors=c.get('errors',[]),error=c.get('error'))
        for c in row['commands']]
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
    for p,h in (row.get('configured_files',{})|row.get('generated_inputs',{})).items():
        require(sha(p)==h,'configured/generated component input changed')
    if 'binary' in row:
        require(sha(row['binary']['path'])==row['binary']['sha256'],'component executable changed')
        require(sha(row['actual_inputs']['path'])==row['actual_inputs']['sha256'],'compiler census changed')


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
    recipe['targets']=recipe['build_targets']=[TARGET]
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
    ROOT.mkdir(mode=0o700)
    for n in ('tmp','cache'): (ROOT/n).mkdir(mode=0o700)
    current(row,committed=False);save(row)
    print('Prepared exact component build;commit before configure.',flush=True)


def build(stage):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=={'configure':'prepared','compile':'configured'}[stage],'wrong component stage')
    frozen=copy.deepcopy(row);scope=owned(ROOT/stage);outputs={}
    try:
        if stage=='configure':
            commands=[('host-controls',['/usr/bin/python3','-B','-m','unittest','tools.sycl.test_compile_census','-v']),
                ('configure',row['recipe']['configure']),
                ('planned',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','commands',TARGET])]
        else: commands=[('compile',['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',TARGET,'-j','2']),
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
            require({v['file'] for v in planned}=={str(P/n) for n in UNITS} and len(planned)==5,
                    'exact five-TU component build required')
            row['planned_compilation']=planned;row['planned_translation_units']=sorted(v['file'] for v in planned)
            row['generated_inputs']={str(p.resolve()):sha(p) for p in (ROOT/'build').rglob('*')
                if p.is_file() and p.suffix in ('.h','.hpp','.c','.cpp','.cxx','.inc')}
            paths=[ROOT/'build'/n for n in ('build.ninja','CMakeCache.txt','compile_commands.json')]
            paths += sorted((ROOT/'build/CMakeFiles').glob('*.ninja'))+[outputs['planned']]
            row['configured_files']={str(p.resolve()):sha(p) for p in paths};row['stage']='configured'
        else:
            row['actual_inputs']=ctx.census(row,outputs['dependencies'],outputs['compile'],minimum_translation_units=5)
            binary=ROOT/'build'/TARGET
            row['binary']=dict(path=str(binary),sha256=sha(binary),bytes=binary.stat().st_size,
                               origin_source_commit=row['native_source_commit'])
            row.update(stage='built',passed=True)
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
    require(sys.argv[1:] in ([s] for s in ('prepare','configure','compile','small','full')),
            'check_qsa16_stream.py prepare|configure|compile|small|full')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1] in ('configure','compile'):build(sys.argv[1])
        else:screen(sys.argv[1])
