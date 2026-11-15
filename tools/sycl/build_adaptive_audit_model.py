#!/usr/bin/env python3
"""Build the owned native engine with opt-in positive-swap correctness checks.

Main-authored; repository LICENSE. Keep552b and replace no active generation.
Only generate.cpp plus the41-case host-qualified audit header differ from cb7.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
import adaptive_residency_sources as source
import build_decode_profile_model as helpers

selected,host,base = helpers.selected,helpers.host,helpers.base
P,B,RESULT = helpers.P,helpers.B,helpers.RESULT
OLD,ROOT = selected.ROOT,B / 'adaptive-audit-model-v1'
KEY = 'adaptive_residency_native_build_v1'
SELF = Path(__file__).resolve()
require,digest,write,signature = helpers.require,helpers.digest,helpers.write,helpers.signature


def compact(row):
    value = base.compact(row);value.pop('commands',None)
    recipe = value.pop('recipe')
    value['recipe'] = dict(helper='tools/sycl/build_adaptive_audit_model.py:recipe',
        sha256=signature(recipe),targets=recipe['targets'],
        scope='Exact owned cb7 recipe at a private root;only opt-in generator/header audit added.')
    if 'actual_link_inputs' in value:
        links = dict(value['actual_link_inputs'])
        links['owned_product_count'] = len(links.pop('owned_build_or_compiler_products'))
        value['actual_link_inputs'] = links
    return value


def save(row):
    write(ROOT / 'record.json',row)
    data = json.loads(RESULT.read_text());data[KEY] = compact(row);write(RESULT,data)


def basis():
    require(digest(OLD / 'record.json') == host.NATIVE_RECORD_SHA, 'original native record changed')
    native = json.loads((OLD / 'record.json').read_text())
    parent,_,before = selected.projection()
    require(len(before)==567 and selected.pins(before)==native['native_sources'] and
            selected.recipe(parent)==native['recipe'], 'recover exact original567sources/recipe')
    data = base.committed(RESULT);test = data['adaptive_residency_audit_host_v1']
    require(test['passed'] and test['complete'] and test['ownership_closed'] and not test['active'] and
            test['source_commit']==source.HEADER_SOURCE and
            test['outcome']['receipt']=='adaptive_residency_audit cases=41 status=PASS' and
            not test['GPU_executed'], 'exact41-case host header qualification required')
    header = selected.blob(source.HEADER_SOURCE,source.HEADER)
    require(digest(P / source.HEADER)==source.HEADER_SHA and
            hashlib.sha256(header).hexdigest()==source.HEADER_SHA, 'freeze exact tested header')
    values = source.projection(before,header)
    for name,expected in test['source_inputs']['project'].items():
        raw = selected.blob(source.HEADER_SOURCE,name)
        require(hashlib.sha256(raw).hexdigest()==expected, 'recover original host fixture/source')
        if name in values:require(hashlib.sha256(values[name]).hexdigest()==expected,
                                 'native header differs from host-qualified header')
    return native,values


def recipe(native):
    value = copy.deepcopy(native['recipe'])
    value['configure'] = [v.replace(str(OLD),str(ROOT)) for v in value['configure']]
    value['environment'] = {k:v.replace(str(OLD),str(ROOT)) for k,v in value['environment'].items()}
    value['retention'] = 'Keep552b and this single immediate positive-swap candidate. Retire after its correctness/clean comparison closes;no third comparison generation. Audit is off in clean timing.'
    return value


def retirement_basis():
    evidence = base.committed(RESULT)['decode_profile_model_build_v1']['compiled_product_retirement']
    require(evidence['complete'] and evidence['recovered_native_files']==568 and
            evidence['fallback_outputs_preserved']==9 and
            digest(evidence['manifest']['path'])==evidence['manifest']['sha256'], 'verified519e retirement required')
    manifest = json.loads(Path(evidence['manifest']['path']).read_text())
    require(manifest['complete'] and all(not Path(p).exists() for p in manifest['roots']) and
            all(not (OLD/n).exists() for n in helpers.RETIRED_PARTS), 'old compiled generations remain')
    return evidence


def inputs(row):
    require(base.controls(row['source_commit'],controller=SELF)==row['controller_inputs'],
            'native audit controller changed')
    native,values = basis()
    require(selected.pins(values)==row['native_sources'] and recipe(native)==row['recipe'] and
            retirement_basis()==row['retirement_basis'], 'native audit source/recipe/retirement changed')
    require(selected.template.measured.build.monitor_qualification(base.committed(RESULT))==
            row['monitor_qualification'], 'finite build monitor qualification changed')
    require(digest(row['external_inputs']['path'])==row['external_inputs']['sha256'], 'external map changed')
    allowed = json.loads(Path(row['external_inputs']['path']).read_text())
    require(all(digest(p)==h for p,h in allowed.items()), 'external compiler/dependency input changed')
    require(all(allowed.get(p)==h for p,h in base.sdk().items()) and
            all(allowed.get(p)==h for p,h in helpers.dependencies().items()), 'SDK/dependency membership changed')
    pin,image = host.image()
    require(pin==row['toolchain_image'] and all(allowed.get(v['realpath'])==v['sha256']
            for v in image['files'].values() if v['kind']=='file'), 'compiler/package image changed')
    require(all(digest(p)==h for p,h in row['fallback_binaries'].items()), 'validated fallback changed')
    for name,raw in values.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes()==raw, 'private source changed: '+name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    allowed.update(row['generated_expected'])
    return allowed


def retention_gate():
    row = json.loads((ROOT / 'record.json').read_text())
    require(retirement_basis()==row['retirement_basis'] and
            all(digest(p)==h for p,h in row['fallback_binaries'].items()), 'retention/fallback admission changed')


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh native audit build')
    native,values = basis();retired = retirement_basis()
    allowed = base.sdk();external = native['external_inputs']
    require(digest(external['path'])==external['sha256'], 'original external map changed')
    allowed.update(json.loads(Path(external['path']).read_text()))
    allowed.update(native['actual_link_inputs']['frozen_inputs']);allowed.update(helpers.dependencies())
    image_pin,image = host.image()
    allowed.update({v['realpath']:v['sha256'] for v in image['files'].values() if v['kind']=='file'})
    require(all(digest(p)==h for p,h in allowed.items()), 'pin actual external inputs before build')
    revision = base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    plan = recipe(native)
    row = dict(stage='prepared',source_commit=revision,
        controller_inputs=base.controls(revision,controller=SELF),native_sources=selected.pins(values),
        selected_basis=dict(record_sha256=host.NATIVE_RECORD_SHA,source_commit=native['source_commit'],
            original_binaries=native['binaries'],recovered_files=567,
            changed=[source.GENERATOR,source.HEADER],unchanged_native_kernel_CPU_runtime_prefill_files=566),
        host_header_qualification=dict(source_commit=source.HEADER_SOURCE,sha256=source.HEADER_SHA,cases=41,
            scope='Payload/map host state-machine only;new native dispatch/loan attachments unqualified.'),
        recipe=plan,toolchain_image=image_pin,retirement_basis=retired,
        external_inputs=dict(path=str(ROOT / 'external-inputs.json')),fallback_binaries=native['fallback_binaries'],
        monitor_qualification=native['monitor_qualification'],
        generated_expected={p.replace(str(OLD),str(ROOT)):h for p,h in native['generated_expected'].items()},
        commands=[['configure',plan['configure']],
            ['compile',['/usr/bin/cmake','--build',str(ROOT / 'build'),'--target',*plan['targets'],'-j','4']],
            ['dependencies',['/usr/bin/ninja','-C',str(ROOT / 'build'),'-t','deps']]],
        budgets=dict(wall=1800,cpu=7200,rss_cap=12<<30,text_cap=8<<20,file_cap=512<<20,total_cap=2<<30),
        whole_wall_seconds=3600,host_plan=dict(modes=[0,1]),commands_executed=[],passed=False,
        complete=False,active=False,ownership_closed=True,GPU_executed=False,model_executed=False,
        inference_adopted=False,
        plan='Commit exact source/recipe/compiler/dependencies before compilation. Keep552b and9fallback outputs;cb7/519e products already retired. Build6same-build targets. Only the generator plus41-case host-qualified header differ from owned567composition;all566other native files unchanged. Native audits check real replacement identities/payload/completion/publication,actual GPU plan/CPU jobs and post-positive occupied loan/refill. Audit defaults off;clean comparisons use the same new binary with static0 versus positive swaps in separate immutable processes. After compilation/actual input/flag checks,commit and prove exact5fixture reuse or run fresh finite component checks before a prospective native audit request. Preserve checked refill-record failure and runtime safety. No source construction/compilation/fixture result qualifies model speed,physical saturation,task quality,ordinary provider teardown or262144/RESTORE.',
        retention=plan['retention'])
    ROOT.mkdir(mode=0o700)
    for name in ('source','tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    for name,raw in values.items():
        path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    write(ROOT/'external-inputs.json',allowed);row['external_inputs']['sha256']=digest(ROOT/'external-inputs.json')
    try:inputs(row)
    except BaseException as error:row.update(stage='failed-prepare',error=repr(error));save(row);raise
    save(row)
    print('Prepared568-source positive-swap audit candidate;commit before compilation.',flush=True)


def context():
    ctx = helpers.context()
    ctx.ROOT,ctx.KEY,ctx.CONTROLLER = ROOT,KEY,SELF
    ctx.inputs,ctx.retention_gate,ctx.compact,ctx.save = inputs,retention_gate,compact,save
    return ctx


def build():
    context().build();row=json.loads((ROOT/'record.json').read_text())
    try:
        database=json.loads(Path(row['compile_database']['path']).read_text())
        for name in (source.GENERATOR,selected.KERNEL[0],'src/core/verify.cpp',selected.FIXTURE):
            units=[u for u in database if u['file']==str(ROOT/'source'/name)]
            require(len(units)==1,'one actual compile unit: '+name)
            flags=shlex.split(units[0]['command'])
            require(all(v in flags for v in ('-O3','-fsycl','--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13')) and
                    not any(v in flags for v in ('-ffast-math','-Ofast','-ffp-contract=fast')), 'native flags changed')
            if name in (selected.KERNEL[0],selected.FIXTURE):require('-ffp-contract=off' in flags,'Q6 contraction changed')
        actual=json.loads(Path(row['compiled_inputs']['path']).read_text())
        require(actual.get(str(ROOT/'source'/source.HEADER))==source.HEADER_SHA, 'tested audit header not compiled')
        row['native_recipe_checks']=dict(passed=True,full_verifier_and_generator_compiled=True,
            audit_header_in_actual_dependencies=True,unchanged_native_kernel_CPU_runtime_prefill_files=566,
            same_build_Q6_fixture=True,GPU_executed=False)
    except BaseException as error:row.update(stage='failed-native-recipe-check',passed=False,complete=False,error=repr(error));raise
    finally:save(row)


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['build']), 'use prepare|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        (prepare if sys.argv[1]=='prepare' else build)()
