#!/usr/bin/env python3
"""Compose the measured Q6 half entry onto the exact 5c model source.

Main-authored, repository LICENSE. The new binary supplies both immutable
decode arms. Keep the validated prefill fallback; retire the closed diagnostic
generation before building its replacement. Compilation is not model adoption.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import build_cpu_service_model as template
import device_fp64_probe as dependencies
import q6_centering_sources as lineage
import verify_window_model as phase

P, B, RESULT, base = template.P, template.B, template.RESULT, template.base
ROOT, OLD = B / 'q6-native-half-model-v1', B / 'decode-service-model-v1'
KEY, OLD_KEY = 'q6_native_half_model_build_v1', 'decode_service_model_build_v1'
SELF = Path(__file__).resolve()
KERNEL_SOURCE = '4947b7746dafb0efde7da9bb2ead49dc89790d89'
POLICY_SOURCE = '517a285094bffa8cd49168282f4ef813d9cbfbdd'
COMPONENT_EVIDENCE = '3e6b836a438b419dcd43e2eab68ba3cc3f7ea798'
HOST_EVIDENCE = '83d859db04248701aacd3211a6ea4bdfd5d63c2b'
OLD_RECORD_SHA = '548a9d2c0f0a2894e2084bf261b2153167a22c0c702c91ba4448504e81227675'
OBSERVER = '96b9cc511b81e5fe4c18e53030019a2ad34c6b4d'
WAIT_LABEL = '1c42b4700f046b6fbcd0cc20338b42c62d0340d2'
KERNEL = phase.Q6_FILES
POLICY = ('include/strata/core/verify.hpp', 'src/core/verify.cpp',
          phase.POLICY, phase.FIXTURE, 'src/program/generate.cpp')
FIXTURE, CMAKE = 'bench/sycl/q6_columns.cpp', 'cmake/sycl_backend.cmake'
CHANGED = (*KERNEL, *POLICY, FIXTURE, CMAKE)
TARGETS = ['strata', 'cpu_service_pool_test', 'iq_single_exact_test',
           'sycl_submission_test', 'sycl_verifier_retirement_test', 'sycl_q6_columns_bench']
require, digest, write = base.require, base.digest, base.write
blob, merge, pins = lineage.blob, lineage.merge, lineage.pins


def historical(revision, key):
    return json.loads(blob(revision, str(RESULT.relative_to(P))))[key]


def baseline():
    """Recover every materialized input, without executing historical gates."""
    _, _, values = lineage.baseline()
    for n in (*KERNEL, lineage.CENTER_HEADER):
        values[n] = blob(lineage.COMPONENT_SOURCE, n)
    for n in lineage.SELECTOR_FILES:
        values[n] = merge(values[n], n, lineage.SELECTOR_SOURCE + '^', lineage.SELECTOR_SOURCE)
    center = json.loads((B / 'q6-centering-model-v2/record.json').read_text())
    require(len(values) == 564 and pins(values) == center['native_sources'],
            'recover all original centering-model bytes')
    existing = ('include/strata/sycl/cpu_wait_profile.hpp', 'src/core/verify.cpp',
                'src/program/generate.cpp')
    added = ('include/strata/sycl/decode_service_profile.hpp',
             'tests/core/decode_service_profile_test.cpp')
    for n in existing:
        values[n] = merge(values[n], n, OBSERVER + '^', OBSERVER)
    for n in added:
        require(n not in values, 'unexpected existing observer source')
        values[n] = blob(OBSERVER, n)
    for n in (added[0], 'src/core/verify.cpp'):
        values[n] = merge(values[n], n, WAIT_LABEL + '^', WAIT_LABEL)
    require(digest(OLD / 'record.json') == OLD_RECORD_SHA, 'original full 5c record changed')
    old = json.loads((OLD / 'record.json').read_text())
    recorded = historical('96886718660d2761eb09e0bf192dc59d71788d31', OLD_KEY)
    require(len(values) == 566 and pins(values) == old['native_sources'] and
            base.compact(old)['native_sources'] == recorded['native_sources'] and
            old['source_commit'] == '70169d3881ab588d45f7df7baa132b4293f00970' and
            old['binaries']['strata']['sha256'] ==
                '5c5e25d04e8ea8a8abcd9d26ed8f70a32d78b8aa93a89a49e7232ec8f47f6d76',
            'Git reconstruction differs from actual 566-file 5c source')
    return old, values


def qualification():
    component = historical(COMPONENT_EVIDENCE, 'q6_native_half_component_v3')
    host = historical(HOST_EVIDENCE, 'q6_native_half_policy_host_v1')
    require(component['source_commit'] == KERNEL_SOURCE and component['passed'] and
            component['complete'] and component['ownership_closed'] and not component['active'] and
            len(component['processes']) == 3 and component['decision']['qualified_shapes'] == 27 and
            component['decision']['individual_timing_samples'] == 1050 and
            sum(s['all_three_faster'] for s in component['decision']['shapes']) == 27,
            'closed repeated full-entry component qualification required')
    require(host['source_commit'] == POLICY_SOURCE and host['passed'] and host['complete'] and
            host['ownership_closed'] and not host['active'] and not host['GPU_executed'] and
            len(host['results']) == 2 and all(r['exit_code'] == 0 for r in host['results']),
            'closed exact C++ policy/cache qualification required')
    return dict(component_commit=COMPONENT_EVIDENCE, host_commit=HOST_EVIDENCE,
        component_sha256=lineage.signature(component), host_sha256=lineage.signature(host),
        component_winners=27, incumbent_intersection=26, model_qualified=False)


def projection():
    old, before = baseline()
    values = dict(before)
    for n in KERNEL:
        values[n] = blob(KERNEL_SOURCE, n)
    for n in POLICY:
        values[n] = merge(before[n], n, POLICY_SOURCE + '^', POLICY_SOURCE)
    require(values[phase.POLICY] == blob(POLICY_SOURCE, phase.POLICY) and
            values[phase.FIXTURE] == blob(POLICY_SOURCE, phase.FIXTURE),
            'actual selected policy/host fixture differs from the qualified bytes')
    values[FIXTURE] = blob(KERNEL_SOURCE, FIXTURE)
    marker = b'add_executable(sycl_submission_test EXCLUDE_FROM_ALL tests/sycl/compat.cpp)'
    addition = (b'add_executable(sycl_q6_columns_bench EXCLUDE_FROM_ALL bench/sycl/q6_columns.cpp)\n'
                b'target_link_libraries(sycl_q6_columns_bench PRIVATE strata_kernels)\n'
                b'target_compile_options(sycl_q6_columns_bench PRIVATE -fno-fast-math -ffp-contract=off)\n')
    require(before[CMAKE].count(marker) == 1 and FIXTURE not in before, 'one new same-build fixture')
    values[CMAKE] = before[CMAKE].replace(marker, addition + marker)
    require(len(values) == 567 and set(values) - set(before) == {FIXTURE} and
            {n for n in values if values[n] != before.get(n)} == set(CHANGED),
            'unplanned source composition')
    require(all(values[n] == raw for n, raw in before.items()
                if n.startswith(('src/prefill/', 'src/kernels/cpu/', 'src/sycl/'))),
            'preserve all prefill,CPU and runtime sources')
    return old, before, values


def compact(row):
    value = base.compact(row)
    value.pop('commands', None)
    plan = value.pop('recipe')
    value['recipe'] = dict(helper='tools/sycl/build_q6_native_half_model.py:recipe',
        sha256=lineage.signature(plan), targets=plan['targets'],
        scope='Exact 5c recipe with private paths,new same-build Q6 fixture,and default-off half/audit settings.')
    if 'host_result' in value:
        value['host_result']={n:v for n,v in value['host_result'].items() if n!='commands'}
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def recipe(old):
    value = copy.deepcopy(old['recipe'])
    value['configure'] = [s.replace(str(OLD), str(ROOT)) for s in value['configure']]
    value['environment'] = {n:s.replace(str(OLD), str(ROOT)) for n,s in value['environment'].items()}
    value['environment'].update(STRATA_SYCL_DECODE_Q6_NATIVE_HALF='0', STRATA_SYCL_Q6_ROUTE_AUDIT='0')
    value['targets'] = list(TARGETS)
    value['retention'] = ('Replace the closed 5c diagnostic products with one same-binary half0/1 comparison. '
        'Keep552b and all eight fallback fixtures. Retire this candidate after comparison unless its '
        'remaining model,prefill/TTFT,agent/task and physical262144/RESTORE gates qualify it.')
    return value


def inputs(row):
    require(base.controls(row['source_commit'], controller=SELF) == row['controller_inputs'],
            'commit and freeze all transitive build controls')
    old, before, values = projection()
    require(pins(values) == row['native_sources'] and recipe(old) == row['recipe'] and
            qualification() == row['qualification'], 'native source,recipe or prerequisite changed')
    require(template.measured.build.monitor_qualification(base.committed(RESULT)) ==
            row['monitor_qualification'], 'qualified build monitor changed')
    require(digest(row['external_inputs']['path']) == row['external_inputs']['sha256'],
            'prebuild external input map changed')
    # Reuse the fallback's original SDK/dependency Git checks, not merely the
    # headers which happened to appear in the last diagnostic's .d files.
    fallback_path = template.selected.model.ROOT / 'record.json'
    require(digest(fallback_path) == row['fallback_record_sha256'], 'fallback source/recipe record changed')
    fallback = json.loads(fallback_path.read_text())
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    allowed = template.selected.model.inputs(fallback, reuse_revision=revision,
        controller_changes=template.selected.NATIVE_CONTROLS)
    allowed = {p:h for p,h in allowed.items() if not Path(p).is_relative_to(template.selected.model.ROOT)}
    allowed.update(json.loads(Path(row['external_inputs']['path']).read_text()))
    require(all(digest(p) == h for p,h in allowed.items()), 'installed SDK/link inputs changed')
    require(digest(row['toolchain_image']['path']) == row['toolchain_image']['sha256'],
            'prebuild compiler image changed')
    image = json.loads(Path(row['toolchain_image']['path']).read_text())
    require(dependencies.toolchain() == image, 'compiler/package membership or bytes changed')
    allowed.update({v['realpath']:v['sha256'] for v in image['files'].values() if v['kind'] == 'file'})
    for n,raw in values.items():
        path = ROOT / 'source' / n
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'private source changed: ' + n)
        allowed[str(path.resolve())] = row['native_sources'][n]
    allowed.update(row['generated_expected'])
    return allowed


def retention_gate():
    row = json.loads((ROOT / 'record.json').read_text())
    retired = row.get('old_retirement', {})
    require(retired.get('completed') is True and not (OLD / 'source').exists() and
            not (OLD / 'build').exists() and digest(retired['manifest']) == retired['sha256'],
            'recover/retire the closed 5c products before building a replacement')
    require(all(digest(p) == h for p,h in row['fallback_binaries'].items()), 'validated fallback changed')


def context():
    spec = importlib.util.spec_from_file_location('strata_q6_half_build', template.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.CONTROLLER = ROOT, KEY, SELF
    ctx.source = SimpleNamespace(NATIVE=POLICY_SOURCE, PROGRAM=lineage.observer.PROGRAM,
        POOL=lineage.observer.POOL, HEADER=lineage.observer.HEADER, FIXTURES=lineage.observer.FIXTURES)
    ctx.inputs, ctx.retention_gate, ctx.compact, ctx.save = inputs, retention_gate, compact, save
    return ctx


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh preparation')
    old, before, values = projection()
    require(all(digest(v['path']) == v['sha256'] for v in old['binaries'].values()),
            'inspect original 5c executables before source-only retirement')
    require(all(digest(OLD / 'source' / n) == h for n,h in old['native_sources'].items()),
            'materialized old source differs from the recovered Git construction')
    asset = old['compiled_inputs']; require(digest(asset['path']) == asset['sha256'], 'old actual inputs changed')
    actual = json.loads(Path(asset['path']).read_text()); require(len(actual) == asset['files'], 'old input count')
    require(all(digest(p) == h for p,h in actual.items()), 'old actual compiler inputs changed')
    external = {p:h for p,h in actual.items() if not Path(p).is_relative_to(OLD)}
    external.update(old['actual_link_inputs']['frozen_inputs'])
    require(all(digest(p) == h for p,h in external.items()), 'external original linker input changed')
    require(digest(old['compile_database']['path']) == old['compile_database']['sha256'], 'old actual compiler database changed')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    controls = base.controls(revision, controller=SELF)
    image = dependencies.toolchain(); plan = recipe(old)
    fallback = json.loads((template.selected.model.ROOT / 'record.json').read_text())
    protected = {v['path']:v['sha256'] for v in fallback['binaries'].values()}
    require(len(protected) == 9 and all(digest(p) == h for p,h in protected.items()), 'protect all552b outputs')
    row = dict(stage='prepared', source_commit=revision, controller_inputs=controls,
        native_sources=pins(values), recipe=plan, qualification=qualification(),
        selected_basis=dict(record_sha256=OLD_RECORD_SHA, original_engine=old['binaries']['strata'],
            recovered_source_files=len(before), recovered_source_aggregate=lineage.signature(pins(before)),
            changed=list(CHANGED), same_binary_comparison='Half0/1 in separate immutable processes;centering0,columns1 in both.'),
        fallback_binaries=protected,
        fallback_record_sha256=digest(template.selected.model.ROOT / 'record.json'),
        monitor_qualification=template.measured.build.monitor_qualification(base.committed(RESULT)),
        external_inputs=dict(path=str(ROOT / 'external-inputs.json')),
        toolchain_image=dict(path=str(ROOT / 'toolchain.json'), files=len(image['files']), packages=image['packages']),
        generated_expected={p.replace(str(OLD),str(ROOT)):h for p,h in old['generated_expected'].items()},
        commands=[['configure',plan['configure']],['compile',['/usr/bin/cmake','--build',str(ROOT/'build'),
            '--target',*TARGETS,'-j','4']],['dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps']]],
        budgets=dict(wall=1800,cpu=7200,rss_cap=12<<30,text_cap=8<<20,file_cap=512<<20,total_cap=2<<30),
        host_plan=dict(modes=[0,1],scope='Same-build real CPU pool parity with unchanged IQ3 kernels;no GPU or speed qualification.'),
        whole_wall_seconds=3600, commands_executed=[], passed=False,complete=False,active=False,
        ownership_closed=True,GPU_executed=False,model_executed=False,inference_adopted=False,
        qualification_plan=dict(host='Same-build IQ3/pool parity;unchanged qualified26-cell policy bytes.',
            device='One small strict traced submission and32-case Q6 native-half smoke,then phase-cache checks;normal0,quiet validation,no new faults,closed owners.',
            model='Commit the exact phase/T/shape capture/replay reader and bounded thinking-on model-entry plan before execution. Require Q6 head and actual completed decode route;prompt reference stays independent.',
            speed='After model entry,commit counterbalanced three independent pairs with thinking-on >=1K generated tokens for observed MTP-strong/weak classes,bench.sh long and agent-session;prefill/TTFT judged independently.',
            stopping='Stop first source,compiler,dependency,link,budget,owner,numerical,validation or fresh-fault failure;no retry or refreshed hashes.',
            limits='Build/host/component evidence does not establish model gain,physical saturation,ordinary teardown,task quality or full262144/tails/RESTORE.'))
    ROOT.mkdir(mode=0o700)
    for n in ['source','tmp','cache']:(ROOT/n).mkdir(mode=0o700)
    write(ROOT/'toolchain.json',image);write(ROOT/'external-inputs.json',external)
    for n in ['toolchain_image','external_inputs']:row[n]['sha256']=digest(row[n]['path'])
    for n,raw in values.items():
        p=ROOT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    write(ROOT/'record.json',row)
    try:inputs(row)
    except BaseException as e:
        row.update(stage='failed-prepare',error=repr(e));save(row);raise
    save(row);print('Prepared exact567-file Q6 half model. Commit,retire old5c products,commit,then build.',flush=True)


def retire():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY] == compact(row) and row['stage']=='prepared' and
            'old_retirement' not in row and not (ROOT/'retire-attempt.json').exists(),
            'one committed source-only retirement;inspect an existing attempt')
    write(ROOT/'retire-attempt.json',dict(operation='retire',source_commit=row['source_commit']))
    try:
        inputs(row)
        old,before,_=projection()
        data=base.committed(RESULT)
        closures={}
        for key in ['decode_iq3_audit_speed_v2','decode_service_overhead_v3','q6_native_half_component_v3']:
            item=data[key]
            require(item['ownership_closed'] and not item['active'], 'old native consumer is still open: '+key)
            closures[key]=lineage.signature(item)
        require(all(base.supervision.closed(e) for e in old['commands_executed']), 'old compiler ownership')
        require(all(digest(v['path'])==v['sha256'] for v in old['binaries'].values()), 'old executables changed')
        # These are the recorded controls/recipe for the exact final 5c ELF,
        # including the bench/ import roots; a nearby tree is not recovery.
        for n,h in old['controller_inputs'].items():
            require(hashlib.sha256(blob(old['source_commit'],n)).hexdigest()==h,
                    'original build control is not recoverable: '+n)
        original_inputs=json.loads(Path(old['compiled_inputs']['path']).read_text())
        require(digest(old['compiled_inputs']['path'])==old['compiled_inputs']['sha256'] and
                len(original_inputs)==old['compiled_inputs']['files'], 'old actual input inventory changed')
        allowed=inputs(row);original_pins=pins(before)
        for p,h in original_inputs.items():
            name=Path(p)
            if name.is_relative_to(OLD/'source'):
                require(original_pins[str(name.relative_to(OLD/'source'))]==h,'unrecoverable actual source')
            elif name.is_relative_to(OLD/'build'):
                require(old['generated_expected'].get(p)==h,'unrecoverable generated build input')
            else:require(allowed.get(str(name.resolve()))==h,'unrecoverable external build input: '+p)
            require(digest(p)==h,'old actual input changed before retirement: '+p)
        require(digest(old['compile_database']['path'])==old['compile_database']['sha256'], 'old actual recipe changed')
        require(all(digest(p)==h for p,h in row['fallback_binaries'].items()), 'fallback changed')
        files=[]
        for p in sorted(OLD.rglob('*')):
            if p==OLD/'record.json' or p.is_dir() and not p.is_symlink():continue
            item=dict(path=str(p),reason='Closed 5c diagnostic generation;new same-binary half0/1 comparison replaces its last consumer. Exact original source/recipe recovered;keep the original shared record.')
            if p.is_symlink():item.update(symlink=p.readlink().as_posix(),bytes=p.lstat().st_size)
            else:item.update(sha256=digest(p),bytes=p.stat().st_size)
            files.append(item)
        require(files and len(files)<8192 and not OLD.is_symlink(),'bounded exact old-generation selection')
        manifest=ROOT/'old-retirement-manifest.json'
        proof=dict(original_record_sha256=OLD_RECORD_SHA,recovered_project_files=len(before),
            actual_compiler_inputs=len(original_inputs),original_controls=len(old['controller_inputs']),
            recipe_sha256=lineage.signature(old['recipe']),all_match=True)
        write(manifest,dict(files=files,source_recovery=proof,consumer_closures=closures,completed=False))
        for item in files:
            p=Path(item['path'])
            require(p.readlink().as_posix()==item['symlink'] if 'symlink' in item else digest(p)==item['sha256'],
                    'old artifact changed during retirement')
        for p in OLD.iterdir():
            if p==OLD/'record.json':continue
            if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
            else:p.unlink()
        require(list(OLD.iterdir())==[OLD/'record.json'] and digest(OLD/'record.json')==OLD_RECORD_SHA and
                all(digest(p)==h for p,h in row['fallback_binaries'].items()),'retirement/fallback post-check')
        written=json.loads(manifest.read_text());written['completed']=True;write(manifest,written)
        row['old_retirement']=dict(completed=True,manifest=str(manifest),sha256=digest(manifest),
            files=len(files),bytes=sum(v['bytes'] for v in files),source_recovery=proof,
            retained_original_record=str(OLD/'record.json'),inference_generations_after_retirement=1,
            next='Build the single replacement;half0/1 share its source and binary. Preserve the552b fallback.')
    except BaseException as e:
        row.update(stage='failed-retirement',error=repr(e));raise
    finally:save(row)
    print('Closed5c products retired after exact recovery;validated552b unchanged. Commit before build.',flush=True)


def build(ctx):
    ctx.build()
    row=json.loads((ROOT/'record.json').read_text())
    try:
        database=json.loads(Path(row['compile_database']['path']).read_text())
        for n in (*KERNEL[:1], 'src/core/verify.cpp', 'src/program/generate.cpp', FIXTURE):
            units=[u for u in database if u['file']==str(ROOT/'source'/n)]
            require(len(units)==1,'one actual unit: '+n)
            flags=shlex.split(units[0]['command'])
            require(all(f in flags for f in ['-O3','-fsycl','--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13']) and
                    not any(f in flags for f in ['-ffast-math','-Ofast','-ffp-contract=fast']), 'native flags changed')
            if n in [KERNEL[0],FIXTURE]:require('-ffp-contract=off' in flags,'preserve Q6 contraction policy')
        row['native_recipe_checks']=dict(passed=True,full_verifier_and_generator_compiled=True,
            same_build_Q6_fixture=True,baseline_prefill_CPU_runtime_unchanged=True,GPU_executed=False)
    except BaseException as e:
        row.update(stage='failed-native-recipe-check',passed=False,complete=False,error=repr(e));raise
    finally:save(row)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ['source','prepare','retire','build','host','verify'],
            'use source|prepare|retire|build|host|verify')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1]=='source':
            _,before,values=projection();print(json.dumps(dict(original_files=len(before),files=len(values),
                changed=list(CHANGED),source_aggregate=lineage.signature(pins(values)),GPU_executed=False)))
        elif sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='retire':retire()
        elif sys.argv[1]=='build':build(context())
        elif sys.argv[1]=='host':context().host()
        else:context().verify()
