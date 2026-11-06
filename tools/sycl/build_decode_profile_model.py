#!/usr/bin/env python3
"""Build the selected native engine plus optional drained decode gates.

Main-authored; repository LICENSE. Preserve552b. Replace the closed cb7
generation only after exact source/recipe recovery and consumer closure.
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
import adaptive_residency_host as host
import build_cache_map_model as fallback
import build_q6_native_half_model as selected
import decode_profile_sources as source

P, B, RESULT, base = selected.P, selected.B, selected.RESULT, selected.base
ROOT, OLD = B / 'decode-profile-model-v1', selected.ROOT
KEY = 'decode_profile_model_build_v1'
SELF = Path(__file__).resolve()
require, digest, write = base.require, base.digest, base.write
signature = selected.lineage.signature
historical, COMPONENT_EVIDENCE = selected.historical, selected.COMPONENT_EVIDENCE
RETIRED_PARTS = ('build', 'source', 'tmp', 'cache', 'owned-build', 'host')


def compact(row):
    value = base.compact(row)
    value.pop('commands', None)
    recipe = value.pop('recipe')
    value['recipe'] = dict(helper='tools/sycl/build_decode_profile_model.py:recipe',
        sha256=signature(recipe), targets=recipe['targets'],
        scope='Exact selected native recipe at a new private root;only optional host decode gates added.')
    if 'host_result' in value:
        value['host_result'] = {n:v for n,v in value['host_result'].items() if n != 'commands'}
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def basis():
    require(digest(OLD / 'record.json') == host.NATIVE_RECORD_SHA, 'original cb7 build record changed')
    native = json.loads((OLD / 'record.json').read_text())
    parent, _, values = selected.projection()
    require(selected.pins(values) == native['native_sources'] and
            selected.recipe(parent) == native['recipe'] and len(values) == 567,
            'recover all exact567 native inputs and original recipe')
    data = base.committed(RESULT)
    width = data['mtp_width_generation_speed_v1']
    require(width['stage'] == 'closed' and width['passed'] and width['complete'] and
            width['ownership_closed'] and not width['active'] and len(width['processes']) == 6 and
            all(v['passed'] and v['complete'] and v['ownership_closed'] and
                v['native_exit_code'] == 0 and not v['fresh_faults'] for v in width['processes']),
            'close/commit all six width-process consumers first')
    for key in ('q6_native_half_collector_active_v1', 'compat_collector_window_check_v1',
                'decode_profile_gate_host_v1'):
        row = data[key]
        require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'],
                'close/commit prerequisite: ' + key)
    gate = data['decode_profile_gate_host_v1']
    require(gate['source_commit'] == source.HEADER_SOURCE and
            gate['source_inputs']['project'][source.HEADER] == source.HEADER_SHA and
            gate['outcome']['receipt'] == 'decode_profile_gate cases=45 status=PASS' and
            not gate['GPU_executed'], 'exact host-qualified gate header required')
    base.guards.no_model_owners([])
    return native, values


def projection():
    native, before = basis()
    header = selected.blob(source.HEADER_SOURCE, source.HEADER)
    require((P / source.HEADER).read_bytes() == header, 'freeze exact tested header')
    return native, before, source.projection(before, header)


def recipe(native):
    value = copy.deepcopy(native['recipe'])
    value['configure'] = [v.replace(str(OLD), str(ROOT)) for v in value['configure']]
    value['environment'] = {n:v.replace(str(OLD), str(ROOT)) for n,v in value['environment'].items()}
    value['retention'] = 'Keep552b fallback plus this single opt-in diagnostic generation. Retire after the current model-attribution/next optimization comparison;no third engine generation. Gates are disabled for clean speed.'
    return value


def dependencies():
    """Recover full pinned third-party Git inputs,not just previously used headers."""
    full, _ = fallback.projection(worker=True)
    files = {}
    for group in full['dependencies']:
        recovered = {n:hashlib.sha256(raw).hexdigest() for n,raw in base.basis.build.blobs(
            Path(group['root']), group['revision'], group['source_sha256'])}
        require(recovered == group['source_sha256'], 'third-party Git recovery changed')
        for n,h in recovered.items():
            path = (Path(group['root']) / n).resolve()
            require(digest(path) == h, 'actual third-party input changed: ' + str(path))
            files[str(path)] = h
    return files


def inputs(row):
    require(base.controls(row['source_commit'], controller=SELF) == row['controller_inputs'],
            'freeze exact transitive diagnostic build controls')
    native, _, values = projection()
    require(selected.pins(values) == row['native_sources'] and recipe(native) == row['recipe'],
            'diagnostic source/recipe changed')
    require(selected.template.measured.build.monitor_qualification(base.committed(RESULT)) ==
            row['monitor_qualification'], 'qualified finite build monitor changed')
    require(digest(row['external_inputs']['path']) == row['external_inputs']['sha256'],
            'prebuild external input map changed')
    allowed = json.loads(Path(row['external_inputs']['path']).read_text())
    require(all(digest(p) == h for p,h in allowed.items()), 'external build input changed')
    # Full SDK and dependency membership/recovery remain checked independently.
    require(all(allowed.get(p) == h for p,h in base.sdk().items()), 'original complete SDK changed')
    require(all(allowed.get(p) == h for p,h in dependencies().items()), 'dependency source changed')
    pin, image = host.image()
    require(pin == row['toolchain_image'] and all(allowed.get(v['realpath']) == v['sha256']
            for v in image['files'].values() if v['kind'] == 'file'), 'compiler/package membership changed')
    require(all(digest(p) == h for p,h in row['fallback_binaries'].items()), 'validated fallback changed')
    for n,raw in values.items():
        p = ROOT / 'source' / n
        require(p.is_file() and not p.is_symlink() and p.read_bytes() == raw, 'private source changed: ' + n)
        allowed[str(p.resolve())] = row['native_sources'][n]
    allowed.update(row['generated_expected'])
    return allowed


def retention_gate():
    row = json.loads((ROOT / 'record.json').read_text())
    retired = row.get('old_retirement', {})
    require(retired.get('complete') is True and
            digest(retired['manifest']) == retired['sha256'] and
            all(not (OLD / n).exists() for n in RETIRED_PARTS) and
            all(not Path(v['path']).exists() for v in json.loads((OLD / 'record.json').read_text())['binaries'].values()),
            'recover and physically retire closed cb7 products before replacement build')
    require(all(digest(p) == h for p,h in row['fallback_binaries'].items()), 'preserve all nine fallback outputs')


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh diagnostic generation')
    native, before, values = projection()
    require(all(digest(v['path']) == v['sha256'] for v in native['binaries'].values()) and
            all((OLD / 'source' / n).read_bytes() == raw for n,raw in before.items()),
            'inspect exact selected final native generation before retirement')
    require(digest(native['compiled_inputs']['path']) == native['compiled_inputs']['sha256'] and
            digest(native['compile_database']['path']) == native['compile_database']['sha256'],
            'original actual compiler/recipe evidence changed')
    allowed = base.sdk()
    original_external = native['external_inputs']
    require(digest(original_external['path']) == original_external['sha256'], 'original external map changed')
    allowed.update(json.loads(Path(original_external['path']).read_text()))
    allowed.update(native['actual_link_inputs']['frozen_inputs'])
    allowed.update(dependencies())
    image_pin, image = host.image()
    allowed.update({v['realpath']:v['sha256'] for v in image['files'].values() if v['kind'] == 'file'})
    require(all(digest(p) == h for p,h in allowed.items()), 'pin external inputs before build')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', source_commit=revision,
        controller_inputs=base.controls(revision, controller=SELF), native_sources=selected.pins(values),
        selected_basis=dict(record_sha256=host.NATIVE_RECORD_SHA,
            source_commit=native['source_commit'], binaries=native['binaries'],
            recovered_files=567, recovered_sha256=signature(native['native_sources']),
            changed=[source.GENERATOR, source.HEADER],
            prefill_CPU_GPU_kernels_and_runtime_unchanged=True),
        recipe=recipe(native), toolchain_image=image_pin,
        external_inputs=dict(path=str(ROOT / 'external-inputs.json')),
        fallback_binaries=native['fallback_binaries'],
        monitor_qualification=native['monitor_qualification'],
        generated_expected={p.replace(str(OLD),str(ROOT)):h for p,h in native['generated_expected'].items()},
        commands=[['configure',recipe(native)['configure']],
            ['compile',['/usr/bin/cmake','--build',str(ROOT / 'build'),'--target',*native['recipe']['targets'],'-j','4']],
            ['dependencies',['/usr/bin/ninja','-C',str(ROOT / 'build'),'-t','deps']]],
        budgets=dict(wall=1800,cpu=7200,rss_cap=12<<30,text_cap=8<<20,file_cap=512<<20,total_cap=2<<30),
        whole_wall_seconds=3600, host_plan=dict(modes=[0,1]),
        commands_executed=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False, inference_adopted=False,
        plan='Commit source/recipe;recover and retire closed cb7 outputs while preserving552b;commit retirement;build six same-build targets. Only generate.cpp and the host-qualified header change. After normal0/closed compiler and full actual header/link checks,commit before CPU parity and small traced device/entry checks. No model or clean-speed qualification from compilation. New model collector plan must retain active construction,fully drained0/8/16/24 gates,strict before-main validation,finite trace budget and actual round/host-flow evidence. Full generation remains thinking-on with an actual answer;diagnostic timing is not clean TG.',
        retention=recipe(native)['retention'])
    ROOT.mkdir(mode=0o700)
    for n in ('source','tmp','cache'): (ROOT / n).mkdir(mode=0o700)
    for n,raw in values.items():
        p = ROOT / 'source' / n; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
    write(ROOT / 'external-inputs.json', allowed); row['external_inputs']['sha256'] = digest(ROOT / 'external-inputs.json')
    try: inputs(row)
    except BaseException as error:
        row.update(stage='failed-prepare', error=repr(error)); save(row); raise
    save(row)
    print('Prepared568-source decode-gate diagnostic. Commit,retire cb7,commit,then build.',flush=True)


def retire():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == compact(row) and row['stage'] == 'prepared' and
            'old_retirement' not in row and not (ROOT / 'retire-attempt.json').exists(), 'one committed retirement')
    write(ROOT / 'retire-attempt.json',dict(source_commit=row['source_commit']))
    try:
        allowed = inputs(row); native, before = basis()
        for n,h in native['controller_inputs'].items():
            require(hashlib.sha256(selected.blob(native['source_commit'],n)).hexdigest() == h,
                    'original final build control unrecoverable: ' + n)
        require(all(base.supervision.closed(c) for c in native['commands_executed']), 'old build owners not closed')
        require(all(digest(v['path']) == v['sha256'] for v in native['binaries'].values()), 'old final executable changed')
        actual = json.loads(Path(native['compiled_inputs']['path']).read_text())
        require(len(actual) == native['compiled_inputs']['files'] and
                digest(native['compiled_inputs']['path']) == native['compiled_inputs']['sha256'], 'original actual compiler map changed')
        for p,h in actual.items():
            path = Path(p)
            if path.is_relative_to(OLD / 'source'):
                require(hashlib.sha256(before[str(path.relative_to(OLD / 'source'))]).hexdigest() == h,
                        'actual final source not recovered')
            elif path.is_relative_to(OLD / 'build'):
                require(native['generated_expected'].get(p) == h, 'original generated input not recovered')
            else:
                require(allowed.get(str(path.resolve())) == h, 'original external input not recovered: ' + p)
            require(digest(p) == h, 'actual original input changed: ' + p)
        require(digest(native['compile_database']['path']) == native['compile_database']['sha256'], 'original final recipe changed')
        files = []
        for name in RETIRED_PARTS:
            root = OLD / name
            require(not root.is_symlink(), 'retirement root symlink')
            for p in sorted(root.rglob('*')) if root.exists() else []:
                if p.is_dir() and not p.is_symlink(): continue
                item = dict(path=str(p), bytes=p.lstat().st_size,
                    reason='Closed cb7 generation;exact original567sources,controls,recipe and actual inputs recovered;new opt-in diagnostic replaces its ended consumers')
                if p.is_symlink(): item['symlink'] = p.readlink().as_posix()
                else: item['sha256'] = digest(p)
                files.append(item)
        require(files and len(files) <= 8192 and sum(v['bytes'] for v in files) <= 4<<30, 'finite retirement selection')
        manifest = ROOT / 'old-retirement-manifest.json'
        proof = dict(original_record_sha256=host.NATIVE_RECORD_SHA,
            recovered_native_files=567, recovered_controls=len(native['controller_inputs']),
            original_recipe_sha256=signature(native['recipe']), actual_compiler_inputs=len(actual), all_match=True)
        write(manifest,dict(files=files,source_recovery=proof,complete=False))
        for item in files:
            p = Path(item['path'])
            require(p.readlink().as_posix() == item['symlink'] if 'symlink' in item else digest(p) == item['sha256'],
                    'old output changed during serialized retirement')
        for name in RETIRED_PARTS:
            if (OLD / name).exists(): shutil.rmtree(OLD / name)
        check = json.loads(manifest.read_text()); check['complete'] = True; write(manifest,check)
        row['old_retirement'] = dict(complete=True,manifest=str(manifest),sha256=digest(manifest),
            files=len(files),bytes=sum(v['bytes'] for v in files),source_recovery=proof,
            retained='Original source/recipe record and shared compiler image remain provenance consumers;552b and all nine fallback outputs are preserved.')
        retention_gate_after = all(not (OLD / n).exists() for n in RETIRED_PARTS)
        require(retention_gate_after and all(digest(p) == h for p,h in row['fallback_binaries'].items()), 'retirement/fallback post-check')
    except BaseException as error:
        row.update(stage='failed-retirement',error=repr(error)); raise
    finally: save(row)
    print('Closed cb7 products retired;552b fallback intact. Commit before build.',flush=True)


def context():
    spec = importlib.util.spec_from_file_location('strata_decode_gate_build', selected.template.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.CONTROLLER = ROOT, KEY, SELF
    ctx.source = SimpleNamespace(PROGRAM=source.GENERATOR,
        POOL=selected.template.source.POOL, HEADER=selected.template.source.HEADER,
        FIXTURES=selected.template.source.FIXTURES)
    ctx.inputs, ctx.retention_gate, ctx.compact, ctx.save = inputs, retention_gate, compact, save
    return ctx


def build():
    context().build()
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        database = json.loads(Path(row['compile_database']['path']).read_text())
        for n in (source.GENERATOR, *selected.KERNEL[:1], 'src/core/verify.cpp', selected.FIXTURE):
            units = [u for u in database if u['file'] == str(ROOT / 'source' / n)]
            require(len(units) == 1, 'one actual compile unit: ' + n)
            flags = shlex.split(units[0]['command'])
            require(all(f in flags for f in ('-O3','-fsycl','--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13')) and
                    not any(f in flags for f in ('-ffast-math','-Ofast','-ffp-contract=fast')),
                    'native compile flags changed: ' + n)
            if n in (selected.KERNEL[0],selected.FIXTURE):
                require('-ffp-contract=off' in flags, 'Q6 contraction policy changed')
        actual = json.loads(Path(row['compiled_inputs']['path']).read_text())
        require(actual.get(str(ROOT / 'source' / source.HEADER)) == source.HEADER_SHA,
                'host-qualified gate header absent from actual compiler inputs')
        row['native_recipe_checks'] = dict(passed=True,full_verifier_and_generator_compiled=True,
            same_build_Q6_fixture=True,prefill_CPU_GPU_kernels_and_runtime_unchanged=True,
            gate_header_in_actual_dependencies=True,GPU_executed=False)
    except BaseException as error:
        row.update(stage='failed-native-recipe-check',passed=False,complete=False,error=repr(error)); raise
    finally: save(row)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('source','prepare','retire','build','host','verify'),
            'use source|prepare|retire|build|host|verify')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1] == 'source':
            _, before, values = projection()
            print(json.dumps(dict(files=len(values),changed=[n for n in values if values[n] != before.get(n)],
                sha256=signature(selected.pins(values)),GPU_executed=False)))
        elif sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'retire': retire()
        elif sys.argv[1] == 'build': build()
        else: getattr(context(),sys.argv[1])()
