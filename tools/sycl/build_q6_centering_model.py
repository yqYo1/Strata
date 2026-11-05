#!/usr/bin/env python3
"""Build the scoped decode centering candidate with the qualified supervisor.

Main-authored, repository LICENSE. Recover b310 exactly, preserve the validated
552b prefill fallback, and retire closed b310 products before building its
replacement. No model or GPU execution is authorized by compilation alone.
"""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import build_cpu_service_model as template
import device_fp64_probe as dependency_image
import q6_centering_sources as source

P, B, RESULT, base = template.P, template.B, template.RESULT, template.base
ROOT = B / 'q6-centering-model-v1'
KEY = 'q6_centering_model_build_v1'
SELF = Path(__file__).resolve()
OLD = source.original.ROOT
TARGETS = source.original.TARGETS
require, digest, write = base.require, base.digest, base.write


def source_basis():
    _, old, values = source.projection()
    return old, dict(original_source_commit=source.BASE_SOURCE,
        original_recipe_commit='1f396eeccdb69e0fe01fe7cb26c4b0683ba4d84e',
        original_result_commit='08537c8e3277dff7cd9d0c47223dbf3eb9c45cee',
        original_record_sha256=digest(OLD / 'record.json'),
        original_engine_sha256=old['binaries']['strata']['sha256'],
        original_source_aggregate=source.signature(old['native_sources']),
        candidate_source_aggregate=source.signature(source.pins(values)),
        changed=list(source.CHANGED), component_and_policy=source.qualification(),
        scope='Recoverable source/recipe and component evidence only;fresh full-verifier,device and model qualification required.')


def recipe(old, root):
    old_root = str(OLD)
    value = copy.deepcopy(old['recipe'])
    value['configure'] = [x.replace(old_root, str(root)) for x in value['configure']]
    flags = [i for i, x in enumerate(value['configure']) if x.startswith('-DCMAKE_CXX_FLAGS=')]
    require(len(flags) == 1, 'one explicit native C++ flag set')
    value['configure'][flags[0]] += ' --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'
    value['environment'] = {k: v.replace(old_root, str(root))
                            for k, v in value['environment'].items()}
    value['environment'].update(STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1',
        STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_DECODE_Q6_CENTERING='0')
    value['targets'] = list(TARGETS)
    value['retention'] = ('Replacement for closed b310 comparison products;preserve552b and its fixtures. '
        'This generation is not qualified by compilation. Fresh host/device/model gates precede '
        'same-binary decode-centering0/1 timing and independent prefill/TTFT checks.')
    return value


def toolchain(row):
    asset = row['toolchain_image']
    require(digest(asset['path']) == asset['sha256'], 'prebuild toolchain image changed')
    image = json.loads(Path(asset['path']).read_text())
    require(dependency_image.toolchain() == image, 'installed toolchain/package membership or bytes changed')
    return {v['realpath']: v['sha256'] for v in image['files'].values() if v['kind'] == 'file'}


def inputs(row):
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    require(base.controls(row['source_commit'], controller=SELF) == row['controller_inputs'],
            'commit and freeze complete centering build controls')
    old, proof = source_basis()
    _, _, values = source.projection()
    require(proof == row['selected_basis'] and source.pins(values) == row['native_sources']
            and recipe(old, ROOT) == row['recipe'], 'centering native construction/recipe changed')
    require(template.measured.build.monitor_qualification(base.committed(RESULT)) ==
            row['monitor_qualification'], 'qualified build supervisor changed')
    fallback = json.loads((template.selected.model.ROOT / 'record.json').read_text())
    allowed = template.selected.model.inputs(fallback, reuse_revision=revision,
        controller_changes=template.selected.NATIVE_CONTROLS)
    allowed = {p: h for p, h in allowed.items()
               if not Path(p).is_relative_to(template.selected.model.ROOT)}
    allowed.update(toolchain(row))
    for name, raw in values.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'frozen candidate native input changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    allowed.update(row['generated_expected'])
    return allowed


def retention_gate():
    review = base.committed(RESULT)['retained_builds_review_20261103']
    retired = review.get('q6_centering_model_replacement', {})
    require(retired.get('completed') and retired.get('original_inputs_recovered')
            and retired.get('next_root') == str(ROOT) and retired.get('old_root') == str(OLD),
            'recover/retire the closed b310 generation before its replacement')
    require(digest(retired['manifest_path']) == retired['manifest_sha256'],
            'closed-generation retirement manifest changed')
    manifest = json.loads(Path(retired['manifest_path']).read_text())
    require(manifest['completed'] and manifest['files']
            and all(not Path(v['path']).exists() for v in manifest['files'])
            and not (OLD / 'build').exists() and not (OLD / 'source').exists(),
            'closed b310 build products/source snapshot still exist')
    fallback = json.loads((template.selected.model.ROOT / 'record.json').read_text())
    require(all(Path(v['path']).is_file() and digest(v['path']) == v['sha256']
                for v in fallback['binaries'].values()), 'protect all validated552b fallback outputs')
    return review


def context():
    spec = importlib.util.spec_from_file_location('strata_centering_native_build', template.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.CONTROLLER = ROOT, KEY, SELF
    ctx.source = SimpleNamespace(NATIVE=source.SELECTOR_SOURCE,
        PROGRAM=source.observer.PROGRAM, POOL=source.observer.POOL,
        HEADER=source.observer.HEADER, FIXTURES=source.observer.FIXTURES)
    ctx.inputs, ctx.retention_gate = inputs, retention_gate
    return ctx


def prepare(ctx):
    require(not ROOT.exists() and KEY not in base.committed(RESULT),
            'one fresh model build;inspect an existing receipt rather than retry')
    old, proof = source_basis()
    _, _, values = source.projection()
    require(old['passed'] and old['complete'] and old['ownership_closed'] and not old['active']
            and all(digest(v['path']) == v['sha256'] for v in old['binaries'].values()),
            'inspect the actual closed original generation before source-only retirement')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    controls = base.controls(revision, controller=SELF)
    image = dependency_image.toolchain()
    plan = recipe(old, ROOT)
    row = dict(stage='prepared', source_commit=revision, controller_inputs=controls,
        native_sources=source.pins(values), selected_basis=proof, recipe=plan,
        monitor_qualification=template.measured.build.monitor_qualification(base.committed(RESULT)),
        toolchain_image=dict(path=str(ROOT / 'toolchain.json'), files=len(image['files']),
                             bytes=image['bytes'], packages=image['packages']),
        generated_expected={p.replace(str(OLD), str(ROOT)): h
                            for p, h in old['generated_expected'].items()},
        commands=[['configure', plan['configure']],
            ['compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'),
                         '--target', *TARGETS, '-j', '4']],
            ['dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps']]],
        budgets=dict(wall=1800, cpu=7200, rss_cap=12 << 30, text_cap=8 << 20,
                     file_cap=512 << 20, total_cap=2 << 30), whole_wall_seconds=3600,
        qualification_plan=dict(
            host='Same-build exact IQ3 and actual pool0/1 checks;Q6 policy/cache separately passes at dc2299.',
            device='Fresh strict traced submission/phase checks on this generation before model requests.',
            model='Fresh thinking-on32K short-output route checks with centering0 and1;require actual per-phase centered-call receipts and normal closure before long comparison.',
            speed='Prospectively commit three counterbalanced independent pairs on both >=1K-output MTP classes using bench.sh. Judge each workload separately;prefill/TTFT remain independent.',
            stopping='Stop on any source,executable,provider,numerical,validation,owner,fault or budget failure;no automatic retries or hash refresh.',
            remaining='Agent/task quality,physical262144/tails/RESTORE and inference adoption stay pending.'),
        retention=plan['retention'], commands_executed=[], active=False, ownership_closed=True,
        passed=False, complete=False, GPU_executed=False, model_executed=False,
        observer_overhead_qualified=False, inference_adopted=False)
    ROOT.mkdir(mode=0o700)
    for folder in ('tmp', 'cache', 'source'):
        (ROOT / folder).mkdir(mode=0o700)
    write(ROOT / 'toolchain.json', image)
    row['toolchain_image']['sha256'] = digest(ROOT / 'toolchain.json')
    for name, raw in values.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    ctx.save(row)
    try:
        inputs(row)
    except BaseException as error:
        row.update(stage='failed-prepare', error=repr(error))
        ctx.save(row)
        raise
    print('Prepared scoped Q6-centering source/recipe. Commit before build;no GPU/model execution.', flush=True)


def build(ctx):
    ctx.build()
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        database = json.loads(Path(row['compile_database']['path']).read_text())
        for name in ('src/kernels/sycl/native_mmvq.cpp', 'src/core/verify.cpp'):
            units = [u for u in database if u['file'] == str(ROOT / 'source' / name)]
            require(len(units) == 1, 'one actual candidate native unit required: ' + name)
            flags = shlex.split(units[0]['command'])
            require(all(f in flags for f in ('-O3', '-fsycl',
                        '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13')),
                    'unplanned native Q6/verifier flags')
            if name.endswith('native_mmvq.cpp'):
                require('-ffp-contract=off' in flags, 'preserve native Q6 contraction policy')
        actual = json.loads(Path(row['compiled_inputs']['path']).read_text())
        require(actual[str(ROOT / 'source' / source.CENTER_HEADER)] ==
                row['native_sources'][source.CENTER_HEADER], 'actual Q6 helper dependency missing')
        row['centering_native_recipe_checks'] = dict(passed=True,
            q6_and_full_verifier_compiled=True, actual_centering_header=True,
            GPU_executed=False, model_executed=False)
    except BaseException as error:
        row.update(stage='failed-native-recipe-check', passed=False, complete=False, error=repr(error))
        raise
    finally:
        ctx.save(row)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'verify'),
            'usage: build_q6_centering_model.py prepare|build|verify')
    ctx = context()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare(ctx)
        elif sys.argv[1] == 'build': build(ctx)
        else: ctx.verify()
