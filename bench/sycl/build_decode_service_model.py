#!/usr/bin/env python3
"""Build the disjoint-service diagnostic from the actual closed generation.

Main-authored; repository LICENSE. Reuse the qualified owned build supervisor.
Compilation changes no model qualification, speed decision or physical ceiling.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import build_cpu_service_model as template
import device_fp64_probe as dependency_image
import decode_service_sources as source

B, RESULT, base = source.B, template.RESULT, template.base
ROOT, OLD = source.ROOT, source.SOURCE_ROOT
KEY = 'decode_service_model_build_v1'
SELF = Path(__file__).resolve()
previous = source.previous
TARGETS = previous.original.TARGETS
require, digest, write = base.require, base.digest, base.write


def controls(revision):
    # The bench/ entrypoints import shared tools/ helpers through sys.path.
    # Pin those import roots explicitly; same-directory AST discovery alone
    # does not cover that boundary.
    pins = {}
    for entry in (SELF, Path(source.__file__), Path(previous.__file__),
                  Path(template.__file__), Path(dependency_image.__file__)):
        pins.update(base.controls(revision, controller=entry))
    for name in ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md',
                 'bench/sycl/decode_wait_bridge.cpp'):
        raw = previous.blob(revision, name)
        require((P / name).read_bytes() == raw, 'commit exact build input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def source_basis():
    native, values = source.projection()
    cohort = source.cohort()
    proof = dict(original_native_source=source.NATIVE,
        original_record_sha256=digest(OLD / 'record.json'),
        original_engine_sha256=source.ENGINE,
        cohort_sha256=previous.signature(cohort),
        centering_selected=cohort['decision']['centering_model_TG_gain_qualified'],
        native_input_aggregate=previous.signature(native['native_sources']),
        diagnostic_input_aggregate=previous.signature(previous.pins(values)),
        observer_delta=source.PATCH, wait_label_delta=source.WAIT_LABEL_PATCH,
        changed=list(source.CHANGED),
        scope='Source recovery and original decision only;fresh host/device/model and observer-overhead qualification required.')
    return native, values, proof


def toolchain(row):
    item = row['toolchain_image']
    require(digest(item['path']) == item['sha256'], 'diagnostic toolchain image changed')
    image = json.loads(Path(item['path']).read_text())
    require(dependency_image.toolchain() == image, 'installed toolchain/package inputs changed')
    return {v['realpath']: v['sha256'] for v in image['files'].values() if v['kind'] == 'file'}


def inputs(row):
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    require(controls(row['source_commit']) == row['controller_inputs'], 'diagnostic controls changed')
    native, values, proof = source_basis()
    require(proof == row['selected_basis'] and previous.pins(values) == row['native_sources']
        and source.recipe(native) == row['recipe'], 'diagnostic source/recipe changed')
    require(template.measured.build.monitor_qualification(base.committed(RESULT)) ==
        row['monitor_qualification'], 'qualified build supervisor changed')
    fallback = json.loads((template.selected.model.ROOT / 'record.json').read_text())
    allowed = template.selected.model.inputs(fallback, reuse_revision=revision,
        controller_changes=template.selected.NATIVE_CONTROLS)
    allowed = {n: h for n, h in allowed.items()
               if not Path(n).is_relative_to(template.selected.model.ROOT)}
    allowed.update(toolchain(row))
    for name, raw in values.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'diagnostic native input changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    allowed.update(row['generated_expected'])
    return allowed


def retention_gate():
    data = base.committed(RESULT)
    review = data['retained_builds_review_20261103']
    retired = review.get('decode_service_model_replacement', {})
    require(retired.get('completed') is True and retired.get('original_inputs_recovered') is True
        and retired.get('next_root') == str(ROOT) and retired.get('old_root') == str(OLD)
        and retired.get('inference_generations_after_retirement') == 1,
        'close all a1b0 consumers and recover/retire it before building its replacement')
    # A retirement declaration alone does not close another controller's job.
    # Bind all named live-byte consumers to their actual committed outcomes.
    names = ('q6_centering_model_entry_arm0_v1', 'q6_centering_model_entry_arm1_v1',
             'q6_centering_generation_speed_v1', 'q6_f16_widen_device_v2')
    closures = {}
    for name in names:
        item = data.get(name, {})
        require(item.get('stage') == 'closed' and item.get('complete') is True
            and item.get('passed') is True and item.get('ownership_closed') is True
            and item.get('active') is False and item.get('native_source_commit') == source.NATIVE,
            'original native consumer still open/unqualified: ' + name)
        closures[name] = previous.signature(item)
    require(retired.get('consumer_closures') == closures,
            'retirement must bind the exact named consumer outcomes')
    require(data[names[-1]]['GPU_executed'] is True
        and data[names[-1]]['semantic_result']['passed'] is True
        and data[names[-1]]['fresh_faults'] == []
        and data[names[-1]]['boot_unchanged'] is True,
        'standalone widening screen still needs its complete safe device outcome')
    require(digest(retired['manifest_path']) == retired['manifest_sha256'],
            'diagnostic replacement retirement manifest changed')
    manifest = json.loads(Path(retired['manifest_path']).read_text())
    require(manifest['completed'] is True and manifest['files']
        and all(not Path(v['path']).exists() for v in manifest['files'])
        and not (OLD / 'source').exists() and not (OLD / 'build').exists(),
        'the retired source/build products still exist')
    fallback = json.loads((template.selected.model.ROOT / 'record.json').read_text())
    require(all(Path(v['path']).is_file() and digest(v['path']) == v['sha256']
        for v in fallback['binaries'].values()), 'preserve every validated552b fallback output')


def context():
    spec = importlib.util.spec_from_file_location('strata_disjoint_service_build', template.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.CONTROLLER = ROOT, KEY, SELF
    ctx.source = SimpleNamespace(NATIVE=source.NATIVE, PROGRAM=previous.observer.PROGRAM,
        POOL=previous.observer.POOL, HEADER=previous.observer.HEADER,
        FIXTURES=previous.observer.FIXTURES)
    ctx.inputs, ctx.retention_gate = inputs, retention_gate
    original = ctx.compact

    def compact(row):
        value = original(row)
        value['recipe']['helper'] = 'bench/sycl/decode_service_sources.py:recipe'
        return value

    ctx.compact = compact
    return ctx


def prepare(ctx):
    require(not ROOT.exists() and KEY not in base.committed(RESULT),
            'one fresh diagnostic build;inspect an existing receipt rather than retry')
    native, values, proof = source_basis()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins, image, plan = controls(revision), dependency_image.toolchain(), source.recipe(native)
    require(plan['targets'] == list(TARGETS), 'preserve the exact five native build targets')
    row = dict(stage='prepared', source_commit=revision, controller_inputs=pins,
        native_sources=previous.pins(values), selected_basis=proof, recipe=plan,
        monitor_qualification=template.measured.build.monitor_qualification(base.committed(RESULT)),
        toolchain_image=dict(path=str(ROOT / 'toolchain.json'), files=len(image['files']),
            bytes=image['bytes'], packages=image['packages']),
        generated_expected={n.replace(str(OLD), str(ROOT)): h
                            for n, h in native['generated_expected'].items()},
        commands=[['configure', plan['configure']],
            ['compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'),
                         '--target', *TARGETS, '-j', '4']],
            ['dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps']]],
        budgets=dict(wall=1800, cpu=7200, rss_cap=12 << 30, text_cap=8 << 20,
            file_cap=512 << 20, total_cap=2 << 30), whole_wall_seconds=3600,
        qualification_plan=dict(
            host='Fresh same-build IQ3/pool checks plus separately compiled seven-case accounting and five-case actual wait-bridge fixtures;pin their dependencies and executable identities.',
            device='Fresh bounded traced submission/phase checks with normal closure and no fresh fault before model execution.',
            model='Fresh thinking-on32K model entry with complete source/ELF/provider/body/cache/Q6/IQ3 joins and disjoint-ledger parser gates.',
            observers='Separate same-binary profile0/1 processes;CPU_SERVICE1 and CPU_WORK_CENSUS1 in both,immutable before exec. Check clock resolution/read cost,closed nesting,wall/callerCPU/workerCPU conservation and overhead. Clean speed stays separate.',
            limits='Wait/submission/copy envelopes do not measure physical bus traffic or hardware saturation. No performance adoption,70TG,agent/task,physical262144/tails/RESTORE or ordinary teardown qualification from compilation.',
            stopping='Stop first input,budget,owner,compiler,dependency,link or output failure;no retry or hash refresh.'),
        retention=plan['retention'], commands_executed=[], active=False, ownership_closed=True,
        passed=False, complete=False, GPU_executed=False, model_executed=False,
        observer_overhead_qualified=False, inference_adopted=False)
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    write(ROOT / 'toolchain.json', image)
    row['toolchain_image']['sha256'] = digest(ROOT / 'toolchain.json')
    for name, raw in values.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write(ROOT / 'record.json', row)
    try:
        inputs(row)
    except BaseException as error:
        row.update(stage='failed-prepare', error=repr(error))
        ctx.save(row)
        raise
    ctx.save(row)
    print('Prepared diagnostic source/recipe;commit before build. No GPU/model qualification.', flush=True)


def build(ctx):
    ctx.build()
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        actual = json.loads(Path(row['compiled_inputs']['path']).read_text())
        for name in ('include/strata/sycl/decode_service_profile.hpp',
                     'include/strata/sycl/cpu_wait_profile.hpp'):
            require(actual[str(ROOT / 'source' / name)] == row['native_sources'][name],
                    'actual diagnostic header dependency missing: ' + name)
        units = json.loads(Path(row['compile_database']['path']).read_text())
        for name in ('src/program/generate.cpp', 'src/core/verify.cpp'):
            matches = [u for u in units if u['file'] == str(ROOT / 'source' / name)]
            require(len(matches) == 1, 'one actual diagnostic production unit: ' + name)
            flags = shlex.split(matches[0]['command'])
            require(all(f in flags for f in ('-O3', '-fsycl',
                '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
                '-DSTRATA_CPU_SERVICE_PROFILE=1')), 'diagnostic compile flags changed')
        row['diagnostic_native_recipe_checks'] = dict(passed=True,
            actual_profile_and_wait_bridge_headers=True, GPU_executed=False, model_executed=False)
    except BaseException as error:
        row.update(stage='failed-native-recipe-check', passed=False, complete=False, error=repr(error))
        raise
    finally:
        ctx.save(row)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'verify'),
            'use prepare|build|verify')
    ctx = context()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare(ctx)
        elif sys.argv[1] == 'build': build(ctx)
        else: ctx.verify()
