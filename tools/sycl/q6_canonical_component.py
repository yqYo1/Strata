#!/usr/bin/env python3
"""Build/check canonical-layout C4 without an additional weight allocation.

Reuse the current compiler census and the immutable f978 component execution
protocol. The latter keeps its source/ownership/graph/oracle/timing gates;
this adapter binds current build inputs and additionally checks the exact
canonical kernel trace. Main owns all serialized stages. Repository LICENSE.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import build_compat_profile as base
import q6_dpas_component as reader
import measurement_guards as guards
import measurement_owner as ownership
import ur_validation as validation

P, B = base.P, base.B
ROOT, KEY = B / 'q6-canonical-component-v4', 'q6_canonical_component_v4'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
PROTOCOL = 'f97893576704e6679c89ea6c4613d912665a0824'
PROTOCOL_PATH = 'tools/sycl/q6_dpas_component.py'
KERNEL = '_ZTSN6strata7kernels9component12_GLOBAL__N_117Q6DpasC4CanonicalE'
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
require, digest, write = guards.require, base.digest, base.write
UNITS = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
    'src/kernels/sycl/native_mmvq.cpp', 'src/kernels/sycl/native_q5_head.cpp',
    'src/kernels/sycl/native_q3_esimd.cpp', 'src/kernels/sycl/native_q4_esimd.cpp',
    'src/kernels/sycl/native_q2_esimd.cpp', 'src/kernels/sycl/native_iq4xs_esimd.cpp',
    'src/kernels/sycl/native_iq_esimd.cpp', 'src/kernels/sycl/native_mmq_xmx.cpp',
    'src/kernels/sycl/iq.cpp', 'src/kernels/sycl/q6_dpas_component.cpp',
    'bench/sycl/q6_canonical.cpp')


def image(blob):
    spec = importlib.util.spec_from_file_location('canonical_q6_image_reader', reader.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.KERNEL_ENTRY = KERNEL
    return module.embedded_dpas(blob, allow_shared_image=True)


def context():
    spec = importlib.util.spec_from_file_location('canonical_q6_build', base.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.RESULT = ROOT, KEY, RESULT
    ctx.TARGET, ctx.CPP, ctx.UNITS = 'sycl_q6_canonical_component', UNITS[-1], UNITS
    ctx.PREFIXES = ('CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'bench/sycl', 'tools/sycl')
    ctx.BUDGET = dict(wall=1200, cpu=3600, rss_cap=8 << 30, text_cap=16 << 20,
                      file_cap=128 << 20, total_cap=768 << 20)

    def current(row, committed=True):
        if committed:
            require(validation.committed(RESULT, (KEY,)) == ctx.compact(row), 'commit the exact canonical stage')
        _, identity = ctx.basis()
        require(identity == row['selected_native'], 'qualified runtime source/feature basis changed')
        ctx.build.check_inputs(row)
        require(all(n in row['project']['source_sha256'] for n in UNITS) and
            'tools/sycl/q6_canonical_component.py' in row['project']['source_sha256'], 'component source missing')
        require(row['recipe']['targets'] == row['recipe']['build_targets'] == [ctx.TARGET] and
            not row['recipe']['environment'].get('LD_PRELOAD') and digest(PYTHON) == row['python_sha256'],
            'frozen build/Python policy changed')
        require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
            all(digest(p) == sha for p, sha in row['runtime_pins'].items()), 'runtime bytes changed')
        require(guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot changed')
        require(hashlib.sha256(ctx.source.git(P, 'show', PROTOCOL + ':' + PROTOCOL_PATH)).hexdigest() ==
            row['protocol_sha256'], 'immutable execution protocol changed')
        if 'binary' in row:
            require(digest(row['binary']['path']) == row['binary']['sha256'], 'canonical executable changed')
        if 'actual_inputs' in row:
            inputs = row['actual_inputs']
            require(digest(inputs['path']) == inputs['sha256'], 'actual compiler census changed')
            data = json.loads(Path(inputs['path']).read_text())
            require(all(digest(p) == sha for field in ('compiled', 'generated', 'links')
                for p, sha in data[field].items()), 'actual compiler/link bytes changed')
        if 'codegen' in row:
            require(image(Path(row['binary']['path']).read_bytes()) == row['codegen'], 'direct DPAS image changed')
    ctx.current = current
    return ctx


def prepare(ctx):
    data = validation.committed(RESULT)
    rejected = data['q6_head_generation_speed_v4']
    require(rejected['stage'] == 'failed' and len(rejected['processes']) == 2 and
        rejected['processes'][0]['passed'] and not rejected['processes'][1]['passed'] and
        rejected['memory_admission']['native_exit_code'] == 1 and
        rejected['memory_admission']['long_generation_samples'] == 0 and
        rejected['memory_admission']['cleanup']['all_original_PIDs_absent'], 'preserve actual capacity refusal')
    require(not ROOT.exists() and KEY not in data, 'fresh component only;no retries')
    native, identity = ctx.basis()
    qualified = data['q6_head_model_entry_v1']
    require(qualified['passed'] and qualified['complete'] and qualified['ownership_closed'], 'qualified runtime required')
    manifest = qualified['runtime_manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'qualified runtime manifest changed')
    runtime = json.loads(Path(manifest['path']).read_text())
    recipe = copy.deepcopy(native['recipe'])
    recipe['configure'] = [v.replace(native['project']['root'], str(P)).replace(
        str(ctx.feature.NATIVE), str(ROOT)) for v in recipe['configure']]
    recipe['environment'] = {k: v.replace(str(ctx.feature.NATIVE), str(ROOT))
                             for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = [ctx.TARGET]
    recipe.pop('host_modes', None)
    require(recipe['configure'][recipe['configure'].index('-S') + 1] == str(P) and
        recipe['configure'][recipe['configure'].index('-B') + 1] == str(ROOT / 'build'), 'isolated build paths')
    environment = dict(qualified['environment'])
    for key in ('LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER'): environment.pop(key, None)
    environment.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        PYTHONPYCACHEPREFIX=str(ROOT / 'cache/python'), STRATA_SYCL_DECODE_Q6_DPAS_HEAD='0',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr')
    require(environment['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
        environment['EnableDirectSubmission'] == '0' and environment['NEOReadDebugKeys'] == '1' and
        environment['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and
        environment['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == environment['ZES_ENABLE_SYSMAN'] == '0',
        'immutable qualified safety required')
    source = ctx.source.git(P, 'show', PROTOCOL + ':' + PROTOCOL_PATH)
    row = dict(stage='prepared', project=ctx.source.pin(P, ctx.PREFIXES),
        dependencies=copy.deepcopy(native['dependencies']), recipe=recipe,
        toolchain=native['toolchain'], external_inputs=native['external_inputs'],
        fallbacks={name: native['fallbacks'][name] for name in ('prefill', 'decode')} |
                  dict(selected_feature=identity['engine']), selected_native=identity,
        environment=environment, runtime_manifest=manifest, runtime_pins=runtime,
        native_validation_probe=qualified['probe'], python_sha256=digest(PYTHON),
        protocol_source=PROTOCOL, protocol_sha256=hashlib.sha256(source).hexdigest(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        commands=[], results=[], timing=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False, budgets=ctx.BUDGET,
        host_budget=dict(wall=60, cpu=60, rss_cap=512 << 20, text_cap=512 << 10,
            file_cap=2 << 20, total_cap=1024 << 20),
        native_budget=dict(wall=600, cpu=1200, rss_cap=8 << 30, text_cap=16 << 20,
            file_cap=128 << 20, total_cap=1024 << 20),
        plan=dict(question='Can direct canonical-layout C4 beat the current production column entry without extra weights?',
            motivation='Packed head was6.55283x faster as a component but its additional521472000device bytes refuse1256cache slots at131072context. Preserve that failed whole-model comparison.',
            reference='Same-source native_q6_k_columns_mmvq via ordinary one-node compatibility graph;candidate reads the identical canonical allocation. No repack or additional GPU weight array.',
            order='22CPU census/linked-image/protocol controls;fresh14TU build and exact canonical DPAS image;12CPU shape/alignment controls;small traced3shape/2seed;full2shape/2seed;three timing processes0,1,2. Commit every closed stage;stop first failure,no retry.',
            shapes='Small256x1,256x17,512x33;representative2560x640 and2560x248320;C4only,C1rejected before device access.',
            numerical='All outputs finite and forward-envelope comparison;independent byte scalar oracle on every small row and543distributed/head-tail full-head rows,guards,input immutability,C1reference preserved. Same f978 envelope;no bitwise output requirement.',
            timing='Same f978 asynchronous completion discipline,2warmup+5batches/arm/process;orderR/C,C/R,R/C;4096small or256full launches/batch;>=10ms and>=300timer-overhead units;retain individual samples and spreads.',
            acceptance='Per shape all3 process median ratios must favor candidate;no minimum percentage floor. Do not promote failed or slower shapes. Component result cannot qualify model speed,quality,physical262144/RESTORE or adoption.',
            observers='Small full UR trace and parameter validation;full numeric/timing retain parameter validation and qualified preload,remove API trace;pre-main and post-device/pre-fixture real owner/provider gates. No environment mutation.',
            memory='One canonical weight allocation shared by both graphs;0additional candidate device/host weight bytes. Kernel has no allocation,barrier,collective or CPU arithmetic fallback;16-lane masked halfword gathers handle unpadded N tails.',
            provenance='Main-authored canonical-load adaptation of our current C4 math;harness/execution protocol from own f97893576704e6679c89ea6c4613d912665a0824. Retain existing MIT GGML operand-layout notice. No XeStrata source.'),
        retention='One temporary component executable,not another engine generation. Keep compact numeric/owner/provider/timing evidence and recoverable source/recipe. Retire normal logs/JIT/build products after component closure. Preserve0b model reference,90f feature and552b/e1b phase guards;legacy56ELF holds unchanged.')
    ctx.current(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    ctx.save(row)
    print('Prepared canonical-only component;commit before controls/build.', flush=True)


def admit(ctx):
    """Prospective controller-only admission; never rerun or relabel failed v3."""
    data = validation.committed(RESULT)
    previous = data['q6_canonical_component_v3']
    require(previous['stage'] == 'failed' and previous['ownership_closed'] and not previous['GPU_executed']
        and len(previous['commands']) == 3 and all(c['exit_code'] == 0 and c['ownership_closed']
        for c in previous['commands']), 'preserve original completed compile/failed admission')
    require(not ROOT.exists() and KEY not in data, 'fresh controller-only admission only')
    original_root = B / 'q6-canonical-component-v3'
    original = json.loads((original_root / 'record.json').read_text())
    proof = previous['census_failure']; artifact = proof['compiled_artifact']
    require(original['census_failure'] == proof and digest(artifact['path']) == artifact['sha256']
        and Path(artifact['path']).stat().st_size == artifact['bytes'], 'original compiled ELF changed')
    for name, field in [('build/compile_commands.json', 'compile_database_sha256'),
                        ('owned/dependencies.stdout', 'dependency_listing_sha256'),
                        ('owned/compile.stdout', 'linker_output_sha256')]:
        require(digest(original_root / name) == proof[field], 'original compiler evidence changed')
    project = ctx.source.pin(P, ctx.PREFIXES)
    old, new = original['project']['source_sha256'], project['source_sha256']
    changed = {p for p in old if new.get(p) != old[p]}
    allowed = {'tools/sycl/build_compat_profile.py', 'tools/sycl/q6_dpas_component.py',
               'tools/sycl/q6_canonical_component.py'}
    require(changed == allowed and set(new) - set(old) == {'tools/sycl/test_compile_census.py'},
            'only named controller/parser/test changes;all native source and recipe inputs must match')
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
        ctx.source.blobs(P, original['project']['revision'], old)}
    require(recovered == old, 'original native source must be recoverable from Git')
    legacy_build = context(); legacy_build.ROOT = original_root
    actual = legacy_build.census(original, original_root / 'owned/dependencies.stdout',
                                 original_root / 'owned/compile.stdout')
    row = copy.deepcopy(original)
    for key in ('error', 'reader_controls', 'census_failure'): row.pop(key, None)
    row.update(project=project, actual_inputs=actual, binary=artifact,
        native_source_commit=original['project']['revision'], admission_source_commit=project['revision'],
        codegen=image(Path(artifact['path']).read_bytes()), stage='prepared', compilation_passed=True,
        commands=[], results=[], timing=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False,
        qualification_reuse=dict(original_failed_key='q6_canonical_component_v3',
            original_evidence_commit='d979167dbfb2248b9038e59037bbb702cc77b424',
            native_source_commit=original['project']['revision'], original_project_sha256=original['project']['aggregate_sha256'],
            binary_sha256=artifact['sha256'], changed_controller_files=sorted(changed),
            exact_native_inputs_and_recipe=True, native_recompiled=False, GPU_qualification_inherited=False,
            scope='Fresh selected14TU/compiler/link census and exact canonical callgraph from the unchanged11.6MB ELF. Failed v3 stays failed. New22CPU controls and all GPU stages still required.'))
    row['environment'] = {k: v.replace(str(original_root), str(ROOT)) for k, v in row['environment'].items()}
    row['environment'].update(STRATA_MEASUREMENT_ROOT=str(ROOT),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    # The retained build is outside this new raw-output root. No need to give
    # its diagnostic processes the old build tree's aggregate byte allowance.
    row['host_budget']['total_cap'] = 64 << 20
    row['native_budget'].update(file_cap=16 << 20, total_cap=64 << 20)
    row['plan']['order'] = 'Reuse exact original native artifact after scoped source/recipe/census/image proof. Run22CPU controls,12CPU native guards,small traced checks,full checks and3fresh timing processes. No native rebuild or GPU result inherited.'
    row['plan']['source_identity'] = 'native_source_commit is cdbd68796;project/admission source pins later controller inputs. Do not label the existing ELF with the later source revision.'
    ctx.current(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    ctx.save(row)
    print('Prepared independent controller-only admission;commit before22CPU controls. Native source/ELF unchanged.', flush=True)


def controls(ctx):
    row = json.loads((ROOT / 'record.json').read_text()); ctx.current(row)
    require(row['stage'] == 'prepared' and 'reader_controls' not in row, 'fresh CPU controls only')
    command = [PYTHON, '-B', '-m', 'unittest', '-v', 'tools.sycl.test_q6_dpas_component',
               'tools.sycl.test_compile_census']
    scope = None
    def persist():
        write(ROOT / 'controls-execution.json', dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    try:
        scope = ownership.Owner(ROOT / 'reader-controls', persist)
        entry, out, err = scope.run('reader-controls', command, row['recipe']['environment'], P, **row['host_budget'])
        text = out.read_text() + err.read_text()
        passed = bool(ownership.closed(entry) and entry['exit_code'] == 0 and
            'Ran 22 tests' in text and '\nOK\n' in text and 'skipped' not in text)
        row['reader_controls'] = dict(passed=passed, tests=22, command=command,
            exit_code=entry['exit_code'], ownership_closed=ownership.closed(entry), GPU_executed=False)
        require(passed, 'linked-image/protocol CPU controls failed')
        if row.get('qualification_reuse'): row['stage'] = 'built'
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        row['active'] = bool(scope and scope.active)
        row['ownership_closed'] = not row['active'] and (scope is None or
            all(ownership.closed(c) for c in scope.commands))
        persist(); ctx.save(row)
    require(row['stage'] != 'failed' and row['ownership_closed'], row.get('error', 'CPU controls not closed'))
    print('22CPU controls passed;commit before the next stage.', flush=True)


def run(ctx, stage):
    if stage == 'build':
        row = json.loads((ROOT / 'record.json').read_text())
        require(not row.get('qualification_reuse'), 'reused native artifact must not be rebuilt')
        require(row['reader_controls']['passed'], 'CPU controls first')
        ctx.compile_fixture()
        row = json.loads((ROOT / 'record.json').read_text())
        try:
            row['codegen'] = image(Path(row['binary']['path']).read_bytes())
            row.update(compilation_passed=True, passed=False, complete=False)
        except BaseException as error:
            row.update(stage='failed', passed=False, complete=False, error=repr(error))
        ctx.save(row); require(row['stage'] == 'built', row.get('error', 'direct image check failed'))
        return
    if stage in ('check', 'time0', 'time1', 'time2'):
        require(json.loads((ROOT / 'record.json').read_text()).get('canonical_kernel_trace_passed'),
                'fresh canonical trace qualification required before launch')
    raw = ctx.source.git(P, 'show', PROTOCOL + ':' + PROTOCOL_PATH)
    protocol = ModuleType('canonical_q6_execution_protocol')
    protocol.__file__ = str(P / PROTOCOL_PATH)
    exec(compile(raw, PROTOCOL + ':' + PROTOCOL_PATH, 'exec'), protocol.__dict__)
    protocol.ROOT, protocol.KEY, protocol.RESULT, protocol.TARGET = ROOT, KEY, RESULT, ctx.TARGET
    protocol.current, protocol.save, protocol.compact = ctx.current, ctx.save, ctx.compact
    protocol.execute(stage)
    if stage == 'check-small':
        row = json.loads((ROOT / 'record.json').read_text())
        text = (ROOT / ('v7-owned-' + stage) / 'native.stderr').read_text()
        passed = 'Q6DpasC4Canonical' in text
        row['canonical_kernel_trace_passed'] = passed
        if not passed: row.update(stage='failed', passed=False, complete=False, error='exact canonical kernel trace missing')
        ctx.save(row); require(passed, 'exact canonical kernel trace missing')


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in
        ('prepare', 'admit', 'controls', 'build', 'host', 'check-small', 'check', 'time0', 'time1', 'time2'), 'canonical component stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        ctx = context(); stage = sys.argv[1]
        if stage == 'prepare': prepare(ctx)
        elif stage == 'admit': admit(ctx)
        elif stage == 'controls': controls(ctx)
        else: run(ctx, stage)
