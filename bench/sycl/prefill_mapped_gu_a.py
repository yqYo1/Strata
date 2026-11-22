"""Compare gather plus GU against the qualified mapped GU operator on B570."""

import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
from statistics import median
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_mixed_gather_measure as reference
from prefill_attention_bank_component import dependency_paths
from run_q6_columns import validation_log

bank, validation = reference.bank, reference.validation
owner, require = reference.owner, reference.require
P, B, RESULT = bank.P, bank.B, bank.RESULT
ROOT, KEY = B / 'prefill-mapped-gu-timing-v1', 'prefill_mapped_gu_timing_v1'
OPERATOR_ROOT, OPERATOR_KEY = B / 'prefill-mapped-gu-operator-v1', 'prefill_mapped_gu_operator_v1'
OPERATOR_RESULT = '6b2a568991a61ba479a70decb7d8d44cbc28ae46'
PARENT = reference.coherent_model.MODEL_ROOT
SRC, BUILD = PARENT / 'source', PARENT / 'build'
PROBE_SRC = ROOT / 'probe-source'
THIS = str(Path(__file__).resolve().relative_to(P))
WITNESS_SOURCE = '19999c6c1df5ce11f12478958904d602c5eff9ac'
WITNESS_RESULT = 'b30fb35e4c237a24a6915c5c3025fb9eb99168a0'
WITNESS_KEY = 'prefill_mapped_gu_a_v5'
KERNEL = 'src/prefill/sycl/vllm_xe2/native16.cpp'
HEADER = 'src/prefill/sycl/vllm_xe2/native16.hpp'
FIXTURE = 'bench/sycl/moe_gu_swiglu.cpp'
TARGET = 'sycl_moe_gu_swiglu_bench'
IMAGE, HOST = 'strata_vllm_xe2_image.o', 'strata_vllm_xe2_host.o'
DEFINE = '-DSTRATA_PREFILL_GU_A_PROBE'
SMOKE_CASES = (('compact', 3, 1, 0, 0), ('wide', 3, 1, 0, 1))
SHAPES = [(n, clustered) for n in (2334, 5963, 8192) for clustered in (0, 1)]
PROCESSES, ROUNDS, PILOT_REPEATS, REPEAT_CAP = 3, 6, 8, 2048
BATCH_FLOOR_MS, TARGET_MS, CLOCK_CAP_US = 100, 180, 100
PLAN = dict(
    interval='Wall time from first host call through the final checked event on the in-order compute queue. '
             'Compare scalar production mixed gather plus native fused GU with mapped native GU. Fix W, '
             'bank, source map, DPAS geometry, B loading, precision and output bytes in each pair. '
             'The model dispatcher, dequantization and all other layers are outside this component interval.',
    shapes=[list(s) for s in SHAPES], processes=PROCESSES, pairs_per_process_shape=ROUNDS,
    order='Rotate shape order by two per process. Start with (process+shape_index)%2, then alternate '
          'arm order per pair. Arm 0 is gather plus native GU; arm 1 is mapped GU.',
    repeat='After exact warmed checks, eight calls in each pilot. Use the same repeat count for the pair, '
           'ceil(target_ms*pilot_repeats/min(pilot_ms)), within [1,repeat_cap]. '
           'Minimum eligible duration applies to both arms of every pair. Keep all short batches '
           'with performance eligibility false for the affected workload. No replacement.',
    target_ms=TARGET_MS, pilot_repeats=PILOT_REPEATS, repeat_cap=REPEAT_CAP, min_batch_ms=BATCH_FLOOR_MS,
    timer='Maximum mean of three windows of 4096 back-to-back steady_clock pairs must be at most '
          '0.1% of the minimum eligible duration. Apply no time subtraction.',
    maximum_clock_mean_us=CLOCK_CAP_US,
    validation='Trace a fresh two-case smoke on the relinked fixture. Time separate unprofiled fresh '
               'processes. Before timing each shape, check gather oracle, BF16 and FP16 captures, '
               'no-capture outputs and guards. Check warmed output bits, status and output guards '
               'after each timed batch. Resets and checks are outside the clock.',
    advance='For a layout at a given size, require complete eligible batches and at least two of '
            'three process medians of paired mapped/gather ratios below one with four of six paired '
            'wins each. The aggregate paired median must also be below one. Both layouts must '
            'pass to consider that size for route review. This is component evidence only.',
    retention='Reuse the admitted operator device and host objects. Keep the latest comparison '
              'fixture and compact samples until the component decision. Preserve the model '
              'prefill and decode references.',
)
BOOT = Path('/proc/sys/kernel/random/boot_id')
BUDGET = dict(wall=480, cpu=840, rss_cap=12 << 30, text_cap=16 << 20,
              file_cap=64 << 20, total_cap=64 << 20)
# SYCL link briefly retains both device bitcode and the extracted archive inputs.
BUILD_BUDGET = dict(BUDGET, total_cap=384 << 20)
sha, signature, write = bank.sha, bank.signature, bank.stage.write


def recipe(root=ROOT):
    """Keep the original BMG device-finalization, host slice and final link modes."""
    probe_src = root / 'probe-source'
    entries = json.loads((BUILD / 'compile_commands.json').read_bytes())
    commands, outputs = {}, {}
    for label, source in (('kernel', KERNEL), ('fixture', FIXTURE)):
        entry, = (x for x in entries if x['file'] == str(SRC / source))
        cmd = shlex.split(entry['command'])
        require(Path(entry['directory']) == BUILD and '-fsycl' in cmd and DEFINE not in cmd and
                cmd[-4:] == ['-o', entry['output'], '-c', entry['file']], 'changed compile rule for ' + source)
        outputs[label] = entry['output']
        # The copied translation unit still needs its original sibling headers.
        commands[label] = [cmd[0], DEFINE, '-iquote', str((SRC / source).parent),
                           '-I' + str((probe_src / HEADER).parent), *cmd[1:-4],
                           '-MD', '-MF', str(root / (label + '.d')), '-o', str(root / (label + '.o')),
                           '-c', str(probe_src / source)]

    lines = subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', TARGET], cwd=BUILD,
                                    text=True, timeout=25).splitlines()
    raw, = (line for line in lines if ' -fsycl-link ' in line and ' -unbundle ' in line and IMAGE in line)
    words = shlex.split(raw)
    require(words[:3] == ['cd', str(BUILD), '&&'] and words.count('&&') == 2,
            'changed native image/host rule')
    split = words.index('&&', 3)
    aot, unbundle = words[3:split], words[split + 1:]
    old_kernel = (BUILD / outputs['kernel']).resolve()
    require(Path(aot[-3]).resolve() == old_kernel and aot[-2:] == ['-o', str(BUILD / IMAGE)],
            'unexpected device registration inputs')
    commands['image'] = [*aot[:-3], str(root / 'kernel.o'), '-o', str(root / IMAGE)]
    bundle_input, = (i for i, word in enumerate(unbundle) if word.startswith('-input='))
    bundle_output, = (i for i, word in enumerate(unbundle) if word.startswith('-output='))
    require(Path(unbundle[bundle_input].removeprefix('-input=')).resolve() == old_kernel
            and unbundle[bundle_output] == '-output=' + str(BUILD / HOST), 'unexpected host slice inputs')
    unbundle[bundle_input] = '-input=' + str(root / 'kernel.o')
    unbundle[bundle_output] = '-output=' + str(root / HOST)
    commands['host-slice'] = unbundle

    words = shlex.split(lines[-1])
    require(words[:2] == [':', '&&'] and words[-2:] == ['&&', ':'] and words.count(TARGET) == 1 and
            words[words.index('-o') + 1] == TARGET and '-Wl,-t' in words, 'checked fixture link rule required')
    changed = {outputs['fixture']: str(root / 'fixture.o'), TARGET: str(root / TARGET),
               IMAGE: str(root / IMAGE), HOST: str(root / HOST)}
    require(all(words.count(old) == 1 for old in changed), 'fixture or native registration missing from final link')
    commands['link'] = [words[2], '-v', *(changed.get(v, v) for v in words[3:-2])]
    return commands


def probe_overlay(native, revision):
    guard = rb'^#ifdef STRATA_PREFILL_GU_A_PROBE\n.*?^#endif\n'
    overlay = {}
    for name, guarded_blocks in ((KERNEL, 3), (HEADER, 2), (FIXTURE, 3)):
        original = reference.fixture.git_file(WITNESS_SOURCE + '^', name)
        candidate = reference.fixture.git_file(revision, name)
        require(original == (SRC / name).read_bytes() and sha(SRC / name) == native['source_files'][name],
                'retained source must match the probe parent at ' + name)
        require(candidate == (P / name).read_bytes(), 'mapped GU source differs from committed revision')
        stripped, count = re.subn(guard, b'', candidate, flags=re.M | re.S)
        require(count == guarded_blocks and list(filter(bytes.strip, stripped.splitlines())) ==
                list(filter(bytes.strip, original.splitlines())), 'unrelated native source change at ' + name)
        overlay[name] = candidate
    return overlay


def archived_probe(revision, key):
    subprocess.run(['git', 'merge-base', '--is-ancestor', revision, 'HEAD'], cwd=P, check=True, timeout=15)
    row = json.loads(reference.fixture.git_file(revision, str(RESULT.relative_to(P))))[key]
    require(row['stage'] == 'closed' and row['complete'] and row['passed'] and row['ownership_closed']
            and not row['active'] and all(s['passed'] and s['fresh_faults'] == [] for s in row['stages']),
            'missing original closed probe: ' + key)
    return row


def parent_inputs(native):
    allowed = {p: h for group in ('compiled', 'generated', 'links', 'objects', 'produced')
               for p, h in native['actual_inputs'][group].items()}
    allowed.update({str(SRC / p): h for p, h in native['source_files'].items()})
    external = native['external_inputs']
    require(sha(external['path']) == external['sha256'], 'qualified build tool inputs changed')
    allowed.update(json.loads(Path(external['path']).read_bytes()))
    return allowed


def recover_probe(root, original, allowed, native, build):
    """Verify the old recipe, compiled source and retained link evidence against Git."""
    row = bank.previous.read(root / 'record.json')
    require(compact(row) == original and row['proof']['parent'] == build['build_sha256']
            and row['proof']['engine'] == native['binaries']['strata']
            and signature((row['build_env'], row['runtime_env'])) == row['environment_sha256'],
            'retained operator build record changed')
    proof = row['proof']
    expected_env = dict(native['recipe']['environment'], TMPDIR=str(root / 'tmp'),
                        XDG_CACHE_HOME=str(root / 'cache'))
    require(row['commands'] == recipe(root) and row['build_env'] == expected_env
            and sha(proof['recipe_reader']['path']) == proof['recipe_reader']['sha256'],
            'retained operator build recipe changed')
    previous = dict(allowed)
    for path, digest in proof['probe_inputs'].items():
        raw = reference.fixture.git_file(proof['probe_source'], path)
        copy = root / 'probe-source' / path
        require(hashlib.sha256(raw).hexdigest() == digest and copy.read_bytes() == raw,
                'retained source no longer matches Git: ' + path)
        previous[str(copy)] = digest
    require(set(proof['probe_inputs']) == {KERNEL, HEADER, FIXTURE}
            and row['compiler_inputs'] == {
                k: compiled_inputs(previous, k, s, root) for k, s in (('kernel', KERNEL), ('fixture', FIXTURE))},
            'retained compiler input closure changed')
    products = row['produced']
    require(set(products) == {str(root / file) for file in ('kernel.o', 'fixture.o', IMAGE, HOST, TARGET)}
            and row['binary'] == dict(path=str(root / TARGET), sha256=products[str(root / TARGET)])
            and all(sha(path) == digest for path, digest in products.items())
            and all((previous | products).get(path) == digest and sha(path) == digest
                    for path, digest in row['link_inputs'].items()),
            'retained operator link inputs or outputs changed')
    return row


def load_inputs(revision):
    native, speed, _, build = reference.coherent_inputs()
    witness = archived_probe(WITNESS_RESULT, WITNESS_KEY)
    old = archived_probe(OPERATOR_RESULT, OPERATOR_KEY)
    require(witness['proof']['probe_source'] == WITNESS_SOURCE
            and old['proof']['previous_witness'] == dict(revision=WITNESS_RESULT, executable=witness['binary'])
            and [s['completed_cases'] for s in old['stages']] == [22, 2, 24, 2, 10],
            'the admitted operator must match the pinned fragment witness and cases')
    allowed = parent_inputs(native)
    previous = recover_probe(OPERATOR_ROOT, old, allowed, native, build)
    overlay = probe_overlay(native, revision)
    sources = {n: hashlib.sha256(raw).hexdigest() for n, raw in overlay.items()}
    require(all(sources[n] == old['proof']['probe_inputs'][n] for n in (KERNEL, HEADER)),
            'timing fixture must reuse exact qualified native source and declarations')
    reuse = {str(ROOT / n): dict(source=str(OPERATOR_ROOT / n), sha256=old['produced'][str(OPERATOR_ROOT / n)])
             for n in (IMAGE, HOST)}
    overlay.pop(KERNEL)
    allowed.update({str(PROBE_SRC / p): sources[p] for p in overlay})
    commands = {name: cmd for name, cmd in recipe().items() if name in ('fixture', 'link')}
    for cmd in commands.values():
        program = Path(cmd[0]).resolve()
        require(allowed.get(str(program), allowed.get(cmd[0])) == sha(program), 'new compiler or linker tool')
    probe = speed['native_validation_probe']
    env = {k: v.replace(str(OPERATOR_ROOT), str(ROOT)) for k, v in previous['runtime_env'].items()
           if not k.startswith('UR_LOG_') and k not in ('UR_ENABLE_LAYERS', 'LD_PRELOAD', 'LD_DEBUG', 'STRATA_TRACE',
                                                        'STRATA_UR_PROBE_EXPECTED_LOADER')
           and not (k.startswith('STRATA_') and k.endswith(('_AUDIT', '_PROFILE', '_OBSERVER', '_CENSUS')))}
    build_env = dict(native['recipe']['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    require(env.get('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD') == '1' and env.get('EnableDirectSubmission') == '0'
            and env.get('NEOReadDebugKeys') == '1' and speed['boot'] == BOOT.read_text().strip()
            and speed['boot'] == old['proof']['boot'] and native['driver'] == old['proof']['driver']
            and probe == old['proof']['validation'] and all(sha(a['path']) == a['sha256'] for a in probe.values()),
            'runtime or qualified validation changed')
    proof = dict(engine=native['binaries']['strata'], parent=build['build_sha256'],
                 native_source_commit=native['native_source_commit'], probe_source=revision, probe_inputs=sources,
                 qualified_operator=dict(revision=OPERATOR_RESULT, executable=old['binary'],
                                         native_overlay_commit=old['proof']['probe_source']),
                 previous_witness=old['proof']['previous_witness'], reused_objects=reuse,
                 boot=speed['boot'], driver=native['driver'], validation=probe,
                 runtime_manifest=native['foundation']['runtime_manifest'])
    return proof, allowed, commands, build_env, env, overlay


def controls(revision=None):
    scripts = [str(Path(m.__file__).resolve().relative_to(P)) for m in
               (reference, reference.fixture, reference.coherent_model, reference.coherent_model.kv)]
    pins = bank.stage.source.pin(P, [THIS, *scripts, 'tools/sycl'])
    require({THIS, *scripts} <= pins['source_sha256'].keys(), 'commit the complete controller first')
    sources = pins['source_sha256']
    if revision is not None:
        historical = {p: hashlib.sha256(raw).hexdigest()
                      for p, raw in bank.stage.source.blobs(P, revision, sources)}
        require(historical == sources, 'original controller inputs changed')
    return dict(revision=revision or pins['revision'], sha256=signature(sources))


def compact(row):
    return {name: val for name, val in row.items() if name not in ('build_env', 'runtime_env')}


def save(row):
    write(ROOT / 'record.json', row)
    public = bank.previous.read(RESULT)
    public[KEY] = compact(row)
    write(RESULT, public)


def prepare(inputs):
    proof, _, commands, build_env, runtime_env, overlay = inputs
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh mapped GU timing fixture only')
    row = dict(stage='prepared', controller=controls(), proof=proof, commands=commands,
        build_env=build_env, runtime_env=runtime_env, environment_sha256=signature((build_env, runtime_env)),
        stages=[], active=False, complete=False, passed=False, ownership_closed=True,
        plan=PLAN,
        budgets=BUDGET, build_budgets=BUILD_BUDGET)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in overlay.items():
        path = PROBE_SRC / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(raw)
    for path, item in proof['reused_objects'].items():
        shutil.copyfile(item['source'], path)
        require(sha(path) == item['sha256'], 'qualified object changed during staging')
    save(row)


def check(row, inputs):
    proof, _, commands, build_env, runtime_env, overlay = inputs
    require(validation.committed(RESULT, (KEY,)) == compact(row) and row['controller'] == controls(row['controller']['revision']) and
            row['proof'] == proof and row['commands'] == commands and row['build_env'] == build_env and
            row['runtime_env'] == runtime_env and row['environment_sha256'] == signature((build_env, runtime_env))
            and row['budgets'] == BUDGET and row['build_budgets'] == BUILD_BUDGET and row['plan'] == PLAN
            and not row['active'] and row['ownership_closed'],
            'commit exact prepared result and keep fixture inputs pinned')
    require(all((PROBE_SRC / p).read_bytes() == raw for p, raw in overlay.items()), 'private probe source changed')
    require(all(sha(path) == item['sha256'] for path, item in proof['reused_objects'].items()),
            'reused operator object changed')


def check_link(text, trace, allowed, generated, root):
    inputs = [root / name for name in ('fixture.o', IMAGE, HOST)]
    require(all(sha(p) == generated[str(p)] for p in inputs), 'driver input changed during link')
    host_slices = {p.name: sorted(reference.fixture.fixture_host_slices(trace, p)) for p in inputs}
    expected_slices = {path for paths in host_slices.values() for path in paths}
    require(len(expected_slices) == sum(map(len, host_slices.values())), 'ambiguous driver host slices')
    observed = {}
    slices = set()
    for line in text.splitlines():
        token = line.strip()
        if not token or len(token) > 4096 or any(c.isspace() for c in token):
            continue
        path = Path(token)
        if not path.is_absolute() and path.suffix not in ('.a', '.o', '.so'):
            continue
        path = (BUILD / path).resolve()
        if str(path) in expected_slices:
            slices.add(str(path))
        elif path.is_file():
            input_sha = generated.get(str(path), allowed.get(str(path)))
            require(input_sha == sha(path), 'unexpected link input ' + str(path))
            observed[str(path)] = input_sha
        else:
            require(path.is_relative_to(root / 'tmp') and path.suffix == '.o', 'unowned link temporary')
    require(expected_slices == slices, 'incomplete actual fixture, AOT image or host object link input')
    return observed, host_slices


def compiled_inputs(allowed, label, source, root=ROOT):
    paths = dependency_paths((root / (label + '.d')).read_text())
    probe_src = root / 'probe-source'
    required = {str(probe_src / HEADER), str(probe_src / source)}
    if label == 'kernel':
        required.add(str((SRC / KERNEL).with_name('xe_gemm_bsd.hpp')))
    require(required <= paths, 'actual probe translation unit or header missing')
    require(all(sha(p) == allowed.get(p) for p in paths), 'changed actual compiler input')
    return dict(count=len(paths), sha256=signature({p: allowed[p] for p in paths}))


def build(row, allowed):
    require(row['stage'] == 'prepared', 'inspect the existing build attempt')
    output = ROOT / 'compile'
    job = owner.Owner(output, lambda: write(output / 'execution.json', job.commands))
    outputs = {path: item['sha256'] for path, item in row['proof']['reused_objects'].items()}
    try:
        for label, product in (('fixture', 'fixture.o'), ('link', TARGET)):
            entry, out, err = job.run(label, row['commands'][label], row['build_env'], BUILD, **row['build_budgets'])
            if not owner.closed(entry) or entry['exit_code'] != 0:
                raise RuntimeError(label + ' failed: ' + (entry.get('error') or err.read_text()[-900:]))
            if label == 'fixture':
                dependencies = {'fixture': compiled_inputs(allowed, label, FIXTURE)}
            if label == 'link':
                links, slices = check_link(out.read_text(), err.read_text(), allowed, outputs, ROOT)
                archives = {str((BUILD / v).resolve()) for v in row['commands']['link'] if v.endswith('.a')}
                require(archives and archives <= links.keys(), 'consumed runtime/engine archives missing')
            outputs[str(ROOT / product)] = sha(ROOT / product)
        row.update(stage='built', binary=dict(path=str(ROOT / TARGET), sha256=outputs[str(ROOT / TARGET)]),
                   compiler_inputs=dependencies, link_inputs=links, link_host_slices=slices, produced=outputs)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row.update(active=bool(job.active), ownership_closed=not job.active and all(owner.closed(c) for c in job.commands))
        save(row)


def read_smoke(text):
    lines = text.splitlines()
    expected = [
        f'prefill_mapped_gu rows={n} tile={tile} split={split} clustered={clustered} invalid={bad} '
        'gather=exact BF16_capture=exact FP16_output=exact null_capture=exact guards=PASS'
        for tile, n, split, clustered, bad in SMOKE_CASES
    ]
    require('{"kind":"topology","work_items":384,"status":"PASS"}' in lines, 'missing subgroup proof')
    end = 'fixture_status=PASS mode=--prefill-mapped-gu-smoke'
    require([line for line in lines if line.startswith(('prefill_mapped_gu', 'fixture_status='))]
            == [*expected, end], 'mapped GU smoke output or coverage differs')
    return dict(completed_cases=len(SMOKE_CASES))


def read_timing(text, process):
    lines = text.splitlines()
    require([line for line in lines if line.startswith('fixture_status=')] ==
            [f'fixture_status=PASS mode=--prefill-mapped-gu-time process={process}'], 'missing timing closure')
    records = [json.loads(line) for line in lines if line.startswith('{')]
    require(records[0] == dict(kind='topology', work_items=384, status='PASS'), 'timing topology mismatch')
    clock = records[1]
    clock_us = clock['maximum_mean_us']
    require(clock['kind'] == 'timer' and clock['clock_pairs'] == 4096 and math.isfinite(clock_us) and clock_us >= 0,
            'invalid timing clock control')
    frames = iter(records[2:])
    shapes = []
    for visit in range(len(SHAPES)):
        shape = (visit + 2 * process) % len(SHAPES)
        rows, clustered = SHAPES[shape]
        first_arm = (process + shape) % 2
        common = dict(process=process, rows=rows, clustered=clustered)
        cal = next(frames)
        pilot = cal['pilot_ms']
        repeat = cal['repeat']
        require(all(cal[k] == v for k, v in common.items()) and cal['kind'] == 'calibration'
                and cal['first_arm'] == first_arm and cal['pilot_repeats'] == PILOT_REPEATS
                and len(pilot) == 2 and all(v > 0 and math.isfinite(v) for v in pilot)
                and type(repeat) is int and 1 <= repeat <= REPEAT_CAP
                and repeat == math.ceil(TARGET_MS * PILOT_REPEATS / min(pilot)), 'timing pilot or order mismatch')
        pairs = []
        for round_index in range(ROUNDS):
            pair = [None, None]
            for order in range(2):
                arm = first_arm ^ (round_index % 2) ^ order
                frame = next(frames)
                ms = frame['batch_ms']
                require({key: value for key, value in frame.items() if key != 'batch_ms'} ==
                        dict(common, kind='gu_batch', round=round_index, order=order, arm=arm,
                             repeat=repeat, output_exact=True)
                        and frame['output_exact'] is True and ms > 0 and math.isfinite(ms),
                        'timed workload, order or exact output differs')
                pair[arm] = ms
            pairs.append(pair)
        shapes.append(dict(rows=rows, clustered=clustered, pilot_ms=pilot, repeat=repeat, paired_batch_ms=pairs))
    require(next(frames, None) is None, 'unexpected timing output')
    return dict(maximum_mean_clock_us=clock_us, shapes=shapes)


def compare(processes):
    require([c['process'] for c in processes] == list(range(PROCESSES)), 'incomplete independent processes')
    shapes = []
    for rows, clustered in SHAPES:
        ratios, summaries = [], []
        for process in processes:
            case, = (s for s in process['shapes'] if (s['rows'], s['clustered']) == (rows, clustered))
            pairs = case['paired_batch_ms']
            paired_ratios = [mapped / gathered for gathered, mapped in pairs]
            short = sum(ms < BATCH_FLOOR_MS for pair in pairs for ms in pair)
            ratios.extend(paired_ratios)
            summaries.append(dict(
                gathered_ms=median(gathered for gathered, _ in pairs) / case['repeat'],
                mapped_ms=median(mapped for _, mapped in pairs) / case['repeat'],
                median_paired_ratio=median(paired_ratios), paired_wins=sum(r < 1 for r in paired_ratios),
                short_batches=short, clock_eligible=process['maximum_mean_clock_us'] <= CLOCK_CAP_US))
        eligible = all(s['short_batches'] == 0 and s['clock_eligible'] for s in summaries)
        gains = sum(s['median_paired_ratio'] < 1 and s['paired_wins'] >= 4 for s in summaries)
        shapes.append(dict(rows=rows, clustered=clustered, processes=summaries, eligible=eligible,
                           median_paired_ratio=median(ratios),
                           repeated_gain=eligible and gains >= 2 and median(ratios) < 1))
    admitted = [rows for rows, clustered in SHAPES if clustered == 0
                and all(s['repeated_gain'] for s in shapes if s['rows'] == rows)]
    return dict(shapes=shapes, sizes_eligible_for_route_review=admitted, model_qualified=False, adopted=False)


def run_child(row, stage, process=None):
    traced = stage == 'smoke'
    name = stage if traced else 'time-' + str(process)
    root = ROOT / name
    root.mkdir(mode=0o700)
    for d in ('tmp', 'cache'):
        (root / d).mkdir(mode=0o700)
    env = {k: v.replace(str(ROOT), str(root)) for k, v in row['runtime_env'].items()}
    probe = row['proof']['validation']
    if traced:
        env.update(LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                   LD_DEBUG='libs', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:stderr',
                   UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    job = owner.Owner(root / 'owned', lambda: write(root / 'execution.json', job.commands))
    sample = dict(stage=stage, environment_sha256=signature(env))
    if process is not None:
        sample['process'] = process
    try:
        executable = row['binary']
        require(sha(executable['path']) == executable['sha256'], 'timing executable changed')
        argv = [executable['path'], '--prefill-mapped-gu-smoke'] if traced else                [executable['path'], '--prefill-mapped-gu-time', str(process)]
        entry, out, err = job.run(name, argv, env, P, **BUDGET)
        sample['native_exit_code'] = entry['exit_code']
        require(owner.closed(entry) and entry['exit_code'] == 0, 'mapped GU process failed: ' + err.read_text()[-700:])
        if traced:
            sample.update(read_smoke(out.read_text()))
            log = err.read_text()
            sample['validation'] = validation_log(log, entry['pid'], probe['loader']['path'], traced=True)
            v = root / 'validation.log'
            require((not v.exists() or not v.stat().st_size) and 'urEnqueueKernelLaunch' in log,
                    'small GU validation did not close cleanly')
            sample['providers'] = bank.device.loaded_providers(
                log, dict(environment=env, native_validation_probe=probe,
                          runtime_manifest=row['proof']['runtime_manifest']))
        else:
            sample.update(read_timing(out.read_text(), process))
        sample['host_probe_accepted'] = True
    except BaseException as error:
        sample['error'] = repr(error)
        raise
    finally:
        validation.finish(sample, job, cursor, row['proof']['boot'])
        sample['passed'] &= sample.get('driver_after') == row['proof']['driver']
        row['stages'].append(sample)
        row.update(active=bool(job.active), ownership_closed=sample['ownership_closed'])
    require(sample['passed'], 'mapped GU process postcondition failed')


def run(row, stage, allowed):
    require(row['stage'] == ('built' if stage == 'smoke' else 'checked') and
            [s['stage'] for s in row['stages']] == ([] if stage == 'smoke' else ['smoke']) and
            all(s['passed'] for s in row['stages']) and
            row['compiler_inputs'] == {'fixture': compiled_inputs(allowed, 'fixture', FIXTURE)} and
            row['binary'] == dict(path=str(ROOT / TARGET), sha256=row['produced'][str(ROOT / TARGET)]) and
            all(sha(path) == h for path, h in (row['produced'] | row['link_inputs']).items()),
            'closed matching build and exact prior stages required')
    try:
        if stage == 'smoke':
            run_child(row, stage)
            row['stage'] = 'checked'
        else:
            for process in range(PROCESSES):
                run_child(row, stage, process)
            row.update(stage='closed', complete=True, passed=True, decision=compare(row['stages'][1:]))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', 'build', 'smoke', 'timing'),
            'select a mapped GU timing stage and the qualified interpreter')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare(load_inputs(controls()['revision']))
        else:
            row = bank.previous.read(ROOT / 'record.json')
            inputs = load_inputs(row['proof']['probe_source'])
            check(row, inputs)
            if action == 'build':
                build(row, inputs[1])
            else:
                run(row, action, inputs[1])
