"""Build and qualify the isolated page-local BF16 key layout on Arc B570."""

import fcntl
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_build as reference
import prefill_qsa_groups as qsa
import prefill_qsa_key_page_source as source

bank, require = reference.bank, reference.require
validation = bank.basis.validation
ROOT = bank.B / 'prefill-qsa-key-page-v1'
KEY = 'prefill_qsa_key_page_v1'
TARGET = qsa.TARGET
FIXTURE = qsa.BENCH
GENERATED = 'src/kernels/sycl/qsa_key_page_component.cpp'
SPEED = 'prefill_kv_upload_speed_v2'
STAGES = ('build', 'smoke', 'boundary', '0', '1', '2')
SHAPES = {
    'smoke': ((97, 3, 33, 'gather'),),
    'boundary': ((262143, 3, 33, 'clustered'), (262144, 3, 33, 'clustered')),
    'timing': ((78453, 8192, 33, 'gather'), (78453, 8192, 2051, 'gather'),
               (78453, 8192, 2051, 'clustered'), (78453, 8192, 2051, 'blocks4')),
}
NAMES = ('ordered_exp', 'key_page')


def reference_inputs():
    native = bank.previous.read(reference.ROOT / 'record.json')
    reference.context().check(native)
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'], 'qualified V2 build required')
    speed = bank.previous.read(bank.B / SPEED.replace('_', '-') / 'record.json')
    measured = validation.committed(bank.RESULT, (SPEED,))
    for name in ('binary', 'driver', 'boot', 'native_source_commit', 'native_validation_probe', 'runtime_manifest'):
        require(speed[name] == measured[name], 'V2 runtime evidence differs: ' + name)
    require(measured['complete'] and measured['passed'] and measured['ownership_closed']
            and not measured['active'] and measured['binary'] == native['binaries']['strata']
            and measured['driver'] == native['driver']
            and measured['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and measured['environment_template_sha256'] == bank.signature(speed['environment_template']),
            'closed V2 timing and exact runtime template required')
    probe = speed['native_validation_probe']
    for obj in (speed['runtime_manifest'], native['foundation']['runtime_manifest'],
                probe['library'], probe['loader']):
        require(bank.sha(obj['path']) == obj['sha256'], 'runtime or probe changed: ' + obj['path'])
    return native, speed


def recipe(native, helper):
    db = helper.BUILD / 'compile_commands.json'
    require(bank.sha(db) == native['configured_files'][str(db)], 'V2 compiler commands changed')
    original = helper.SOURCE / source.KERNEL
    require(bank.sha(original) == native['source_files'][source.KERNEL], 'measured QSA source changed')
    fixture = helper.git_file('HEAD', FIXTURE)
    require(fixture == (bank.P / FIXTURE).read_bytes(), 'commit the component fixture first')
    overlay = {GENERATED: source.generate(original.read_bytes()), FIXTURE: fixture}
    units = bank.previous.read(db)
    commands = []
    objects = {}
    for original_path, path, target in ((source.KERNEL, GENERATED, 'strata_kernels'),
                                        (FIXTURE, FIXTURE, TARGET)):
        unit, = [u for u in units if u['file'] == str(helper.SOURCE / original_path)
                 and u['output'].startswith(f'CMakeFiles/{target}.dir/')]
        argv = shlex.split(unit['command'])
        require(unit['directory'] == str(helper.BUILD) and '-fsycl' in argv and '-ffp-contract=off' in argv
                and argv[-4:] == ['-o', unit['output'], '-c', unit['file']], 'unexpected SYCL compile input')
        if path == FIXTURE:
            old = '-DSTRATA_QSA_BENCH_ORDERED_EXP=1'
            require(argv.count(old) == 1, 'mandatory fixture definition changed')
            argv[argv.index(old)] = '-DSTRATA_QSA_PREFILL_KEY_PAGE_COMPONENT=1'
        obj = str(ROOT / (Path(path).name + '.o'))
        commands.append([*argv[:-4], '-MD', '-MF', obj + '.d', '-o', obj, '-c', str(ROOT / path)])
        objects[path] = (unit['output'], obj)

    ninja = '/usr/bin/ninja'
    require(bank.sha(ninja) == native['build_tools'][ninja], 'configured Ninja changed')
    link = subprocess.check_output([ninja, '-t', 'commands', TARGET],
                                   cwd=helper.BUILD, text=True, timeout=20).splitlines()[-1]
    words = shlex.split(link)
    old_object, new_object = objects[FIXTURE]
    require(words[:2] == [':', '&&'] and words[-2:] == ['&&', ':'] and '-Wl,-t' in words
            and words.count(old_object) == 1 and words[words.index('-o') + 1] == TARGET
            and [w for w in words if w.endswith('.a')] ==
                ['libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a'],
            'original layout fixture link recipe changed')
    argv = words[2:-2]
    argv[argv.index('-o') + 1] = str(ROOT / TARGET)
    index = argv.index(old_object)
    argv[index:index + 1] = [new_object, objects[GENERATED][1]]
    commands.append([argv[0], '-v', *argv[1:]])
    return overlay, commands


def read_samples(text, action):
    timed = action in ('0', '1', '2')
    process = int(action) if timed else 0
    shapes = SHAPES['timing' if timed else action]
    lines = text.splitlines()
    timer, = [line for line in lines if line.startswith('timer ')]
    t = re.fullmatch(r'timer calls=10000 median_ns=([.\d]+) p99_ns=(\d+)', timer)
    require(t and 0 <= float(t[1]) < 1000 and float(t[1]) <= int(t[2]) < 5000,
            'ineligible steady-clock overhead')
    require(lines.count('fixture_status=PASS model_quality=UNRUN model_PP=UNRUN '
                        'physical262144_model=UNRUN faults=EXTERNAL_REQUIRED') == 1, 'incomplete fixture')
    samples = [line for line in lines if line.startswith('sample ')]
    require(len(samples) == len(shapes) * 12, 'wrong number of paired intervals')
    cases = [line for line in lines if line.startswith('case ')]
    expected_cases = [
        f'case cells={cells} queries={n} cap={cap} page_size=4 pattern={pattern} '
        'samples=12 full_bits=PASS guards=PASS inputs=PASS status=PASS'
        for cells, n, cap, pattern in shapes]
    require(cases == expected_cases, 'missing shape, bitwise, guard or input validation')
    results = []
    for i, (cells, n, cap, pattern) in enumerate(shapes):
        pairs = []
        for pair in range(6):
            intervals = {}
            for order in range(2):
                arm = (process + pair + order) % 2
                name = NAMES[arm]
                head = (f'sample process={process} cells={cells} queries={n} cap={cap} pattern={pattern} '
                        f'round={pair} order={order} variant={arm} implementation={name} milliseconds=')
                s = samples[i * 12 + pair * 2 + order]
                m = re.fullmatch(re.escape(head) + r'(\S+)' +
                                 f' compared={n * 24 * 256} exact_bits=PASS guards=PASS status=PASS', s)
                require(m, 'timed arm, shape or validation differs')
                ms = float(m[1])
                floor = 0.5 if cap == 33 else 10
                require(math.isfinite(ms) and 0 < ms < 60000 and (not timed or ms >= floor),
                        'interval outside planned timer envelope')
                intervals[name] = ms
            pairs.append(intervals)
        row = dict(cells=cells, queries=n, width=cap, pattern=pattern)
        if timed:
            row.update(pairs_ms=pairs, median_paired_ratio=statistics.median(
                p['key_page'] / p['ordered_exp'] for p in pairs))
        results.append(row)
    return dict(timer_p99_ns=int(t[2]), shapes=results)


def execute(action):
    native, speed = reference_inputs()
    helper = bank.basis.core.source.device.load_module('qsa_key_page_fixture_build', qsa.device.fixture.__file__)
    helper.SOURCE, helper.BUILD = reference.ROOT / 'source', reference.ROOT / 'build'
    overlay, commands = recipe(native, helper)
    allowed = helper.build_inputs(native, ROOT, overlay, commands)
    paths = {FIXTURE, str(Path(__file__).resolve().relative_to(bank.P))}
    for module in tuple(sys.modules.values()):
        f = getattr(module, '__file__', None)
        if f and Path(f).resolve().is_relative_to(bank.P / 'bench' / 'sycl'):
            paths.add(str(Path(f).resolve().relative_to(bank.P)))
    pins = bank.stage.source.pin(bank.P, [*sorted(paths), 'tools/sycl'])
    require(paths <= pins['source_sha256'].keys(), 'commit the component controls first')
    plan = dict(commands=commands, sources=bank.stage.hashed(overlay), controls=pins['aggregate_sha256'],
                parent=bank.signature(bank.builder.compact(native)),
                native_source_commit=native['native_source_commit'],
                build_environment_sha256=bank.signature(dict(native['recipe']['environment'],
                    TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))),
                runtime_template_sha256=bank.signature(speed['environment_template']), order=list(STAGES),
                shapes={mode: [list(shape) for shape in sizes] for mode, sizes in SHAPES.items()},
                timing_pairs_per_shape=6, arm_order='(process + pair + order) % 2',
                minimum_timing_ms={'33': 0.5, '2051': 10},
                gate='Require a lower paired median in all three blocks4 timing processes. '
                     'Keep spread and clustered measurements for a separate coverage decision.')
    if action == 'recipe':
        print(f'{len(overlay)} committed or generated sources, {len(commands)} commands, no build')
        return
    result = validation.committed(bank.RESULT)
    if action == 'prepare':
        require(KEY not in result and not ROOT.exists(), 'inspect the existing QSA component')
        result[KEY] = dict(plan=plan, controller_commit=pins['revision'], measured_model=False, stages=[])
        bank.stage.write(bank.RESULT, result)
        return
    row = result[KEY]
    prior = bank.signature(row)
    require(row['plan'] == plan and [s['stage'] for s in row['stages']] == list(STAGES[:STAGES.index(action)])
            and all(s['passed'] and s['ownership_closed'] for s in row['stages']), 'plan or prior stage differs')
    stage = dict(stage=action, passed=False)
    receipt = ROOT / 'record.json'
    try:
        if action == 'build':
            require(not ROOT.exists(), 'inspect the existing fixture build')
            helper.build_fixture(ROOT, TARGET, native, overlay, commands, allowed, {
                'controller_commit': row['controller_commit'], 'plan_sha256': bank.signature(plan)})
            build = bank.previous.read(receipt)
            require(build['complete'] and build['ownership_closed'], 'fixture build did not close')
            stage.update(passed=True, build_sha256=bank.sha(receipt), binary=build['binary'])
        else:
            build = bank.previous.read(receipt)
            first = row['stages'][0]
            binary = build['binary']
            require(build['complete'] and build['ownership_closed'] and bank.sha(receipt) == first['build_sha256']
                    and binary == first['binary'] and bank.sha(binary['path']) == binary['sha256']
                    and all(bank.sha(p) == h for k in ('compiled', 'linked') for p, h in build[k].items())
                    and all(bank.sha(ROOT / p) == h for p, h in build['driver_inputs'].items()),
                    'fixture or its measured inputs changed')
            arguments = ['--' + action] if action in ('smoke', 'boundary') else ['--component', action]
            qsa.measure_fixture(action, arguments, binary, native, speed, stage, ROOT,
                                lambda stdout: read_samples(stdout, action))
            require(stage['passed'], 'fixture correctness, owner or driver gate failed')
    except BaseException as exc:
        stage['error'] = repr(exc)
        raise
    finally:
        if action == 'build' and receipt.is_file():
            stage['ownership_closed'] = bank.previous.read(receipt)['ownership_closed']
        row['stages'].append(stage)
        if action == '2' and stage['passed']:
            ratios = [next(s['median_paired_ratio'] for s in p['shapes'] if s['pattern'] == 'blocks4')
                      for p in row['stages'][-3:]]
            row.update(grouped_time_ratios=ratios, component_performance_gate=all(r < 1 for r in ratios))
        current = bank.previous.read(bank.RESULT)
        require(bank.signature(current.get(KEY)) == prior, 'QSA result changed during execution')
        current[KEY] = row
        bank.stage.write(bank.RESULT, current)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('recipe', 'prepare', *STAGES) and sys.executable == bank.basis.reference.model.PYTHON,
            'use the pinned benchmark interpreter and one QSA stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        execute(action)
