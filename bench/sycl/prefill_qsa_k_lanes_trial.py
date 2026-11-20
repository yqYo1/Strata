"""Compare one long-prefill attention change at a time with the measured ordered-exp kernel."""

import fcntl
import hashlib
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import build_prefill_mixed_gather as fixture
import prefill_qsa_head_major_source as head_major
import prefill_qsa_k_lanes_source as lanes
import qsa_prefill_probability_trial as ordered

bank, require = fixture.bank, fixture.require
P, B, BUILD = bank.P, bank.B, fixture.BUILD
REFERENCE = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
BENCH = 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
ARCHIVES = ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')
STAGES = ('build', 'smoke', '0', '1', '2')
CASES = ((33, 'gather'), (2051, 'gather'), (2051, 'clustered'), (2051, 'blocks4'))


@dataclass(frozen=True)
class Trial:
    arm: str
    generator: ModuleType
    revision: str
    bench_revision: str
    kernel_sha256: str
    bench_sha256: str

    @property
    def root(self):
        return B / ('prefill-qsa-' + self.arm.replace('_', '-') + '-v1')

    @property
    def target(self):
        return 'sycl_qsa_prefill_' + self.arm + '_bench'


K_LANES = Trial('k_lanes4', lanes, 'c11c99275e191c828a375148a52543885213280b',
                '5c58ff1758243725da00c8ec4c91137da7b77aeb',
                '5d131f75ab26ad0e9366324e610127060da35bce198f3f87cbc72672ce83e3c2',
                '2ea5fb3b94ef593a832769bce0c16e4ba7abd72a701355b7bbcbd732dffa413b')
HEAD_MAJOR = Trial('head_major', head_major, 'f0b231aec0ec84149dd4bc7e5afbae1caa419335',
                   'f0b231aec0ec84149dd4bc7e5afbae1caa419335',
                   '0d46692e0b56e837ce611147284c27ced33a69ace2b7f5d203bd4287d323fc69',
                   '2434dee9e5481f9b9e842b18fea31eaf15345de5efb985b7d6091b48201e84eb')


def recipe(parent, trial):
    source = fixture.SOURCE
    original = (source / REFERENCE).read_bytes()
    generator = str(Path(trial.generator.__file__).resolve().relative_to(P))
    require(original == fixture.git_file(trial.revision, REFERENCE)
            and fixture.git_file(trial.revision, generator) == (P / generator).read_bytes(),
            'the measured ordered-exp kernel and pinned generator are required')
    generated = 'qsa_prefill_' + trial.arm + '.cpp'
    overlay = {generated: trial.generator.candidate(original), BENCH: fixture.git_file(trial.bench_revision, BENCH)}
    require(hashlib.sha256(overlay[generated]).hexdigest() == trial.kernel_sha256
            and hashlib.sha256(overlay[BENCH]).hexdigest() == trial.bench_sha256,
            'the generated kernel and pinned page-four fixture must match their measured inputs')
    ordered.kernel_arguments(BUILD)  # Both kernel targets have the actual production compile flags.
    units = bank.previous.read(BUILD / 'compile_commands.json')
    commands, replacements = [], {}
    for target, old, new in (('sycl_qsa_prefill_pv_kernel', bank.PV_KERNEL, generated),
                             ('sycl_qsa_prefill_pv_bench', BENCH, BENCH)):
        obj = 'CMakeFiles/' + target + '.dir/' + old + '.o'
        unit, = [u for u in units if u['output'] == obj]
        args = shlex.split(unit['command'])
        require(unit['directory'] == str(BUILD) and args[-4:] == ['-o', obj, '-c', str(source / old)],
                'unexpected compile unit or arguments')
        flags = args[:-4]
        if old == BENCH:
            flags[flags.index('-DSTRATA_QSA_PREFILL_PV_COMPONENT=1')] = f'-DSTRATA_QSA_PREFILL_{trial.arm.upper()}_COMPONENT=1'
            flags.remove('-I' + str(source / 'bench/sycl'))
        out = str(trial.root / (Path(new).name + '.o'))
        replacements[obj] = out
        commands.append([*flags, '-MD', '-MF', out + '.d', '-o', out, '-c', str(trial.root / new)])
    old_target = 'sycl_qsa_prefill_pv_bench'
    raw = subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', old_target],
                                  cwd=BUILD, text=True, timeout=20).splitlines()[-1]
    link = shlex.split(raw)
    require(link[:3] == [':', '&&', commands[0][0]] and link[-2:] == ['&&', ':']
            and link[link.index(next(iter(replacements))):-2] == [*replacements, '-o', old_target, *ARCHIVES]
            and '-Wl,-t' in link and '-fsycl' in link, 'unexpected parent fixture link recipe')
    replacements[old_target] = str(trial.root / trial.target)
    commands.append([link[2], '-v', *(replacements.get(arg, arg) for arg in link[3:-2])])
    require(all(parent['actual_inputs']['produced'].get(str(BUILD / a)) == bank.sha(BUILD / a) for a in ARCHIVES),
            'original compiled QSA or runtime archive differs')
    return overlay, commands


def checked_samples(text, process, smoke, trial):
    def rows(label):
        return [ordered.data_fields(line) for line in text.splitlines() if line.startswith(label + ' ')]
    timer, = rows('timer')
    require(int(timer['calls']) == 10000 and 0 <= float(timer['median_ns']) <= int(timer['p99_ns']) < 5000
            and float(timer['median_ns']) < 1000, 'timer overhead exceeds the declared limit')
    cases = CASES[:1] if smoke else CASES
    cells, queries = (97, 3) if smoke else (78453, 8192)
    samples, markers = rows('sample'), rows('case')
    require(len(samples) == 12 * len(cases) and len(markers) == len(cases), 'incomplete QSA comparison')
    adjacent = rows('adjacent_queries')
    require(len(adjacent) == (len(cases) if trial == HEAD_MAJOR else 0), 'unexpected locality evidence')
    names = ('ordered_exp', trial.arm)
    points = []
    for i, (cap, pattern) in enumerate(cases):
        common = dict(cells=str(cells), queries=str(queries), cap=str(cap), pattern=pattern)
        require(markers[i] == dict(common, page_size='4', samples='12', full_bits='PASS', guards='PASS', inputs='PASS',
                                   status='PASS'), 'case extent or full bit checks differ')
        overlap = {}
        if trial == HEAD_MAJOR:
            counts = adjacent[i].copy()
            shared, valid = int(counts.pop('shared_ids')), int(counts.pop('valid_ids'))
            require(counts == common and 0 <= shared <= valid <= (queries - 1) * cap and valid > 0,
                    'wrong adjacent-query population or shape')
            fraction = shared / valid
            if cap == 2051:
                lower, upper = {'gather': (0, 0.10), 'clustered': (0.90, 1), 'blocks4': (0.40, 0.50)}[pattern]
                require(lower <= fraction <= upper, 'actual valid-ID overlap differs from the planned workload')
            overlap = dict(adjacent_shared_ids=shared, adjacent_valid_ids=valid)
        arms = {name: [] for name in names}
        for j, sample in enumerate(samples[i * 12:(i + 1) * 12]):
            pair, position = divmod(j, 2)
            variant = (process + pair + position) % 2
            ms = float(sample.pop('milliseconds'))
            require(sample == dict(common, process=str(process), round=str(pair), order=str(position),
                                   variant=str(variant), implementation=names[variant], compared=str(queries*24*256),
                                   exact_bits='PASS', guards='PASS', status='PASS'), 'actual arm, order or bits differ')
            floor = 0 if smoke else (0.5 if cap == 33 else 10)
            require(floor <= ms < 10000, 'invalid measured duration')
            arms[names[variant]].append(ms)
        if not smoke:
            points.append(dict(cap=cap, pattern=pattern, samples_ms=arms, **overlap,
                               median_time_ratio=statistics.median(arms[trial.arm]) / statistics.median(arms['ordered_exp'])))
    return dict(clock=timer, points=points, validated_cases=len(cases))


def measure(action, record, parent, speed, stage, trial):
    directory = trial.root / ('measure-' + action)
    directory.mkdir(mode=0o700)
    for folder in ('tmp', 'cache'):
        (directory / folder).mkdir(mode=0o700)
    owner = bank.builder.ownership.Owner(directory / 'owned', lambda: bank.stage.write(directory / 'execution.json', owner.commands))
    seed, probe = speed['environment_template'], speed['native_validation_probe']
    env = {k: v.replace(seed['STRATA_MEASUREMENT_ROOT'], str(directory)) for k, v in seed.items()}
    env.pop('STRATA_TRACE', None)
    env.pop('UR_LOG_TRACING', None)
    env.update(STRATA_TEST_MAIN_GATES='0', LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'], UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(directory / 'validation.log'))
    smoke = action == 'smoke'
    if smoke:
        env.update(LD_DEBUG='libs', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:stderr')
    argv = [record['binary']['path'], *(['--smoke'] if smoke else ['--component', action])]
    stage.update(command=argv, environment_sha256=bank.signature(env))
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    try:
        cmd, out, err = owner.run(action, argv, env, P, **(bank.device.SMALL if smoke else bank.device.LARGE))
        require(bank.builder.ownership.closed(cmd) and cmd['exit_code'] == 0, 'fixture failed: ' + err.read_text()[-800:])
        text, log = out.read_text(), err.read_text()
        proof = bank.basis.validation.preload_evidence(log, cmd['pid'])
        layer = directory / 'validation.log'
        device, = re.findall(r'^device=([^\n]+) driver=(\S+)$', text, re.M)
        require('B570' in device[0] and proof['loader'] == probe['loader']['path']
                and (not layer.exists() or layer.stat().st_size == 0) and 'fixture_status=PASS' in text,
                'wrong device, validation or incomplete fixture')
        bank.device.diagnostics.diagnostics(log, cmd['pid'])
        if smoke:
            require('urEnqueueKernelLaunch' in log, 'missing first-use UR trace')
            stage['providers'] = bank.device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
                                                              runtime_manifest=parent['foundation']['runtime_manifest']))
        stage.update(**checked_samples(text, 0 if smoke else int(action), smoke, trial),
                     native_exit_code=0, validation=proof, device=device, host_probe_accepted=True)
    finally:
        bank.basis.validation.finish(stage, owner, cursor, speed['boot'])
        stage['passed'] = stage['passed'] and stage.get('driver_after') == parent['driver']
    require(stage['passed'], 'post-run fault, ownership or driver check failed')


def execute(action, trial):
    parent, _ = ordered.parent()
    root, key = trial.root, 'prefill_qsa_' + trial.arm + '_v1'
    overlay, commands = recipe(parent, trial)
    sources = [str(Path(path).resolve().relative_to(P)) for path in
               (__file__, fixture.__file__, lanes.__file__, head_major.__file__)]
    pins = bank.stage.source.pin(P, [*sources, 'tools/sycl'])['source_sha256']
    require(all(p in pins for p in sources), 'commit the fixture and its controllers first')
    controls = bank.signature(pins)
    allowed = fixture.build_inputs(parent, root, overlay, commands)
    if action == 'check':
        print(f'Original ordered-exp flags, {len(commands)-1} isolated compiles, {len(ARCHIVES)} parent archives; '
              f'recipe {bank.signature(commands)}')
        return
    result = bank.previous.read(bank.RESULT)
    if action == 'build':
        require(key not in result and not root.exists(), 'inspect the previous QSA comparison')
        row = dict(controls=controls, reference=parent['binaries']['strata'], source=trial.revision,
                   fixture_source=trial.bench_revision, build_recipe=bank.signature(commands), stages=[], model_tested=False)
    else:
        row = bank.basis.validation.committed(bank.RESULT, (key,))
        require(row == result[key] and row['controls'] == controls and row['build_recipe'] == bank.signature(commands)
                and row['source'] == trial.revision and row['fixture_source'] == trial.bench_revision
                and row['reference'] == parent['binaries']['strata']
                and [s['stage'] for s in row['stages']] == list(STAGES[:STAGES.index(action)])
                and all(s['passed'] for s in row['stages']), 'commit the preceding stage and preserve the inputs')
    stage = dict(stage=action, passed=False)
    try:
        if action == 'build':
            fixture.build_fixture(root, trial.target, parent, overlay, commands, allowed,
                                  dict(driver=bank.sha(__file__), controls=controls, candidate=trial.revision, base=bank.INPUT_SOURCE))
            record = bank.previous.read(root / 'record.json')
            require(record['complete'] and record['ownership_closed'], 'incomplete component build')
            stage.update(build_sha256=bank.sha(root / 'record.json'), binary=record['binary'],
                         compiler_inputs=len(record['compiled']), linker_inputs=len(record['linked']),
                         passed=True, ownership_closed=True)
        else:
            record = bank.previous.read(root / 'record.json')
            built = row['stages'][0]
            require(built['build_sha256'] == bank.sha(root / 'record.json') and record['binary'] == built['binary']
                    and bank.sha(record['binary']['path']) == record['binary']['sha256']
                    and record['parent'] == bank.signature(bank.builder.compact(parent))
                    and all(bank.sha(p) == h for group in ('compiled', 'linked') for p, h in record[group].items())
                    and all(bank.sha(root / p) == h for p, h in record['driver_inputs'].items()),
                    'the compiled comparison or one of its inputs changed')
            speed = bank.previous.read(bank.basis.reference.ROOT / 'record.json')
            measure(action, record, parent, speed, stage, trial)
    except BaseException as error:
        stage['error'] = repr(error)
        raise
    finally:
        if action == 'build' and (root / 'record.json').is_file():
            stage['ownership_closed'] = bank.previous.read(root / 'record.json')['ownership_closed']
        row['stages'].append(stage)
        if action == STAGES[-1] and stage['passed']:
            wins = [all(s['points'][i]['median_time_ratio'] < 1 for s in row['stages'][2:]) for i in range(len(CASES))]
            row['faster_each_process'] = wins
            row['wide_model_trial_eligible'] = all(wins[1:]) if trial == HEAD_MAJOR else wins[1] and wins[3]
        previous = bank.previous.read(bank.RESULT)
        require(previous.get(key) == result.get(key), 'comparison changed while running')
        previous[key] = row
        bank.stage.write(bank.RESULT, previous)


if __name__ == '__main__':
    trial = HEAD_MAJOR if sys.argv[1:2] == ['--head-major'] else K_LANES
    action, = sys.argv[2:] if trial == HEAD_MAJOR else sys.argv[1:]
    require(action in ('check', *STAGES), 'select a planned component stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        execute(action, trial)
