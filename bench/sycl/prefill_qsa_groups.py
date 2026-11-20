"""Measure the actual long-prefill selector and ordered attention with two ID encodings."""

import fcntl
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_mixed_gather_measure as device
from official_model_sources import merge_delta

bank, require = device.bank, device.require
BASE = '43cac31ede2a7482689ff9d60a0fadf2c08e1b4e'
FIRST_BUILD = 'ce7e64c4d53e0dbb7b8cd09570ae0b0b7af30bfa'
SOURCE = '5c02d5b0a16eb7b86871e40a4fbd3eb17486d5bb'
ROOT = bank.B / 'prefill-qsa-groups-v2'
KEY = 'prefill_qsa_groups_v2'
TARGET = 'sycl_qsa_bf16_prefix_ordered_exp_bench'
BENCH = 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
SELECT = 'src/kernels/sycl/qsa_select.cpp'
ATTEND = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
HEADERS = ('include/strata/prefill/qsa_group_ids.hpp', 'include/strata/kernels/qsa_prefill_groups.hpp',
           'bench/sycl/qsa_prefill_groups/selection_fixture.hpp')
STAGES = ('build', 'smoke', '0', '1', '2')


def recipe(helper, native):
    src, build = helper.SOURCE, helper.BUILD
    for path in (ATTEND, SELECT, BENCH):
        require(native['source_files'][path] == bank.sha(src / path), 'changed build input: ' + path)
    require((src / SELECT).read_bytes() == helper.git_file(BASE, SELECT), 'selector differs from measured parent')
    require(all(helper.git_file(SOURCE, name) == helper.git_file(FIRST_BUILD, name)
                for name in (SELECT, ATTEND, *HEADERS[:2])), 'only the fixture may change for the sliced comparison')
    caller = (src/'src/prefill/prefill.cpp').read_text()
    planner = (src/'include/strata/prefill/scratch_allocation.hpp').read_text()
    require('int64_t sel_batch = 256, max_blocks = 0;' in caller and
            'for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch)' in caller and
            'uint64_t(cells / index_block) + 2' in planner, 'production selector storage or launch shape changed')
    original = (src / ATTEND).read_bytes()
    projected = merge_delta(original, ATTEND, BASE, SOURCE)
    require(merge_delta(projected, ATTEND, SOURCE, BASE) == original, 'unrelated attention source change')
    overlay = {name: helper.git_file(SOURCE, name) for name in (*HEADERS, BENCH, SELECT)}
    overlay[ATTEND] = projected
    db = bank.previous.read(build / 'compile_commands.json')
    commands, objects, old = [], {}, {}
    for name in (SELECT, ATTEND, BENCH):
        target = TARGET if name == BENCH else 'strata_kernels'
        unit, = [u for u in db if u['file'] == str(src / name) and
                 u['output'].startswith(f'CMakeFiles/{target}.dir/')]
        args = shlex.split(unit['command'])
        require(unit['directory'] == str(build) and args[-4:] == ['-o', unit['output'], '-c', unit['file']]
                and '-fsycl' in args and '-ffp-contract=off' in args, 'wrong native recipe: ' + name)
        if name == BENCH:
            args[args.index('-DSTRATA_QSA_BENCH_ORDERED_EXP=1')] = '-DSTRATA_QSA_PREFILL_GROUPS_COMPONENT=1'
        obj = ROOT / Path(unit['output']).name
        old[name], objects[name] = unit['output'], str(obj)
        commands.append([args[0], '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'bench/sycl'),
                         *args[1:-4], '-MD', '-MF', str(obj) + '.d', '-o', str(obj), '-c', str(ROOT / name)])

    archive = 'libstrata_kernels.a'
    members = subprocess.check_output(['/usr/bin/ar', 't', archive], cwd=build, text=True, timeout=20).splitlines()
    paths = [Path(p) for p in native['actual_inputs']['objects'] if Path(p).is_relative_to(build/'CMakeFiles/strata_kernels.dir')]
    original_members = {p.name: str(p) for p in paths}
    require(len(paths) == len(original_members) == len(members) and set(members) == original_members.keys(),
            'unexpected kernel archive members')
    for name in (SELECT, ATTEND):
        member = Path(old[name]).name
        require(original_members[member] == str(build/old[name]), 'wrong kernel archive input')
        original_members[member] = objects[name]
    commands.append(['/usr/bin/ar', 'rcsD', str(ROOT/archive), *(original_members[m] for m in members)])
    link = shlex.split(subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', TARGET],
                       cwd=build, text=True, timeout=20).splitlines()[-1])
    replacements = {old[BENCH]: objects[BENCH], archive: str(ROOT/archive), TARGET: str(ROOT/TARGET)}
    require(link[:2] == [':', '&&'] and link[-2:] == ['&&', ':'] and '-Wl,-t' in link
            and link[link.index('-o') + 1] == TARGET and all(link.count(k) == 1 for k in replacements),
            'unexpected component link')
    commands.append([link[2], '-v', *(replacements.get(a, a) for a in link[3:-2])])
    return overlay, commands


def samples(stdout, stage):
    smoke = stage == 'smoke'
    ordinal = 0 if smoke else int(stage)
    geometry = ((97, 16, 33), (401, 259, 33)) if smoke else ((39260, 6491, 2051),)
    lines = stdout.splitlines()
    timer, = [s for s in lines if s.startswith('timer ')]
    median_ns, p99_ns = re.fullmatch(r'timer calls=10000 median_ns=([.\d]+) p99_ns=(\d+)', timer).groups()
    require(0 <= float(median_ns) <= int(p99_ns) < 5000 and float(median_ns) < 1000, 'clock overhead')
    require(sum(line.startswith('sample ') for line in lines) == 12 * len(geometry)
            and sum(line.startswith('selector_geometry ') for line in lines) == len(geometry),
            'unexpected sample or geometry count')
    for cells, queries, cap in geometry:
        marker = (f'selector_geometry queries={queries} subbatches={(queries+255)//256} '
                  f'maximum_rows={min(queries,256)} last_rows={(queries-1)%256+1} score_stride=65538')
        require(lines.count(marker) == 1, 'selector slicing or score capacity differs')
        prefix = f'sample process={ordinal} cells={cells} queries={queries} cap={cap} pattern=gather '
        observed = [line for line in lines if line.startswith(prefix)]
        require(len(observed) == 12 and lines.count(f'case cells={cells} queries={queries} cap={cap} page_size=4 '
                'pattern=gather samples=12 full_bits=PASS guards=PASS inputs=PASS status=PASS') == 1, 'missing exact outputs')
        pairs = []
        for rnd in range(6):
            pair = {}
            for pos in range(2):
                arm = (ordinal + rnd + pos) % 2
                name = 'groups' if arm else 'explicit'
                start = prefix + f'round={rnd} order={pos} variant={arm} implementation={name} milliseconds='
                m = re.fullmatch(re.escape(start) + r'(\S+)' + f' compared={queries*24*256} exact_bits=PASS guards=PASS status=PASS',
                                 observed[2 * rnd + pos])
                require(m, 'wrong producer/consumer, sample or pair order')
                ms = float(m[1])
                require(math.isfinite(ms) and 0 < ms < 60000 and
                        (smoke or (ms >= 10 and int(p99_ns) < ms * 10000)), 'ineligible timed interval')
                pair[name] = ms
            pairs.append(pair)
    require(lines.count('fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED') == 1,
            'missing fixture completion')
    return dict(shapes=geometry, timer_p99_ns=int(p99_ns),
                **({} if smoke else dict(pairs_ms=pairs, median_paired_ratio=statistics.median(
                    p['groups']/p['explicit'] for p in pairs))))


def measure(action, binary, native, speed, stage):
    directory = ROOT / ('measure-' + action)
    directory.mkdir(mode=0o700)
    for key in ('tmp', 'cache'):
        (directory/key).mkdir(mode=0o700)
    owner = device.owner.Owner(directory/'owned', lambda: bank.stage.write(directory/'execution.json', owner.commands))
    seed, probe = speed['environment_template'], speed['native_validation_probe']
    env = {k: v.replace(seed['STRATA_MEASUREMENT_ROOT'], str(directory)) for k,v in seed.items()}
    for k in ('STRATA_TRACE', 'LD_PRELOAD', 'LD_DEBUG', 'STRATA_UR_PROBE_EXPECTED_LOADER',
              'UR_ENABLE_LAYERS', 'UR_LOG_TRACING', 'UR_LOG_VALIDATION'):
        env.pop(k, None)
    env.update(STRATA_TEST_MAIN_GATES='0', STRATA_SYCL_SELECT_RADIX4='0', STRATA_SYCL_SELECT_CYCLIC='0')
    smoke = action == 'smoke'
    if smoke:
        env.update(LD_DEBUG='libs', LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                   UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION', UR_LOG_TRACING='level:info;flush:info;output:stderr',
                   UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(directory/'validation.log'))
    cursor, = [l.removeprefix('-- cursor: ') for l in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if l.startswith('-- cursor: ')]
    argv = [binary['path'], *(['--smoke'] if smoke else ['--component', action])]
    stage.update(environment_sha256=bank.signature(env), argv=argv)
    try:
        cmd, out, err = owner.run(action, argv, env, bank.P, **(bank.device.SMALL if smoke else bank.device.LARGE))
        require(device.owner.closed(cmd) and cmd['exit_code'] == 0, 'QSA fixture failed: ' + err.read_text()[-900:])
        stdout, log = out.read_text(), err.read_text()
        gpu, = re.findall(r'^device=(.+) driver=(\S+)$', stdout, re.M)
        require('B570' in gpu[0], 'wrong GPU')
        if smoke:
            evidence = device.validation.preload_evidence(log, cmd['pid'])
            layer = directory/'validation.log'
            require(evidence['loader'] == probe['loader']['path'] and (not layer.exists() or layer.stat().st_size == 0)
                    and 'urEnqueueKernelLaunch' in log, 'missing trace or validation error')
            bank.device.diagnostics.diagnostics(log, cmd['pid'])
            stage['providers'] = bank.device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
                                                              runtime_manifest=native['foundation']['runtime_manifest']))
        else:
            require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|cleanup failed|terminate called|strata UR validation:',log),
                    'unexpected observer or runtime error')
        stage.update(**samples(stdout, action), native_exit_code=0, host_probe_accepted=True, device=gpu[0], driver=gpu[1])
    finally:
        device.validation.finish(stage, owner, cursor, speed['boot'])
        stage['passed'] = stage['passed'] and stage.get('driver_after') == native['driver']


def execute(action):
    native, speed, parent, build = device.coherent_inputs()
    helper = bank.basis.core.source.device.load_module('qsa_groups_build', device.fixture.__file__)
    helper.SOURCE, helper.BUILD = parent/'source', parent/'build'
    overlay, commands = recipe(helper, native)
    allowed = helper.build_inputs(native, ROOT, overlay, commands)
    controls = [str(Path(m).resolve().relative_to(bank.P)) for m in
                (__file__, device.__file__, device.fixture.__file__, device.coherent_model.__file__, device.coherent_model.kv.__file__)]
    pins = bank.stage.source.pin(bank.P, [*controls, 'tools/sycl'])
    identity = dict(source=SOURCE, parent_build=build['build_sha256'], overlay=bank.stage.hashed(overlay),
                    recipe=bank.signature(commands), controls=pins['aggregate_sha256'],
                    benchmark_environment_sha256=bank.signature(speed['environment_template']), runtime=speed['runtime_manifest']['sha256'])
    if action == 'recipe':
        print(f'{len(overlay)} pinned sources, {len(commands)} build commands, reversible attention projection')
        return
    result = device.validation.committed(bank.RESULT)
    if action == 'build':
        require(KEY not in result and not ROOT.exists(), 'inspect existing QSA generation')
        row = dict(**identity, controller_commit=pins['revision'], model_tested=False, stages=[])
    else:
        row = result[KEY]
        require(all(row[k] == v for k,v in identity.items()) and
                [s['stage'] for s in row['stages']] == list(STAGES[:STAGES.index(action)])
                and all(s['passed'] and s['ownership_closed'] for s in row['stages']), 'previous gate or input changed')
    previous = bank.signature(result.get(KEY))
    stage = dict(stage=action, passed=False)
    try:
        if action == 'build':
            helper.build_fixture(ROOT, TARGET, native, overlay, commands, allowed, identity)
            record = bank.previous.read(ROOT/'record.json')
            require(record['ownership_closed'] and record['complete'], 'incomplete QSA build')
            stage.update(passed=True, build_sha256=bank.sha(ROOT/'record.json'), binary=record['binary'])
        else:
            record = bank.previous.read(ROOT/'record.json')
            built = row['stages'][0]
            require(record['binary'] == built['binary'] and bank.sha(ROOT/'record.json') == built['build_sha256']
                    and bank.sha(record['binary']['path']) == record['binary']['sha256']
                    and all(bank.sha(p) == h for k in ('compiled','linked') for p,h in record[k].items())
                    and all(bank.sha(ROOT/p) == h for p,h in record['driver_inputs'].items()), 'QSA build or executable changed')
            measure(action, record['binary'], native, speed, stage)
            require(stage['passed'], 'post-run fault, driver or ownership failure')
    except BaseException as exc:
        stage['error'] = repr(exc)
        raise
    finally:
        if action == 'build' and (ROOT/'record.json').is_file():
            stage['ownership_closed'] = bank.previous.read(ROOT/'record.json')['ownership_closed']
        row['stages'].append(stage)
        if action == '2' and stage['passed']:
            row['component_performance_gate'] = all(s['median_paired_ratio'] < 1 for s in row['stages'][-3:])
        current = bank.previous.read(bank.RESULT)
        require(bank.signature(current.get(KEY)) == previous, 'QSA result changed during execution')
        current[KEY] = row
        bank.stage.write(bank.RESULT, current)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in (*STAGES, 'recipe') and sys.executable == bank.basis.reference.model.PYTHON,
            'select one pinned QSA stage')
    if action == 'recipe':
        execute(action)
    else:
        with (bank.B/'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bank.builder.guards.no_model_owners([])
            execute(action)
