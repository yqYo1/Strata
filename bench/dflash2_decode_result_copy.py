"""Measure mapped reads versus an ordered copy in the same T4 combine graph."""

import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_primary_retained as retained

bank, basis, require = retained.bank, retained.basis, retained.require
sha, write = bank.sha, bank.stage.write
PRIMARY = retained.PRIMARY.ROOT
KEY = 'dflash2_decode_result_copy_v1'
ROOT = bank.B / KEY.replace('_', '-')
BUILD = ROOT / 'fixture-build'
TARGET = 'sycl_decode_moe_combine_test'
FIXTURE = 'tests/sycl/decode_moe_combine.cpp'
REVISION = 'c4bf3b970d842f9d8dedc37887812b6f20fd8c6a'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
LINK_DRIVER = 'bench/sycl/build_prefill_mixed_gather.py'
LIBRARIES = ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')
ORDER = ('validate', 'time-forward', 'time-reverse', 'time-forward')
HITS = (4, 10, 20)
REPLAYS, BATCHES, COPY_BYTES = 512, 4, 409600
SUMMARY = 'decode MoE combine numerical=PASS guards=PASS nodes=1/1 completed=1'
LIMITS = dict(wall=180, cpu=600, rss_cap=4 << 30, text_cap=24 << 20,
              file_cap=32 << 20, total_cap=64 << 20)
PLAN = dict(order=list(ORDER), hits=list(HITS), replays=REPLAYS, paired_batches=BATCHES,
            copy_bytes_per_replay=COPY_BYTES, budgets=LIMITS,
            scope='Same fused T4 reduction, inputs, hit map and ordered queue. Compare mapped reads '
                  'with one complete pinned-host copy to device storage followed by the same graph.',
            acceptance='Bit-exact scalar oracle, guards, poisoned source and invalid destination list '
                       'before timing. Each batch must last at least 10 ms with at most 1% empty-clock cost. '
                       'Report ratios per hit count and independent process. Keep all samples.',
            decision='Advance a hit count only if every independent process has smaller median '
                     'completion time both from launch and from CPU publication. This is component evidence.',
            retention='Retire this fixture build and raw output at the comparison closure; '
                      'keep the independently validated serving engine unchanged.')


def inputs(revision):
    selected = retained.baseline()
    parent = bank.previous.read(PRIMARY / 'record.json')
    fixture = basis.validation.git('show', REVISION + ':' + FIXTURE)
    require(fixture == (bank.P / FIXTURE).read_bytes()
            == basis.validation.git('show', revision + ':' + FIXTURE), 'exact copy fixture source required')
    paths = (THIS, retained.THIS, 'bench/sycl', 'tools/sycl', *bank.basis.POLICIES)
    pins = bank.stage.source.pin(bank.P, paths)['source_sha256']
    official = basis.core.source.device.load_module('decode_result_copy_controls',
                                                    bank.P / 'tools/sycl/build_official_model.py')
    pins.update(official.controls(revision, controller=bank.P / THIS))
    original = dict(bank.stage.source.blobs(bank.P, revision, pins))
    require(bank.stage.hashed(original) == pins, 'fixture controller or helper changed')

    parent_dir = PRIMARY / 'build'
    source = PRIMARY / 'source' / FIXTURE
    database = parent_dir / 'compile_commands.json'
    unit, = [u for u in json.loads(database.read_text()) if u['file'] == str(source)]
    compiler = shlex.split(unit['command'])
    obj = unit['output']
    require(Path(unit['directory']) == parent_dir and compiler[-4:] == ['-o', obj, '-c', str(source)],
            'unexpected CMake fixture compile command')
    object_path = BUILD / Path(obj).name
    command = [*compiler[:-4], '-MD', '-MF', str(object_path) + '.d',
               '-o', str(object_path), '-c', str(BUILD / FIXTURE)]
    ninja = '/usr/bin/ninja'
    require(sha(ninja) == parent['build_tools'][ninja], 'selected Ninja tool changed')
    output = subprocess.check_output([ninja, '-t', 'commands', TARGET],
                                     cwd=parent_dir, text=True, timeout=20)
    link = shlex.split(output.splitlines()[-1])
    require(link[:2] == [':', '&&'] and link[-2:] == ['&&', ':']
            and link[2] == compiler[0] and link.count(obj) == 1
            and link[link.index('-o') + 1] == TARGET and '-Wl,-t' in link
            and tuple(p for p in link if p.endswith('.a')) == LIBRARIES,
            'unexpected fixture object or CMake link recipe')
    replacements = {obj: str(object_path), TARGET: str(BUILD / TARGET)}
    commands = [command, [link[2], '-v', *(replacements.get(word, word) for word in link[3:-2])]]
    archives = {str(parent_dir / name): parent['actual_inputs']['produced'][str(parent_dir / name)]
                for name in LIBRARIES}
    require(all(sha(p) == h for p, h in archives.items()), 'selected archive bytes changed')
    fields = {k: selected[k] for k in ('boot', 'driver', 'native_source_commit', 'native_build_sha256',
                                        'native_validation_probe', 'runtime_manifest')}
    fields.update(source_commit=revision, source_revision=REVISION, fixture_sha256=sha(bank.P / FIXTURE),
                  controller_sha256=bank.signature(pins), compile_database_sha256=sha(database),
                  commands=commands, archives=archives, plan=PLAN,
                  parent_environment_sha256=bank.signature(parent['recipe']['environment']))
    return fields, parent, fixture


def save(row):
    write(ROOT / 'record.json', row)
    entries = bank.previous.read(bank.RESULT)
    entries[KEY] = row
    write(bank.RESULT, entries)


def current(row):
    require(basis.validation.committed(bank.RESULT, (KEY,)) == row
            and not row['active'] and row['ownership_closed'], 'commit the preceding fixture stage')
    fields, parent, fixture = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items()), 'the frozen component recipe changed')
    return parent, fixture


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
            'one fresh copy component comparison required')
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    fields, _, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    save(dict(fields, stage='prepared', processes=[], complete=False, passed=False,
              ownership_closed=True, active=False))


def build(row, parent, fixture):
    require(row['stage'] == 'prepared' and not BUILD.exists(), 'one fresh fixture link required')
    # This helper verifies actual compiler headers, linker inputs and temporary
    # host slices. Reuse the already verified archives without rebuilding them.
    driver = basis.core.source.device.load_module('decode_result_copy_link', bank.P / LINK_DRIVER)
    driver.BUILD, driver.SOURCE = PRIMARY / 'build', PRIMARY / 'source'
    overlay = {FIXTURE: fixture}
    try:
        allowed = driver.build_inputs(parent, BUILD, overlay, row['commands'])
        driver.build_fixture(BUILD, TARGET, parent, overlay, row['commands'], allowed,
                             dict(fixture_source_commit=REVISION, fixture_sha256=row['fixture_sha256']))
        receipt = bank.previous.read(BUILD / 'record.json')
        require(receipt['complete'] and receipt['ownership_closed'] and receipt['commands'] == row['commands']
                and all(receipt['linked'].get(p) == h for p, h in row['archives'].items()),
                'fixture link did not consume the qualified archives')
        row.update(stage='built', binary=receipt['binary'], build_receipt_sha256=sha(BUILD / 'record.json'),
                   binary_bytes=Path(receipt['binary']['path']).stat().st_size,
                   compiled_inputs=len(receipt['compiled']), linked_inputs=len(receipt['linked']))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        if (BUILD / 'record.json').exists():
            row['ownership_closed'] = bank.previous.read(BUILD / 'record.json')['ownership_closed']
        raise
    finally:
        save(row)


def check_binary(row):
    path = BUILD / 'record.json'
    require(sha(path) == row['build_receipt_sha256'], 'fixture build receipt changed')
    proof = bank.previous.read(path)
    binary = Path(row['binary']['path'])
    require(proof['binary'] == row['binary'] and binary.stat().st_size == row['binary_bytes']
            and sha(binary) == row['binary']['sha256']
            and all(sha(p) == h for group in ('compiled', 'linked') for p, h in proof[group].items()),
            'fixture executable or actual build input changed')


def samples(text, mode):
    lines = text.strip().splitlines()
    require(lines and lines[-1] == SUMMARY, 'missing fixture oracle and guard proof')
    if mode == 'validate':
        require(lines == [SUMMARY], 'unexpected validation output')
        return []
    pattern = (r'decode MoE combine hits=(\d+) batch=(\d+) arm=([01]) replays=512 '
               r'ms=(\d+\.\d+) full_ms=(\d+\.\d+) empty_ms=(\d+\.\d+) cpu_copy_bytes_per_replay=(\d+)')
    parsed = []
    for line in lines[:-1]:
        match = re.fullmatch(pattern, line)
        require(match is not None, 'invalid fixture timing row')
        hit, batch, arm = map(int, match.groups()[:3])
        elapsed, full, empty = map(float, match.groups()[3:6])
        size = int(match[7])
        require(math.isfinite(full) and size == (COPY_BYTES if arm else 0)
                and full >= elapsed >= 10 and empty <= elapsed * .01,
                'timing boundary or requested copy bytes differ')
        parsed.append(dict(hits=hit, batch=batch, arm=arm, ms=elapsed, full_ms=full,
                           empty_ms=empty, copy_bytes_per_replay=size))
    reverse = int(mode == 'time-reverse')
    expected = [(h, b, a) for h in HITS for b in range(BATCHES)
                for a in (reverse ^ (b % 2), 1 ^ reverse ^ (b % 2))]
    require([(s['hits'], s['batch'], s['arm']) for s in parsed] == expected, 'incomplete alternating timing pairs')
    return parsed


def device_case(row, parent, index):
    check_binary(row)
    mode = ORDER[index]
    case = ROOT / str(index)
    case.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (case / name).mkdir(mode=0o700)
    probe = row['native_validation_probe']
    env = dict(parent['recipe']['environment'], TMPDIR=str(case / 'tmp'), XDG_CACHE_HOME=str(case / 'cache'),
               LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               LD_DEBUG='libs', UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(case / 'validation.log'))
    env.pop('UR_LOG_TRACING', None)
    env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    if mode == 'validate':
        env['UR_ENABLE_LAYERS'] = 'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION'
        env['UR_LOG_TRACING'] = 'level:info;flush:info;output:stderr'
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['NEOReadDebugKeys'] == '1'
            and env['EnableDirectSubmission'] == '0', 'runtime safety settings changed')
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    owner = bank.builder.ownership.Owner(case / 'owned', lambda: write(case / 'ownership.json', owner.commands))
    command = [row['binary']['path'], mode, '--compare-copy']
    result = dict(index=index, mode=mode, command=command, environment_sha256=bank.signature(env),
                  ownership_closed=False)
    row['processes'].append(result)
    try:
        entry, out, err = owner.run(mode, command, env, case, **LIMITS)
        errors = err.read_text()
        require(bank.builder.ownership.closed(entry) and entry['exit_code'] == 0,
                'fixture execution failed: ' + errors[-1200:])
        validation = basis.validation.preload_evidence(errors, entry['pid'])
        warning = case / 'validation.log'
        require(validation['loader'] == probe['loader']['path']
                and (not warning.exists() or warning.stat().st_size == 0)
                and (mode != 'validate' or 'urEnqueue' in errors), 'incomplete runtime or API validation')
        result['providers'] = bank.device.loaded_providers(errors, dict(row, environment=env))
        result['samples'] = samples(out.read_text(), mode)
        leader, = [item for item in entry['owners'] if item['pid'] == entry['pid']]
        result.update(host_probe_accepted=True, validation=validation, native_identity=leader)
    except Exception as error:
        result['error'] = repr(error)
    finally:
        basis.validation.finish(result, owner, cursor, row['boot'])
        result['passed'] &= result.get('driver_after') == row['driver']
        write(case / 'outcome.json', result)
    return result


def measure(row, parent, action):
    start = len(row['processes'])
    indices = range(1, len(ORDER)) if action == 'time' else range(1)
    require((row['stage'], start) == (('validated', 1) if action == 'time' else ('built', 0)),
            'commit the built fixture and device validation in order')
    try:
        for index in indices:
            result = device_case(row, parent, index)
            require(result['passed'], result.get('error', 'fixture device or ownership gate failed'))
        if action == 'validate':
            row['stage'] = 'validated'
            return
        owners = {(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in row['processes']}
        require(len(owners) == len(ORDER), 'fixture comparisons need independent processes')
        ratios = {}
        for hit in HITS:
            ratios[str(hit)] = {}
            for field in ('ms', 'full_ms'):
                ratios[str(hit)][field] = []
                for p in row['processes'][1:]:
                    base, new = [statistics.median(s[field] for s in p['samples'] if s['hits'] == hit and s['arm'] == arm)
                                 for arm in (0, 1)]
                    ratios[str(hit)][field].append(new / base)
        eligible = [int(h) for h, metrics in ratios.items() if all(r < 1 for r in (*metrics['ms'], *metrics['full_ms']))]
        row.update(stage='closed', complete=True, passed=True, candidate_over_control=ratios,
                   eligible_hit_counts=eligible, model_speed_qualified=False)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row['ownership_closed'] = all(p['ownership_closed'] for p in row['processes'])
        save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'build', 'validate', 'time'),
            'use the pinned interpreter and one component stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            row = bank.previous.read(ROOT / 'record.json')
            parent, fixture = current(row)
            if action == 'build':
                build(row, parent, fixture)
            else:
                measure(row, parent, action)
