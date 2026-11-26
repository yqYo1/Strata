"""Measure the existing scalar and cooperative KV resolvers at verifier width three."""

import fcntl
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/sycl'))
import runtime_probe
import dflash2_primary_retained as retained

bank, basis, require = retained.bank, retained.basis, retained.require
sha, write, validation = bank.sha, bank.stage.write, basis.validation
owner_api = bank.builder.ownership
PRIMARY = retained.PRIMARY.ROOT
KEY = 'decode_kv_width3_v1'
ROOT = bank.B / KEY.replace('_', '-')
BUILD = ROOT / 'fixture-build'
TARGET = 'sycl_kv_copy_groups_bench'
FIXTURE = 'tools/sycl/kv_copy_groups_bench.cpp'
RESOLVERS = ('src/kernels/sycl/kv_stream.cpp', 'src/kernels/sycl/decode_kv_stream.cpp')
CONTRACT = (*RESOLVERS, 'tools/sycl/kv_copy_groups_oracle.hpp',
            'include/strata/kernels/kv_stream.hpp', 'include/strata/kernels/kv_stream_copy_groups.hpp')
LINK_DRIVER = 'bench/sycl/build_prefill_mixed_gather.py'
ARCHIVE = 'libstrata_sycl_runtime.a'
ARMS = ('scalar', 'cooperative', 'cooperative', 'scalar', 'scalar', 'cooperative')
MISSES = (0, 384, 1539)
STAGES = ('smoke', 'validate', *(str(i) for i in range(len(ARMS))))
LIMITS = dict(cpu=1800, rss_cap=4 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)
WALL = dict(smoke=120, validate=600, benchmark=900)
PLAN = dict(
    stages=list(STAGES), arms=list(ARMS), misses=list(MISSES), samples_per_arm=9,
    scope='One Arc fixture. Change only the width-three resolver for the captured resolve and copy graph. '
          'Both arms use K8V8, 65,536 logical pages, 5,120 resident pages and capacity 2,051 per query.',
    validation='Trace the small direct and graph cases. Repeat the full-shape graph with parameter validation. '
               'Check ordered maps, counters, payloads, guards, physical last cell and malformed width.',
    interval='Sum of isolated graph launch and completion intervals with completed map resets outside timing. '
             'Charge all five timer calls. No API trace during timing.',
    decision='Require all three paired process medians to improve at each miss count before model dispatch work.',
    wall_seconds=WALL, limits=LIMITS,
    retention='Retire the private fixture and routine logs after the result and build recovery are verified. '
              'This comparison does not consume or replace another serving binary.')


def inputs(revision):
    live = retained.baseline()
    parent = bank.previous.read(PRIMARY / 'record.json')
    driver = basis.core.source.device.load_module('decode_kv_width3_link', bank.P / LINK_DRIVER)
    controls = {FIXTURE, LINK_DRIVER, Path(__file__).resolve().relative_to(bank.P).as_posix(),
                'tools/sycl/measurement_environment.json'}
    for module in tuple(sys.modules.values()):
        origin = getattr(module, '__file__', None)
        if origin and (path := Path(origin).resolve()).is_relative_to(bank.P) and path.suffix == '.py':
            controls.add(path.relative_to(bank.P).as_posix())
    files = dict(bank.stage.source.blobs(bank.P, revision, sorted(controls)))
    require(files.keys() == controls and all((bank.P / p).read_bytes() == raw for p, raw in files.items()),
            'commit the exact fixture and its controllers')
    original = dict(bank.stage.source.blobs(bank.P, live['native_source_commit'], CONTRACT))
    require(original.keys() == set(CONTRACT)
            and all((PRIMARY / 'source' / p).read_bytes() == (bank.P / p).read_bytes() == raw
                for p, raw in original.items()), 'retained resolver or fixture contracts changed')

    parent_build = PRIMARY / 'build'
    database = parent_build / 'compile_commands.json'
    units = json.loads(database.read_bytes())
    replacements, commands = {}, []
    for name in (FIXTURE, *RESOLVERS):
        source = str(PRIMARY / 'source' / name)
        unit, = [u for u in units if u['file'] == source and f'{TARGET}.dir/' in u['output']]
        argv = shlex.split(unit['command'])
        obj = unit['output']
        require(Path(unit['directory']) == parent_build and argv[-4:] == ['-o', obj, '-c', source]
                and all(flag in argv for flag in ('-fsycl', '-fno-fast-math', '-ffp-contract=off'))
                and not any(flag in argv for flag in ('-ffast-math', '-Ofast')),
                'unexpected retained KV compiler command')
        dest = str(BUILD / Path(obj).name)
        replacements[obj] = dest
        quote = []
        if name == FIXTURE:
            source = str(BUILD / FIXTURE)
            quote = ['-iquote', str(PRIMARY / 'source' / 'tools/sycl')]
        commands.append([*argv[:-4], *quote, '-MD', '-MF', dest + '.d', '-o', dest, '-c', source])
    ninja = '/usr/bin/ninja'
    require(sha(ninja) == parent['build_tools'][ninja], 'retained Ninja changed')
    lines = subprocess.check_output([ninja, '-t', 'commands', TARGET], cwd=parent_build, text=True, timeout=20)
    link = shlex.split(lines.splitlines()[-1])
    require(link[:2] == [':', '&&'] and link[-2:] == ['&&', ':'] and '-Wl,-t' in link
            and all(link.count(obj) == 1 for obj in replacements) and link[link.index('-o') + 1] == TARGET
            and all(argv[0] == link[2] for argv in commands)
            and [p for p in link if p.endswith('.a')] == [ARCHIVE], 'unexpected component link recipe')
    replacements[TARGET] = str(BUILD / TARGET)
    commands.append([link[2], '-v', *(replacements.get(x, x) for x in link[3:-2])])

    archive = str(parent_build / ARCHIVE)
    archive_sha = parent['actual_inputs']['produced'][archive]
    require(sha(archive) == archive_sha, 'retained runtime archive changed')
    fields = {k: live[k] for k in ('boot', 'driver', 'native_source_commit',
                                    'native_build_sha256', 'native_validation_probe', 'runtime_manifest')}
    fields.update(source_commit=revision, sources=bank.stage.hashed(files),
                  resolver_source=bank.stage.hashed(original), compile_database_sha256=sha(database),
                  parent_environment_sha256=bank.signature(parent['recipe']['environment']),
                  driver_library=live['environment_template']['ZE_ENABLE_ALT_DRIVERS'],
                  archives={archive: archive_sha}, commands=commands, plan=PLAN)
    driver.BUILD, driver.SOURCE = parent_build, PRIMARY / 'source'
    return fields, parent, driver, {FIXTURE: files[FIXTURE]}


def save(row):
    write(ROOT / 'record.json', row)
    results = bank.previous.read(bank.RESULT)
    results[KEY] = row
    write(bank.RESULT, results)


def build(row, parent, driver, overlay):
    require(row['stage'] == 'prepared' and not BUILD.exists(), 'one fresh component build required')
    allowed = driver.build_inputs(parent, BUILD, overlay, row['commands'])
    driver.build_fixture(BUILD, TARGET, parent, overlay, row['commands'], allowed,
                         dict(source_commit=row['source_commit'], overlay_sha256=bank.stage.hashed(overlay)))
    receipt = bank.previous.read(BUILD / 'record.json')
    require(receipt['complete'] and receipt['ownership_closed'] and receipt['commands'] == row['commands']
            and all(receipt['linked'].get(p) == h for p, h in row['archives'].items()),
            'fixture link did not consume the qualified runtime archive')
    binary = receipt['binary']
    row.update(stage='built', binary=dict(binary, bytes=Path(binary['path']).stat().st_size),
               build_receipt_sha256=sha(BUILD / 'record.json'),
               compiled_inputs=len(receipt['compiled']), linked_inputs=len(receipt['linked']))


def check_binary(row):
    receipt_path = BUILD / 'record.json'
    require(sha(receipt_path) == row['build_receipt_sha256'], 'fixture build receipt changed')
    receipt = bank.previous.read(receipt_path)
    binary = row['binary']
    require(receipt['binary'] == {k: binary[k] for k in ('path', 'sha256')}
            and Path(binary['path']).stat().st_size == binary['bytes'] and sha(binary['path']) == binary['sha256']
            and all(sha(p) == h for group in ('compiled', 'linked') for p, h in receipt[group].items()),
            'fixture executable or actual build inputs changed')


def observations(lines, stage):
    if stage == 'smoke':
        require(lines == ['planner_width3_smoke=PASS direct_cases=2 graph_replays=2'],
                'incomplete small width-three fixture')
        return dict(direct_cases=2, graph_replays=2)
    if stage == 'validate':
        recorded = [json.loads(line) for line in lines]
        graphs = recorded[:10]
        expected_graphs = [(c, p) for c in (False, True) for p in range(5)]
        require([(s['cooperative'], s['replay']) for s in graphs] == expected_graphs
                and all(type(s['cooperative']) is bool and type(s['fingerprint_fnv64']) is int
                        and s['phase'] == 'planner_context_graph' and s['fmt'] == 1 and s['nq'] == 3
                        and s['last_cell'] == 262143 and s['oracle_exact'] is True for s in graphs)
                and [s['fingerprint_fnv64'] for s in graphs[:5]] == [s['fingerprint_fnv64'] for s in graphs[5:]],
                'missing or mismatched width-three graph parity')
        expected = [dict(phase='planner_perf_cases', nq=3, missing=m, cooperative=c,
                         replays=2, oracle_exact=True) for m in MISSES for c in (False, True)]
        require(recorded[10:] == expected, 'miss-case validation census differs')
        return dict(graph_fingerprints=[s['fingerprint_fnv64'] for s in graphs[:5]], exact_miss_cases=expected)
    arm = ARMS[int(stage)]
    marker = ('planner_benchmark=PASS samples=9 timing=sum_of_isolated_graph_launch_and_wait '
              'reset_and_oracle=excluded physical_bus_bytes=unknown')
    require(len(lines) == 10 and lines[-1] == marker, 'incomplete width-three timings')
    samples = [json.loads(line) for line in lines[:-1]]
    for s, (missing, rep) in zip(samples, ((m, i) for m in MISSES for i in range(3)), strict=True):
        calls = s['calls']
        require(s['phase'] == 'planner_sample' and s['cooperative'] is (arm == 'cooperative')
                and (s['nq'], s['missing'], s['rep'], s['unique_pages'], s['graph_nodes']) == (3, missing, rep, 1539, 2)
                and s['oracle_exact'] is True and s['requested_host_payload_bytes_per_call'] == missing * 4224
                and type(calls) is int and 64 <= calls <= 16384 and calls & (calls - 1) == 0,
                'wrong timing arm, geometry or call count')
        for name in ('elapsed_ns', 'ns_per_call', 'clock_pair_ns', 'timer_fraction', 'calibration_ns', 'warmup_ns'):
            require(type(s[name]) in (int, float) and math.isfinite(s[name]) and s[name] > 0,
                    'nonfinite or nonpositive time: ' + name)
        require(s['elapsed_ns'] >= 250_000_000 and s['warmup_ns'] >= 500_000_000
                and s['calibration_ns'] >= 750_000_000 and s['timer_fraction'] <= .01
                and math.isclose(s['ns_per_call'], s['elapsed_ns'] / calls, rel_tol=1e-12)
                and math.isclose(s['timer_fraction'], 5 * s['clock_pair_ns'] * calls / s['elapsed_ns'], rel_tol=1e-12),
                'ineligible shape time or inconsistent timer arithmetic')
    return dict(arm=arm, samples=samples)


def device_case(row, parent, stage):
    check_binary(row)
    root = ROOT / stage
    root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (root / name).mkdir(mode=0o700)
    probe = row['native_validation_probe']
    env = dict(parent['recipe']['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
               LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               ZE_ENABLE_ALT_DRIVERS=row['driver_library'], STRATA_SYCL_KV_COPY_GROUPS='0',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    require(all(env[k] == v for k, v in (
        ('ONEAPI_DEVICE_SELECTOR', 'level_zero:gpu'), ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', '1'),
        ('NEOReadDebugKeys', '1'), ('EnableDirectSubmission', '0'), ('STRATA_SYCL_VERIFY_NATIVE_CAPTURE', '0')
    )), 'qualified fixture runtime settings changed')
    if stage != 'smoke':
        env.pop('UR_LOG_TRACING', None)
        env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    verb = stage if stage in ('smoke', 'validate') else 'benchmark'
    argv = [row['binary']['path'], '--planner-width3-' + verb]
    if verb == 'benchmark':
        argv.append(ARMS[int(stage)])
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    owner = owner_api.Owner(root / 'owned', lambda: write(root / 'ownership.json', owner.commands))
    result = dict(stage=stage, argv=argv, environment_sha256=bank.signature(env))
    providers = {str(Path(p).resolve()): h for p, h in bank.previous.read(Path(row['runtime_manifest']['path'])).items()}
    providers.update({str(Path(a['path']).resolve()): a['sha256'] for a in (*probe.values(), row['binary'])})
    required = {str(Path(p).resolve()) for p in
                (env['UR_ADAPTERS_FORCE_LOAD'], row['driver_library'], probe['loader']['path'], probe['library']['path'])}

    def interact(io, entry):
        deadline = time.monotonic() + WALL[verb] - 25
        lines = []
        while True:
            remaining = deadline - time.monotonic()
            require(remaining > 0 and len(lines) < 32, 'bounded width-three fixture response required')
            line = io.line(remaining)
            if line == 'KV_PLAN_WAIT':
                break
            lines.append(line)
        result.update(observations(lines, stage))
        mapped = runtime_probe.executable_mappings(Path(f'/proc/{entry["pid"]}/maps').read_text())
        require(required <= mapped <= providers.keys() and len(mapped) >= 8 and
                all(sha(p) == providers[p] for p in mapped), 'actual loaded fixture providers differ')
        result['provider_sha256'] = bank.signature({p: providers[p] for p in sorted(mapped)})
        io.send(b'QUIT\n')
        io.proc.stdin.close()
        io.exit(20)

    try:
        entry, _, err = owner.run('fixture', argv, env, root, interaction=interact, wall=WALL[verb], **LIMITS)
        errors = err.read_text()
        warning = root / 'validation.log'
        actual_probe = validation.preload_evidence(errors, entry['pid'])
        errors_seen = set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', errors))
        allowed_errors = {'UR_RESULT_ERROR_INVALID_NULL_HANDLE'} if stage == 'smoke' else set()
        if stage == 'smoke':
            calls = r'^\s*<--- urEnqueueCommandBufferExp\(.* -> UR_RESULT_SUCCESS;'
            count = len(re.findall(calls, errors, re.M))
            require(count == 2, 'expected two successful traced graph replays')
            result['traced_graph_replays'] = count
        require(owner_api.closed(entry) and entry['exit_code'] == 0 and
                'PASS bounded component scope only' in errors and
                errors_seen <= allowed_errors
                and not re.search(r'SYCL exception|^FAIL ', errors, re.M)
                and (not warning.exists() or warning.stat().st_size == 0)
                and Path(actual_probe['loader']).resolve() == Path(probe['loader']['path']).resolve(),
                'fixture completion, trace or parameter validation failed')
        check_binary(row)
        identity, = [item for item in entry['owners'] if item['pid'] == entry['pid']]
        result.update(host_probe_accepted=True, validation=actual_probe, native_identity=identity, native_exit_code=0)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        validation.finish(result, owner, cursor, row['boot'])
        result['passed'] &= result.get('driver_after') == row['driver']
        write(root / 'outcome.json', result)
    return result


def decision(processes):
    ids = {(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}
    require(len(ids) == len(STAGES) and [p['stage'] for p in processes] == list(STAGES),
            'all independent component processes required')
    cells = []
    for missing in MISSES:
        medians = [statistics.median(s['ns_per_call'] for s in p['samples'] if s['missing'] == missing)
                   for p in processes[2:]]
        ratios = [medians[1] / medians[0], medians[2] / medians[3], medians[5] / medians[4]]
        cells.append(dict(missing=missing, process_medians_ns=medians, paired_time_ratios=ratios,
                          faster=all(ratio < 1 for ratio in ratios)))
    return dict(cells=cells, eligible_for_model_check=all(c['faster'] for c in cells))


def main(action):
    if action == 'prepare':
        prior = validation.committed(bank.RESULT, ('dflash2_decode_kv_speed_v1',))
        require(prior['stage'] in ('closed', 'failed') and not prior['active'] and prior['ownership_closed'],
                'close the existing DFlash2 comparison before preparing this fixture')
        require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh width-three run')
        revision = validation.git('rev-parse', 'HEAD').decode().strip()
        fields, _, _, _ = inputs(revision)
        ROOT.mkdir(mode=0o700)
        save(dict(fields, stage='prepared', processes=[], complete=False, passed=False,
                  ownership_closed=True, model_qualified=False))
        return
    row = bank.previous.read(ROOT / 'record.json')
    require(validation.committed(bank.RESULT, (KEY,)) == row and row['ownership_closed'],
            'commit the exact preceding component stage')
    fields, parent, driver, overlay = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items()), 'frozen fixture inputs changed')
    if action != 'build':
        index = len(row['processes'])
        require(index < len(STAGES) and action == STAGES[index]
                and row['stage'] == ('built' if index == 0 else STAGES[index - 1]),
                'run only the next committed case')
    try:
        if action == 'build':
            build(row, parent, driver, overlay)
        else:
            result = device_case(row, parent, action)
            row['processes'].append(result)
            row['ownership_closed'] = result['ownership_closed']
            require(result['passed'], result.get('error', 'device or ownership gate failed'))
            row['stage'] = action
            if action == STAGES[-1]:
                row.update(stage='closed', complete=True, passed=True, decision=decision(row['processes']))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        if action == 'build' and (BUILD / 'record.json').is_file():
            row['ownership_closed'] = bank.previous.read(BUILD / 'record.json')['ownership_closed']
        raise
    finally:
        save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'build', *STAGES),
            'use the pinned interpreter and select the next component stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        main(action)
