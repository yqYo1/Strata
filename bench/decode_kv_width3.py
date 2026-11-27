"""Measure KV resolvers at verifier width three on the retained Arc recipe."""

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
import current_pti_admission as pti
import decode_window_trace
import runtime_probe
import dflash2_primary_retained as retained

bank, basis, require = retained.bank, retained.basis, retained.require
sha, write, validation = bank.sha, bank.stage.write, basis.validation
owner_api = bank.builder.ownership
PRIMARY = retained.PRIMARY.ROOT
KEY = 'decode_kv_width3_v1'
CLOCK_WRAP_KEY = 'decode_kv_clock_wrap_v1'
ROOT = bank.B / KEY.replace('_', '-')
BUILD = ROOT / 'fixture-build'
TARGET = 'sycl_kv_copy_groups_bench'
FIXTURE = 'tools/sycl/kv_copy_groups_bench.cpp'
RESOLVERS = ('src/kernels/sycl/kv_stream.cpp', 'src/kernels/sycl/decode_kv_stream.cpp')
HEADER = 'include/strata/kernels/kv_stream.hpp'
CONTRACT = (*RESOLVERS, 'tools/sycl/kv_copy_groups_oracle.hpp',
            HEADER, 'include/strata/kernels/kv_stream_copy_groups.hpp')
LINK_DRIVER = 'bench/sycl/build_prefill_mixed_gather.py'
ARCHIVE = 'libstrata_sycl_runtime.a'
ARMS = ('scalar', 'cooperative', 'cooperative', 'scalar', 'scalar', 'cooperative')
MISSES = (0, 384, 1539)
STAGES = ('smoke', 'validate', *(str(i) for i in range(len(ARMS))))
TRACE_CASES = {'trace-scalar-0': ('scalar', 0), 'trace-cooperative-0': ('cooperative', 0),
               'trace-cooperative-384': ('cooperative', 384), 'trace-scalar-384': ('scalar', 384)}
TRACE_FLAGS = ('--device-timing', '--chrome-kernel-logging', '--chrome-call-logging',
               '--chrome-event-buffer-size', '128')
LIMITS = dict(cpu=1800, rss_cap=4 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)
WALL = dict(smoke=120, validate=600, benchmark=900, trace=180)
PLAN = dict(
    stages=list(STAGES), arms=list(ARMS), misses=list(MISSES), samples_per_arm=9,
    scope='One Arc fixture. Change only the width-three resolver for the captured resolve and copy graph. '
          'Both arms use K8V8, 65,536 logical pages, 5,120 resident pages and capacity 2,051 per query.',
    validation='Trace the small direct and graph cases. Repeat the full-shape graph with parameter validation. '
               'Check ordered maps, counters, payloads, guards, physical last cell and malformed width.',
    interval='Sum of isolated graph launch and completion intervals with completed map resets outside timing. '
             'Charge all five timer calls. No API trace during timing.',
    decision='Require all three paired process medians to improve at each miss count before model dispatch work.',
    trace=dict(cases=[dict(stage=stage, arm=arm, missing=missing)
                      for stage, (arm, missing) in TRACE_CASES.items()],
               flags=list(TRACE_FLAGS), replays=5, clock='CLOCK_MONOTONIC_RAW',
               criterion='Two ordered, distinct, fully contained native kernels per replay, with stable names '
                         'and no memory copy, fill or boundary-crossing GPU events. Open collection at exec. '
                         'Diagnostic only, independent of the clean timing decision.'),
    wall_seconds=WALL, limits=LIMITS,
    retention='Keep the private fixture through the bounded resolver and copy kernel trace. '
              'Retire the fixture and routine logs after the decisions and build recovery are verified. '
              'Reuse the retained runtime archive and keep the selected serving executable.')
CLOCK_WRAP_PLAN = dict(
    PLAN,
    clock_wrap=True,
    arms=['cooperative', 'clock-wrap', 'clock-wrap', 'cooperative', 'cooperative', 'clock-wrap'],
    misses=[0, 16, 64, 384],
    samples_per_arm=12,
    scope='Width-three K8V8 cooperative reference versus conditional clock-hand wrap. '
          'Keep 65,536 logical pages, 5,120 resident pages, capacity 2,051 and interleaved misses.',
    validation='Trace second-chance wrap, protected overflow and reset of an invalid hand. '
               'Compare exact full-shape maps, payloads and physical last-cell repeats with parameter validation.',
    decision='All three paired process medians must improve at each nonzero miss count. The zero-miss candidate '
             'median must be no larger than the slowest of the three matched zero-miss controls.',
    trace=None,
    retention='Retain the private fixture until its decision and original inputs are recoverable. '
              'Keep the committed batch samples and retire the closed raw output.')


def inputs(revision, plan=PLAN, build_dir=BUILD):
    live = retained.baseline()
    parent = bank.previous.read(PRIMARY / 'record.json')
    driver = basis.core.source.device.load_module('decode_kv_width3_link', bank.P / LINK_DRIVER)
    overlay_paths = (FIXTURE, RESOLVERS[1], HEADER) if plan.get('clock_wrap') else (FIXTURE,)
    controls = {*overlay_paths, LINK_DRIVER, Path(__file__).resolve().relative_to(bank.P).as_posix(),
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
            and all((PRIMARY / 'source' / p).read_bytes() == raw
                    and (p in overlay_paths or (bank.P / p).read_bytes() == raw)
                    for p, raw in original.items()), 'retained resolver or fixture contracts changed')
    overlay = {path: files[path] for path in overlay_paths}
    require(all(overlay[path] != original[path] for path in overlay if path in original),
            'clock wrap requires the committed resolver and declaration overlays')

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
        dest = str(build_dir / Path(obj).name)
        replacements[obj] = dest
        if name in overlay:
            source = str(build_dir / name)
        quote = []
        if name == FIXTURE:
            quote = ['-iquote', str(PRIMARY / 'source' / 'tools/sycl')]
        if HEADER in overlay:
            argv = [argv[0], '-I', str(build_dir / 'include'), *argv[1:]]
        commands.append([*argv[:-4], *quote, '-MD', '-MF', dest + '.d', '-o', dest, '-c', source])
    ninja = '/usr/bin/ninja'
    require(sha(ninja) == parent['build_tools'][ninja], 'retained Ninja changed')
    lines = subprocess.check_output([ninja, '-t', 'commands', TARGET], cwd=parent_build, text=True, timeout=20)
    link = shlex.split(lines.splitlines()[-1])
    require(link[:2] == [':', '&&'] and link[-2:] == ['&&', ':'] and '-Wl,-t' in link
            and all(link.count(obj) == 1 for obj in replacements) and link[link.index('-o') + 1] == TARGET
            and all(argv[0] == link[2] for argv in commands)
            and [p for p in link if p.endswith('.a')] == [ARCHIVE], 'unexpected component link recipe')
    replacements[TARGET] = str(build_dir / TARGET)
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
                  archives={archive: archive_sha}, commands=commands, plan=plan)
    driver.BUILD, driver.SOURCE = parent_build, PRIMARY / 'source'
    return fields, parent, driver, overlay


def save(row, root=ROOT, key=KEY):
    write(root / 'record.json', row)
    results = bank.previous.read(bank.RESULT)
    results[key] = row
    write(bank.RESULT, results)


def build(row, parent, driver, overlay, build_dir=BUILD):
    require(row['stage'] == 'prepared' and not build_dir.exists(), 'one fresh component build required')
    allowed = driver.build_inputs(parent, build_dir, overlay, row['commands'])
    driver.build_fixture(build_dir, TARGET, parent, overlay, row['commands'], allowed,
                         dict(source_commit=row['source_commit'], overlay_sha256=bank.stage.hashed(overlay)))
    receipt = bank.previous.read(build_dir / 'record.json')
    require(receipt['complete'] and receipt['ownership_closed'] and receipt['commands'] == row['commands']
            and all(receipt['linked'].get(p) == h for p, h in row['archives'].items()),
            'fixture link did not consume the qualified runtime archive')
    binary = receipt['binary']
    row.update(stage='built', binary=dict(binary, bytes=Path(binary['path']).stat().st_size),
               build_receipt_sha256=sha(build_dir / 'record.json'),
               compiled_inputs=len(receipt['compiled']), linked_inputs=len(receipt['linked']))


def check_binary(row, build_dir=BUILD):
    receipt_path = build_dir / 'record.json'
    require(sha(receipt_path) == row['build_receipt_sha256'], 'fixture build receipt changed')
    receipt = bank.previous.read(receipt_path)
    binary = row['binary']
    require(receipt['binary'] == {k: binary[k] for k in ('path', 'sha256')}
            and Path(binary['path']).stat().st_size == binary['bytes'] and sha(binary['path']) == binary['sha256']
            and all(sha(p) == h for group in ('compiled', 'linked') for p, h in receipt[group].items()),
            'fixture executable or actual build inputs changed')


def matches_arm(sample, clock_wrap, candidate):
    return (sample['cooperative'] is (clock_wrap or candidate)
            and sample.get('clock_wrap', False) is (clock_wrap and candidate))


def observations(lines, stage, plan=PLAN):
    clock_wrap = plan.get('clock_wrap', False)
    if stage == 'smoke':
        name, count = ('clock_wrap', 3) if clock_wrap else ('width3', 2)
        require(lines == [f'planner_{name}_smoke=PASS direct_cases={count} graph_replays={count}'],
                'incomplete small KV fixture')
        return dict(direct_cases=count, graph_replays=count)
    if stage == 'validate':
        recorded = [json.loads(line) for line in lines]
        graphs = recorded[:10]
        require(len(graphs) == 10 and all(
                    matches_arm(sample, clock_wrap, index >= 5) and sample['replay'] == index % 5
                    for index, sample in enumerate(graphs))
                and all(type(s['fingerprint_fnv64']) is int
                        and s['phase'] == 'planner_context_graph' and s['fmt'] == 1 and s['nq'] == 3
                        and s['last_cell'] == 262143 and s['oracle_exact'] is True for s in graphs)
                and [s['fingerprint_fnv64'] for s in graphs[:5]] == [s['fingerprint_fnv64'] for s in graphs[5:]],
                'missing or mismatched width-three graph parity')
        expected = []
        for missing in plan['misses']:
            for candidate in (False, True):
                case = dict(phase='planner_perf_cases', nq=3, missing=missing,
                            cooperative=(clock_wrap or candidate), replays=2, oracle_exact=True)
                if clock_wrap:
                    case['miss_pattern'] = 'interleaved'
                    if candidate:
                        case['clock_wrap'] = True
                expected.append(case)
        require(recorded[10:] == expected, 'miss-case validation census differs')
        return dict(graph_fingerprints=[s['fingerprint_fnv64'] for s in graphs[:5]], exact_miss_cases=expected)
    arm = plan['arms'][int(stage)]
    count = plan['samples_per_arm']
    marker = (f'planner_benchmark=PASS samples={count} timing=sum_of_isolated_graph_launch_and_wait '
              'reset_and_oracle=excluded physical_bus_bytes=unknown')
    require(len(lines) == count + 1 and lines[-1] == marker, 'incomplete width-three timings')
    samples = [json.loads(line) for line in lines[:-1]]
    for s, (missing, rep) in zip(samples, ((m, i) for m in plan['misses'] for i in range(3)), strict=True):
        calls = s['calls']
        require(s['phase'] == 'planner_sample' and matches_arm(s, clock_wrap, arm != plan['arms'][0])
                and s.get('miss_pattern') == ('interleaved' if clock_wrap else None)
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


def trace_windows(lines, stage):
    arm, missing = TRACE_CASES[stage]
    windows = [json.loads(line) for line in lines]
    require(len(windows) == 6 and windows.pop() == dict(phase='planner_width3_trace',
                replays=5, nq=3, graph_nodes=2, missing=missing, cooperative=arm == 'cooperative',
                requested_host_payload_bytes_per_call=missing * 4224, oracle_exact=True),
            'incomplete trace fixture or payload oracle')
    previous = 0
    for index, window in enumerate(windows):
        begin, end = window['begin_raw_ns'], window['end_raw_ns']
        require(window['phase'] == 'planner_width3_replay' and window['replay'] == index
                and type(begin) is int and type(end) is int and previous < begin < end,
                'five ordered, disjoint native RAW-clock replays required')
        previous = end
    return dict(arm=arm, missing=missing, replay_windows=windows)


def trace_services(events, native_pid, windows, capture_begin, capture_end):
    kernels = [event for event in events if event.get('ph') == 'X' and event.get('cat') == 'gpu_op'
               and decode_window_trace.kind(event['name']) == 'kernel']
    samples, identities = [], set()
    kernel_names = None
    for window in windows:
        begin, end = window['begin_raw_ns'], window['end_raw_ns']
        summary = decode_window_trace.summarize(events, native_pid=native_pid,
            begin_ns=begin, end_ns=end, capture_begin_ns=capture_begin, capture_end_ns=capture_end)
        counts = summary['operation_counts']
        require(counts.get('kernel') == 2 and not counts.get('copy') and not counts.get('fill')
                and not summary['boundary_crossing_counts'], 'unexpected device work inside graph replay')
        first, second = sorted((event for event in kernels
            if begin / 1000 <= event['ts'] and event['ts'] + event['dur'] <= end / 1000),
            key=lambda event: event['ts'])
        pair = {str(event['args']['id']) for event in (first, second)}
        names = [event['name'].split('[SIMD', 1)[0] for event in (first, second)]
        gap = second['ts'] - (first['ts'] + first['dur'])
        require(len(pair) == 2 and not pair & identities and gap >= 0 and names[0] != names[1]
                and (kernel_names is None or names == kernel_names),
                'expected one stable ordered resolver/copy pair per replay')
        kernel_names = names
        identities.update(pair)
        samples.append(dict(replay=window['replay'], native_ids=[str(event['args']['id']) for event in (first, second)],
            resolver_us=first['dur'], copy_us=second['dur'], inter_kernel_gap_us=gap,
            wall_us=(end - begin) / 1000))
    return dict(kernel_names=kernel_names, samples=samples)


def trace_provider(row):
    evidence = pti.committed_json(pti.RESULT, bank.P)
    tools = pti.verify_tools(evidence)
    probe = evidence['unitrace_current_runtime_probe_v2']
    require(probe['stage'] == 'closed' and probe['ownership_closed'] and probe['passed']
            and probe['boot'] == row['boot'] and probe['driver'] == row['driver']
            and probe['preload'] == row['native_validation_probe']
            and probe['admission']['runtime_manifest']['sha256'] == row['runtime_manifest']['sha256']
            and probe['admission']['tools']['artifacts'] == tools['artifacts']
            and [result['label'] for result in probe['results']] == ['control', 'trace']
            and all(result['passed'] for result in probe['results'])
            and probe['results'][-1]['raw_clock_mapping_checked'],
            'closed current-runtime clock and collector proof required')
    return dict(artifacts=tools['artifacts'], source_commit=tools['collector_source_commit'],
                probe_sha256=bank.signature(probe))


def device_case(row, parent, stage, profile=None, run_root=ROOT):
    build_dir = run_root / 'fixture-build'
    check_binary(row, build_dir)
    plan = row['plan']
    root = run_root / stage
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
    verb = 'trace' if profile else stage if stage in ('smoke', 'validate') else 'benchmark'
    fixture = 'clock-wrap' if plan.get('clock_wrap') else 'width3'
    argv = [row['binary']['path'], f'--planner-{fixture}-{verb}']
    if verb == 'benchmark':
        argv.append(plan['arms'][int(stage)])
    if profile:
        arm, missing = TRACE_CASES[stage]
        argv.extend((arm, str(missing)))
        argv = [profile['artifacts']['unitrace']['path'], *TRACE_FLAGS, *argv]
        (root / 'trace').mkdir(mode=0o700)
        require(env['ZES_ENABLE_SYSMAN'] == '0' and env.get('ZET_ENABLE_METRICS', '0') != '1',
                'trace uses the qualified provider without metrics activation')
        env.update(STRATA_UNITRACE_TARGET_PRELOAD=env.pop('LD_PRELOAD'), UNITRACE_SystemTime='1',
                   UNITRACE_FollowChildProcess='1', UNITRACE_TraceOutputDir=str(root / 'trace'))
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    owner = owner_api.Owner(root / 'owned', lambda: write(root / 'ownership.json', owner.commands))
    result = dict(stage=stage, argv=argv, environment_sha256=bank.signature(env))
    providers = {str(Path(p).resolve()): h for p, h in bank.previous.read(Path(row['runtime_manifest']['path'])).items()}
    providers.update({str(Path(a['path']).resolve()): a['sha256'] for a in (*probe.values(), row['binary'])})
    required = {str(Path(p).resolve()) for p in
                (env['UR_ADAPTERS_FORCE_LOAD'], row['driver_library'], probe['loader']['path'], probe['library']['path'])}
    required.add(str(Path(row['binary']['path']).resolve()))
    if profile:
        collector = profile['artifacts']['libunitrace_tool.so']
        path = str(Path(collector['path']).resolve())
        providers[path] = collector['sha256']
        required.add(path)

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
        result.update(trace_windows(lines, stage) if profile else observations(lines, stage, plan))
        require(Path(f'/proc/{entry["pid"]}/exe').resolve() == Path(row['binary']['path']).resolve(),
                'native fixture did not exec under the owned PID')
        mapped = runtime_probe.executable_mappings(Path(f'/proc/{entry["pid"]}/maps').read_text())
        require(required <= mapped <= providers.keys() and len(mapped) >= 8 and
                all(sha(p) == providers[p] for p in mapped), 'actual loaded fixture providers differ')
        result['provider_sha256'] = bank.signature({p: providers[p] for p in sorted(mapped)})
        io.send(b'QUIT\n')
        io.proc.stdin.close()
        io.exit(20)

    try:
        capture_begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) if profile else None
        entry, _, err = owner.run('fixture', argv, env, root, interaction=interact, wall=WALL[verb], **LIMITS)
        capture_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) if profile else None
        errors = err.read_text()
        warning = root / 'validation.log'
        actual_probe = validation.preload_evidence(errors, entry['pid'])
        errors_seen = set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', errors))
        allowed_errors = {'UR_RESULT_ERROR_INVALID_NULL_HANDLE'} if stage == 'smoke' else set()
        if stage == 'smoke':
            calls = r'^\s*<--- urEnqueueCommandBufferExp\(.* -> UR_RESULT_SUCCESS;'
            count = len(re.findall(calls, errors, re.M))
            require(count == result['graph_replays'], 'expected all traced graph replays')
            result['traced_graph_replays'] = count
        require(owner_api.closed(entry) and entry['exit_code'] == 0 and
                'PASS bounded component scope only' in errors and
                errors_seen <= allowed_errors
                and not re.search(r'SYCL exception|^FAIL ', errors, re.M)
                and (not warning.exists() or warning.stat().st_size == 0)
                and Path(actual_probe['loader']).resolve() == Path(probe['loader']['path']).resolve(),
                'fixture completion, trace or parameter validation failed')
        if profile:
            require(not re.search(r'^\s*\[(?:ERROR|WARN(?:ING)?)\]|\b(?:dropped|lost)\s+\d+\s+events',
                                  errors, re.M | re.I), 'collector reported a warning or loss')
            require(all(sha(item['path']) == item['sha256'] for item in profile['artifacts'].values()),
                    'collector or launcher changed during fixture profiling')
            paths = list((root / 'trace').glob('*.json'))
            require(len(paths) == 1 and 0 < paths[0].stat().st_size <= LIMITS['file_cap'],
                    'one bounded completed timeline required')
            events = json.loads(paths[0].read_bytes())['traceEvents']
            timings = trace_services(events, entry['pid'], result['replay_windows'], capture_begin, capture_end)
            result.update(trace_sha256=sha(paths[0]), trace_services=timings,
                          capture_raw_ns=[capture_begin, capture_end], trace_provider=profile)
        check_binary(row, build_dir)
        identity, = [item for item in entry['owners'] if item['pid'] == entry['pid']]
        result.update(host_probe_accepted=True, validation=actual_probe, native_identity=identity, native_exit_code=0)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        validation.finish(result, owner, cursor, row['boot'])
        result['passed'] &= result.get('driver_after') == row['driver']
        write(root / 'outcome.json', result)
    return result


def decision(processes, plan=PLAN):
    ids = {(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}
    stages = plan['stages']
    require(len(ids) == len(stages) and [p['stage'] for p in processes] == stages,
            'all independent component processes required')
    cells = []
    for missing in plan['misses']:
        medians = [statistics.median(s['ns_per_call'] for s in p['samples'] if s['missing'] == missing)
                   for p in processes[2:]]
        ratios = [medians[1] / medians[0], medians[2] / medians[3], medians[5] / medians[4]]
        cells.append(dict(missing=missing, process_medians_ns=medians, paired_time_ratios=ratios,
                          faster=all(ratio < 1 for ratio in ratios)))
    if plan.get('clock_wrap'):
        zero = cells[0]['process_medians_ns']
        zero_miss_preserved = statistics.median(zero[i] for i in (1, 2, 5)) <= max(zero[i] for i in (0, 3, 4))
        return dict(cells=cells, zero_miss_preserved=zero_miss_preserved,
                    eligible_for_model_check=zero_miss_preserved and all(c['faster'] for c in cells[1:]))
    return dict(cells=cells, eligible_for_model_check=all(c['faster'] for c in cells))


def main(action, clock_wrap=False):
    key, plan = (CLOCK_WRAP_KEY, CLOCK_WRAP_PLAN) if clock_wrap else (KEY, PLAN)
    root = bank.B / key.replace('_', '-')
    build_dir = root / 'fixture-build'
    stages = plan['stages']
    if clock_wrap:
        prior = bank.previous.read(bank.B / 'decode-kv-resident-batch-v1' / 'record.json')
        original = bank.previous.historic('11dff2bb5d7e6db8f8628d007efc7b39f0bbbafb', 'decode_kv_resident_batch_v1')
        require(prior == original and prior['stage'] == 'closed' and prior['ownership_closed']
                and not prior['decision']['eligible_for_model_check'],
                'require the exact closed resident KV experiment')

    if action == 'prepare':
        prior = validation.committed(bank.RESULT, ('dflash2_decode_kv_speed_v1',))
        require(prior['stage'] in ('closed', 'failed') and not prior['active'] and prior['ownership_closed'],
                'close the existing DFlash2 comparison before preparing this fixture')
        require(not root.exists() and key not in validation.committed(bank.RESULT), 'one fresh width-three run')
        revision = validation.git('rev-parse', 'HEAD').decode().strip()
        fields, _, _, _ = inputs(revision, plan, build_dir)
        root.mkdir(mode=0o700)
        save(dict(fields, stage='prepared', processes=[], complete=False, passed=False,
                  ownership_closed=True, model_qualified=False), root, key)
        return
    row = bank.previous.read(root / 'record.json')
    require(validation.committed(bank.RESULT, (key,)) == row and row['ownership_closed'],
            'commit the exact preceding component stage')
    fields, parent, driver, overlay = inputs(row['source_commit'], plan, build_dir)
    require(all(row[k] == v for k, v in fields.items()), 'frozen fixture inputs changed')
    if action in TRACE_CASES:
        require(not clock_wrap, 'clock wrap timing uses no kernel trace stages')
        traces = row.setdefault('traces', [])
        require(row['stage'] == 'closed' and row['complete'] and row['passed']
                and len(traces) < len(TRACE_CASES) and action == list(TRACE_CASES)[len(traces)]
                and all(result['passed'] for result in traces), 'run only the next trace after clean closure')
        profile = trace_provider(row)
        result = device_case(row, parent, action, profile, root)
        traces.append(result)
        row['ownership_closed'] = result['ownership_closed']
        save(row, root, key)
        require(result['passed'], result.get('error', 'trace or ownership gate failed'))
        return
    if action != 'build':
        index = len(row['processes'])
        require(index < len(stages) and action == stages[index]
                and row['stage'] == ('built' if index == 0 else stages[index - 1]),
                'run only the next committed case')
    try:
        if action == 'build':
            build(row, parent, driver, overlay, build_dir)
        else:
            result = device_case(row, parent, action, run_root=root)
            row['processes'].append(result)
            row['ownership_closed'] = result['ownership_closed']
            require(result['passed'], result.get('error', 'device or ownership gate failed'))
            row['stage'] = action
            if action == stages[-1]:
                row.update(stage='closed', complete=True, passed=True,
                           decision=decision(row['processes'], plan))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        if action == 'build' and (build_dir / 'record.json').is_file():
            row['ownership_closed'] = bank.previous.read(build_dir / 'record.json')['ownership_closed']
        raise
    finally:
        save(row, root, key)


if __name__ == '__main__':
    clock_wrap = sys.argv[1:2] == ['--clock-wrap']
    args = sys.argv[2:] if clock_wrap else sys.argv[1:]
    traces = () if clock_wrap else tuple(TRACE_CASES)
    require(sys.executable == basis.reference.model.PYTHON and len(args) == 1
            and args[0] in ('prepare', 'build', *STAGES, *traces),
            'use the pinned interpreter and select the next component stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        main(args[0], clock_wrap)
