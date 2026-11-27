#!/usr/bin/env python3
"""Pin and build the standalone canonical two-column Q6 comparison fixture."""

import fcntl
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import build_q6_c5_component as existing
from q6_canonical_component import batch_size, fields, number, validated_trace

base, P, B, RESULT = existing.base, existing.P, existing.B, existing.RESULT
require, sha = existing.require, existing.sha
sys.path.insert(0, str(P / 'bench'))
import dflash2_decode_t4_build as source_build

ROOT = B / 'q6-t4-c2-component-v2'
KEY = 'q6_t4_c2_component_build_v2'
BASE_BUILD = '2dac4cbf2'
BASE_DEVICE = '460628c2a'
FOUNDATION = B / 'coherent-prefill-native-v1' / 'record.json'
THIS = str(Path(__file__).resolve().relative_to(P))
PREFIXES = ('CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'bench/sycl',
            'tools/sycl', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md', source_build.THIS,
            source_build.fixture.THIS, source_build.retained.THIS)
CONTROL = ('--host-split-c2', 'HOST_SPLIT_C2 passed=1 cases=5 GPU=0')
ARC_GATE = 'dflash2_decode_t4_shared_plan_v2'
T4_SPEED = 'dflash2_decode_t4_speed_v1'
CAPTURE_COMMIT = '4c324029d'
CAPTURE_KEY = 'dflash2_decode_t4_production_dpas_v1'
SHAPES = ((512, 33), (6144, 2560), (2560, 6144), (2560, 10240), (2560, 12288))
TIMED_SHAPES = SHAPES[1:]
ORDER = (('c4', 0), ('columns', 0), ('columns', 1), ('c4', 1), ('c4', 2), ('columns', 2))
SUMMARY = 'SPLIT_C2_CHECK passed=1 cases=10 bitwise=1 guards=1 isolated_halves=20'
TIMING_SUMMARY = 'TIMING passed=1 samples=40 shapes=4 model_speed=0 physical_traffic=0 cache=one-repeated-matrix'
ARC_LIMITS = dict(wall=180, cpu=600, rss_cap=4 << 30, text_cap=24 << 20,
                  file_cap=32 << 20, total_cap=64 << 20)
HOST_OUTPUT = ROOT / 'host-check'


def reference():
    bank = source_build.bank
    receipt = source_build.ROOT / 'record.json'
    native = bank.previous.read(receipt)
    archived = bank.previous.historic(BASE_BUILD, source_build.KEY)
    require(source_build.retained.closed(native, 'built')
            and source_build.builder.compact(native) == archived, 'recover the committed T4 recipe and native identity')
    require(sha(FOUNDATION) == native['foundation']['original_record_sha256'],
            'the original compiler foundation changed')
    arc = bank.previous.historic(BASE_DEVICE, ARC_GATE)
    require(source_build.retained.closed(arc, 'closed')
            and arc['native_build_sha256'] == bank.signature(archived)
            and arc['driver'] == native['driver'] and arc['runtime_manifest'] == native['runtime_manifest'],
            'the validated T4 runtime and recipe differ')
    runtime = {path: digest for path, digest in native['source_files'].items()
               if path.startswith(('src/sycl/', 'include/strata/sycl/'))}
    require(runtime and all(sha(P / path) == digest for path, digest in runtime.items()),
            'the component must reuse the verified SYCL runtime source')
    identity = dict(record=dict(path=str(receipt), sha256=sha(receipt)),
                    build_commit=BASE_BUILD, source_commit=native['native_source_commit'],
                    reference_binary=native['binaries']['strata'], runtime_sources=runtime,
                    device_commit=BASE_DEVICE, runtime_manifest=arc['runtime_manifest'],
                    driver=arc['driver'], boot=arc['boot'], validation_probe=arc['native_validation_probe'])
    return native, bank.previous.read(FOUNDATION), identity


def arc_environment(env, output_root):
    return dict(env, TMPDIR=str(output_root / 'tmp'), XDG_CACHE_HOME=str(output_root / 'cache'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(output_root / 'validation.log'))


def numerical_plan(ctx):
    arc = source_build.bank.previous.historic(BASE_DEVICE, ARC_GATE)
    legacy_root = str(B / ARC_GATE.replace('_', '-'))
    env = {key: value.replace(legacy_root, str(ROOT)) for key, value in arc['environment'].items()}
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['NEOReadDebugKeys'] == '1'
            and env['EnableDirectSubmission'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0',
            'preserve the verified Arc safety settings')
    output_root = ROOT / 'arc-check'
    return dict(command=[str(ROOT / 'build' / ctx.TARGET), '--check-split-c2'],
                output_root=str(output_root), environment=arc_environment(env, output_root),
                limits=ARC_LIMITS, shapes=[list(s) for s in SHAPES], summary=SUMMARY)


def timing_plan(ctx):
    arc = numerical_plan(ctx)
    env = dict(arc['environment'], UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
    for name in ('UR_LOG_TRACING', 'UR_LOG_VALIDATION', 'TMPDIR', 'XDG_CACHE_HOME'):
        env.pop(name)
    return dict(stages=[
        dict(mode=f'time-{i}', reference=reference, repetition=repetition,
             command=[arc['command'][0], f'--time-split-c2-{reference}', str(repetition)],
             output_root=str(ROOT / f'arc-time-{i}'))
        for i, (reference, repetition) in enumerate(ORDER)],
        environment=env,
        shapes=[list(shape) for shape in TIMED_SHAPES], limits=dict(ARC_LIMITS, wall=300),
        scope='Two independent one-node graph launches versus canonical C4 and versus the existing '
              'two-launch column path, on repeated matrices. Preserve parameter validation and runtime safety.',
        eligibility='Five batches per arm per shape per process, after two warm-up batches. Each batch '
                    'contains 1024 complete four-column replays and lasts at least 10 ms and 300 timer units. '
                    'Record submission, completion and caller CPU separately. Keep every sample.',
        decision='For each shape require at least two of three faster C2 medians and a geometric time '
                 'ratio below one versus the existing column pair for a follow-up. Use C4 to price the '
                 'split. Scope any follow-up to eligible shapes; model draft yield and speed need a new test.',
        stop='Stop on an identity, numerical, allocation, timing, validation, ownership, fault or budget failure.')


def closed_comparison(identity):
    bank = source_build.bank
    revision = source_build.source.git(P, 'rev-parse', 'refs/remotes/origin/feature/sycl').decode().strip()
    row = bank.previous.historic(revision, T4_SPEED)
    require(source_build.retained.closed(row, 'closed') and row['binary'] == identity['reference_binary']
            and len(row['processes']) == 9 and all(
                entry['complete'] and entry['passed'] and entry['ownership_closed']
                and entry['native_exit_code'] == 0 and entry['fresh_faults'] == []
                for entry in row['processes']),
            'close and publish all nine owned model processes before the standalone comparison')
    return dict(commit=revision, result_sha256=bank.signature(row), processes=len(row['processes']),
                speed_screen_passed=row['decision']['speed_screen_passed'],
                timing_only_work_matched=row['decision']['timing_only_work_matched'])


def dense_capture(identity):
    bank = source_build.bank
    row = bank.previous.historic(CAPTURE_COMMIT, CAPTURE_KEY)
    require(source_build.retained.closed(row, 'closed') and row['production_dpas_preserved']
            and row['plan']['binary'] == identity['reference_binary'],
            'the dense capture must be from the same production DPAS engine')
    tables = {arm: {(k, n): count for k, n, count in row['outcomes'][arm]['dense_capture']}
              for arm in ('stock', 'after-cpu', 'early')}
    require(tables['early'] == tables['after-cpu']
            and all(set(table) == set(TIMED_SHAPES) for table in tables.values()),
            'the original capture must match the two split routes and four timing shapes')
    shapes = []
    for k, n in TIMED_SHAPES:
        displaced = tables['stock'][k, n] - tables['early'][k, n]
        require(displaced > 0, 'the split must displace stock C4 calls at every captured shape')
        shapes.append(dict(shape=f'{k}x{n}', displaced=displaced,
                           payload_bytes=displaced * n * (k // 256) * 210))
    return dict(commit=CAPTURE_COMMIT, result_sha256=bank.signature(row), shapes=shapes,
                scope='Static C4 calls displaced in one checked width-four capture. Payload assumes one '
                      'canonical weight visit per displaced call. Hardware traffic and long-request '
                      'route frequency require separate observations.')


def context():
    ctx = existing.context()
    ctx.ROOT, ctx.KEY = ROOT, KEY

    def current(row, committed=True):
        if committed:
            proof = source_build.validation.committed(RESULT, (KEY,))
            require(ctx.compact(row) == proof, 'commit the exact C2 component stage')
        _, _, identity = reference()
        require(row['selected_native'] == identity and {*ctx.UNITS, THIS, existing.THIS}
                <= row['project']['source_sha256'].keys(), 'C2 build provenance or inputs changed')
        ctx.build.check_inputs(row)
        require(row['recipe']['targets'] == row['recipe']['build_targets'] == [ctx.TARGET]
                and not row['recipe']['environment'].get('LD_PRELOAD')
                and row['plan']['host_output'] == str(HOST_OUTPUT)
                and row['plan']['arc'] == numerical_plan(ctx)
                and row['plan']['timing'] == timing_plan(ctx), 'use the frozen standalone Q6 component plan')
        if 'binary' in row:
            binary = row['binary']
            require(Path(binary['path']).stat().st_size == binary['bytes']
                    and sha(binary['path']) == binary['sha256'], 'C2 fixture executable changed')
            actual = row['actual_inputs']
            require(sha(actual['path']) == actual['sha256'], 'the actual C2 compiler input census changed')
            inputs = json.loads(Path(actual['path']).read_text())
            require(all(sha(path) == digest for kind in ('compiled', 'generated', 'links')
                        for path, digest in inputs[kind].items()), 'an actual C2 build input changed')

    ctx.current = current
    return ctx


def prepare(ctx):
    require(not ROOT.exists() and KEY not in source_build.validation.committed(RESULT),
            'prepare one fresh C2 component build')
    native, compiler, identity = reference()
    cohort = closed_comparison(identity)
    capture = dense_capture(identity)
    previous_root = str(source_build.ROOT)
    args = [value.replace(previous_root + '/source', str(P)).replace(previous_root, str(ROOT))
            for value in native['recipe']['configure']]
    require(args[args.index('-S') + 1] == str(P) and args[args.index('-B') + 1] == str(ROOT / 'build'),
            'separate fixture and engine build paths required')
    recipe = dict(configure=args, environment={k: v.replace(previous_root, str(ROOT))
                                               for k, v in native['recipe']['environment'].items()},
                  targets=[ctx.TARGET], build_targets=[ctx.TARGET])
    row = dict(stage='prepared', active=False, passed=False, complete=False, ownership_closed=True,
               commands=[], fallbacks={}, selected_native=identity, project=ctx.source.pin(P, PREFIXES),
               dependencies=compiler['dependencies'], toolchain=compiler['toolchain'],
               external_inputs=native['external_inputs'], recipe=recipe, budgets=ctx.BUDGET,
               host_budget=dict(wall=120, cpu=120, rss_cap=1 << 30, text_cap=1 << 20,
                                file_cap=2 << 20, total_cap=8 << 20),
               plan=dict(compile_target=ctx.TARGET, initial_native_checks=['--host', '--host-c5', CONTROL[0]],
                         host_output=str(HOST_OUTPUT),
                         scope='Build the fixture only with the retained production DPC++ recipe.',
                         arc=numerical_plan(ctx), timing=timing_plan(ctx), closed_model_comparison=cohort,
                         dense_capture=capture,
                         next='Commit the same-build Arc bitwise and guard check before timing either control.'),
               timing=[],
               retention='One standalone fixture through the C2 component decision. No serving engine build.')
    ctx.current(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    ctx.save(row)


def read_numerical(lines, shapes, reference):
    require(sum(line.startswith('DEVICE name=') and 'B570' in line for line in lines) == 1,
            'the fixture must identify the Arc B570')
    admissions = [fields(line, 'ADMISSION ') for line in lines if line.startswith('ADMISSION ')]
    require(tuple((number(item, 'k'), number(item, 'n')) for item in admissions) == shapes,
            'C2 fixture shape admissions differ')
    expected_route = 'canonical_c4' if reference == 'c4' else 'columns_c2_pair'
    control_graphs = 1 if reference == 'c4' else 2
    for item, (k, n) in zip(admissions, shapes):
        weights = n * (k // 256) * 210
        require(item['graph_counts'] == f'{control_graphs}/2' and item['reference'] == expected_route
                and number(item, 'graphs') == control_graphs + 2 and number(item, 'nodes_each') == 1
                and item['scope'] == 'prequantized-canonical-C4'
                and number(item, 'additional_candidate_weight_bytes') == number(item, 'packed_bytes') == 0
                and number(item, 'canonical_bytes') == weights
                and number(item, 'device_payload_bytes') == weights + (k // 32) * 36 * 4 + n * 32
                and number(item, 'max_alloc_bytes') >= weights,
                'C2 allocation, reference, or graph admission differs')
    values = [fields(line, 'NUMERIC ') for line in lines if line.startswith('NUMERIC ')]
    require([(number(v, 'k'), number(v, 'n')) for v in values] ==
            [s for s in shapes for _ in range(2)], 'complete ordered per-shape numerical checks required')
    for v in values:
        n = number(v, 'n')
        error = float(v['max_abs_error'])
        require(v['passed'] == v['guards'] == '1' and number(v, 'columns') == 4
                and number(v, 'outputs') == 4 * n and min(n, 512) <= number(v, 'oracle_rows') <= n
                and (n > 1024 or number(v, 'oracle_rows') == n)
                and math.isfinite(error) and error >= 0 and (reference != 'c4' or error == 0),
                'numerical or guard outcome differs')
    return len(values)


def read_timing(lines, reference, repetition):
    numerical_cases = read_numerical(lines, TIMED_SHAPES, reference)
    clocks = [fields(line, 'CLOCK ') for line in lines if line.startswith('CLOCK ')]
    require(len(clocks) == 1 and number(clocks[0], 'process') == repetition
            and number(clocks[0], 'floor_ns') == 10_000_000, 'timing control missing or changed')
    timer = number(clocks[0], 'median_stamp_ns')
    require(timer > 0, 'timer overhead must be positive')
    rows = []
    for line in lines:
        if not line.startswith('SAMPLE '):
            continue
        f = fields(line, 'SAMPLE ')
        v = {key: number(f, key) for key in
             ('process', 'arm', 'sample', 'k', 'n', 'columns', 'launches', 'graph_launches',
              'nodes_per_graph', 'wall_ns', 'cpu_ns', 'submit_wall_ns', 'submit_cpu_ns', 'wait_wall_ns', 'wait_cpu_ns')}
        batch = batch_size(v['n'])
        graphs = 1 if reference == 'c4' and v['arm'] == 0 else 2
        require(v['process'] == repetition and v['columns'] == 4 and batch == v['launches'] == 1024
                and v['graph_launches'] == graphs * batch and v['nodes_per_graph'] == 1
                and f['route'] == 'compat_graph'
                and f['reference'] == ('canonical_c4' if reference == 'c4' else 'columns_c2_pair'),
                'actual timed route or extent differs')
        require(v['wall_ns'] >= max(10_000_000, 300 * timer)
                and v['wall_ns'] == v['submit_wall_ns'] + v['wait_wall_ns']
                and v['cpu_ns'] == v['submit_cpu_ns'] + v['wait_cpu_ns']
                and v['cpu_ns'] <= v['wall_ns'] + 10 * timer, 'timing floor or accounting differs')
        rows.append(v)
    order = (1, 0) if repetition == 1 else (0, 1)
    require([(v['k'], v['n'], v['arm'], v['sample']) for v in rows] ==
            [(k, n, arm, sample) for k, n in TIMED_SHAPES for arm in order for sample in range(5)]
            and [line for line in lines if line.startswith('TIMING ')] == [TIMING_SUMMARY],
            'all ordered timing samples and the completion footer are required')
    medians = {f'{k}x{n}': [statistics.median(v['wall_ns'] / v['launches'] for v in rows
                                            if (v['k'], v['n'], v['arm']) == (k, n, arm))
                            for arm in (0, 1)] for k, n in TIMED_SHAPES}
    return dict(samples=rows, median_ns_per_four_columns=medians, timer_overhead_ns=timer,
                numerical_cases=numerical_cases)


def decide(outcomes, capture):
    require(tuple((v['reference'], v['repetition']) for v in outcomes) == ORDER
            and all(v['passed'] for v in outcomes), 'six accepted timing processes required')
    summary = {}
    for k, n in TIMED_SHAPES:
        shape = f'{k}x{n}'
        comparisons = {}
        for reference in ('c4', 'columns'):
            medians = [v['evidence']['median_ns_per_four_columns'][shape]
                       for v in outcomes if v['reference'] == reference]
            ratios = [candidate / control for control, candidate in medians]
            comparisons[reference] = dict(candidate_over_reference=ratios,
                                          geometric_ratio=statistics.geometric_mean(ratios),
                                          faster=sum(r < 1 for r in ratios))
        summary[shape] = comparisons
    counts = {item['shape']: item['displaced'] for item in capture['shapes']}
    require(counts.keys() == summary.keys() and all(count > 0 for count in counts.values()),
            'complete captured shape weights required')
    fixture_projection = []
    for outcome in outcomes:
        medians = outcome['evidence']['median_ns_per_four_columns']
        delta = sum(count * (medians[shape][1] - medians[shape][0]) for shape, count in counts.items())
        fixture_projection.append(dict(reference=outcome['reference'], repetition=outcome['repetition'],
                                       candidate_minus_reference_ns=delta))
    return dict(shapes=summary, eligible_shapes=[shape for shape, values in summary.items()
                if values['columns']['faster'] >= 2 and values['columns']['geometric_ratio'] < 1],
                capture_weighted_fixture_deltas=fixture_projection,
                projection_scope='Sum of isolated repeated-matrix deltas for the displaced capture counts. '
                                 'Context, graph overlap, cache and other model work are outside this projection.',
                model_speed_qualified=False, model_draft_yield_qualified=False, adopted=False)


def run_arc(ctx, mode):
    bank, validation = source_build.bank, source_build.validation
    row = bank.previous.read(ROOT / 'record.json')
    ctx.current(row)
    identity = row['selected_native']
    timing = mode != 'check'
    if timing:
        schedule = row['plan']['timing']
        stages = schedule['stages']
        index = [stage['mode'] for stage in stages].index(mode)
        plan = dict(stages[index], limits=schedule['limits'])
        plan['environment'] = arc_environment(schedule['environment'], Path(plan['output_root']))
        require(row.get('numerical', {}).get('passed') and len(row['timing']) == index
                and all(v['passed'] for v in row['timing']), 'commit numerical and all prior timing outcomes first')
    else:
        plan = row['plan']['arc']
        require('numerical' not in row and not row['timing'], 'require a fresh numerical check')
    output_root = Path(plan['output_root'])
    require(row['stage'] == 'built' and row['passed'] and row['ownership_closed']
            and row['host']['passed'] and row['host']['ownership_closed']
            and plan['command'][0] == row['binary']['path']
            and output_root.parent == ROOT and not output_root.exists(),
            'require the same-build host outcomes and fresh Arc stage')
    runtime = identity['runtime_manifest']
    pins = bank.previous.read(Path(runtime['path']))
    require(sha(runtime['path']) == runtime['sha256']
            and len(pins) == runtime['files'] and all(sha(path) == digest for path, digest in pins.items())
            and all(sha(v['path']) == v['sha256'] for v in identity['validation_probe'].values())
            and base.feature.guards.driver_guard() == identity['driver']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity['boot'],
            'the qualified C2 runtime, probe or driver changed')
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    output_root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (output_root / name).mkdir(mode=0o700)
    owner = base.ownership.Owner(output_root / 'owned', lambda: ctx.write(
        output_root / 'execution.json', dict(active=bool(owner.active), commands=owner.commands)))
    outcome = dict(mode=mode, host_probe_accepted=False)
    try:
        entry, out, err = owner.run(mode, plan['command'], plan['environment'], P, **plan['limits'])
        outcome['native_exit_code'] = entry.get('exit_code')
        require(base.ownership.closed(entry) and outcome['native_exit_code'] == 0,
                'the Arc C2 fixture did not complete')
        lines = out.read_text().splitlines()
        if timing:
            outcome.update(reference=plan['reference'], repetition=plan['repetition'],
                           evidence=read_timing(lines, plan['reference'], plan['repetition']))
        else:
            require(read_numerical(lines, SHAPES, 'c4') == 10 and lines.count(plan['summary']) == 1,
                    'the full bitwise and independent-half numerical checks are required')
            outcome.update(summary=SUMMARY, shapes=plan['shapes'])
        log = err.read_text()
        proof, body = validated_trace(log, entry['pid'])
        warning = Path(plan['environment']['UR_LOG_VALIDATION'].split('output:file,', 1)[1])
        require(proof['loader'] == identity['validation_probe']['loader']['path']
                and (timing or 'urEnqueueCommandBufferExp' in body)
                and (not warning.exists() or warning.stat().st_size == 0), 'C2 runtime validation differs')
        providers = bank.device.loaded_providers(log, dict(runtime_manifest=runtime,
            native_validation_probe=identity['validation_probe'], environment=plan['environment']))
        outcome.update(host_probe_accepted=True, validation=proof, providers=providers)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, identity['boot'])
        outcome['passed'] &= outcome.get('driver_after') == identity['driver']
        if timing:
            row['timing'].append(outcome)
        else:
            row['numerical'] = outcome
        row['active'], row['ownership_closed'] = bool(owner.active), outcome['ownership_closed']
        if not outcome['passed']:
            row.update(stage='failed', passed=False, complete=False)
        elif len(row['timing']) == len(ORDER):
            row.update(stage='closed', complete=True, decision=decide(row['timing'], row['plan']['dense_capture']))
        ctx.save(row)
    require(outcome['passed'], outcome.get('error', 'C2 runtime, timing or ownership gate failed'))


if __name__ == '__main__':
    stages = ('prepare', 'build', 'host', 'check', *(f'time-{i}' for i in range(len(ORDER))))
    require(sys.executable == source_build.bank.basis.reference.model.PYTHON
            and len(sys.argv) == 2 and sys.argv[1] in stages,
            'use the pinned interpreter and one C2 component stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.feature.guards.no_model_owners([])
        ctx = context()
        if sys.argv[1] == 'prepare':
            prepare(ctx)
        elif sys.argv[1] == 'build':
            ctx.compile_fixture()
        elif sys.argv[1] == 'host':
            existing.host(ctx, extra_controls=(CONTROL,), output_root=HOST_OUTPUT)
        else:
            run_arc(ctx, sys.argv[1])
