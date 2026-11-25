#!/usr/bin/env python3
"""Qualify and measure the standalone five-column Q6 fixture."""

import fcntl
import json
import math
from pathlib import Path
import select
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import benchmark_runtime
import build_q6_c5_component as build
import measurement_guards as guards
import measurement_owner as ownership
import q6_canonical_component as reference
import ur_validation as validation

P, B, RESULT = build.P, build.B, build.RESULT
ROOT = B / 'q6-canonical-c5-check-v2'
KEY = 'q6_canonical_c5_component_check_v2'
PACKED_ROOT = B / 'q6-packed-c5-check-v1'
PACKED_KEY = 'q6_packed_c5_component_check_v1'
VALIDATION_REFERENCE = '58aabd0d294b527412e29d4b68c3e1e557c68deb'
CANONICAL_REFERENCE = 'f282d174d'
THIS = str(Path(__file__).resolve().relative_to(P))
PACKED_HEAD = ((2560, 248320),)
TIME_STAGES = ('time0', 'time1', 'time2')
STAGES = ('clean-small', *TIME_STAGES)
PACKED_STAGES = ('trace-small', 'validate-full', *TIME_STAGES)
SAFETY = {'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD': '1', 'NEOReadDebugKeys': '1',
          'EnableDirectSubmission': '0', 'STRATA_SYCL_PREFILL_SAME_QUEUE': '1',
          'STRATA_SYCL_VERIFY_NATIVE_CAPTURE': '0', 'SYCL_CACHE_PERSISTENT': '0', 'ZES_ENABLE_SYSMAN': '0'}
require, sha, write = build.require, build.sha, build.base.write
fields, number = reference.fields, reference.number


def numerical(text, shapes, packed=False):
    lines = text.splitlines()
    admission = [fields(line, 'ADMISSION ') for line in lines if line.startswith('ADMISSION ')]
    checks = [fields(line, 'NUMERIC ') for line in lines if line.startswith('NUMERIC ')]
    require([(number(v, 'k'), number(v, 'n')) for v in admission] == list(shapes)
            and [(number(v, 'k'), number(v, 'n')) for v in checks] == [s for s in shapes for _ in range(2)],
            'complete ordered five-column admissions and checks required')
    for f, (k, n) in zip(admission, shapes, strict=True):
        weight_bytes = n * (k // 256) * 210
        packed_bytes = ((n + 15) // 16) * 16 * (k // 256) * 210 if packed else 0
        require(number(f, 'canonical_bytes') == weight_bytes
                and number(f, 'device_payload_bytes') == weight_bytes + packed_bytes + (k // 32) * 36 * 5 + n * 40
                and number(f, 'packed_bytes') == number(f, 'additional_candidate_weight_bytes') == packed_bytes
                and number(f, 'max_alloc_bytes') >= max(weight_bytes, packed_bytes)
                and number(f, 'graphs') == 2 and number(f, 'nodes_each') == 1
                and f['reference'] == ('canonical_c5' if packed else 'generic')
                and f['scope'] == ('prequantized-packed-C5' if packed else 'prequantized-canonical-C5'),
                'five-column storage or dispatch disagrees with the fixture plan')
    result = []
    for f, (k, n) in zip(checks, (s for s in shapes for _ in range(2)), strict=True):
        error = float(f['max_abs_error'])
        oracle = number(f, 'oracle_rows')
        require(f['passed'] == f['guards'] == '1' and number(f, 'columns') == 5
                and number(f, 'outputs') == 5 * n and min(n, 512) <= oracle <= n
                and (n > 1024 or oracle == n) and math.isfinite(error) and error >= 0,
                'five-column numerical, oracle or guard outcome differs')
        result.append(dict(k=k, n=n, checked_outputs=n * 5, oracle_rows=oracle, max_abs_error=error))
    return result


def timing(text, index, shapes=reference.SHAPES, packed=False):
    lines = text.splitlines()
    clock, = (fields(line, 'CLOCK ') for line in lines if line.startswith('CLOCK '))
    timer = number(clock, 'median_stamp_ns')
    require(timer > 0 and number(clock, 'process') == index and number(clock, 'floor_ns') == 10_000_000,
            'timer identity or floor differs')
    arms = (1, 0) if index == 1 else (0, 1)
    expected = [(k, n, arm, sample) for k, n in shapes for arm in arms for sample in range(5)]
    samples = [fields(line, 'SAMPLE ') for line in lines if line.startswith('SAMPLE ')]
    require([tuple(number(f, key) for key in ('k', 'n', 'arm', 'sample')) for f in samples] == expected,
            'ordered timing samples differ')
    require(lines.count(f'TIMING passed=1 samples={len(expected)} shapes={len(shapes)} model_speed=0 physical_traffic=0 '
                        'cache=one-repeated-matrix') == 1, 'complete timing footer required')
    groups = {f'{k}x{n}': {'0': [], '1': []} for k, n in shapes}
    for f, (k, n, arm, _) in zip(samples, expected, strict=True):
        launches = reference.batch_size(n)
        wall = number(f, 'wall_ns')
        cpu = number(f, 'cpu_ns')
        submit = number(f, 'submit_wall_ns')
        require(number(f, 'columns') == 5 and number(f, 'process') == index
                and number(f, 'launches') == number(f, 'graph_launches') == launches
                and number(f, 'nodes_per_graph') == 1 and f['route'] == 'compat_graph'
                and f['reference'] == ('canonical_c5' if packed else 'generic')
                and wall >= max(10_000_000, 300 * timer) and wall == submit + number(f, 'wait_wall_ns')
                and cpu == number(f, 'submit_cpu_ns') + number(f, 'wait_cpu_ns') and cpu <= wall,
                'timed route, launch extent or clock accounting differs')
        groups[f'{k}x{n}'][str(arm)].append(dict(wall_ns=wall, cpu_ns=cpu, submit_wall_ns=submit, launches=launches))
    medians = {key: {arm: statistics.median(v['wall_ns'] / v['launches'] for v in rows)
                     for arm, rows in by_arm.items()} for key, by_arm in groups.items()}
    return dict(index=index, timer_overhead_ns=timer, samples=groups, median_ns=medians)


def device_plan(ctx):
    fixture_record = ctx.ROOT / 'record.json'
    fixture = json.loads(fixture_record.read_text())
    ctx.current(fixture)
    require(fixture['stage'] == 'built' and fixture['passed'] and fixture['host']['passed']
            and fixture['host']['ownership_closed'], 'same-build host checks required')
    path = str(RESULT.relative_to(P))
    checked = json.loads(validation.git('show', VALIDATION_REFERENCE + ':' + path))
    previous = checked['q6_canonical_c5_component_check_v1']
    observed = checked['q6_c5_clean_provider_probe_v1']
    require(previous['stage'] == 'failed' and previous['binary'] == observed['binary']
            and [v['mode'] for v in previous['outcomes']] == ['small', 'full', 'time0']
            and all(v['passed'] and v['ownership_closed'] and not v['fresh_faults'] for v in previous['outcomes'][:2])
            and len(previous['outcomes'][0]['numerical']) == 8 and len(previous['outcomes'][1]['numerical']) == 12
            and not previous['outcomes'][-1]['passed'], 'recover the exact fixture numerical checks and original refusal')
    require(observed['stage'] == 'closed' and observed['outcome']['passed'] and observed['outcome']['ownership_closed']
            and not observed['outcome']['fresh_faults'], 'recover the closed clean runtime observation')
    measured = json.loads(validation.git('show', build.REFERENCE + ':' + path))['dflash2_snapshot_regular_code_v1']
    require(measured['complete'] and measured['passed'] and measured['ownership_closed'], 'qualified runtime reference required')
    runtime = measured['runtime_manifest']
    require(sha(runtime['path']) == runtime['sha256'], 'runtime manifest changed')
    pins = json.loads(Path(runtime['path']).read_text())
    require(len(pins) == runtime['files'] and all(sha(p) == h for p, h in pins.items()), 'actual runtime input changed')
    proof = measured['native_validation_probe']
    require(all(sha(v['path']) == v['sha256'] for v in proof.values()), 'qualified validation preload changed')
    driver = guards.driver_guard()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(driver == measured['driver'] and boot == measured['boot'] and not driver['devcoredump'],
            'driver or boot differs from the current qualification')
    ze_driver, = (p for p in pins if Path(p).name.startswith('libze_intel_gpu.so.') and str(Path(p).resolve()) == p)
    after_device = observed['outcome']['phases'][1]
    require(after_device['missing'] == [str(Path(proof['loader']['path']).resolve())]
            and not after_device['extra'] and not after_device['changed_environment_names']
            and observed['runtime_manifest'] == runtime and observed['driver'] == driver and observed['boot'] == boot,
            'the clean adapter and driver must have been observed with the same provider profile')
    root = PACKED_ROOT if ctx.PACKED else ROOT
    env = dict(fixture['recipe']['environment'], ZE_ENABLE_ALT_DRIVERS=ze_driver,
        STRATA_Q6_DPAS_GATE='1', TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
    for name in ('UR_ENABLE_LAYERS', 'UR_LOG_TRACING', 'UR_LOG_VALIDATION'):
        env.pop(name)
    require(set(env) == set(observed['environment']) and all(env[k] == value for k, value in observed['environment'].items()
                    if k not in ('TMPDIR', 'XDG_CACHE_HOME')), 'clean environment differs from the observed provider profile')
    require(all(env.get(k) == value for k, value in SAFETY.items()) and env['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:gpu',
            'preserve qualified runtime safety settings')
    plan = dict(binary=fixture['binary'], build_record_sha256=sha(fixture_record),
        driver=driver, boot=boot, runtime_manifest=runtime, validation_reference=VALIDATION_REFERENCE,
        environment=env)
    if ctx.PACKED:
        canonical = json.loads(validation.git('show', CANONICAL_REFERENCE + ':' + path))[KEY]
        require(canonical['stage'] == 'closed' and canonical['binary'] == observed['binary']
                and canonical['runtime_manifest'] == runtime and canonical['driver'] == driver
                and canonical['boot'] == boot and all(v['passed'] for v in canonical['outcomes'])
                and '2560x248320' in canonical['decision']['eligible_shapes'],
                'recover the qualified canonical C5 full-head comparison')
        require(fixture['host']['outcomes'][-1]['result'] == 'HOST_PACKED_C5 passed=1 cases=7 GPU=0',
                'same-build packed fixture guard checks required')
        plan.update(canonical_reference=CANONICAL_REFERENCE, validation_probe=proof)
    else:
        require(previous['binary'] == fixture['binary'], 'original canonical fixture required')
    return plan


def save(row, root, key):
    write(root / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[key] = row
    write(RESULT, data)


def arguments(stage, packed):
    if stage in TIME_STAGES:
        return ['--time-packed-c5' if packed else '--time-c5', stage[-1]]
    return [{'clean-small': '--check-c5-small', 'trace-small': '--check-packed-c5-small',
             'validate-full': '--check-packed-c5'}[stage]]


def prepare(ctx, root, key, stages):
    require(not root.exists() and key not in validation.committed(RESULT), 'fresh device check required')
    source = build.base.source.pin(P, (THIS,))
    row = dict(stage='ready', source_commit=source['revision'], controller_sha256=source['source_sha256'][THIS],
        python=dict(path=sys.executable, sha256=sha(sys.executable)), **device_plan(ctx), active=False, outcomes=[],
        plan=dict(order=list(stages), modes=[' '.join(arguments(stage, ctx.PACKED)) for stage in stages],
            timing='Same five-column Q6 reference and DPAS fixture. Compare each head or dense shape independently. Require at least 10 ms and 300 timer stamps per batch. Record all eligible samples. Require a lower candidate median in each of three fresh processes before a model trial.',
            scope='Reuse the same fixture’s traced small and production-sized validation. Check four small shapes with the observed clean provider, then time separate clean processes including graph submission and completion. Preserve input and output guards.'),
        budgets=dict(wall=900, cpu=1800, rss_cap=8 << 30, text_cap=32 << 20, file_cap=32 << 20, total_cap=768 << 20))
    if ctx.PACKED:
        row['plan'].update(
            timing='Compare only 2560x248320 canonical/packed DPAS at five columns. Require five eligible 256-launch graph batches per arm across each of three fresh processes and a lower candidate median in every process.',
            scope='Trace the small numerical check, validate the full head, then time unobserved graph completion. Keep canaries and the scalar oracle. This is a standalone layout comparison.')
    root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (root / name).mkdir(mode=0o700)
    save(row, root, key)


def run(ctx, root, key, stages, stage):
    row = json.loads((root / 'record.json').read_text())
    require(row == validation.committed(RESULT, (key,)) and row['stage'] in ('ready', 'running')
            and not row['active'] and row['plan']['order'] == list(stages) and len(row['outcomes']) == stages.index(stage)
            and all(r['passed'] and r['ownership_closed'] for r in row['outcomes']), 'commit and verify the preceding stage')
    require(sha(P / THIS) == row['controller_sha256'] and (P / THIS).read_bytes() ==
            validation.git('show', row['source_commit'] + ':' + THIS)
            and row['python'] == dict(path=sys.executable, sha256=sha(sys.executable)), 'controller or interpreter changed')
    require(all(row[k] == v for k, v in device_plan(ctx).items()), 'fixture or runtime profile changed')
    args = arguments(stage, ctx.PACKED)
    require(row['plan']['modes'][stages.index(stage)] == ' '.join(args),
            'fixture arguments differ from the committed plan')
    env = dict(row['environment'])
    timed = stage in TIME_STAGES
    small = stage in ('clean-small', 'trace-small')
    diagnostic_stage = stage in ('trace-small', 'validate-full')
    log = root / f'{stage}-validation.log'
    if diagnostic_stage:
        proof = row['validation_probe']
        env.update(LD_PRELOAD=proof['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=proof['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(log))
        if small:
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_TRACING,' + env['UR_ENABLE_LAYERS']
            env['UR_LOG_TRACING'] = 'level:info;flush:info;output:stderr'
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    pins = json.loads(Path(row['runtime_manifest']['path']).read_text())
    result = dict(mode=stage, provider_admission=[], host_probe_accepted=False)
    if ctx.PACKED:
        result['environment_sha256'] = build.base.signature(env)

    def persist():
        write(root / (stage + '-execution.json'), dict(active=bool(owner.active), commands=owner.commands, outcome=result))

    def interact(io, command):
        for phase, tag in enumerate(('STRATA_Q6_DPAS_GATE ', 'STRATA_Q6_DPAS_DEVICE_GATE ')):
            line = io.line(180)
            while not line.startswith(tag):
                line = io.line(180)
            pid = command['pid']
            proc = Path('/proc', str(pid))
            identity = ownership.identity(pid)
            require(line == tag + 'pid=' + str(pid) and str((proc / 'exe').resolve()) == row['binary']['path']
                    and identity and all(identity[k] == command['direct_identity'][k] for k in ('pid', 'sid', 'start_ticks')),
                    'owned process or executable differs at the fixture gate')
            actual = dict(v.split('=', 1) for v in (proc / 'environ').read_text().split('\0') if '=' in v)
            mapped = set(benchmark_runtime.code_maps(pid))
            providers = [env['UR_ADAPTERS_FORCE_LOAD'], env['ZE_ENABLE_ALT_DRIVERS']] if phase else []
            if diagnostic_stage:
                providers.extend((env['LD_PRELOAD'], env['STRATA_UR_PROBE_EXPECTED_LOADER']))
            required = {str(Path(p).resolve()) for p in providers}
            require(actual == env and required <= mapped and mapped <= (set(pins) | {row['binary']['path']}),
                    'unexpected runtime providers or environment at fixture gate')
            result['native_identity'] = identity
            result['provider_admission'].append(dict(phase=phase, mapped_files=len(mapped)))
            io.send(b'continue\n')
        io.proc.stdin.close()
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)

    owner = ownership.Owner(root / ('owned-' + stage), persist)
    try:
        command, out, err = owner.run(stage, [row['binary']['path'], *args], env, P, interaction=interact, **row['budgets'])
        result['native_exit_code'] = command.get('exit_code')
        require(ownership.closed(command) and command['exit_code'] == 0, 'owned fixture failed')
        diagnostic = err.read_text()
        if diagnostic_stage:
            result['validation'], body = reference.validated_trace(diagnostic, command['pid'])
            require(result['validation']['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER']
                    and (not log.exists() or log.stat().st_size == 0), 'loader or parameter validation differs')
            if small:
                require(all(name in body for name in ('Q6DpasCanonicalColumns', 'Q6DpasColumns', 'urEnqueueCommandBufferExp')),
                        'both packed and canonical C5 graph routes must run')
        else:
            require(not any(value in diagnostic for value in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'strata UR validation:')),
                    'runtime error or validation observer in clean execution')
        text = out.read_text()
        shapes = reference.SMALL if small else PACKED_HEAD if ctx.PACKED else reference.SHAPES
        result['numerical'] = numerical(text, shapes, ctx.PACKED)
        if timed:
            result['timing'] = timing(text, int(stage[-1]), shapes, ctx.PACKED)
        else:
            require(text.splitlines().count(f'CHECK passed=1 small={int(small)} cases={2 * len(shapes)} '
                    'route=compat_graph columns=5') == 1
                    and text.splitlines().count('C1_CONTROL passed=1 dpas_launched=0') ==
                        (0 if ctx.PACKED else len(shapes)),
                    'complete five-column numerical checks or one-column controls missing')
        result['host_probe_accepted'] = True
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        validation.finish(result, owner, cursor, row['boot'])
        result['passed'] = result['passed'] and result.get('driver_after') == row['driver']
        row['outcomes'].append(result)
        row['active'] = bool(owner.active)
        row['stage'] = 'failed' if not result['passed'] else 'closed' if stage == 'time2' else 'running'
        if row['stage'] == 'closed':
            row['decision'] = reference.decision([r['timing'] for r in row['outcomes'] if 'timing' in r], shapes)
            if ctx.PACKED:
                row['decision']['next'] = ('Check width-five head retirement and cache borrowing as separate changes.'
                    if row['decision']['eligible_shapes'] else 'Keep the canonical C5 selection and close this speed trial.')
            else:
                row['decision']['next'] = 'Evaluate only independently faster C5 decode shapes in a matched model trial.'
        persist()
        save(row, root, key)
    require(result['passed'], result.get('error', 'fixture or driver qualification failed'))


if __name__ == '__main__':
    packed = sys.argv[2:] == ['--packed']
    root, key, stages = (PACKED_ROOT, PACKED_KEY, PACKED_STAGES) if packed else (ROOT, KEY, STAGES)
    require(len(sys.argv) in (2, 3) and sys.argv[2:] in ([], ['--packed'])
            and sys.argv[1] in ('prepare', *stages), 'select a planned C5 fixture stage with optional --packed')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        ctx = build.context(packed)
        if sys.argv[1] == 'prepare':
            prepare(ctx, root, key, stages)
        else:
            run(ctx, root, key, stages, sys.argv[1])
