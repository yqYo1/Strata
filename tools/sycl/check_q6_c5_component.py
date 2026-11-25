#!/usr/bin/env python3
"""Qualify and measure the standalone five-column canonical Q6 fixture."""

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
ROOT = B / 'q6-canonical-c5-check-v1'
KEY = 'q6_canonical_c5_component_check_v1'
THIS = str(Path(__file__).resolve().relative_to(P))
STAGES = ('small', 'full', 'time0', 'time1', 'time2')
SAFETY = {'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD': '1', 'NEOReadDebugKeys': '1',
          'EnableDirectSubmission': '0', 'STRATA_SYCL_PREFILL_SAME_QUEUE': '1',
          'STRATA_SYCL_VERIFY_NATIVE_CAPTURE': '0', 'SYCL_CACHE_PERSISTENT': '0', 'ZES_ENABLE_SYSMAN': '0'}
require, sha, write = build.require, build.sha, build.base.write
fields, number = reference.fields, reference.number


def numerical(text, shapes):
    lines = text.splitlines()
    admission = [fields(line, 'ADMISSION ') for line in lines if line.startswith('ADMISSION ')]
    checks = [fields(line, 'NUMERIC ') for line in lines if line.startswith('NUMERIC ')]
    require([(number(v, 'k'), number(v, 'n')) for v in admission] == list(shapes)
            and [(number(v, 'k'), number(v, 'n')) for v in checks] == [s for s in shapes for _ in range(2)],
            'complete ordered five-column admissions and checks required')
    for f, (k, n) in zip(admission, shapes, strict=True):
        weight_bytes = n * (k // 256) * 210
        require(number(f, 'canonical_bytes') == weight_bytes
                and number(f, 'device_payload_bytes') == weight_bytes + (k // 32) * 36 * 5 + n * 40
                and number(f, 'packed_bytes') == number(f, 'additional_candidate_weight_bytes') == 0
                and number(f, 'max_alloc_bytes') >= weight_bytes
                and number(f, 'graphs') == 2 and number(f, 'nodes_each') == 1
                and f['reference'] == 'generic' and f['scope'] == 'prequantized-canonical-C5',
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


def timing(text, index):
    lines = text.splitlines()
    clock, = (fields(line, 'CLOCK ') for line in lines if line.startswith('CLOCK '))
    timer = number(clock, 'median_stamp_ns')
    require(timer > 0 and number(clock, 'process') == index and number(clock, 'floor_ns') == 10_000_000,
            'timer identity or floor differs')
    arms = (1, 0) if index == 1 else (0, 1)
    expected = [(k, n, arm, sample) for k, n in reference.SHAPES for arm in arms for sample in range(5)]
    samples = [fields(line, 'SAMPLE ') for line in lines if line.startswith('SAMPLE ')]
    require([tuple(number(f, key) for key in ('k', 'n', 'arm', 'sample')) for f in samples] == expected,
            'ordered timing samples differ')
    require(lines.count('TIMING passed=1 samples=60 shapes=6 model_speed=0 physical_traffic=0 '
                        'cache=one-repeated-matrix') == 1, 'complete timing footer required')
    groups = {f'{k}x{n}': {'0': [], '1': []} for k, n in reference.SHAPES}
    for f, (k, n, arm, _) in zip(samples, expected, strict=True):
        launches = reference.batch_size(n)
        wall = number(f, 'wall_ns')
        cpu = number(f, 'cpu_ns')
        submit = number(f, 'submit_wall_ns')
        require(number(f, 'columns') == 5 and number(f, 'process') == index
                and number(f, 'launches') == number(f, 'graph_launches') == launches
                and number(f, 'nodes_per_graph') == 1 and f['route'] == 'compat_graph' and f['reference'] == 'generic'
                and wall >= max(10_000_000, 300 * timer) and wall == submit + number(f, 'wait_wall_ns')
                and cpu == number(f, 'submit_cpu_ns') + number(f, 'wait_cpu_ns') and cpu <= wall,
                'timed route, launch extent or clock accounting differs')
        groups[f'{k}x{n}'][str(arm)].append(dict(wall_ns=wall, cpu_ns=cpu, submit_wall_ns=submit, launches=launches))
    medians = {key: {arm: statistics.median(v['wall_ns'] / v['launches'] for v in rows)
                     for arm, rows in by_arm.items()} for key, by_arm in groups.items()}
    return dict(index=index, timer_overhead_ns=timer, samples=groups, median_ns=medians)


def device_plan():
    fixture = json.loads((build.ROOT / 'record.json').read_text())
    build.context().current(fixture)
    require(fixture['stage'] == 'built' and fixture['passed'] and fixture['host']['passed']
            and fixture['host']['ownership_closed'], 'same-build host checks required')
    path = str(RESULT.relative_to(P))
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
    env = dict(fixture['recipe']['environment'], LD_PRELOAD=proof['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=proof['loader']['path'], ZE_ENABLE_ALT_DRIVERS=ze_driver,
        STRATA_Q6_DPAS_GATE='1', TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    require(all(env.get(k) == value for k, value in SAFETY.items()) and env['ONEAPI_DEVICE_SELECTOR'] == 'level_zero:gpu',
            'preserve qualified runtime safety settings')
    return dict(binary=fixture['binary'], build_record_sha256=sha(build.ROOT / 'record.json'),
        driver=driver, boot=boot, runtime_manifest=runtime, validation_probe=proof, environment=env)


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = row
    write(RESULT, data)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh device check required')
    source = build.base.source.pin(P, (THIS,))
    row = dict(stage='ready', source_commit=source['revision'], controller_sha256=source['source_sha256'][THIS],
        python=dict(path=sys.executable, sha256=sha(sys.executable)), **device_plan(), active=False, outcomes=[],
        plan=dict(order=list(STAGES), modes=['--check-c5-small', '--check-c5', '--time-c5 0', '--time-c5 1', '--time-c5 2'],
            timing='Same five-column Q6 reference and DPAS fixture. Compare each head or dense shape independently. Require at least 10 ms and 300 timer stamps per batch. Record all eligible samples. Require a lower candidate median in each of three fresh processes before a model trial.',
            scope='Persistent cache disabled. Trace and validate the four small shapes, then use parameter validation for the production-sized check. Time separate clean processes, including graph submission and completion. Preserve input and output guards.'),
        budgets=dict(wall=900, cpu=1800, rss_cap=8 << 30, text_cap=32 << 20, file_cap=32 << 20, total_cap=768 << 20))
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    save(row)


def run(stage):
    row = json.loads((ROOT / 'record.json').read_text())
    require(row == validation.committed(RESULT, (KEY,)) and row['stage'] in ('ready', 'running')
            and not row['active'] and len(row['outcomes']) == STAGES.index(stage)
            and all(r['passed'] and r['ownership_closed'] for r in row['outcomes']), 'commit and verify the preceding stage')
    require(sha(P / THIS) == row['controller_sha256'] and (P / THIS).read_bytes() ==
            validation.git('show', row['source_commit'] + ':' + THIS)
            and row['python'] == dict(path=sys.executable, sha256=sha(sys.executable)), 'controller or interpreter changed')
    require(all(row[k] == v for k, v in device_plan().items()), 'fixture or runtime profile changed')
    log = ROOT / (stage + '-validation.log')
    env = dict(row['environment'], UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(log))
    timed = stage.startswith('time')
    if timed:
        for name in ('LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER', 'UR_ENABLE_LAYERS',
                     'UR_LOG_TRACING', 'UR_LOG_VALIDATION'):
            env.pop(name)
    elif stage == 'full':
        env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
        env.pop('UR_LOG_TRACING', None)
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    pins = json.loads(Path(row['runtime_manifest']['path']).read_text())
    result = dict(mode=stage, provider_admission=[], host_probe_accepted=False)

    def persist():
        write(ROOT / (stage + '-execution.json'), dict(active=bool(owner.active), commands=owner.commands, outcome=result))

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
            required = [env['LD_PRELOAD'], env['STRATA_UR_PROBE_EXPECTED_LOADER']] if not timed else []
            if phase:
                required.extend((row['validation_probe']['loader']['path'], env['UR_ADAPTERS_FORCE_LOAD'], env['ZE_ENABLE_ALT_DRIVERS']))
            required = {str(Path(p).resolve()) for p in required}
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

    owner = ownership.Owner(ROOT / ('owned-' + stage), persist)
    modes = {'small': ['--check-c5-small'], 'full': ['--check-c5']}
    args = modes.get(stage, ['--time-c5', stage[-1]])
    try:
        command, out, err = owner.run(stage, [row['binary']['path'], *args], env, P, interaction=interact, **row['budgets'])
        result['native_exit_code'] = command.get('exit_code')
        require(ownership.closed(command) and command['exit_code'] == 0, 'owned fixture failed')
        diagnostic = err.read_text()
        if timed:
            require(not any(value in diagnostic for value in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'strata UR validation:')),
                    'runtime error or validation observer in clean timing')
        else:
            proof, body = reference.validated_trace(diagnostic, command['pid'])
            result['validation'] = proof
            require(proof['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'] and (not log.exists() or log.stat().st_size == 0),
                    'parameter validation or loader changed')
        text = out.read_text()
        shapes = reference.SMALL if stage == 'small' else reference.SHAPES
        result['numerical'] = numerical(text, shapes)
        if timed:
            result['timing'] = timing(text, int(stage[-1]))
        else:
            require(text.splitlines().count(f'CHECK passed=1 small={int(stage == "small")} cases={2 * len(shapes)} '
                    'route=compat_graph columns=5') == 1
                    and text.splitlines().count('C1_CONTROL passed=1 dpas_launched=0') == len(shapes),
                    'complete five-column numerical and one-column controls required')
            if stage == 'small':
                require('Q6DpasCanonicalColumns' in body and 'urEnqueueCommandBufferExp' in body, 'actual C5 graph route required')
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
            row['decision'] = reference.decision([r['timing'] for r in row['outcomes'] if 'timing' in r])
            row['decision']['next'] = 'Evaluate only independently faster C5 decode shapes in a matched model trial.'
        persist()
        save(row)
    require(result['passed'], result.get('error', 'fixture or driver qualification failed'))


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', *STAGES), 'select a planned C5 fixture stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            run(sys.argv[1])
