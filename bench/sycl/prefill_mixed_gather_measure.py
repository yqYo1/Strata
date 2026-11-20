"""Compare prefill expert gathers on the same B570 fixture and stable row maps."""

import fcntl
import itertools
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import build_prefill_mixed_gather as fixture
import prefill_large_gather_model_source as coherent_model
import prefill_resident_input_model as model

bank, require = fixture.bank, fixture.require
owner, validation = bank.builder.ownership, bank.basis.validation
BUILD_KEY = 'prefill_mixed_gather_dispatch_build_v1'
KEY = 'prefill_mixed_gather_dispatch_v1'
COHERENT_KEY = 'prefill_large_gather_device_v1'
STAGES = ('contiguous', 'ragged', '0', '1', '2')
SHAPES = tuple(itertools.product((2334, 8192), (1, 16)))


def check_samples(text, process):
    lines = text.splitlines()
    clock, = [line for line in lines if line.startswith('gather_clock ')]
    def fields(line):
        return dict(token.split('=', 1) for token in line.split()[1:])
    timer = fields(clock)
    require(int(timer['samples']) == 1024 and 0 <= int(timer['median_upper_ns']) < 1000
            and int(timer['median_upper_ns']) <= int(timer['p99_ns']) < 5000, 'ineligible timer')
    samples = [fields(line) for line in lines if line.startswith('gather_sample ')]
    expected = list(itertools.product(SHAPES, range(6), range(2)))
    times = {shape: {1: [], 8: []} for shape in SHAPES}
    for sample, ((rows, cluster), pair, position) in zip(samples, expected, strict=True):
        words = 8 if (process + pair + position) % 2 else 1
        keys = ('process', 'rows', 'cluster', 'jobs', 'first', 'pair', 'order', 'words')
        require(tuple(int(sample[k]) for k in keys) ==
                (process, rows, cluster, 64, (process * 13 + pair * 9) % 64, pair, position, words)
                and sample['last_output_bits'] == 'PASS', 'unmatched gather sample or route')
        ms = float(sample['ms'])
        require(math.isfinite(ms) and 0.25 <= ms < 2500, 'ineligible batch duration')
        times[(rows, cluster)][words].append(ms)
    markers = [line for line in lines if line.startswith('gather_case ')]
    require(markers == [f'gather_case rows={r} cluster={c} batches=12 input_bits=PASS source_map=PASS guards=PASS'
                        for r, c in SHAPES], 'incomplete shape, map or guard checks')
    return [dict(rows=r, cluster=c, scalar_ms=t[1], vec8_ms=t[8],
                 median_time_ratio=statistics.median(t[8]) / statistics.median(t[1]))
            for (r, c), t in times.items()]


def component_inputs():
    native, speed, parent_identity, _ = model.inputs()
    build = validation.committed(bank.RESULT, (BUILD_KEY,))
    root = fixture.ROOT
    record = bank.previous.read(root / 'record.json')
    require(build['passed'] and build['complete'] and record['complete'] and record['ownership_closed']
            and bank.sha(root / 'record.json') == build['build_sha256']
            and parent_identity == build['parent_sha256'] and record['binary'] == build['binary']
            and bank.sha(record['binary']['path']) == build['binary']['sha256']
            and all(bank.sha(p) == h for group in ('compiled', 'linked') for p, h in record[group].items())
            and all(bank.sha(root / p) == h for p, h in record['driver_inputs'].items()),
            'build or measured inputs changed')
    return native, speed, root, build


def coherent_inputs():
    root = coherent_model.MODEL_ROOT
    native = bank.previous.read(root / 'record.json')
    coherent_model.check_build(native)
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed'],
            'complete the coherent model build and its host control first')
    speed = bank.previous.read(bank.basis.reference.ROOT / 'record.json')
    build = dict(build_sha256=bank.signature(bank.builder.compact(native)), binary=native['binaries'][fixture.TARGET])
    return native, speed, root, build


def measure(action, coherent=False):
    native, speed, root, build = coherent_inputs() if coherent else component_inputs()
    key = COHERENT_KEY if coherent else KEY
    stages = STAGES[:2] if coherent else STAGES
    scripts = [str(Path(f).resolve().relative_to(bank.P))
               for f in (__file__, fixture.__file__, coherent_model.__file__)]
    pins = bank.stage.source.pin(bank.P, [*scripts, 'tools/sycl'])['source_sha256']
    require(all(p in pins for p in scripts), 'commit the fixture and its build controllers')
    pin_digest = bank.signature(pins)
    result = bank.previous.read(bank.RESULT)
    i = stages.index(action)
    prior_digest = bank.signature(result.get(key))
    if i:
        result[key] = validation.committed(bank.RESULT, (key,))
    else:
        require(key not in result, 'inspect the previous gather attempt')
        result[key] = dict(build_sha256=build['build_sha256'], controls=pin_digest, stages=[],
                           source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                           measured_model=False)
        if coherent:
            result[key].update(binary=build['binary'], native_source_commit=native['native_source_commit'])
    row = result[key]
    require(row['controls'] == pin_digest and row['build_sha256'] == build['build_sha256']
            and [s['stage'] for s in row['stages']] == list(stages[:i])
            and all(s['passed'] for s in row['stages']), 'previous stage, plan or inputs changed')
    directory = root / ('measure-' + action)
    directory.mkdir(mode=0o700)
    for folder in ('tmp', 'cache'):
        (directory / folder).mkdir(mode=0o700)
    child = owner.Owner(directory / 'owned', lambda: bank.stage.write(directory / 'execution.json', child.commands))
    seed, probe = speed['environment_template'], speed['native_validation_probe']
    env = {k: v.replace(seed['STRATA_MEASUREMENT_ROOT'], str(directory)) for k, v in seed.items()}
    env.pop('UR_LOG_TRACING', None)
    env.pop('STRATA_TRACE', None)
    env.update(STRATA_TEST_MAIN_GATES='0', LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(directory / 'validation.log'))
    if i == 0:
        env.update(LD_DEBUG='libs', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:stderr')
    case = 'mixed-gather-' + ('vec8-' if i < 2 else 'dispatch-') + action
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    stage = dict(stage=action, case=case, environment_sha256=bank.signature(env))
    try:
        cmd, out, err = child.run(case, [build['binary']['path'], '--case', case], env, bank.P,
                                  **(bank.device.SMALL if i < 2 else bank.device.LARGE))
        require(owner.closed(cmd) and cmd['exit_code'] == 0, 'fixture failed: ' + err.read_text()[-600:])
        text, log = out.read_text(), err.read_text()
        proof = validation.preload_evidence(log, cmd['pid'])
        layer = directory / 'validation.log'
        devices = re.findall(r'^device=([^\n]+) driver=(\S+)$', text, re.M)
        require(proof['loader'] == probe['loader']['path'] and (not layer.exists() or not layer.stat().st_size)
                and len(devices) == 1 and 'Intel' in devices[0][0] and 'B570' in devices[0][0]
                and 'fixture_status=PASS model_PP=UNRUN' in text,
                'wrong device or incomplete fixture/validation result')
        bank.device.diagnostics.diagnostics(log, cmd['pid'])
        if i < 2:
            rows, primary, n, offset = (16384, 16384, 8192, 0) if i == 0 else (16383, 8192, 8191, 8192)
            require(f'vec8 words=8 job_rows={n} offset={offset} primary_mod16=4 tail_mod16=8 output_mod16=12 PASS'
                    in text and f'mixed_gather rows={rows} primary_rows={primary} compared_values={n * 2560} '
                    in text and 'admission_rejects=6 invalid_rows=4' in text
                    and 'inputs_immutable=1 guards=PASS PASS' in text, 'incomplete numerical evidence')
            if i == 0:
                require('urEnqueueKernelLaunch' in log, 'first-use launch trace missing')
                if coherent:
                    stage['providers'] = bank.device.loaded_providers(log, dict(environment=env,
                        native_validation_probe=probe, runtime_manifest=native['foundation']['runtime_manifest']))
            stage.update(compared_values=n * 2560, invalid_rows=4, admission_rejects=6)
        else:
            stage['samples'] = check_samples(text, int(action))
        stage.update(device=devices[0][0], runtime_driver=devices[0][1], native_exit_code=0, host_probe_accepted=True)
    except BaseException as error:
        stage['error'] = repr(error)
        raise
    finally:
        validation.finish(stage, child, cursor, speed['boot'])
        stage['passed'] = stage['passed'] and stage['driver_after'] == native['driver']
        row['stages'].append(stage)
        if not coherent and stage['passed'] and action == STAGES[-1]:
            row['faster_each_process'] = [all(s['samples'][j]['median_time_ratio'] < 1 for s in row['stages'][2:])
                                          for j in range(len(SHAPES))]
        current = bank.previous.read(bank.RESULT)
        require(bank.signature(current.get(key)) == prior_digest, 'gather record changed during the run')
        current[key] = row
        bank.stage.write(bank.RESULT, current)
    require(stage['passed'], 'post-run platform, ownership or fault check failed')


if __name__ == '__main__':
    coherent = sys.argv[1:2] == ['--coherent']
    action, = sys.argv[2:] if coherent else sys.argv[1:]
    require(action in (STAGES[:2] if coherent else STAGES), 'select a planned gather stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        measure(action, coherent)
