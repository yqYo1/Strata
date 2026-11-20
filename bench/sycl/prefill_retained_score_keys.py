"""Compare cold and suffix-only FP16 index-key packing inside the long-prefill selector."""

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
ROOT = bank.B / 'prefill-retained-score-keys-v1'
KEY = 'prefill_retained_score_keys_v1'
BASE = '9639531fddd1a43bd3aba38e2940b518f291a7ef'
SOURCE = 'a2904848647fc6f79b11570b98059a030be06c76'
TARGET = 'sycl_qsa_score_gemm_test'
HEADER = 'include/strata/prefill/qsa_score_gemm.hpp'
CALLER = 'src/prefill/prefill.cpp'
ADMISSION = 'src/prefill/layer_major.inc'
FIXTURE = 'tests/sycl/qsa_score_gemm.cpp'
PACKER = 'src/prefill/sycl/qsa_score_gemm.cpp'
UNITS = {CALLER: 'strata_prefill', PACKER: 'strata_prefill', FIXTURE: TARGET}
STAGES = ('build', 'smoke', 'check', '0', '1', '2')


def recipe(helper, parent):
    src, build = helper.SOURCE, helper.BUILD
    overlay = {}
    for name in (HEADER, ADMISSION, *UNITS):
        original = (src / name).read_bytes()
        require(bank.sha(src / name) == parent['source_files'][name], 'parent input changed: ' + name)
        overlay[name] = merge_delta(original, name, BASE, SOURCE)
        require(merge_delta(overlay[name], name, SOURCE, BASE) == original, 'irreversible projection: ' + name)
    # The archive can reuse other objects: changed declarations are free functions.
    def scratch_definition(raw):
        return raw.split(b'struct Scratch {', 1)[1].split(b'};', 1)[0]
    require(scratch_definition(overlay[HEADER]) == scratch_definition((src/HEADER).read_bytes()),
            'prefill score workspace definition changed')
    db = bank.previous.read(build / 'compile_commands.json')
    commands, originals, objects = [], {}, {}
    for name, target in UNITS.items():
        u, = [v for v in db if v['file'] == str(src / name) and
              v['output'].startswith(f'CMakeFiles/{target}.dir/')]
        args = shlex.split(u['command'])
        require(u['directory'] == str(build) and args[-4:] == ['-o', u['output'], '-c', u['file']]
                and '-fsycl' in args and '-fno-fast-math' in args, 'unexpected compile recipe: ' + name)
        obj = ROOT / ('fixture.o' if name == FIXTURE else Path(u['output']).name)
        objects[name], originals[name] = str(obj), u['output']
        commands.append([args[0], '-I' + str(ROOT / 'include'), '-iquote', str((src / name).parent),
                         *args[1:-4], '-MD', '-MF', str(obj) + '.d', '-o', str(obj), '-c', str(ROOT / name)])
    replacements = {originals[FIXTURE]: objects[FIXTURE], TARGET: str(ROOT / TARGET)}
    target = 'strata_prefill'
    archive = 'lib' + target + '.a'
    members = subprocess.check_output(['/usr/bin/ar', 't', archive], cwd=build, text=True, timeout=20).splitlines()
    archive_dir = build / ('CMakeFiles/' + target + '.dir')
    paths = [Path(p) for p in parent['actual_inputs']['objects'] if Path(p).is_relative_to(archive_dir)]
    by_name = {p.name: str(p) for p in paths}
    require(len(paths) == len(members) == len(by_name) and set(members) == by_name.keys(),
            'archive membership differs: ' + target)
    for unit, owner in UNITS.items():
        if owner == target:
            member = Path(originals[unit]).name
            require(by_name[member] == str(build / originals[unit]), 'wrong baseline object: ' + unit)
            by_name[member] = objects[unit]
    replacements[archive] = str(ROOT / archive)
    commands.append(['/usr/bin/ar', 'rcsD', replacements[archive], *(by_name[m] for m in members)])
    link = shlex.split(subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', TARGET],
                       cwd=build, text=True, timeout=20).splitlines()[-1])
    require(link[:2] == [':', '&&'] and link[-2:] == ['&&', ':'] and '-Wl,-t' in link
            and link[link.index('-o')+1] == TARGET and all(link.count(k) == 1 for k in replacements),
            'unexpected score fixture link')
    commands.append([link[2], '-v', *(replacements.get(arg, arg) for arg in link[3:-2])])
    return overlay, commands


def samples(output, ordinal):
    controls = re.findall(r'^score_keys_timer_ms=(\S+)$', output, re.M)
    require(len(controls) == 1, 'missing timer control')
    control = float(controls[0])
    rows = re.findall(r'^score_keys_sample round=(\d+) slot=(\d+) reuse=([01]) rows=(\d+) chunks=(\d+) ms=(\S+)$',
                      output, re.M)
    require(len(rows) == 10 and math.isfinite(control) and control >= 0, 'missing paired samples')
    pairs = []
    for rnd in range(5):
        pair = {}
        for slot, entry in enumerate(rows[2*rnd:2*rnd+2]):
            idx, order, arm, count, chunks = map(int, entry[:5])
            ms = float(entry[5])
            require((idx, order, arm, count, chunks) == (rnd, slot, (ordinal+rnd+slot)%2, 2800, 12)
                    and math.isfinite(ms) and ms >= 100 and control <= .01*ms, 'wrong order or ineligible batch')
            pair['retained' if arm else 'cold'] = ms
        pairs.append(pair)
    return dict(timer_control_ms=control, batches_ms=pairs,
                median_paired_time_ratio=statistics.median(p['retained']/p['cold'] for p in pairs))


def measure(action, binary, native, speed, stage):
    smoke, timed = action == 'smoke', action.isdecimal()
    directory = ROOT / ('run-' + action)
    for name in ('tmp', 'cache'):
        (directory / name).mkdir(mode=0o700, parents=True)
    owner = device.owner.Owner(directory/'owned', lambda: bank.stage.write(directory/'execution.json', owner.commands))
    seed, probe = speed['environment_template'], speed['native_validation_probe']
    env = {k: v.replace(seed['STRATA_MEASUREMENT_ROOT'], str(directory)) for k,v in seed.items()}
    for k in ('STRATA_TRACE', 'UR_ENABLE_LAYERS', 'LD_DEBUG', 'LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER',
              'UR_LOG_TRACING', 'UR_LOG_VALIDATION'):
        env.pop(k, None)
    require(tuple(env.get(k) for k in ('STRATA_SYCL_PREFILL_SCORE_GEMM', 'STRATA_SYCL_SELECT_RADIX4',
            'STRATA_SYCL_SELECT_CYCLIC', 'STRATA_SYCL_PREFILL_TOPK_EARLY', 'STRATA_SYCL_SCORE_QUERY_TILE',
            'STRATA_SYCL_SCORE_HEAD_PARALLEL')) == ('1','1','0','0','8','1') and 'STRATA_SELECT_OLD' not in env,
            'production score GEMM and selector required')
    env['STRATA_SYCL_PREFILL_RETAIN_SCORE_KEYS'] = '1'
    if not timed:
        env.update(LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                   UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(directory/'validation.log'))
    if smoke:
        env.update(LD_DEBUG='libs', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:stderr')
    # Expose kernel dispatch only in the smoke. Clean arms alternate through explicit function arguments.
    if smoke:
        env['STRATA_TRACE'] = '1'
    args = ['--gemm', '1', '--tile', '8', '--head-parallel', '1']
    args += ['--key-benchmark', action] if timed else ['--key-sequence', '1' if smoke else '2']
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    stage['environment_sha256'] = bank.signature(env)
    try:
        cmd, out, err = owner.run(action, [binary['path'], *args], env, bank.P,
                                  **(bank.device.SMALL if smoke else bank.device.LARGE))
        require(device.owner.closed(cmd) and cmd['exit_code'] == 0, 'score-key fixture failed: ' + err.read_text()[-900:])
        output, log = out.read_text(), err.read_text()
        gpu, = re.findall(r'^device=(.+) driver=(\S+)$', output, re.M)
        batches, capacity = (8, 130) if smoke else (10, 65538)
        expected = (f'prefill_score_keys batches={batches} capacity_blocks={capacity} '
                    'scores_ids_exact=1 packed_pad_guards=1 reset=1 bounds=1 sticky=1')
        require('Intel' in gpu[0] and 'B570' in gpu[0] and output.splitlines().count(expected) == 1
                and output.splitlines().count('fixture_status=PASS closed=1') == 1, 'incomplete B570 output')
        if not timed:
            proof = device.validation.preload_evidence(log, cmd['pid'])
            layer = directory/'validation.log'
            require(proof['loader'] == probe['loader']['path'] and (not layer.exists() or layer.stat().st_size == 0),
                    'validation failed')
            stage['validation'] = proof
        if smoke:
            require('urEnqueueKernelLaunch' in log, 'missing traced GPU launch')
            stage['providers'] = bank.device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
                                                runtime_manifest=native['foundation']['runtime_manifest']))
        if timed:
            stage.update(samples(output, int(action)))
        if timed:
            require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|cleanup failed|terminate called|strata UR validation:', log),
                    'unexpected diagnostic or observer in the clean timing process')
        else:
            bank.device.diagnostics.diagnostics(log, cmd['pid'])
        stage.update(device=gpu[0], driver=gpu[1], native_pid=cmd['pid'], native_exit_code=0, compared_batches=batches, score_stride=capacity,
                     host_probe_accepted=True, clean_timing=timed)
    finally:
        device.validation.finish(stage, owner, cursor, speed['boot'])
        stage['passed'] = stage['passed'] and stage.get('driver_after') == native['driver']


def execute(action):
    native, speed, parent, build = device.coherent_inputs()
    helper = bank.basis.core.source.device.load_module('score_keys_build', device.fixture.__file__)
    helper.SOURCE, helper.BUILD = parent/'source', parent/'build'
    overlay, commands = recipe(helper, native)
    allowed = helper.build_inputs(native, ROOT, overlay, commands)
    controls = {str(Path(__file__).resolve().relative_to(bank.P))}
    for mod in tuple(sys.modules.values()):
        filename = getattr(mod, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(bank.P/'bench'/'sycl') and path.suffix == '.py':
                controls.add(str(path.relative_to(bank.P)))
    pins = bank.stage.source.pin(bank.P, [*sorted(controls), 'tools/sycl'])
    require(controls <= pins['source_sha256'].keys(), 'uncommitted controller or helper')
    identity = dict(source=SOURCE, parent_build=build['build_sha256'], overlay=bank.stage.hashed(overlay),
                    recipe=bank.signature(commands), controls=pins['aggregate_sha256'],
                    benchmark_environment_sha256=bank.signature(speed['environment_template']), runtime=speed['runtime_manifest']['sha256'])
    if action == 'recipe':
        print(f'{len(overlay)} pinned source files, {len(commands)} commands, reversible projection')
        return
    result = device.validation.committed(bank.RESULT)
    if action == 'build':
        require(KEY not in result and not ROOT.exists(), 'inspect the existing score-key generation')
        row = dict(**identity, controller_commit=pins['revision'], stages=[], model_tested=False)
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
            require(record['complete'] and record['ownership_closed'], 'incomplete score-key build')
            stage.update(passed=True, build_sha256=bank.sha(ROOT/'record.json'), binary=record['binary'])
        else:
            record = bank.previous.read(ROOT/'record.json')
            built = row['stages'][0]
            require(bank.sha(ROOT/'record.json') == built['build_sha256'] and record['binary'] == built['binary']
                    and bank.sha(record['binary']['path']) == record['binary']['sha256']
                    and all(bank.sha(p) == h for group in ('compiled','linked') for p,h in record[group].items())
                    and all(bank.sha(ROOT/p) == h for p,h in record['driver_inputs'].items()), 'score-key executable or inputs changed')
            measure(action, record['binary'], native, speed, stage)
            require(stage['passed'], 'post-run driver, fault or ownership failure')
    except BaseException as exc:
        stage['error'] = repr(exc)
        raise
    finally:
        if action == 'build' and (ROOT/'record.json').is_file():
            stage['ownership_closed'] = bank.previous.read(ROOT/'record.json')['ownership_closed']
        row['stages'].append(stage)
        if action == '2' and stage['passed']:
            row['component_performance_gate'] = all(s['median_paired_time_ratio'] < 1 for s in row['stages'][-3:])
        current = bank.previous.read(bank.RESULT)
        require(bank.signature(current.get(KEY)) == previous, 'score-key record changed during execution')
        current[KEY] = row
        bank.stage.write(bank.RESULT, current)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in (*STAGES, 'recipe') and sys.executable == bank.basis.reference.model.PYTHON, 'select a pinned score-key stage')
    if action == 'recipe':
        execute(action)
        sys.exit(0)
    with (bank.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        execute(action)
