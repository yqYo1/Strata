"""Check the long-prefill GDN gate producer and recurrence together on the B570."""

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
ROOT = bank.B / 'prefill-gdn-decay-v1'
KEY = 'prefill_gdn_decay_v1'
BASE = '827b4be344cf91c7420b22955b33308eba3b81ab'
SOURCE = '18d0b137e6068677dd222fafeebe9169c13b19b7'
TARGET = 'sycl_gdn_storage_pipeline_bench'
HEADER = 'include/strata/prefill/kernels.hpp'
CALLER = 'src/prefill/prefill.cpp'
FIXTURE = 'bench/sycl/gdn_storage_pipeline.cpp'
UNITS = {'src/kernels/sycl/fused_gdn.cpp': 'strata_kernels',
         'src/prefill/sycl/postops.cpp': 'strata_prefill', CALLER: 'strata_prefill', FIXTURE: TARGET}
STAGES = ('build', 'smoke', 'check', '0', '1', '2')


def recipe(helper, parent):
    src, build = helper.SOURCE, helper.BUILD
    overlay = {}
    for name in (HEADER, *UNITS):
        original = (src / name).read_bytes()
        require(bank.sha(src / name) == parent['source_files'][name], 'parent input changed: ' + name)
        if name == CALLER:
            overlay[name] = merge_delta(original, name, BASE, SOURCE)
            require(merge_delta(overlay[name], name, SOURCE, BASE) == original, 'irreversible caller change')
        else:
            require(helper.git_file(BASE, name) == original, 'wrong measured baseline: ' + name)
            overlay[name] = helper.git_file(SOURCE, name)
    header = overlay[HEADER]
    first = header.index(b'/// Long FP16-storage prefill may publish')
    last = header.index(b'#endif', first)
    require(header[:first] + header[last:] == (src / HEADER).read_bytes()
            and header[first:last].count(b'void prefill_gdn_') == 2
            and b'{' not in header[first:last], 'shared header may only add the two free declarations')

    db = bank.previous.read(build / 'compile_commands.json')
    commands, originals, objects = [], {}, {}
    for name, target in UNITS.items():
        u, = [v for v in db if v['file'] == str(src / name) and
              v['output'].startswith(f'CMakeFiles/{target}.dir/')]
        args = shlex.split(u['command'])
        require(u['directory'] == str(build) and args[-4:] == ['-o', u['output'], '-c', u['file']]
                and '-fsycl' in args and '-fno-fast-math' in args, 'unexpected compile recipe: ' + name)
        obj = ROOT / Path(u['output']).name
        objects[name], originals[name] = str(obj), u['output']
        commands.append([args[0], '-I' + str(ROOT / 'include'), '-iquote', str((src / name).parent),
                         *args[1:-4], '-MD', '-MF', str(obj) + '.d', '-o', str(obj), '-c', str(ROOT / name)])
    replacements = {originals[FIXTURE]: objects[FIXTURE], TARGET: str(ROOT / TARGET)}
    for target in ('strata_kernels', 'strata_prefill'):
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
            'unexpected GDN fixture link')
    commands.append([link[2], '-v', *(replacements.get(arg, arg) for arg in link[3:-2])])
    return overlay, commands


def samples(output, ordinal):
    controls = re.findall(r'^prefill_decay timer_control_ms=(\S+)$', output, re.M)
    require(len(controls) == 1, 'missing timer control')
    control = float(controls[0])
    rows = re.findall(r'^prefill_decay T=(\d+) round=(\d+) early=([01]) calls=(\d+) ms=(\S+)$', output, re.M)
    require(len(rows) == 10 and math.isfinite(control) and control >= 0, 'missing paired timing samples')
    pairs = []
    for rnd in range(5):
        pair = {}
        for pos, (tokens, idx, arm, count, duration) in enumerate(rows[2*rnd:2*rnd+2]):
            ms = float(duration)
            require((int(tokens), int(idx), int(arm), int(count)) == (8192, rnd, (ordinal+rnd+pos)%2, 16)
                    and math.isfinite(ms) and ms >= 100 and control <= .01*ms, 'wrong order or ineligible batch')
            pair['early' if arm == '1' else 'ordinary'] = ms
        pairs.append(pair)
    return dict(timer_control_ms=control, batches_ms=pairs,
                median_paired_time_ratio=statistics.median(p['early']/p['ordinary'] for p in pairs))


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
    args = ['--decay-bench',action] if timed else ['--decay-' + action]
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    stage['environment_sha256'] = bank.signature(env)
    try:
        cmd, out, err = owner.run(action, [binary['path'], *args], env, bank.P,
                                  **(bank.device.SMALL if smoke else bank.device.LARGE))
        require(device.owner.closed(cmd) and cmd['exit_code'] == 0, 'GDN fixture failed: ' + err.read_text()[-900:])
        output, log = out.read_text(), err.read_text()
        gpu, = re.findall(r'^device=(.+) driver=(\S+)$', output, re.M)
        shapes = re.findall(r'^prefill_decay_exact T=(\d+) passed=1$', output, re.M)
        require('Intel' in gpu[0] and 'B570' in gpu[0] and shapes == (['1','17'] if smoke else ['8192'])
                and output.splitlines().count('fixture_status=PASS;prefill decay only') == 1, 'incomplete B570 output')
        if not timed:
            proof = device.validation.preload_evidence(log, cmd['pid'])
            layer = directory/'validation.log'
            require(proof['loader'] == probe['loader']['path'] and (not layer.exists() or layer.stat().st_size == 0),
                    'validation failed')
            stage['validation'] = proof
        if smoke:
            require('urEnqueueKernelLaunch' in log and re.findall(r'^strata GDN prefill: precomputed_decay=1 tokens=(\d+)$',
                    log, re.M) == ['1','17'], 'actual precomputed recurrence not observed')
            stage['providers'] = bank.device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
                                                runtime_manifest=native['foundation']['runtime_manifest']))
        if timed:
            stage.update(samples(output, int(action)))
        if timed:
            require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|cleanup failed|terminate called|strata UR validation:', log),
                    'unexpected diagnostic or observer in the clean timing process')
        else:
            bank.device.diagnostics.diagnostics(log, cmd['pid'])
        stage.update(device=gpu[0], driver=gpu[1], native_pid=cmd['pid'], native_exit_code=0, compared_rows=shapes,
                     host_probe_accepted=True, clean_timing=timed)
    finally:
        device.validation.finish(stage, owner, cursor, speed['boot'])
        stage['passed'] = stage['passed'] and stage.get('driver_after') == native['driver']


def execute(action):
    native, speed, parent, build = device.coherent_inputs()
    helper = bank.basis.core.source.device.load_module('gdn_decay_build', device.fixture.__file__)
    helper.SOURCE, helper.BUILD = parent/'source', parent/'build'
    overlay, commands = recipe(helper, native)
    allowed = helper.build_inputs(native, ROOT, overlay, commands)
    controls = [str(Path(m).resolve().relative_to(bank.P)) for m in
                (__file__, device.__file__, device.fixture.__file__, device.coherent_model.__file__, device.coherent_model.kv.__file__)]
    pins = bank.stage.source.pin(bank.P, [*controls, 'tools/sycl'])
    identity = dict(source=SOURCE, parent_build=build['build_sha256'], overlay=bank.stage.hashed(overlay),
                    recipe=bank.signature(commands), controls=pins['aggregate_sha256'],
                    benchmark_environment_sha256=bank.signature(speed['environment_template']), runtime=speed['runtime_manifest']['sha256'])
    result = device.validation.committed(bank.RESULT)
    if action == 'build':
        require(KEY not in result and not ROOT.exists(), 'inspect the existing GDN generation')
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
            require(record['complete'] and record['ownership_closed'], 'incomplete GDN build')
            stage.update(passed=True, build_sha256=bank.sha(ROOT/'record.json'), binary=record['binary'])
        else:
            record = bank.previous.read(ROOT/'record.json')
            built = row['stages'][0]
            require(bank.sha(ROOT/'record.json') == built['build_sha256'] and record['binary'] == built['binary']
                    and bank.sha(record['binary']['path']) == record['binary']['sha256']
                    and all(bank.sha(p) == h for group in ('compiled','linked') for p,h in record[group].items())
                    and all(bank.sha(ROOT/p) == h for p,h in record['driver_inputs'].items()), 'GDN executable or inputs changed')
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
        require(bank.signature(current.get(KEY)) == previous, 'GDN record changed during execution')
        current[KEY] = row
        bank.stage.write(bank.RESULT, current)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in STAGES and sys.executable == bank.basis.reference.model.PYTHON, 'select a pinned GDN stage')
    with (bank.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        execute(action)
