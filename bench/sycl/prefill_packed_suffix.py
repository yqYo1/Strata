"""Compile the long-prefill caller and check the packed KV suffix on the B570."""

import fcntl
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_mixed_gather_measure as device

bank, require = device.bank, device.require
parent_root = device.coherent_model.MODEL_ROOT
ROOT = bank.B / 'prefill-packed-suffix-v1'
KEY = 'prefill_packed_suffix_v1'
SOURCE = 'fa2b291f89c3dbbfd68aaa7a44d9af4d18ff41be'
TARGET = 'sycl_qsa_bf16_prefix_test'
HEADER = 'include/strata/kernels/qsa_bf16_prefix.hpp'
KERNEL = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
CALLER = 'src/prefill/prefill.cpp'
TEST = 'tests/sycl/qsa_bf16_prefix.cpp'


def recipe(helper, native):
    source, build = helper.SOURCE, helper.BUILD
    overlay = {}
    for path in (HEADER, KERNEL, CALLER, TEST):
        require((source / path).read_bytes() == helper.git_file(device.coherent_model.DISPATCH_COUNTS, path),
                'the overlay must apply to its actual measured parent: ' + path)
        overlay[path] = helper.git_file(SOURCE, path)
    # All other kernel archive members and the test keep identical type definitions.
    header = overlay[HEADER]
    start, end = header.index(b'// Long prefill with a private'), header.index(b'// Diagnostic-only online state')
    require(header[:start] + header[end:] == (source / HEADER).read_bytes()
            and b'bool prefill_qsa_bf16_ordered_exp_suffix(' in header[start:end], 'expected only a free declaration')
    units = bank.previous.read(build / 'compile_commands.json')
    commands = []
    objects = {}
    for path, target, name in ((KERNEL, 'strata_kernels', 'qsa_bf16_prefix.cpp.o'),
                               (CALLER, 'strata_prefill', 'prefill.cpp.o'), (TEST, TARGET, 'fixture.o')):
        unit, = [u for u in units if u['file'] == str(source / path)
                 and u['output'].startswith('CMakeFiles/' + target + '.dir/')]
        args = shlex.split(unit['command'])
        require(unit['directory'] == str(build) and args[-4:] == ['-o', unit['output'], '-c', unit['file']]
                and '-fsycl' in args, 'expected the production compile unit for ' + path)
        out = str(ROOT / name)
        commands.append([args[0], '-I' + str(ROOT / 'include'), '-iquote', str((source / path).parent),
                         *args[1:-4], '-MD', '-MF', out + '.d', '-o', out, '-c', str(ROOT / path)])
        objects[path] = unit['output']
    archive = 'libstrata_kernels.a'
    members = subprocess.check_output(['/usr/bin/ar', 't', archive], cwd=build, timeout=20, text=True).splitlines()
    originals = {Path(p).name: Path(p) for p in native['actual_inputs']['objects']
                 if Path(p).is_relative_to(build / 'CMakeFiles/strata_kernels.dir')}
    require(len(originals) == len(members) == len(set(members)) and set(members) == originals.keys()
            and originals['qsa_bf16_prefix.cpp.o'] == build / objects[KERNEL], 'kernel archive membership differs')
    originals['qsa_bf16_prefix.cpp.o'] = ROOT / 'qsa_bf16_prefix.cpp.o'
    commands.append(['/usr/bin/ar', 'rcsD', str(ROOT / archive), *(str(originals[n]) for n in members)])
    link = shlex.split(subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', TARGET],
                       cwd=build, timeout=20, text=True).splitlines()[-1])
    require(link[:2] == [':', '&&'] and link[-2:] == ['&&', ':'] and '-Wl,-t' in link
            and link[link.index(objects[TEST]):-2] == [objects[TEST], '-o', TARGET, archive, 'libstrata_core.a',
                                                       'libstrata_sycl_runtime.a'], 'unexpected fixture link inputs')
    replace = {objects[TEST]: str(ROOT / 'fixture.o'), TARGET: str(ROOT / TARGET), archive: str(ROOT / archive)}
    commands.append([link[2], '-v', *(replace.get(a, a) for a in link[3:-2])])
    return overlay, commands


def run_device(record, native, speed, outcome):
    directory = ROOT / 'device'
    for name in ('tmp', 'cache'):
        (directory / name).mkdir(mode=0o700, parents=True)
    owner = device.owner.Owner(directory / 'owned', lambda: bank.stage.write(directory / 'execution.json', owner.commands))
    seed, probe = speed['environment_template'], speed['native_validation_probe']
    env = {k: v.replace(seed['STRATA_MEASUREMENT_ROOT'], str(directory)) for k, v in seed.items()}
    env.update(STRATA_TEST_MAIN_GATES='0', LD_DEBUG='libs', LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(directory / 'validation.log'))
    cursor, = [s.removeprefix('-- cursor: ') for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        timeout=15, text=True).splitlines() if s.startswith('-- cursor: ')]
    outcome['environment_sha256'] = bank.signature(env)
    try:
        cmd, out, err = owner.run('suffix', [record['binary']['path'], '--packed-suffix'], env, bank.P, **bank.device.SMALL)
        require(device.owner.closed(cmd) and cmd['exit_code'] == 0, 'packed suffix fixture: ' + err.read_text()[-900:])
        output, log = out.read_text(), err.read_text()
        gpu, = re.findall(r'^device=([^\n]+) driver=(\S+)$', output, re.M)
        validation = device.validation.preload_evidence(log, cmd['pid'])
        layer = directory / 'validation.log'
        require('B570' in gpu[0] and validation['loader'] == probe['loader']['path'] and
                (not layer.exists() or not layer.stat().st_size) and 'urEnqueueKernelLaunch' in log and
                output.splitlines().count('packed_suffix=PASS page_size=4 live_cells=71 transitions=14 invalid_extents=4') == 1
                and 'fixture_status=PASS' in output, 'missing real device, transitions, rejection or tracing')
        bank.device.diagnostics.diagnostics(log, cmd['pid'])
        providers = bank.device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
                                                           runtime_manifest=native['foundation']['runtime_manifest']))
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=validation,
                       gpu=gpu, providers=providers, transitions=14, invalid_extents=4)
    finally:
        device.validation.finish(outcome, owner, cursor, speed['boot'])
        outcome['passed'] = outcome['passed'] and outcome.get('driver_after') == native['driver']


def execute(action):
    native, speed, _, build = device.coherent_inputs()
    helper = bank.basis.core.source.device.load_module('packed_suffix_build', device.fixture.__file__)
    helper.SOURCE, helper.BUILD = parent_root / 'source', parent_root / 'build'
    overlay, commands = recipe(helper, native)
    allowed = helper.build_inputs(native, ROOT, overlay, commands)
    controls = [str(Path(f).resolve().relative_to(bank.P)) for f in
                (__file__, device.__file__, device.fixture.__file__, device.coherent_model.__file__, device.coherent_model.kv.__file__)]
    pins = bank.stage.source.pin(bank.P, [*controls, 'tools/sycl'])
    require(set(controls) <= pins['source_sha256'].keys(), 'commit every build and run controller')
    identity = dict(source=SOURCE, parent_build=build['build_sha256'], controls=pins['aggregate_sha256'],
                    overlay=bank.stage.hashed(overlay), recipe=bank.signature(commands))
    if action == 'check':
        print('Ready to compile the actual prefill caller, pack kernel and page-four fixture', flush=True)
        return
    before = device.validation.committed(bank.RESULT)
    if action == 'build':
        require(KEY not in before and not ROOT.exists(), 'inspect the earlier packed-suffix generation first')
        row = dict(**identity, controller_commit=pins['revision'], stages=[], model_tested=False)
    else:
        row = before[KEY]
        archived = dict(bank.stage.source.blobs(bank.P, row['controller_commit'], pins['source_sha256']))
        require(bank.stage.hashed(archived) == pins['source_sha256']
                and all(row[k] == v for k, v in identity.items()) and len(row['stages']) == 1
                and row['stages'][0]['passed'] and row['stages'][0]['ownership_closed'], 'need a matching clean build')
    outcome = dict(stage=action, complete=False, passed=False)
    try:
        if action == 'build':
            helper.build_fixture(ROOT, TARGET, native, overlay, commands, allowed, identity)
            record = bank.previous.read(ROOT / 'record.json')
            require(record['complete'] and record['ownership_closed'], 'incomplete component build')
            outcome.update(complete=True, passed=True, ownership_closed=True, build_sha256=bank.sha(ROOT / 'record.json'),
                           binary=record['binary'], caller_object=bank.sha(ROOT / 'prefill.cpp.o'))
        else:
            record = bank.previous.read(ROOT / 'record.json')
            built = row['stages'][0]
            require(bank.sha(ROOT / 'record.json') == built['build_sha256'] and record['binary'] == built['binary']
                    and bank.sha(record['binary']['path']) == record['binary']['sha256']
                    and bank.sha(ROOT / 'prefill.cpp.o') == built['caller_object']
                    and all(bank.sha(p) == h for group in ('compiled', 'linked') for p,h in record[group].items())
                    and all(bank.sha(ROOT / p) == h for p,h in record['driver_inputs'].items()),
                    'original sources, archives, compiler dependencies or final binary changed')
            run_device(record, native, speed, outcome)
            require(outcome['passed'], 'device or ownership failure')
    except BaseException as exc:
        outcome['error'] = repr(exc)
        raise
    finally:
        if action == 'build' and (ROOT / 'record.json').exists():
            outcome['ownership_closed'] = bank.previous.read(ROOT / 'record.json')['ownership_closed']
        updated = bank.previous.read(bank.RESULT)
        require(updated.get(KEY) == before.get(KEY), 'result changed during execution')
        row['stages'].append(outcome)
        updated[KEY] = row
        bank.stage.write(bank.RESULT, updated)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('check', 'build', 'device') and sys.executable == bank.basis.reference.model.PYTHON,
            'use the pinned interpreter and one packed-suffix check')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        execute(action)
