"""Run the primary DFlash2 build's host and bounded device checks."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import dflash2_primary_prefill_build as primary

bank, builder, require = primary.bank, primary.builder, primary.require
device = bank.device
KEY = 'dflash2_primary_checks_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
MEASURED = bank.B / 'dflash2-snapshot-direct-prefill-v1' / 'record.json'
BOOT = Path('/proc/sys/kernel/random/boot_id')
ORDER = ('host', 'taps', 'restore')
COMMANDS = {
    'host': (('sycl_conversation_transfer_test', ()), ('full_context_diagnostic_host_test', ())),
    'taps': (('sycl_dflash2_test', ()),),
    'restore': (('sycl_conversation_snapshot_test', ('--restore-smoke',)),),
}
BUDGETS = {step: device.HOST if step == 'host' else device.SMALL for step in ORDER}


def build():
    row = bank.previous.read(primary.ROOT / 'record.json')
    require(row['stage'] == 'built' and row['passed'] and row['complete']
            and not row['active'] and row['ownership_closed'], 'require the completed primary build')
    native = primary.check(row)
    return row, native['native_validation_probe']


def parameters(native, probe):
    measured = bank.previous.historic(primary.PREFILL, 'dflash2_snapshot_direct_prefill_v1')
    seed = bank.previous.read(MEASURED)
    settings = bank.basis.validation.committed(bank.P / 'tools/sycl/measurement_environment.json')
    env = {k: seed['environment_template'][k] for k in settings['environment'] if not k.startswith('UR_LOG_')}
    for key in ('EnableDirectSubmission', 'NEOReadDebugKeys', 'ONEAPI_DEVICE_SELECTOR',
                'STRATA_SYCL_PREFILL_SAME_QUEUE', 'STRATA_SYCL_VERIFY_NATIVE_CAPTURE',
                'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'ZES_ENABLE_SYSMAN'):
        require(env[key] == settings['environment'][key], 'unexpected fixture safety setting: ' + key)
    runtime = measured['runtime_manifest']
    build_runtime = native['foundation']['runtime_manifest']
    require(bank.signature(seed['environment_template']) == measured['environment_template_sha256']
            and seed['runtime_manifest'] == runtime
            and bank.sha(runtime['path']) == runtime['sha256'] == bank.sha(build_runtime['path'])
            == build_runtime['sha256'] and runtime['files'] == build_runtime['files']
            and seed['native_validation_probe'] == measured['native_validation_probe'] == probe
            and seed['driver'] == measured['driver'] == native['driver']
            and seed['boot'] == measured['boot'] == BOOT.read_text().strip(),
            'recover the measured runtime and loader qualification')
    env.update(LD_DEBUG='libs', LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
               UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
    return dict(runtime_manifest=runtime, native_validation_probe=probe, environment=env,
                boot=measured['boot'], driver=measured['driver'], native_source_commit=native['native_source_commit'],
                build_sha256=bank.signature(builder.compact(native)))


def child_environment(native, envelope, step):
    directory = ROOT / step
    env = native['recipe']['environment'] if step == 'host' else envelope['environment']
    env = dict(env, TMPDIR=str(directory / 'tmp'), XDG_CACHE_HOME=str(directory / 'cache'))
    if step != 'host':
        env['UR_LOG_VALIDATION'] = 'level:warning;flush:warning;output:file,' + str(directory / 'validation.log')
    return env


def plans(native, envelope):
    stages = []
    for step in ORDER:
        commands = []
        for name, args in COMMANDS[step]:
            item = native['binaries'][name]
            commands.append(dict(target=name, argv=[item['path'], *args], binary=item))
        env = child_environment(native, envelope, step)
        stages.append(dict(step=step, commands=commands, environment_sha256=bank.signature(env), budget=BUDGETS[step]))
    return stages


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = {k: v for k, v in row.items() if k != 'environment'}
    bank.stage.write(bank.RESULT, data)


def coverage(name, output):
    if name == 'full_context_diagnostic_host_test':
        return device.cpu_cases(name, output)
    if name == 'sycl_conversation_transfer_test':
        require(output.splitlines().count('conversation_validation_test: 1656 host-only checks passed') == 1,
                'incomplete target-only and MTP host transfer coverage')
        return dict(checks=1656)
    if name == 'sycl_dflash2_test':
        require(output.splitlines().count('dflash2 fixture cases=16 PASS') == 1,
                'incomplete bounded DFlash2 kernel coverage')
        return dict(cases=16)
    count, = re.findall(r'^conversation_snapshot_test: ([1-9]\d*) checks passed \(restore smoke\)$', output, re.M)
    return dict(checks=int(count))


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect any previous primary fixture plan first')
    native, probe = build()
    envelope = parameters(native, probe)
    pin = bank.stage.source.pin(bank.P, (THIS,))
    row = dict(envelope, controller_commit=pin['revision'], controller_sha256=pin['source_sha256'][THIS],
               plans=plans(native, envelope), stage='prepared', results=[], passed=False, complete=False,
               ownership_closed=True, active=False,
               scope='Bounded same-build transfer, host extent, DFlash2 tap and snapshot smoke. Model and speed are separate.')
    ROOT.mkdir(mode=0o700)
    for label in ORDER:
        for child in ('tmp', 'cache'):
            (ROOT / label / child).mkdir(mode=0o700, parents=True)
    save(row)


def execute(step):
    row = bank.previous.read(ROOT / 'record.json')
    committed = bank.basis.validation.committed(bank.RESULT, (KEY,))
    require(committed == {k: v for k, v in row.items() if k != 'environment'} and not row['active']
            and row['ownership_closed'] and row['stage'] == ('prepared', 'host_passed', 'taps_passed')[ORDER.index(step)],
            'commit the exact preceding fixture stage')
    source, = bank.stage.source.blobs(bank.P, row['controller_commit'], (THIS,))
    require(source == (THIS, (bank.P / THIS).read_bytes()) and bank.sha(bank.P / THIS) == row['controller_sha256'],
            'recover the exact fixture controller')
    native, probe = build()
    envelope = parameters(native, probe)
    require(all(row[k] == v for k, v in envelope.items()) and row['plans'] == plans(native, envelope),
            'fixture routes or runtime changed')
    plan = row['plans'][ORDER.index(step)]
    directory = ROOT / step
    env = child_environment(native, envelope, step)
    require(bank.signature(env) == plan['environment_sha256'], 'fixture environment changed')
    cursor, = [l.removeprefix('-- cursor: ') for l in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if l.startswith('-- cursor: ')]
    owner = builder.ownership.Owner(directory / 'owned', lambda: bank.stage.write(
        directory / 'ownership.json', dict(active=owner.active is not None, commands=owner.commands)))
    outcome = dict(step=step, fixtures=[], host_probe_accepted=False)
    try:
        for cmd in plan['commands']:
            name = cmd['target']
            item, out, err = owner.run(name, cmd['argv'], env, bank.P, **plan['budget'])
            require(builder.ownership.closed(item) and item['exit_code'] == 0,
                    name + ' failed: ' + err.read_text(errors='replace')[-1600:])
            evidence = dict(target=name, exit_code=0, ownership_closed=True, **coverage(name, out.read_text()))
            if step != 'host':
                log = err.read_text()
                proof = bank.basis.validation.preload_evidence(log, item['pid'])
                layer = directory / 'validation.log'
                require(proof['loader'] == probe['loader']['path'] and 'urEnqueueKernelLaunch' in log
                        and (not layer.exists() or layer.stat().st_size == 0), 'fixture trace or validation differs')
                device.diagnostics.diagnostics(log, item['pid'])
                evidence['providers'] = device.loaded_providers(log, dict(envelope, environment=env))
                evidence['validation'] = proof
            outcome['fixtures'].append(evidence)
        outcome['host_probe_accepted'] = True
    except Exception as error:
        outcome['error'] = str(error)
    finally:
        bank.basis.validation.finish(outcome, owner, cursor, row['boot'])
        passed = outcome['passed'] and outcome.get('driver_after') == row['driver']
        complete = step == ORDER[-1] and passed
        row['results'].append(outcome)
        row.update(stage=step + '_passed' if passed else 'failed', passed=complete, complete=complete,
                   active=owner.active is not None, ownership_closed=outcome['ownership_closed'])
        save(row)
    require(passed, outcome.get('error', 'fixture validation, driver or ownership failed'))
    print(step + ' checks passed. Commit before the next stage.', flush=True)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', *ORDER), 'select one fixture stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        prepare() if action == 'prepare' else execute(action)
