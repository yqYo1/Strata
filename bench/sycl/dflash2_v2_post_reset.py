"""Check device command completion after the closed V2 DFlash2 stall."""

import fcntl
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_v2_generation as generation
import prefill_split_store as store

v2, bank, basis, require = generation.v2, generation.bank, generation.basis, generation.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_v2_post_reset_v1'
ROOT = bank.B / KEY.replace('_', '-')
FAILURE = '1a418e7bb'
FIXTURE_SOURCE = 'tests/sycl/prefill_attention_bank.cpp'


def inputs(revision):
    base = v2.baseline()
    failure = bank.previous.historic(FAILURE, generation.KEY)
    require(failure['stage'] == 'failed' and failure['ownership_closed'] and not failure['active']
            and failure['binary'] == base['binary'] and failure['processes'][0]['fresh_faults']
            and failure['processes'][0]['boot_unchanged'] and not base['driver']['devcoredump'],
            'require the closed incident on the same boot and driver')
    native = bank.previous.read(v2.upload.build.ROOT / 'record.json')
    fixture = native['binaries'][store.FIXTURE]
    require(bank.sha(fixture['path']) == fixture['sha256'] and Path(fixture['path']).stat().st_size == fixture['bytes'],
            'retained store fixture changed')
    env = generation.environment(base, dict(root=str(ROOT), index=1))
    env = generation.model.native_environment(base, env)
    pins = generation.experiment().pins(revision)
    fixed = [THIS, store.THIS, store.attention.CONTROLLER, FIXTURE_SOURCE]
    pins.update(bank.stage.source.pin(bank.P, fixed)['source_sha256'])
    for name in fixed:
        require((bank.P / name).read_bytes() == generation.suite.gen.git('show', revision + ':' + name),
                'changed device proof input: ' + name)
    plan = dict(source_commit=revision, source_sha256=bank.signature(pins), source_files=len(pins),
                native_source_commit=base['native_source_commit'], native_build_sha256=base['native_build_sha256'],
                fixture=fixture, command=[fixture['path'], 'correctness'], limits=store.attention.BUDGET,
                driver=base['driver'], boot=base['boot'], runtime=base['runtime_manifest'],
                environment_sha256=bank.signature(env), failure=FAILURE,
                scope='Four known BF16 store cases and checked retirements after automatic reset. No model execution.')
    return plan, env, base['native_validation_probe']


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def run(row):
    plan, env, probe = inputs(row['plan']['source_commit'])
    require(plan == row['plan'] and row == basis.validation.committed(bank.RESULT, (KEY,))
            and row['stage'] == 'prepared' and not (ROOT / 'owned').exists(),
            'commit the one-use device proof before launch')
    cursor = next(line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: '))
    owner = store.attention.ownership.Owner(ROOT / 'owned')
    result = dict(host_probe_accepted=False)
    try:
        command, out, err = owner.run('device', plan['command'], env, bank.P, **plan['limits'])
        require(command['exit_code'] == 0 and store.attention.ownership.closed(command), 'device fixture failed')
        basis.comparison.require_quiet_validation(ROOT)
        result.update(store.attention.parse(out.read_text(), 'correctness'), native_exit_code=0,
                      validation=store.validate_runtime(err.read_text(), command['pid'], probe),
                      host_probe_accepted=True)
    except Exception as error:
        result['error'] = repr(error)
    finally:
        basis.validation.finish(result, owner, cursor, plan['boot'])
        passed = result['passed'] = result['passed'] and result.get('driver_after') == plan['driver']
        row.update(stage='closed' if passed else 'failed', passed=passed, complete=passed,
                   active=bool(owner.active), ownership_closed=result['ownership_closed'], result=result)
        save(row)
    print(result, flush=True)
    require(row['passed'], result.get('error', 'post-reset completion proof failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'run') and sys.executable == basis.reference.model.PYTHON, 'use the pinned Python')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT), 'fresh device proof only')
            revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
            plan, _, _ = inputs(revision)
            for folder in ('cache', 'tmp'):
                (ROOT / folder).mkdir(parents=True, mode=0o700)
            save(dict(plan=plan, stage='prepared', passed=False, complete=False, active=False, ownership_closed=True))
        else:
            run(bank.previous.read(ROOT / 'record.json'))
