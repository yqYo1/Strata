"""Qualify attention-store correctness on the measured split-prefill build."""

import fcntl
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_split_upload_compare as upload
import prefill_attention_bank_component as attention
from engine_once import native_environment

bank, basis, require = upload.bank, upload.basis, upload.require
P, RESULT = bank.P, bank.RESULT
THIS = str(Path(__file__).resolve().relative_to(P))
ROOT = bank.B / 'prefill-split-direct-store-component-v1'
KEY = 'prefill_split_direct_store_component_v1'
CPU_EVIDENCE = '1046b416442c889b90c7c9ea759d94970ae897a7'
FIXTURE = 'sycl_prefill_attention_bank_test'


def inputs(revision):
    context = upload.experiment('profile')
    row = bank.previous.read(context.ROOT / 'record.json')
    evidence = bank.previous.historic(CPU_EVIDENCE, context.KEY)
    native, prior, providers, _ = upload.profile_basis(2)
    require(context.compact(row) == evidence and row['stage'] == 'closed'
            and row['complete'] and row['ownership_closed'] and row['passed'] and not row['active']
            and row['binary'] == native['binaries']['strata']
            and row['native_source_commit'] == native['native_source_commit']
            and row['driver'] == native['driver']
            and row['boot'] == attention.BOOT.read_text().strip()
            and row['native_validation_probe'] == prior['probe']
            and all(bank.sha(path) == digest for path, digest in providers.items()),
            'recover the exact closed accounting build and its runtime settings')
    env = upload.model.ordinary_environment(row, str(ROOT), row['observer_admission']['ordinary_control'])
    env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    env = native_environment(row, env)
    pins = context.pins(revision)
    pins.update(bank.stage.source.pin(P, [THIS, attention.CONTROLLER])['source_sha256'])
    fixture = native['binaries'][FIXTURE]
    plan = dict(source_commit=revision, sources_sha256=bank.signature(pins), sources_count=len(pins),
                native_source_commit=native['native_source_commit'],
                native_build_sha256=bank.signature(upload.build.builder.compact(native)),
                engine=native['binaries']['strata'], fixture=fixture,
                command=[fixture['path'], 'correctness'], budgets=attention.BUDGET,
                environment_sha256=bank.signature(env), providers_sha256=bank.signature(providers),
                observer_reference=CPU_EVIDENCE, boot=row['boot'], driver=row['driver'],
                scope='Exact BF16 store bits, guards and checked cancellation on the split-build fixture.')
    return plan, env, row['native_validation_probe']


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    results = bank.previous.read(RESULT)
    results[KEY] = row
    bank.stage.write(RESULT, results)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(RESULT), 'fresh store fixture only')
    revision = bank.stage.source.pin(P, [THIS])['revision']
    plan, _, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    save(dict(plan=plan, stage='prepared', active=False, passed=False,
              complete=False, ownership_closed=True))


def run():
    row = bank.previous.read(ROOT / 'record.json')
    require(row == basis.validation.committed(RESULT, (KEY,)) and row['stage'] == 'prepared'
            and row['ownership_closed'] and not row['active'], 'commit the fresh fixture plan before launch')
    plan, env, probe = inputs(row['plan']['source_commit'])
    require(row['plan'] == plan and bank.sha(plan['fixture']['path']) == plan['fixture']['sha256'],
            'fixture command, sources or runtime changed')
    cursor = next(line.split(': ', 1)[1] for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if line.startswith('-- cursor: '))
    owner = attention.ownership.Owner(ROOT / 'owned')
    result = dict(host_probe_accepted=False)
    try:
        command, output, errors = owner.run('fixture', plan['command'], env, P, **plan['budgets'])
        require(attention.ownership.closed(command) and command['exit_code'] == 0, 'native store fixture failed')
        basis.comparison.require_quiet_validation(ROOT)
        log = errors.read_text()
        validation = basis.validation.preload_evidence(log, command['pid'])
        require(validation['loader'] == probe['loader']['path']
                and not any(error in log for error in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'retirement failed')),
                'unexpected store fixture runtime event')
        result.update(attention.parse(output.read_text(), 'correctness'),
                      native_exit_code=0, validation=validation, host_probe_accepted=True)
    except Exception as error:
        result['error'] = repr(error)
    finally:
        basis.validation.finish(result, owner, cursor, plan['boot'])
        passed = result['passed'] = result['passed'] and result.get('driver_after') == plan['driver']
        row.update(stage='closed' if passed else 'failed', passed=passed, complete=passed,
                   active=bool(owner.active), ownership_closed=result['ownership_closed'], result=result)
        save(row)
    require(row['passed'], result.get('error', 'store fixture did not qualify'))
    print({k: result[k] for k in ('cases', 'compared_BF16', 'fresh_faults')}, flush=True)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'run'),
            'use the benchmark environment and select prepare or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[action]()
