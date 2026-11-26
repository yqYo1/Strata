"""Qualify the same-build ranked refill lease with bounded host and device cases."""

import fcntl
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_decode_ranked_refill_build as build
import dflash2_primary_checks as common

bank, builder, require = build.bank, build.builder, build.require
validation, device = build.validation, bank.device
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_ranked_refill_checks_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('host', 'ranked-refill', 'ranked-refill-copy-failure', 'ranked-refill-publish-failure')
FAIL_STOP = 'Strata SYCL cleanup: ranked refill fixture verified poisoned lease; retaining owners until process exit'
PLAN = dict(order=list(ORDER), scope='Same-build host lease checks and a 130-slot device cache with two replaced rows.',
            checks='Read back prefix, borrowed payloads and both residency tables. Refuse invalid maps before copies. '
                   'After separately injected transfer or publication failure, require blocked readers and the expected '
                   'fail-stop marker. Require no new Xe fault or runtime validation warning in any case.',
            stop='Stop on the first unexpected exit, changed premise or driver fault. Model RESTORE and speed are separate.')


def inputs(revision):
    native = bank.previous.read(build.ROOT / 'record.json')
    require(build.retained.closed(native, 'built'), 'finish the coherent ranked-refill build')
    original = build.check(native)
    envelope = common.parameters(native, original['native_validation_probe'])
    sources = {}
    for name in (THIS, build.THIS, common.THIS):
        path = bank.P / name
        require(path.read_bytes() == validation.git('show', revision + ':' + name), 'commit exact fixture input: ' + name)
        sources[name] = bank.sha(path)
    cases, environments = [], {}
    for name in ORDER:
        host = name == 'host'
        root = ROOT / name
        item = native['binaries'][build.FIXTURES[0 if host else 1]]
        args = [] if host else ['--case', name]
        env = dict(native['recipe']['environment'] if host else envelope['environment'],
                   TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
        if not host:
            env['UR_LOG_VALIDATION'] = 'level:warning;flush:warning;output:file,' + str(root / 'validation.log')
        cases.append(dict(name=name, binary=item, command=[item['path'], *args], expected_exit=int(name.endswith('-failure')),
                          env_sha256=bank.signature(env), limits=device.HOST if host else device.SMALL))
        environments[name] = env
    pinned = {field: envelope[field] for field in ('boot', 'driver', 'native_source_commit', 'build_sha256',
                                                    'native_validation_probe', 'runtime_manifest')}
    return dict(pinned, source_commit=revision, sources=sources, plan=PLAN, cases=cases), environments


def coverage(name, stdout, stderr):
    lines = stdout.splitlines()
    if name == 'host':
        tests = {line for line in lines if line.startswith('prefill_scratch case=') and line.endswith(' status=PASS')}
        require(len(tests) == 40 and 'prefill_scratch case=loan_checked_refill_validation status=PASS' in tests
                and lines[-1] == 'prefill_scratch_fixture cases=40 status=PASS' and not stderr,
                'incomplete same-build host lease checks')
        return dict(cases=len(tests))
    prefix = 'prefill_scratch_device case='
    require(lines.count(prefix + 'ranked_refill_rejects_invalid_maps_before_copy status=PASS') == 1,
            'the invalid-map preflight did not pass')
    if name.endswith('-failure'):
        require(lines.count(f'prefill_ranked_refill failure={name} owners_live=1 status=PASS') == 1
                and FAIL_STOP in stderr.splitlines() and 'prefill_scratch_device_complete=1' not in stdout,
                'the injected failure did not retain cache ownership')
        return dict(fail_stop=True, admission_closed=True)
    require(lines.count(prefix + 'ranked_refill_publishes_payload_and_both_maps status=PASS') == 1
            and lines[-1] == 'prefill_scratch_device_complete=1 model_PP=UNRUN' and FAIL_STOP not in stderr,
            'refill did not verify both maps and payloads')
    return dict(payloads_verified=True, map_verified=True)


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = row
    bank.stage.write(bank.RESULT, data)


def execute(row, name, env):
    case = next(c for c in row['cases'] if c['name'] == name)
    root = ROOT / name
    require(not (root / 'execution.json').exists() and bank.sha(case['binary']['path']) == case['binary']['sha256']
            and bank.signature(env) == case['env_sha256'], 'the exact new fixture case and environment are required')
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: ')]
    owner = builder.ownership.Owner(root / 'owned', lambda: bank.stage.write(
        root / 'execution.json', dict(active=owner.active is not None, commands=owner.commands)))
    outcome = dict(name=name, host_probe_accepted=False)
    try:
        entry, out, err = owner.run(name, case['command'], env, bank.P, **case['limits'])
        outcome['exit_code'] = entry['exit_code']
        require(builder.ownership.closed(entry) and entry['exit_code'] == case['expected_exit'], 'unexpected fixture exit')
        log = err.read_text()
        outcome['coverage'] = coverage(name, out.read_text(), log)
        if name != 'host':
            proof = validation.preload_evidence(log, entry['pid'])
            layer = root / 'validation.log'
            require(proof['loader'] == row['native_validation_probe']['loader']['path']
                    and 'urEnqueueUSMMemcpy' in log and (not layer.exists() or layer.stat().st_size == 0)
                    and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called', log),
                    'missing copy trace or unexpected runtime validation')
            outcome['providers'] = device.loaded_providers(log, dict(row, environment=env))
            outcome['validation'] = proof
        outcome['host_probe_accepted'] = True
    except Exception as error:
        outcome['error'] = str(error)
    finally:
        validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] &= outcome.get('driver_after') == row['driver']
        row['outcomes'][name] = outcome
        complete = outcome['passed'] and name == ORDER[-1]
        row.update(stage='closed' if complete else 'running' if outcome['passed'] else 'failed',
                   passed=complete, complete=complete, ownership_closed=outcome['ownership_closed'],
                   active=owner.active is not None)
        save(row)
    require(outcome['passed'], outcome.get('error', 'ranked refill fixture validation failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned Python and one ranked-refill fixture stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one new fixture cohort')
            revision = validation.git('rev-parse', 'HEAD').decode().strip()
            row, _ = inputs(revision)
            for name in ORDER:
                for child in ('tmp', 'cache'):
                    (ROOT / name / child).mkdir(mode=0o700, parents=True)
            save(dict(row, stage='prepared', outcomes={}, passed=False, complete=False, ownership_closed=True, active=False))
        else:
            row = bank.previous.read(ROOT / 'record.json')
            pinned, environments = inputs(row['source_commit'])
            require(row == validation.committed(bank.RESULT, (KEY,)) and all(row[k] == v for k, v in pinned.items())
                    and row['stage'] in ('prepared', 'running') and row['ownership_closed'] and not row['active']
                    and list(row['outcomes']) == list(ORDER[:ORDER.index(action)]), 'commit the exact next fixture case')
            execute(row, action, environments[action])
