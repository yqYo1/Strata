"""Validate the new shared-plan combine on the candidate's Arc build."""

import fcntl
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_decode_t4_build as build

bank, require, validation = build.bank, build.require, build.validation
KEY = 'dflash2_decode_t4_shared_plan_v2'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
SUMMARY = 'decode MoE combine numerical=PASS guards=PASS nodes=3/1/1 completed=1'
HOST = 'sycl_verify_window_policy_test'
HOST_SUMMARY = ('verify-window-policy: passed selections=30 centering=18 native-half=26 '
                'dpas-head=1 dpas-dense=4 phase-roundtrip=1 invalid-keys=4 retirement=1 route-budget=64')
LIMITS = dict(wall=180, cpu=600, rss_cap=4 << 30, text_cap=24 << 20,
              file_cap=32 << 20, total_cap=64 << 20)
HOST_LIMITS = dict(wall=30, cpu=30, rss_cap=1 << 30, text_cap=256 << 10,
                   file_cap=1 << 20, total_cap=4 << 20)


def fields(revision):
    native = bank.previous.read(build.ROOT / 'record.json')
    require(build.retained.closed(native, 'built'), 'require the closed coherent T4 build')
    build.check(native)
    controls = bank.stage.source.pin(bank.P, [THIS])['source_sha256']
    require(bank.stage.hashed(dict(build.source.blobs(bank.P, revision, controls))) == controls,
            'commit the exact Arc check controller')
    evidence = build.retained.baseline()
    probe = native['native_validation_probe']
    binary = native['binaries'][build.fixture.TARGET]
    env = dict(native['recipe']['environment'], TMPDIR=str(ROOT / 'tmp'),
               XDG_CACHE_HOME=str(ROOT / 'cache'), LD_DEBUG='libs', LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
               UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['NEOReadDebugKeys'] == '1'
            and env['EnableDirectSubmission'] == '0', 'pinned Arc safety envelope changed')
    return dict(source_commit=revision, controller_sources=controls,
                native_source_commit=native['native_source_commit'],
                native_build_sha256=bank.signature(build.builder.compact(native)),
                boot=evidence['boot'], driver=native['driver'], native_validation_probe=probe,
                runtime_manifest=native['runtime_manifest'], binary=binary, environment=env,
                host_binary=native['binaries'][HOST], host_environment=native['recipe']['environment'],
                host_expected=HOST_SUMMARY, host_limits=HOST_LIMITS,
                command=[binary['path'], 'validate', '--check-shared-plan'], limits=LIMITS,
                expected=SUMMARY, resident_counts=[0, 1, 2, 3, 4, 5, 10, 39, 40])


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    results = bank.previous.read(bank.RESULT)
    results[KEY] = row
    bank.stage.write(bank.RESULT, results)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh Arc check')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    planned = fields(revision)
    ROOT.mkdir(mode=0o700)
    for sub in ('tmp', 'cache'):
        (ROOT / sub).mkdir(mode=0o700)
    save(dict(planned, stage='prepared', active=False, ownership_closed=True, passed=False, complete=False))


def execute():
    row = bank.previous.read(ROOT / 'record.json')
    require(validation.committed(bank.RESULT, (KEY,)) == row and row['stage'] == 'prepared'
            and not row['active'] and row['ownership_closed'] and not (ROOT / 'owned').exists(),
            'commit the exact fresh Arc check plan')
    require(all(row[k] == v for k, v in fields(row['source_commit']).items()), 'Arc check inputs changed')
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True,
        timeout=15).splitlines() if line.startswith('-- cursor: ')]
    owner = build.builder.ownership.Owner(ROOT / 'owned',
        lambda: bank.stage.write(ROOT / 'execution.json', owner.commands))
    try:
        host, out, err = owner.run('host-policy', [row['host_binary']['path']],
                                   row['host_environment'], ROOT, **HOST_LIMITS)
        require(build.builder.ownership.closed(host) and host['exit_code'] == 0
                and out.read_text().strip() == HOST_SUMMARY and err.stat().st_size == 0,
                'same-build host policy check failed')
        row['host_summary'] = HOST_SUMMARY
        entry, out, err = owner.run('shared-plan', row['command'], row['environment'], ROOT, **LIMITS)
        require(build.builder.ownership.closed(entry) and entry['exit_code'] == 0,
                'Arc shared-plan fixture did not complete')
        errors = err.read_text()
        observed = validation.preload_evidence(errors, entry['pid'])
        warning = ROOT / 'validation.log'
        require(out.read_text().strip() == SUMMARY and observed['loader'] == row['native_validation_probe']['loader']['path']
                and 'urEnqueue' in errors and (not warning.exists() or warning.stat().st_size == 0),
                'Arc numerical, bounds or runtime gate failed')
        row.update(validation=observed, providers=bank.device.loaded_providers(errors, row),
                   summary=SUMMARY, host_probe_accepted=True, native_exit_code=entry['exit_code'])
    except Exception as exc:
        row['error'] = repr(exc)
    finally:
        validation.finish(row, owner, cursor, row['boot'])
        ready = row['passed'] and row.get('driver_after') == row['driver'] and 'summary' in row
        row.update(stage='closed' if ready else 'failed', passed=ready, complete=ready,
                   active=owner.active is not None)
        save(row)
    require(row['passed'], row.get('error', 'Arc validation or owner closure failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON
            and action in ('prepare', 'validate'), 'one pinned shared-plan check stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        prepare() if action == 'prepare' else execute()
