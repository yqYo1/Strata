"""Build and check the isolated 32-slot SYCL expert-upload fixture."""

import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_primary_checks as checks

primary, bank, builder, device = checks.primary, checks.bank, checks.builder, checks.device
require, validation = checks.require, checks.bank.basis.validation
ROOT = bank.B / 'dflash2-mode16-batch-fixture-v1'
SOURCE = 'tests/sycl/expert_bank.cpp'
ADMISSION = 'src/prefill/layer_major.inc'
TARGET = 'sycl_expert_bank_test'
OBJECT = f'CMakeFiles/{TARGET}.dir/{SOURCE}.o'
SUPPORT = f'CMakeFiles/{TARGET}.dir/src/core/expert_cache.cpp.o'
RUNTIME = 'libstrata_sycl_runtime.a'
BASE_EVIDENCE = '0ab86d36b'
BUILD_BUDGET = dict(wall=120, cpu=180, rss_cap=8 << 30, text_cap=4 << 20,
                    file_cap=32 << 20, total_cap=96 << 20)


def plan():
    require(not subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=bank.P, timeout=15),
            'commit the exact fixture and controller first')
    build = primary.ROOT / 'build'
    record = bank.previous.read(primary.ROOT / 'record.json')
    require(builder.compact(record) == bank.previous.historic(BASE_EVIDENCE, primary.KEY)
            and all(record[k] for k in ('complete', 'passed', 'ownership_closed')) and not record['active'],
            'exact closed primary build required')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    changed = validation.git('diff', '--name-only', record['native_source_commit'], revision, '--',
                             'CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'tests')
    require(set(changed.decode().splitlines()) == {SOURCE, ADMISSION},
            'a fixture, runtime or shared header changed outside the declared unit')
    census = record['actual_inputs']
    inputs = record['configured_files'] | census['compiled'] | census['generated'] | census['links']
    inputs.update({str(build / SUPPORT): census['objects'][str(build / SUPPORT)],
                   str(build / RUNTIME): census['produced'][str(build / RUNTIME)]})
    external = record['external_inputs']
    require(census['uncovered'] == 0 and not census['partial_archive_reuse']
            and bank.sha(external['path']) == external['sha256'],
            'original compile/link audit or SDK manifest changed')
    require(all(bank.sha(p) == digest for p, digest in
                (inputs | bank.previous.read(Path(external['path'])) | record['build_tools']).items()),
            'a retained compile, link or toolchain input changed')

    db = json.loads((build / 'compile_commands.json').read_bytes())
    entry, = (e for e in db if e['output'] == OBJECT)
    commands = subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', TARGET],
                                       cwd=build, text=True, timeout=15)
    link = shlex.split(commands.splitlines()[-1])
    require(link[:2] == [':', '&&'] and link[-2:] == ['&&', ':'], 'unexpected link command wrapper')
    link = link[2:-2]
    compile_argv = shlex.split(entry['command'])
    require(link[0] == compile_argv[0] and all(arg in link for arg in (OBJECT, SUPPORT, RUNTIME, '-lze_loader'))
            and compile_argv[compile_argv.index('-o') + 1] == OBJECT
            and compile_argv[-1] == str(primary.ROOT / 'source' / SOURCE)
            and link[link.index('-o') + 1] == TARGET and Path(entry['directory']) == build,
            'fixture build and link target changed')
    fixture_obj = str(ROOT / 'expert_bank.o')
    compile_argv[-1] = str(bank.P / SOURCE)
    compile_argv[compile_argv.index('-o') + 1] = fixture_obj
    linked = ROOT / TARGET
    link[link.index(OBJECT)] = fixture_obj
    link[link.index('-o') + 1] = str(linked)
    return record, revision, compile_argv, link, linked


def main():
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        record, revision, compile_argv, link, fixture = plan()
        envelope = checks.parameters(record, record['native_validation_probe'])
        require(builder.guards.driver_guard() == envelope['driver'], 'fixture driver changed')
        cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
            ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
            text=True, timeout=15).splitlines() if line.startswith('-- cursor: ')]
        ROOT.mkdir(mode=0o700)
        scratch = {key: str(ROOT / name) for key, name in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'))}
        for path in scratch.values():
            Path(path).mkdir(mode=0o700)
        owner = builder.ownership.Owner(ROOT / 'owned')
        outcome = dict(source_commit=revision, primary_evidence=BASE_EVIDENCE, host_probe_accepted=False)
        build = primary.ROOT / 'build'
        try:
            build_env = dict(record['recipe']['environment'], **scratch)
            for label, argv in (('compile-fixture', compile_argv), ('link-fixture', link)):
                item, _, stderr = owner.run(label, argv, build_env, build, **BUILD_BUDGET)
                require(builder.ownership.closed(item) and item['exit_code'] == 0,
                        label + ' failed: ' + stderr.read_text(errors='replace')[-1600:])
            binary = dict(path=str(fixture), sha256=bank.sha(fixture), bytes=fixture.stat().st_size)
            outcome.update(binary=binary, build_environment_sha256=bank.signature(build_env))
            env = dict(envelope['environment'], **scratch)
            env['UR_LOG_VALIDATION'] = 'level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log')
            require(binary['sha256'] == bank.sha(fixture), 'fixture executable changed before launch')
            item, stdout, stderr = owner.run('device-fixture', [str(fixture), '--split-upload-bank32'],
                                            env, ROOT, **device.SMALL)
            require(builder.ownership.closed(item) and item['exit_code'] == 0,
                    'device fixture failed: ' + stderr.read_text(errors='replace')[-1600:])
            lines = stdout.read_text().splitlines()
            expected = [
                f'split_prefill_upload_fixture PASS stride=2662400 coalesced=0 cancellation={cancel} '
                f'groups={groups} experts={experts} copy_commands={experts} '
                f'payload_bytes={experts * 2662400} guards=PASS checked_drain=PASS'
                for cancel, groups, experts in ((0, 5, 27), (1, 1, 16))
            ]
            log = stderr.read_text()
            probe = validation.preload_evidence(log, item['pid'])
            device.diagnostics.diagnostics(log, item['pid'])
            validation_log = ROOT / 'validation.log'
            require(lines == expected and 'urEnqueueUSMMemcpy' in log
                    and probe['loader'] == envelope['native_validation_probe']['loader']['path']
                    and (not validation_log.exists() or validation_log.stat().st_size == 0),
                    'upload counts, payload, guards, or API validation differ')
            providers = device.loaded_providers(log, dict(envelope, environment=env))
            outcome.update(cases=lines, validation=probe, providers=providers, host_probe_accepted=True)
        except Exception as error:
            outcome['error'] = str(error)
        finally:
            validation.finish(outcome, owner, cursor, envelope['boot'])
            outcome['passed'] = outcome['passed'] and outcome['driver_after'] == envelope['driver']
            outcome['commands'] = [dict(label=cmd['label'], argv=cmd['argv'],
                environment_sha256=bank.signature(cmd['environment']), exit_code=cmd.get('exit_code'),
                ownership_closed=builder.ownership.closed(cmd)) for cmd in owner.commands]
            bank.stage.write(ROOT / 'record.json', outcome)
        print(json.dumps({k: v for k, v in outcome.items() if k != 'commands'}), flush=True)
        require(outcome['passed'], outcome.get('error', 'fixture validation, driver or ownership failed'))


if __name__ == '__main__':
    main()
