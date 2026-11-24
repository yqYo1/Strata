"""Build and run the target-only conversation snapshot fixture on the host."""

import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_build as reference
import measurement_owner as owner

bank = reference.bank
require = reference.require
PREVIOUS = 'dflash2_target_snapshot_host_v1'
KEY = 'dflash2_target_snapshot_host_v2'
ROOT = bank.B / KEY.replace('_', '-')
TARGET = 'sycl_conversation_transfer_test'
COMPILER = Path('/opt/intel/oneapi/compiler/2026.2/bin/icpx')
THIS = str(Path(__file__).resolve().relative_to(bank.P))


def git(*args):
    return subprocess.check_output(['git', '-C', str(bank.P), *args], timeout=30)


def read_inputs():
    base = bank.previous.read(reference.DFLASH2_KV_ROOT / 'record.json')
    require(base['ownership_closed'] and base['stage'] == 'built' and base['passed']
            and bank.basis.validation.committed(bank.RESULT, (reference.DFLASH2_KV_KEY,))
                == reference.builder.compact(base), 'require the committed SYCL build recipe')
    external = base['external_inputs']
    require(bank.sha(external['path']) == external['sha256'], 'changed build dependency inventory')
    return base, bank.previous.read(Path(external['path']))


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    data = bank.previous.read(bank.RESULT)
    data[KEY] = row
    bank.stage.write(bank.RESULT, data)


def source_matches(revision):
    require(not git('diff', '--name-only', revision, '--', '.', ':(exclude)' + str(bank.RESULT.relative_to(bank.P)))
            and not git('ls-files', '--others', '--exclude-standard'),
            'commit all host fixture recipe and source inputs')


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT), 'fresh host fixture')
    prior = bank.basis.validation.committed(bank.RESULT, (PREVIOUS,))
    require(prior['stage'] == 'failed' and prior['ownership_closed'] and not prior['active']
            and 'qsa_attn_pools' in prior['link_failure'], 'preserve the closed first fixture attempt')
    base, allowed = read_inputs()
    revision = git('rev-parse', 'HEAD').decode().strip()
    source_matches(revision)
    recipe = base['recipe']
    old, new = str(reference.DFLASH2_KV_ROOT), str(ROOT)
    source = str(reference.DFLASH2_KV_ROOT / 'source')
    command = [arg.replace(source, str(bank.P)).replace(old, new) for arg in recipe['configure']
               if not arg.startswith('-DSTRATA_QSA_PREFILL_PV_SOURCE:')]
    index = command.index('-DSTRATA_BUILD_CONVERSATION_TESTS:BOOL=OFF')
    command[index] = '-DSTRATA_BUILD_CONVERSATION_TESTS:BOOL=ON'
    environment = {key: val.replace(old, new) for key, val in recipe['environment'].items()}
    environment['ONEAPI_DEVICE_SELECTOR'] = '*:cpu'
    compiler_digest = bank.sha(COMPILER)
    require(allowed.get(str(COMPILER.resolve())) == compiler_digest, 'expected pinned compiler')
    for path, digest in base['build_tools'].items():
        require(bank.sha(path) == digest, 'changed build tool: ' + path)
    ROOT.mkdir(mode=0o700)
    for folder in ('tmp', 'cache'):
        (ROOT / folder).mkdir(mode=0o700)
    save(dict(stage='prepared', source_commit=revision, controller=THIS,
              frozen_tools=dict(base['build_tools'], **{str(COMPILER): compiler_digest}),
              preceding_attempt=PREVIOUS, external_inputs=base['external_inputs'],
              environment=environment, configure=command,
              build=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', TARGET, '--parallel', '3'],
              dependency_command=['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'],
              executable=str(ROOT / 'build' / TARGET), passed=False, active=False, ownership_closed=True,
              scope='Transfer-injected host fixture. Exercise target-only packing, cache reuse, validation and '
                    'byte readback along with the existing MTP and error cases. No native inference launch.'))


def checked_inputs(row, rcpt, allowed):
    build = ROOT / 'build'
    dependency_file = Path(rcpt['deps'])
    paths = {Path(line.strip()).resolve() if Path(line.strip()).is_absolute()
             else (build / line.strip()).resolve()
             for line in dependency_file.read_text().splitlines() if line.startswith('    ') and line.strip()}
    required = {bank.P / 'src/core' / file for file in ('conversation_validation_test.cpp',
                'conversation_state.cpp', 'conversation_snapshot.cpp')}
    require(required <= paths, 'incomplete compiler dependency receipt')
    for line in Path(rcpt['build']).read_text().splitlines():
        if line.startswith('/') and not any(ch.isspace() for ch in line) and Path(line).is_file():
            paths.add(Path(line).resolve())
    external = {}
    for path in paths:
        if path.is_relative_to(bank.P):
            name = str(path.relative_to(bank.P))
            require(git('cat-file', '-t', row['source_commit'] + ':' + name) == b'blob\n',
                    'uncommitted compiler input: ' + name)
        else:
            digest = bank.sha(path)
            require(allowed.get(str(path)) == digest, 'unknown compiler or linker input: ' + str(path))
            external[str(path)] = digest
    require(len(external) > 100, 'incomplete external input list')
    manifest = ROOT / 'verified-inputs.json'
    bank.stage.write(manifest, external)
    return dict(files=len(paths), external=dict(path=str(manifest), sha256=bank.sha(manifest),
                files=len(external)), dependencies_sha256=bank.sha(dependency_file))


def execute(stage):
    base, allowed = read_inputs()
    row = bank.previous.read(ROOT / 'record.json')
    require(row == bank.basis.validation.committed(bank.RESULT, (KEY,))
            and row['stage'] == ('prepared' if stage == 'build' else 'built')
            and row['ownership_closed'] and not row['active']
            and row['external_inputs'] == base['external_inputs'], 'commit exact prior host fixture stage')
    source_matches(row['source_commit'])
    for path, expected in row['frozen_tools'].items():
        require(bank.sha(path) == expected, 'changed compiler or tool: ' + path)

    receipt = ROOT / (stage + '-execution.json')
    require(not receipt.exists(), 'inspect the existing host stage owner')
    owned = None

    def persist():
        bank.stage.write(receipt, dict(active=bool(owned and owned.active),
            commands=owned.commands if owned else []))

    owned = owner.Owner(ROOT / ('owned-' + stage), persist)

    def command(label, argv, wall, cpus=80):
        entry, out, err = owned.run(label, argv, row['environment'], bank.P, wall=wall, cpu=cpus,
                                    rss_cap=3 << 30, text_cap=4 << 20, file_cap=8 << 20, total_cap=48 << 20)
        require(owner.closed(entry) and entry['exit_code'] == 0,
                label + ' failed: ' + (err.read_text(errors='replace') or
                                       out.read_text(errors='replace'))[-1300:])
        return out, err

    try:
        if stage == 'build':
            command('configure', row['configure'], 60)
            out, _ = command('build', row['build'], 120, cpus=240)
            dep, _ = command('deps', row['dependency_command'], 15)
            row['inputs'] = checked_inputs(row, dict(build=out, deps=dep), allowed)
            row['executable_sha256'] = bank.sha(row['executable'])
            row['stage'] = 'built'
        else:
            pinned = row['inputs']['external']
            manifest = Path(pinned['path'])
            require(bank.sha(manifest) == pinned['sha256'], 'changed host build input manifest')
            inputs = bank.previous.read(manifest)
            require(bank.sha(row['executable']) == row['executable_sha256']
                    and len(inputs) == pinned['files']
                    and all(bank.sha(p) == digest for p, digest in inputs.items()),
                    'host fixture build or dependencies changed')
            stdout, stderr = command('snapshot', [row['executable']], 30)
            match = re.fullmatch(r'conversation_validation_test: (\d+) host-only checks passed\n', stdout.read_text())
            require(match and int(match[1]) >= 1000 and not stderr.stat().st_size, 'missing fixture checks')
            row.update(stage='closed', passed=True, checks=int(match[1]))
        source_matches(row['source_commit'])
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row.update(active=bool(owned.active), ownership_closed=owned.active is None
                   and all(owner.closed(entry) for entry in owned.commands))
        save(row)
    print({'stage': row['stage'], 'checks': row.get('checks'), 'ownership_closed': row['ownership_closed']})


def main(action):
    require(action in ('prepare', 'build', 'run'), 'use prepare, build or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            execute(action)


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
