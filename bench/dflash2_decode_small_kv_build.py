"""Build an isolated DFlash2 candidate for the smaller streamed KV pool."""

import fcntl
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
import dflash2_decode_ranked_refill_build as retained_builder
import dflash2_primary_checks as checks

bank, primary, builder = retained_builder.bank, retained_builder.primary, retained_builder.builder
validation, require = retained_builder.validation, retained_builder.require
KEY = 'dflash2_decode_small_kv_build_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PATCH_PARENT = '6f7aeee9745f55659636539b15c5646bbfb0fe65'
PATCH = '4fd9693618949554930ce30b9cf81556e816befe'
PATCH_SHA = '96f8c2b721c27b3b8f19bd3218dd9860a389097da3727fd995889abfe34f35c0'
GENERATOR = 'src/program/generate.cpp'
GENERATOR_SHA = 'f83fbc369a32ac0eefbb015232964b25d868884d86ea76449059b72926acdf99'
FILES = ('include/strata/core/layer.hpp', 'src/core/layer.cpp', GENERATOR, 'tests/sycl/kv.cpp')
TARGET = 'sycl_kv_test'
TARGETS = [*primary.TARGETS, TARGET]
CONTROLLERS = (THIS, checks.THIS, *retained_builder.CONTROLLERS)
FIXTURE = dict(target=TARGET, args=['--dflash2-kv-resident'],
               expected='DFlash2 width4 resident KV: 3072 slots, 2056 distinct pages, payload and eviction passed',
               limits=checks.device.SMALL)


def overlay(payload):
    patch = validation.git('diff', '--binary', PATCH_PARENT, PATCH, '--', *FILES)
    require(sha256(patch).hexdigest() == PATCH_SHA, 'the committed smaller-KV delta changed')
    with TemporaryDirectory(prefix='strata-small-kv-projection-') as temp:
        root = Path(temp)
        for name in FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload[name])
        subprocess.run(['git', 'apply', '--whitespace=error'], input=patch, cwd=root, check=True, timeout=20)
        changed = {name: (root / name).read_bytes() for name in FILES}
    require(sha256(changed[GENERATOR]).hexdigest() == GENERATOR_SHA
            and all(changed[name] == validation.git('show', PATCH + ':' + name) for name in FILES if name != GENERATOR),
            'the isolated generator or other projected sources changed')
    return changed


def inputs(revision):
    reference, base, original, payload, controls = retained_builder.retained_build_inputs(revision, CONTROLLERS)
    changed = overlay(payload)
    payload.update(changed)
    old, new = str(primary.ROOT), str(ROOT)
    recipe = base['recipe']
    configure = [arg.replace(old, new) for arg in recipe['configure']]
    require(recipe['targets'] == primary.TARGETS and configure[2:5] == [str(TREE), '-B', str(ROOT / 'build')],
            'the retained build recipe differs')
    fields = dict(source_commit=revision, native_source_commit=PATCH,
                  native_base_commit=reference['native_source_commit'], source_files=bank.stage.hashed(payload),
                  overlays=bank.stage.hashed(changed), controller_sources=controls, fixture_plan=FIXTURE,
                  recipe=dict(configure=configure, targets=TARGETS,
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}),
                  foundation=base['foundation'], external_inputs=base['external_inputs'], driver=reference['driver'],
                  build_tools=base['build_tools'], base_binary=reference['binary'],
                  base_build_sha256=reference['native_build_sha256'])
    return fields, payload, original, base


def check(row, committed=True):
    if committed:
        require(validation.committed(bank.RESULT, (KEY,)) == builder.compact(row), 'commit the previous build stage')
    expected, _, original, base = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in expected.items())
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(TREE / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'the materialized smaller-KV source or recipe changed')
    if row['stage'] == 'built':
        census = row['actual_inputs']
        require(not census['partial_archive_reuse'] and census['uncovered'] == 0
                and census['fresh_objects'] == base['actual_inputs']['fresh_objects'] + 1
                and all(census['compiled'].get(str(TREE / p)) == row['source_files'][p]
                        for p in set(primary.REQUIRED_SOURCES) | set(FILES))
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in census[group].items())
                and set(row['binaries']) == set(TARGETS)
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require a coherent larger/smaller KV build and fixture')
    return original


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        paths = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        paths.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in paths}
    primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh smaller-KV generation')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    fields, payload, _, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in payload.items():
        path = TREE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False,
               ownership_closed=True, scope='Same retained DFlash2 source with the 12288-cell KV trial and fixture.',
               retention='Keep through the bounded KV check and its subsequent native model gate.')
    check(row, committed=False)
    save(row)


def validate():
    row = bank.previous.read(ROOT / 'record.json')
    original = check(row)
    case = ROOT / 'device'
    require(retained_builder.retained.closed(row, 'built') and 'fixture' not in row and not case.exists(),
            'require a fresh closed build and committed component plan')
    ref = checks.parameters(row, original['native_validation_probe'])
    env = dict(ref['environment'], TMPDIR=str(case / 'tmp'), XDG_CACHE_HOME=str(case / 'cache'),
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(case / 'validation.log'))
    plan = row['fixture_plan']
    binary = row['binaries'][plan['target']]
    command = [binary['path'], *plan['args']]
    cursor, = [line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True,
        timeout=15).splitlines() if line.startswith('-- cursor: ')]
    for name in ('tmp', 'cache'):
        (case / name).mkdir(mode=0o700, parents=True)
    owner = builder.ownership.Owner(case / 'owned', lambda: bank.stage.write(
        case / 'ownership.json', dict(active=owner.active is not None, commands=owner.commands)))
    outcome = dict(binary=binary, command=command, environment_sha256=bank.signature(env),
                   boot=ref['boot'], runtime_manifest=ref['runtime_manifest'], host_probe_accepted=False)
    try:
        entry, out, err = owner.run('resident-kv', command, env, case, **plan['limits'])
        outcome['native_exit_code'] = entry['exit_code']
        require(builder.ownership.closed(entry) and entry['exit_code'] == 0
                and out.read_text().strip() == plan['expected'], 'incomplete resident KV fixture')
        trace = err.read_text()
        probe = validation.preload_evidence(trace, entry['pid'])
        warnings = case / 'validation.log'
        require(probe['loader'] == ref['native_validation_probe']['loader']['path']
                and 'urEnqueueKernelLaunch' in trace and (not warnings.exists() or warnings.stat().st_size == 0)
                and bank.sha(binary['path']) == binary['sha256'], 'invalid Arc trace, fixture or runtime')
        outcome.update(validation=probe, providers=checks.device.loaded_providers(trace, dict(ref, environment=env)),
                       summary=plan['expected'], host_probe_accepted=True)
    except Exception as exc:
        outcome['error'] = repr(exc)
    finally:
        validation.finish(outcome, owner, cursor, ref['boot'])
        outcome['passed'] &= outcome.get('driver_after') == row['driver']
        row.update(fixture=outcome, active=owner.active is not None, ownership_closed=outcome['ownership_closed'])
        save(row)
    require(outcome['passed'], outcome.get('error', 'Arc validation or owner closure failed'))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON
            and action in ('prepare', 'configure', 'compile', 'validate'),
            'use the pinned Python and one smaller-KV build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        if action == 'prepare':
            prepare()
        elif action == 'validate':
            validate()
        else:
            builder.execute(action)
