"""Build the decode-placement candidate from its committed one-file change."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_primary_retained as retained

primary = retained.PRIMARY
bank, builder, require = primary.bank, primary.builder, primary.require
source, validation = bank.stage.source, bank.basis.validation
KEY = 'dflash2_decode_phase_placement_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = Path(__file__).resolve().relative_to(bank.P).as_posix()
PARENT_RECORD = '529948493453e17249bfb64f054aa48e6aa3ef1a'
PARENT_KEY = 'dflash2_decode_t4_prefix_help_native_v1'
PARENT_SOURCE = '2919b3531b9f8f37c97866591a63ea1464d81058'
NATIVE = ('CMakeLists.txt', 'cmake', 'include', 'src', 'third_party', 'tests')
GENERATOR = 'src/program/generate.cpp'
CONTROLLERS = (THIS, retained.THIS, 'bench/dflash2_decode_phase_placement_check.py', 'bench/dflash2_primary_prefill_speed.py',
               'bench/dflash2_primary_checks.py', 'bench/dflash2_primary_context.py', 'data/expert-profile.bin')
REQUIRED = (GENERATOR, 'src/core/expert_cache.cpp', 'src/core/verify.cpp',
            'tests/core/prefill_scratch_test.cpp', 'tests/sycl/prefill_scratch.cpp')
TARGETS = ['strata', 'strata_full_context_diagnostic', 'prefill_scratch_host_test', 'sycl_prefill_scratch_test']


def inputs(revision):
    baseline = retained.baseline()
    parent = bank.previous.historic(PARENT_RECORD, PARENT_KEY)
    require(retained.closed(parent, 'built') and parent['native_source_commit'] == PARENT_SOURCE,
            'require the original coherent source and recipe for the 96K caller-help engine')
    changed = source.git(bank.P, 'diff', '--name-only', PARENT_SOURCE, revision, '--', *NATIVE).decode().splitlines()
    require(changed == [GENERATOR], 'decode placement must change only the serving generator')

    foundation = parent['foundation']
    original_path = builder.original.ROOT / 'record.json'
    require(bank.sha(original_path) == foundation['original_record_sha256'], 'original dependency receipt changed')
    native = bank.previous.read(original_path)
    require(native['external_inputs'] == parent['external_inputs']
            and parent['driver'] == builder.guards.driver_guard()
            and all(bank.sha(path) == h for path, h in (native['toolchain'] | parent['build_tools']).items())
            and all(bank.sha(Path(dep['root']) / path) == h for dep in native['dependencies']
                    for path, h in dep['source_sha256'].items()), 'compiler, dependency, tool or runtime changed')
    files = source.pin(bank.P, [*primary.PREFIXES, *CONTROLLERS])['source_sha256']
    payload = dict(source.blobs(bank.P, revision, files))
    require(bank.stage.hashed(payload) == files, 'commit the exact placement source and controllers')

    recipe = parent['recipe']
    old = str(bank.B / PARENT_KEY.replace('_', '-'))
    new = str(ROOT)
    args = [arg.replace(old, new) for arg in recipe['configure']]
    require(args[2:5] == [str(TREE), '-B', str(ROOT / 'build')], 'parent recipe has unexpected build paths')
    return dict(source_commit=revision, native_source_commit=revision, source_files=files,
                controller_sources={path: files[path] for path in CONTROLLERS},
                parent_record=PARENT_RECORD, parent_source=PARENT_SOURCE, parent_binary=parent['binaries']['strata'],
                retained_binary=baseline['binary'], foundation=foundation, driver=parent['driver'],
                external_inputs=parent['external_inputs'], build_tools=parent['build_tools'],
                recipe=dict(configure=args, targets=TARGETS,
                            environment={k: v.replace(old, new) for k, v in recipe['environment'].items()})), payload, native


def check(row, committed=True):
    if committed:
        require(validation.committed(bank.RESULT, (KEY,)) == builder.compact(row), 'commit the previous build stage')
    fields, _, native = inputs(row['source_commit'])
    require(all(row[key] == value for key, value in fields.items())
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == fields['source_files'].keys()
            and all(bank.sha(TREE / path) == h for path, h in fields['source_files'].items())
            and all(bank.sha(path) == h for path, h in row.get('configured_files', {}).items()),
            'placement source or the configure recipe changed')
    if row['stage'] == 'built':
        census = row['actual_inputs']
        require(not census['partial_archive_reuse'] and census['uncovered'] == 0
                and all(census['compiled'].get(str(TREE / p)) == row['source_files'][p] for p in REQUIRED)
                and all(bank.sha(path) == h for kind in bank.previous.ACTUAL for path, h in census[kind].items())
                and set(row['binaries']) == set(TARGETS)
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require a coherent build and its exact fixtures')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in files}
    primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT),
            'review the existing placement build generation before creating one')
    revision = source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
    fields, payload, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in payload.items():
        path = TREE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    row = dict(fields, stage='prepared', active=False, commands=[], passed=False, complete=False, ownership_closed=True,
               scope='One candidate binary for stock vs decode-only placement. Keep T4 schedule and caller policy fixed.',
               retention='Keep through the payload, retirement, repeated request and paired decode measurements.')
    check(row, committed=False)
    save(row)


def main(action):
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', 'configure', 'compile'),
            'use the pinned Python and select one build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        if action == 'prepare':
            prepare()
        else:
            builder.execute(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
