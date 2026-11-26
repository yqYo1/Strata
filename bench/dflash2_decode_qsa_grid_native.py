"""Build the retained DFlash2 engine with only the decode QSA grid change."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_primary_retained as retained

primary = retained.PRIMARY
bank, builder, require = primary.bank, primary.builder, primary.require
validation, source = bank.basis.validation, bank.stage.source
KEY = 'dflash2_decode_qsa_grid_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
OVERLAY = 'd1bdd6872a635a1c016ace804fe42ced939a3dbc'
FILES = ('include/strata/core/verify.hpp', 'include/strata/core/verify_window_policy.hpp',
         'include/strata/kernels/qsa_select.hpp', 'src/core/verify.cpp',
         'src/kernels/sycl/qsa_select.cpp', 'tests/core/verify_window_policy_test.cpp')
COMPONENT_REVISION = 'a3e2a9a3d'
COMPONENT_KEY = 'decode_qsa_score_grid_v1'
TARGETS = [*primary.TARGETS, 'sycl_verify_window_policy_test']
CONTROLLERS = (THIS, retained.THIS, 'bench/sycl', 'tools/sycl', *bank.basis.POLICIES)


def inputs(revision):
    current = retained.baseline()
    base = bank.previous.read(primary.ROOT / 'record.json')
    comp = bank.previous.historic(COMPONENT_REVISION, COMPONENT_KEY)
    require(comp == validation.committed(bank.RESULT, (COMPONENT_KEY,))
            and retained.closed(comp, 'closed') and comp['profile']['passed']
            and comp['native_source_commit'] == current['native_source_commit']
            and comp['native_build_sha256'] == current['native_build_sha256']
            and len(comp['decision']) == 8 and all(x['eligible'] for x in comp['decision']),
            'require the closed eight-shape Arc component decision')
    original_path = builder.original.ROOT / 'record.json'
    foundation = base['foundation']
    require(bank.sha(original_path) == foundation['original_record_sha256'],
            'the original SDK and recipe record changed')
    original = bank.previous.read(original_path)
    require(original['external_inputs'] == base['external_inputs']
            and all(bank.sha(p) == h for p, h in (original['toolchain'] | base['build_tools']).items()),
            'the exact compiler, tools and SDK are required')
    for pin in original['dependencies']:
        root = Path(pin['root'])
        require(bank.stage.hashed(dict(source.blobs(root, pin['revision'], pin['source_sha256'])))
                == pin['source_sha256']
                and all(bank.sha(root / p) == h for p, h in pin['source_sha256'].items()),
                'the retained dependencies changed: ' + str(root))

    controls = source.pin(bank.P, CONTROLLERS)['source_sha256']
    require(bank.stage.hashed(dict(source.blobs(bank.P, revision, controls))) == controls,
            'commit the build controller and its helpers')
    payload = dict(source.blobs(bank.P, current['native_source_commit'], base['source_files']))
    require(bank.stage.hashed(payload) == base['source_files'], 'the qualified native source differs')
    previous = dict(source.blobs(bank.P, OVERLAY + '^', FILES))
    changed = dict(source.blobs(bank.P, OVERLAY, FILES))
    require(all(payload[p] == previous[p] != changed[p] and (bank.P / p).read_bytes() == changed[p]
                for p in FILES), 'the candidate must apply to the exact verified source preimages')
    payload.update(changed)

    old, new = str(primary.ROOT), str(ROOT)
    recipe = base['recipe']
    configure = [arg.replace(old, new) for arg in recipe['configure']]
    require(configure[2:5] == [str(TREE), '-B', str(ROOT / 'build')]
            and recipe['targets'] == primary.TARGETS, 'the retained build recipe differs')
    fields = dict(source_commit=revision, native_source_commit=OVERLAY,
                  native_base_commit=current['native_source_commit'], overlays=bank.stage.hashed(changed),
                  native_source_identity='Retained primary source plus the six committed grid and policy files.',
                  controller_sources=controls, source_files=bank.stage.hashed(payload),
                  component_revision=COMPONENT_REVISION, component_sha256=bank.signature(comp),
                  recipe=dict(configure=configure, targets=TARGETS,
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}),
                  foundation=foundation, external_inputs=base['external_inputs'], driver=current['driver'],
                  build_tools=base['build_tools'], base_binary=current['binary'],
                  base_build_sha256=current['native_build_sha256'])
    return fields, payload, original, base


def check(row, committed=True):
    if committed:
        require(validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the preceding QSA build stage')
    fields, _, original, base = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items())
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(TREE / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'the materialized build source or recipe changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and actual['fresh_objects'] == base['actual_inputs']['fresh_objects'] + 1
                and all(actual['compiled'].get(str(TREE / p)) == row['source_files'][p]
                        for p in (*primary.REQUIRED_SOURCES, *FILES))
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items())
                and set(row['binaries']) == set(TARGETS)
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require a coherent candidate and policy fixture')
    return original


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        paths = [build / p for p in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        paths.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in paths}
    primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh candidate generation')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    fields, payload, _, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in payload.items():
        target = TREE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False,
               ownership_closed=True, scope='The qualified primary build plus the decode grid dispatch and policy test.',
               retention='Keep the qualified primary fallback until the same-binary DFlash2 checks close.')
    check(row, committed=False)
    save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON
            and action in ('prepare', 'configure', 'compile'), 'use the pinned Python and one build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        if action == 'prepare':
            prepare()
        else:
            builder.execute(action)
