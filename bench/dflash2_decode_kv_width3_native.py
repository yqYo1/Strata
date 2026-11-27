"""Build the isolated DFlash2 width-three resolver trial after its Arc screen."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import decode_kv_width3 as component
import dflash2_decode_qsa_grid_native as grid

bank, builder, require = grid.bank, grid.builder, grid.require
validation, source = grid.validation, grid.source
KEY = 'dflash2_decode_kv_width3_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PATCH = '9df3b8c66ca2e1365578685b01317a4b770fffe4'
FILES = ('include/strata/core/verify.hpp', 'include/strata/core/verify_window_policy.hpp',
         'src/core/verify.cpp', 'tests/core/verify_window_policy_test.cpp')
TARGETS = list(grid.TARGETS)
CONTROLLERS = (THIS, 'bench/decode_kv_width3.py', *grid.CONTROLLERS)
POLICY = 'sycl_verify_window_policy_test'
POLICY_MARKER = ('verify-window-policy: passed selections=30 centering=18 native-half=26 '
                 'dpas-head=1 dpas-dense=4 phase-roundtrip=1 invalid-keys=4 retirement=1 route-budget=64')


def inputs(revision):
    stage = validation.committed(bank.RESULT, (component.KEY,))
    require(stage['stage'] == 'closed' and stage['complete'] and stage['passed'] and stage['ownership_closed']
            and all(p['passed'] and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in stage['processes'])
            and all(not p['fresh_faults'] for p in stage.get('traces', ()))
            and stage['decision'] == component.decision(stage['processes'])
            and stage['decision']['eligible_for_model_check'], 'finish the accepted width-three component screen')
    cohort = validation.committed(bank.RESULT, ('dflash2_decode_kv_speed_v1',))
    require(cohort['stage'] in ('closed', 'failed') and not cohort['active'] and cohort['ownership_closed'],
            'close the original DFlash2 KV comparison')

    # Recover the parent recipe and its source from the committed projection.
    parent = validation.committed(bank.RESULT, (grid.KEY,))
    require(grid.retained.closed(parent, 'built')
            and cohort['binary'] == parent['binaries']['strata']
            and cohort['native_build_sha256'] == bank.signature(parent),
            'recover the exact model comparison build receipt')
    baseline, payload, original, _ = grid.inputs(parent['source_commit'], historical_controls=True)
    require(all(parent[k] == v for k, v in builder.compact(baseline).items())
            and stage['resolver_source'] == {name: baseline['source_files'][name] for name in component.CONTRACT}
            and stage['boot'] == cohort['boot'] and stage['driver'] == cohort['driver'] == parent['driver']
            and stage['runtime_manifest']['sha256'] == cohort['runtime_manifest']['sha256']
            and stage['native_validation_probe'] == cohort['native_validation_probe'],
            'model resolver, provider or recovered recipe differs from the measurement')

    before = dict(source.blobs(bank.P, PATCH + '^', FILES))
    changed = dict(source.blobs(bank.P, PATCH, FILES))
    require(before.keys() == changed.keys() == set(FILES) and all(payload[p] == before[p] != changed[p] for p in FILES),
            'apply the exact four-file width-three overlay to the original preimages')
    require(all((bank.P / p).read_bytes() == changed[p] for p in FILES), 'the reviewed width-three source changed')
    controls = source.pin(bank.P, CONTROLLERS)['source_sha256']
    require(controls == bank.stage.hashed(dict(source.blobs(bank.P, revision, controls))),
            'commit the exact model trial controller and helpers')
    payload.update(changed)

    old, new = str(grid.ROOT), str(ROOT)
    recipe = parent['recipe']
    configure = [arg.replace(old, new) for arg in recipe['configure']]
    require(recipe['targets'] == TARGETS and configure[2:5] == [str(TREE), '-B', str(ROOT / 'build')],
            'unexpected model build recipe')
    fields = dict(source_commit=revision, native_source_commit=PATCH,
                  native_base_commit=parent['native_source_commit'],
                  base_binary=parent['binaries']['strata'], base_build_sha256=cohort['native_build_sha256'],
                  component_source_commit=stage['source_commit'],
                  component_decision_sha256=bank.signature(stage['decision']),
                  overlays=bank.stage.hashed(changed), source_files=bank.stage.hashed(payload),
                  controller_sources=controls, foundation=baseline['foundation'],
                  external_inputs=parent['external_inputs'], driver=parent['driver'], build_tools=parent['build_tools'],
                  recipe=dict(configure=configure, targets=TARGETS,
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}))
    return fields, payload, original, parent


def check(row, committed=True):
    if committed:
        require(validation.committed(bank.RESULT, (KEY,)) == builder.compact(row), 'commit the preceding build stage')
    fields, _, original, parent = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items())
            and row['driver'] == builder.guards.driver_guard()
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(TREE / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'materialized source, runtime or configured recipe changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and actual['fresh_objects'] == parent['actual_inputs']['fresh_objects']
                and all(actual['compiled'].get(str(TREE / p)) == row['source_files'][p]
                        for p in (*grid.primary.REQUIRED_SOURCES, *FILES))
                and all(bank.sha(p) == h for name in bank.previous.ACTUAL for p, h in actual[name].items())
                and set(row['binaries']) == set(TARGETS)
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require a full coherent build and same-build policy fixture')
    return original


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        paths = [build / p for p in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        paths.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in paths}
    grid.primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh model-trial generation')
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
               ownership_closed=True, scope='Isolate the width-three selector on the measured model source.',
               retention='One candidate through same-binary policy, model parity and performance checks. '
                         'Retain the qualified primary fallback through the decision. The exact QSA '
                         'comparison receipt is the source and recipe reference for this build.')
    check(row, committed=False)
    save(row)


def policy_check():
    row = bank.previous.read(ROOT / 'record.json')
    check(row)
    case = ROOT / 'policy'
    require(grid.retained.closed(row, 'built') and 'policy' not in row and not case.exists(),
            'require an untested coherent model build')
    case.mkdir(mode=0o700)
    owner = builder.ownership.Owner(case / 'owned', lambda: bank.stage.write(case / 'ownership.json', owner.commands))
    binary = row['binaries'][POLICY]
    result = dict(binary=binary, passed=False, gpu_executed=False)
    try:
        entry, out, err = owner.run('verify-window-policy', [binary['path']], row['recipe']['environment'], case,
                                   wall=30, cpu=30, rss_cap=1 << 30, text_cap=128 << 10,
                                   file_cap=1 << 20, total_cap=4 << 20)
        require(builder.ownership.closed(entry) and entry['exit_code'] == 0
                and out.read_text().strip() == POLICY_MARKER and err.stat().st_size == 0
                and bank.sha(binary['path']) == binary['sha256'], 'verify the same-build host window policy')
        result.update(passed=True, exit_code=0, marker=POLICY_MARKER)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['ownership_closed'] = owner.active is None and all(builder.ownership.closed(c) for c in owner.commands)
        row.update(policy=result, active=owner.active is not None, ownership_closed=result['ownership_closed'])
        save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', 'configure', 'compile', 'policy'),
            'use the pinned Python and one width-three model build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        if action == 'prepare':
            prepare()
        elif action == 'policy':
            policy_check()
        else:
            builder.execute(action)
