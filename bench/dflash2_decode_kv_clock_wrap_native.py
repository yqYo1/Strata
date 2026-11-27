"""Build the retained DFlash2 engine with only the scalar decode KV change."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_ranked_refill_build as foundation

bank, primary, builder = foundation.bank, foundation.primary, foundation.builder
retained, require = foundation.retained, foundation.require
source, validation = foundation.source, foundation.validation
KEY = 'dflash2_decode_kv_clock_wrap_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
COMPONENT = '9a20db11dc6043cd12e128e61e2afc3c72721176'
COMPONENT_KEY = 'decode_kv_scalar_clock_wrap_v1'
NATIVE_SOURCE = '5a87b380f02e58c0a8bf63607c965aede9f81aa9'
RESOLVER_SOURCE = 'daffa0d18850622e5c4ae4e3409e1013a9593476'
DISPATCH_SOURCE = '35e3b2539bb6aa891fb2582e5504073a7b1c17c3'
RESOLVER_FILES = ('include/strata/kernels/kv_stream.hpp', 'src/kernels/sycl/kv_stream.cpp')
DISPATCH_FILES = ('include/strata/core/verify.hpp', 'include/strata/core/verify_window_policy.hpp',
                  'src/core/verify.cpp', 'tests/core/verify_window_policy_test.cpp')
FILES = (*RESOLVER_FILES, *DISPATCH_FILES)
POLICY = 'sycl_verify_window_policy_test'
TARGETS = [*primary.TARGETS, POLICY]
CONTROLLERS = (THIS, *foundation.CONTROLLERS)
POLICY_STDOUT = ('verify-window-policy: passed selections=30 centering=18 native-half=26 '
                 'dpas-head=1 dpas-dense=4 phase-roundtrip=1 invalid-keys=4 retirement=1 route-budget=64\n')


def inputs(revision):
    reference, base, original, payload, controls = foundation.retained_build_inputs(revision, CONTROLLERS)
    comp = bank.previous.historic(COMPONENT, COMPONENT_KEY)
    component_root = bank.B / COMPONENT_KEY.replace('_', '-')
    require(comp == bank.previous.read(component_root / 'record.json')
            and not (component_root / 'fixture-build').exists()
            and comp['stage'] == 'closed' and comp['complete'] and comp['passed']
            and comp['ownership_closed'] and comp['source_commit'] == RESOLVER_SOURCE
            and comp['native_source_commit'] == reference['native_source_commit']
            and comp['driver'] == reference['driver'] and comp['boot'] == reference['boot']
            and comp['runtime_manifest'] == reference['runtime_manifest']
            and comp['native_validation_probe'] == reference['native_validation_probe']
            and comp['decision']['eligible_for_model_check']
            and all(p['passed'] and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in comp['processes']),
            'require the exact scalar component result and retired fixture')

    base_revision = reference['native_source_commit']
    parent = validation.git('rev-parse', NATIVE_SOURCE + '^').decode().strip()
    names = validation.git('diff', '--name-only', base_revision, NATIVE_SOURCE).decode().splitlines()
    overlays = dict(source.blobs(bank.P, NATIVE_SOURCE, FILES))
    measured = dict(source.blobs(bank.P, RESOLVER_SOURCE, RESOLVER_FILES))
    live = dict(source.blobs(bank.P, DISPATCH_SOURCE, DISPATCH_FILES))
    require(parent == base_revision and set(names) == overlays.keys() == set(FILES)
            and all(payload[p] != overlays[p] for p in FILES)
            and all(overlays[p] == measured[p] == (bank.P / p).read_bytes()
                    and comp['sources'][p] == bank.sha(bank.P / p) for p in RESOLVER_FILES)
            and all(live[p] == (bank.P / p).read_bytes() for p in DISPATCH_FILES),
            'recover only the six scoped resolver and verifier files')
    payload.update(overlays)
    recipe = base['recipe']
    old, new = str(primary.ROOT), str(ROOT)
    configure = [arg.replace(old, new) for arg in recipe['configure']]
    require(recipe['targets'] == primary.TARGETS and configure[2:5] == [str(TREE), '-B', str(ROOT / 'build')],
            'the retained build recipe differs')
    fields = dict(source_commit=revision, native_source_commit=NATIVE_SOURCE,
                  native_base_commit=base_revision, live_dispatch_commit=DISPATCH_SOURCE,
                  component_result_commit=COMPONENT, component_sha256=bank.signature(comp),
                  source_files=bank.stage.hashed(payload), overlays=bank.stage.hashed(overlays),
                  controller_sources=controls, foundation=base['foundation'],
                  external_inputs=base['external_inputs'], build_tools=base['build_tools'],
                  driver=reference['driver'], base_binary=reference['binary'],
                  base_build_sha256=reference['native_build_sha256'],
                  recipe=dict(configure=configure, targets=TARGETS,
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}))
    return fields, payload, original, base


def check(row, committed=True):
    if committed:
        require(validation.committed(bank.RESULT, (KEY,)) == builder.compact(row), 'commit the previous build stage')
    expected, _, original, base = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in expected.items())
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(TREE / name) == digest for name, digest in row['source_files'].items())
            and all(bank.sha(path) == digest for path, digest in row.get('configured_files', {}).items()),
            'materialized clock-wrap source or build configuration changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and actual['fresh_objects'] == base['actual_inputs']['fresh_objects'] + 1
                and all(actual['compiled'].get(str(TREE / p)) == row['source_files'][p]
                        for p in (*primary.REQUIRED_SOURCES, *FILES))
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items())
                and set(row['binaries']) == set(TARGETS)
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require the coherent engine and same-build policy fixture')
    return original


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        paths = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        paths.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in paths}
    primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh clock-wrap model build')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    fields, payload, _, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in payload.items():
        path = TREE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    row = dict(fields, stage='prepared', commands=[], active=False, passed=False,
               complete=False, ownership_closed=True,
               scope='One coherent retained model binary with a decode-only scalar KV clock-wrap selector.',
               retention='Preserve the validated primary. Review this candidate at the first failed gate or decision.')
    check(row, committed=False)
    save(row)


def policy_check():
    row = bank.previous.read(ROOT / 'record.json')
    check(row)
    case = ROOT / 'policy'
    require(retained.closed(row, 'built') and 'policy' not in row and not case.exists(),
            'require a fresh closed build for policy validation')
    case.mkdir(mode=0o700)
    owner_type = builder.ownership
    owner = owner_type.Owner(case / 'owned', lambda: bank.stage.write(case / 'ownership.json', owner.commands))
    binary = row['binaries'][POLICY]
    result = dict(binary=binary, passed=False, gpu_executed=False)
    try:
        entry, out, err = owner.run('verify-window-policy', [binary['path']], row['recipe']['environment'], case,
                                   wall=30, cpu=30, rss_cap=1 << 30, text_cap=128 << 10,
                                   file_cap=1 << 20, total_cap=4 << 20)
        require(owner_type.closed(entry) and entry['exit_code'] == 0 and out.read_text() == POLICY_STDOUT
                and err.stat().st_size == 0 and bank.sha(binary['path']) == binary['sha256'],
                'verify the actual same-build host policy and retirement')
        result.update(passed=True, exit_code=0, marker=POLICY_STDOUT.strip())
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        row['policy'] = result
        row['active'] = owner.active is not None
        row['ownership_closed'] = not row['active'] and all(owner_type.closed(c) for c in owner.commands)
        save(row)


def main(action):
    require(sys.executable == bank.basis.reference.model.PYTHON
            and action in ('prepare', 'configure', 'compile', 'policy'), 'use the pinned Python and one build stage')
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


if __name__ == '__main__':
    main(*sys.argv[1:])
