"""Build the retained DFlash2 engine with the final expert-refill change."""

import fcntl
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
import dflash2_primary_retained as retained

primary = retained.PRIMARY
bank, builder, require = primary.bank, primary.builder, primary.require
validation, source = bank.basis.validation, bank.stage.source
KEY = 'dflash2_decode_ranked_refill_build_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PATCH_PARENT = 'd6ee8c2286edb0fc36417644e681102c583dd47f'
PATCH = '4757120ec4d7c700bb0fdcd9571d1dd37abb8a04'
PATCH_SHA = '54e04db380468d0a0ffde442d0038e6d6f8ed6558bbf9638bce40f74ee7e3231'
BASE = '968e71988953dca3d98037d3b3fec9f903bdfd6f'
PLACEMENT = '5e75c7dbe333d0f4f11d2e7a2b21dcc3f56567e6'
PLACEMENT_KEY = 'dflash2_decode_ranked_generation_blend_v1'
GENERATOR = 'src/program/generate.cpp'
GENERATOR_SHA = '52c77e3ff69566997e81b91663cf5f2d8ceba5352d67f7ccbe7939414e2f09f0'
FILES = ('include/strata/core/cache_scratch_lease.hpp', 'include/strata/core/expert_cache.hpp',
         'src/core/expert_cache.cpp', GENERATOR, 'tests/core/prefill_scratch_test.cpp', 'tests/sycl/prefill_scratch.cpp')
FIXTURES = ('prefill_scratch_host_test', 'sycl_prefill_scratch_test')
TARGETS = [*primary.TARGETS, *FIXTURES]
CONTROLLERS = (THIS, retained.THIS, 'bench/sycl', 'tools/sycl', *bank.basis.POLICIES)


def overlay(payload):
    patch = validation.git('diff', '--binary', PATCH_PARENT, PATCH, '--', *FILES)
    require(sha256(patch).hexdigest() == PATCH_SHA, 'the committed six-file refill delta changed')
    # The live generator also contains a handoff observer. Apply only this
    # delta to the original generator and check the resulting source too.
    with TemporaryDirectory(prefix='strata-ranked-refill-projection-') as directory:
        tree = Path(directory)
        for name in FILES:
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload[name])
        command = ['git', 'apply', '--whitespace=error']
        subprocess.run([*command, '--check'], input=patch, cwd=tree, check=True, timeout=20)
        subprocess.run(command, input=patch, cwd=tree, check=True, timeout=20)
        projected = {name: (tree / name).read_bytes() for name in FILES}
    require(sha256(projected[GENERATOR]).hexdigest() == GENERATOR_SHA
            and all(projected[p] == validation.git('show', PATCH + ':' + p) for p in FILES if p != GENERATOR),
            'ranked refill projection differs from the reviewed source')
    return projected


def inputs(revision):
    reference = retained.baseline()
    base = bank.previous.read(primary.ROOT / 'record.json')
    evidence = bank.previous.historic(PLACEMENT, PLACEMENT_KEY)
    require(evidence == validation.committed(bank.RESULT, (PLACEMENT_KEY,)) and retained.closed(evidence, 'closed')
            and evidence['binary'] == reference['binary'] and evidence['native_source_commit'] == BASE
            and evidence['decision']['sustained_gain_screen'] and not evidence['decision']['prefill_preserved'],
            'require the repeated decode gain and its original prefill decision')
    profiles = evidence['qualified_profiles']
    require(all(bank.sha(p['path']) == p['sha256'] for p in profiles.values()), 'the measured profile inputs changed')

    foundation = base['foundation']
    original_path = builder.original.ROOT / 'record.json'
    require(bank.sha(original_path) == foundation['original_record_sha256'], 'the retained dependency recipe changed')
    original = bank.previous.read(original_path)
    require(original['external_inputs'] == base['external_inputs']
            and all(bank.sha(p) == h for p, h in (original['toolchain'] | base['build_tools']).items()),
            'the exact compiler, tools and SDK are required')
    for dependency in original['dependencies']:
        root, files = Path(dependency['root']), dependency['source_sha256']
        require(bank.stage.hashed(dict(source.blobs(root, dependency['revision'], files))) == files
                and all(bank.sha(root / p) == h for p, h in files.items()), 'retained dependency changed: ' + str(root))
    controls = source.pin(bank.P, CONTROLLERS)['source_sha256']
    require(bank.stage.hashed(dict(source.blobs(bank.P, revision, controls))) == controls,
            'commit the exact controller and build helpers')

    payload = dict(source.blobs(bank.P, BASE, base['source_files']))
    require(bank.stage.hashed(payload) == base['source_files'], 'the retained engine source changed')
    changed = overlay(payload)
    payload.update(changed)
    recipe = base['recipe']
    original_root, new_root = str(primary.ROOT), str(ROOT)
    configure = [arg.replace(original_root, new_root) for arg in recipe['configure']]
    require(recipe['targets'] == primary.TARGETS and configure[2:5] == [str(TREE), '-B', str(ROOT / 'build')],
            'the retained build recipe differs')
    fields = dict(source_commit=revision, native_source_commit=PATCH, native_base_commit=BASE,
                  source_files=bank.stage.hashed(payload), overlays=bank.stage.hashed(changed), controller_sources=controls,
                  baseline_placement_sha256=bank.signature(evidence), profiles=profiles,
                  recipe=dict(configure=configure, targets=TARGETS,
                              environment={k: v.replace(original_root, new_root) for k, v in recipe['environment'].items()}),
                  foundation=foundation, external_inputs=base['external_inputs'], driver=reference['driver'],
                  build_tools=base['build_tools'], base_binary=reference['binary'],
                  base_build_sha256=reference['native_build_sha256'])
    return fields, payload, original, base


def check(row, committed=True):
    if committed:
        require(validation.committed(bank.RESULT, (KEY,)) == builder.compact(row), 'commit the previous build stage')
    fields, _, original, base = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items())
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(TREE / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'the materialized ranked-refill source or build recipe changed')
    if row['stage'] == 'built':
        inputs_found = row['actual_inputs']
        require(not inputs_found['partial_archive_reuse'] and inputs_found['uncovered'] == 0
                and inputs_found['fresh_objects'] == base['actual_inputs']['fresh_objects'] + len(FIXTURES)
                and all(inputs_found['compiled'].get(str(TREE / p)) == row['source_files'][p]
                        for p in set(primary.REQUIRED_SOURCES) | set(FILES))
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in inputs_found[group].items())
                and set(row['binaries']) == set(TARGETS)
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require the coherent engine and both same-build fixtures')
    return original


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        paths = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        paths.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in paths}
    primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh ranked-refill build generation')
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    fields, payload, _, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in payload.items():
        target = TREE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False, ownership_closed=True,
               scope='Retained DFlash2 engine with the six-file final expert-refill change.',
               retention='Candidate for the paired lease, RESTORE and long-generation comparisons.')
    check(row, committed=False)
    save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', 'configure', 'compile'),
            'use the pinned Python and one ranked-refill build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        if action == 'prepare':
            prepare()
        else:
            builder.execute(action)
