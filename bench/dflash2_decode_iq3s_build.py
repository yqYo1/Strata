"""Build the measured DFlash2 source with the isolated CPU IQ3_S selector."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_ranked_refill_build as recipe

bank, primary, builder = recipe.bank, recipe.primary, recipe.builder
validation, require = recipe.validation, recipe.require
KEY = 'dflash2_decode_iq3s_build_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
SELECTOR = 'c0db4198fb62dece41882fd5d10fc460443557dc'
COMPONENT = '14180e83308569b2bf10cd3aa8eba802dc5db8d7'
EVIDENCE = bank.RESULT.with_name('cpu-fixed-expert-iq3s-single-current.json')
OVERLAYS = ('include/strata/kernels/cpu/native_expert.hpp', 'src/kernels/cpu/native_expert.cpp')
KERNEL = 'src/kernels/cpu/iq_single_avx2.cpp'
FIXTURE = 'iq_single_exact_test'
TARGETS = [*primary.TARGETS, FIXTURE]
CONTROLLERS = (THIS, *recipe.CONTROLLERS)


def inputs(revision):
    proof = validation.committed(EVIDENCE)
    require(EVIDENCE.read_bytes() == validation.git('show', COMPONENT + ':' + str(EVIDENCE.relative_to(bank.P))),
            'preserve the original closed IQ3_S replay')
    require(proof['stage'] == 'closed' and proof['ownership_closed'] and not proof['active']
            and not proof['gpu_executed'] and proof['iq3s_single_result']['eligible'],
            'require the repeated fixed CPU replay gain')
    reference, base, original, payload, controls = recipe.retained_build_inputs(revision, CONTROLLERS)
    source = bank.stage.source
    before = dict(source.blobs(bank.P, SELECTOR + '^', OVERLAYS))
    changed = dict(source.blobs(bank.P, SELECTOR, OVERLAYS))
    tested = dict(source.blobs(bank.P, proof['source_commit'], (*OVERLAYS, KERNEL)))
    require(before == {name: payload[name] for name in OVERLAYS}
            and changed == {name: tested[name] for name in OVERLAYS}
            and payload[KERNEL] == tested[KERNEL],
            'project only the replay-tested selector onto the exact primary CPU preimages')
    payload.update(changed)
    old, new = str(primary.ROOT), str(ROOT)
    retained_recipe = base['recipe']
    configure = [arg.replace(old, new) for arg in retained_recipe['configure']]
    require(retained_recipe['targets'] == primary.TARGETS and FIXTURE not in primary.TARGETS
            and configure[2:5] == [str(TREE), '-B', str(ROOT / 'build')], 'relocate the exact primary recipe')
    fields = dict(source_commit=revision, native_base_commit=reference['native_source_commit'],
                  native_source_commit=SELECTOR, component_evidence_commit=COMPONENT,
                  component_sha256=bank.signature(proof), source_files=bank.stage.hashed(payload),
                  overlays=bank.stage.hashed(changed), controller_sources=controls,
                  recipe=dict(configure=configure, targets=TARGETS,
                              environment={k: v.replace(old, new) for k, v in retained_recipe['environment'].items()}),
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
            'the materialized IQ3_S source or build recipe changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        required = (*primary.REQUIRED_SOURCES, *OVERLAYS, KERNEL, 'tests/core/iq_single_exact_test.cpp')
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and actual['fresh_objects'] == base['actual_inputs']['fresh_objects'] + 1
                and all(actual['compiled'].get(str(TREE / p)) == row['source_files'][p] for p in required)
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items())
                and set(row['binaries']) == set(TARGETS)
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require the coherent engine and same-build CPU parity fixture')
    return original


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in files}
    primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(bank.RESULT), 'one fresh IQ3_S build generation')
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
               ownership_closed=True, scope='Primary DFlash2 plus the exact decode-only single-token IQ3_S selector.',
               retention='Keep through CPU parity, bounded model parity and the timed selector comparison.')
    check(row, committed=False)
    save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('prepare', 'configure', 'compile'),
            'use the pinned Python and one IQ3_S build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
        builder.check, builder.save = check, save
        if action == 'prepare':
            prepare()
        else:
            builder.execute(action)
