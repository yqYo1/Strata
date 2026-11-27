"""Project the cold-prefill observer onto the retained DFlash2 build."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_primary_retained as retained

primary = retained.PRIMARY
bank, builder, require = primary.bank, primary.builder, primary.require
source = bank.stage.source
KEY = 'dflash2_cold_host_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
OVERLAY = '8a437b97ac43f59ee28559565436f9c50493df49'
CONTROL = 'b74b6b067913227d0774830d9bf8fdc0f60569aa'
OVERLAYS = ('src/prefill/prefill.cpp', 'src/prefill/layer_major.inc',
            'src/prefill/cold_host_phases.hpp')
CONTROLLERS = (THIS, retained.THIS, retained.speed.THIS, 'bench/sycl', 'tools/sycl',
               *bank.basis.POLICIES)


def projection(base):
    revision = base['native_source_commit']
    sources = base['source_files']
    payload = dict(source.blobs(bank.P, revision, sources))
    require(bank.stage.hashed(payload) == sources, 'recover the original primary build inputs')
    existing = [p for p in OVERLAYS if p in payload]
    before = dict(source.blobs(bank.P, OVERLAY + '^', existing))
    overlay = dict(source.blobs(bank.P, OVERLAY, OVERLAYS))
    require(existing == list(OVERLAYS[:2]) and before == {p: payload[p] for p in existing}
            and all(raw != payload.get(p) for p, raw in overlay.items()),
            'the observer must apply to the exact primary source preimages')

    recipe = base['recipe']
    prior, current = str(primary.ROOT), str(ROOT)
    configure = [arg.replace(prior, current) for arg in recipe['configure']]
    require(configure[2:5] == [str(TREE), '-B', str(ROOT / 'build')]
            and recipe['targets'] == primary.TARGETS, 'relocate only the retained recipe paths')
    return payload | overlay, dict(
        configure=configure, targets=recipe['targets'],
        environment={k: v.replace(prior, current) for k, v in recipe['environment'].items()})


def inputs(revision):
    reference = retained.baseline()
    base = bank.previous.read(primary.ROOT / 'record.json')
    foundation = base['foundation']
    control = bank.previous.historic(CONTROL, 'dflash2_primary_prefill_cpu_wait_v1')
    require(retained.closed(control, 'closed')
            and control['binary'] == reference['binary']
            and control['native_build_sha256'] == reference['native_build_sha256']
            and control['decision']['warm_observer_gate'],
            'require the completed 118785-token control on the retained engine')
    original_path = builder.original.ROOT / 'record.json'
    require(bank.sha(original_path) == foundation['original_record_sha256'],
            'the retained dependency recipe changed')
    original = bank.previous.read(original_path)
    require(original['external_inputs'] == base['external_inputs']
            and all(bank.sha(p) == h for p, h in (original['toolchain'] | base['build_tools']).items()),
            'the retained compiler, build tools or SDK changed')
    for dep in original['dependencies']:
        root = Path(dep['root'])
        require(bank.stage.hashed(dict(source.blobs(root, dep['revision'], dep['source_sha256'])))
                == dep['source_sha256']
                and all(bank.sha(root / p) == h for p, h in dep['source_sha256'].items()),
                'the retained dependency source changed: ' + str(root))

    controls = source.pin(bank.P, CONTROLLERS)['source_sha256']
    require(bank.stage.hashed(dict(source.blobs(bank.P, revision, controls))) == controls,
            'the committed build controller or transitive helper changed')
    payload, recipe = projection(base)
    hashes = bank.stage.hashed(payload)
    fields = dict(source_commit=revision, native_base_commit=base['native_source_commit'],
                  overlay_commit=OVERLAY, overlays={p: hashes[p] for p in OVERLAYS},
                  controller_sources=controls, source_files=hashes, recipe=recipe,
                  foundation=foundation, external_inputs=base['external_inputs'], driver=reference['driver'],
                  build_tools=base['build_tools'], base_build_sha256=reference['native_build_sha256'],
                  base_binary=reference['binary'], attribution_control=CONTROL)
    return fields, payload, original, base


def check(row, committed=True):
    if committed:
        require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the preceding observer build stage')
    fields, _, original, base = inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items())
            and {p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(TREE / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'the materialized observer source or build inputs changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        required = (*primary.REQUIRED_SOURCES, *OVERLAYS)
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and actual['fresh_objects'] == base['actual_inputs']['fresh_objects']
                and all(actual['compiled'].get(str(TREE / p)) == row['source_files'][p] for p in required)
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items())
                and row['binaries'].keys() == base['binaries'].keys()
                and all(Path(b['path']).stat().st_size == b['bytes'] and bank.sha(b['path']) == b['sha256']
                        for b in row['binaries'].values()), 'require a fresh engine with the mapped observer')
    return original


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        names = ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')
        paths = [build / name for name in names]
        paths.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in paths}
    primary.publish(row)


def prepare():
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect any existing observer generation before preparing')
    revision = source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
    fields, payload, _, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in payload.items():
        target = TREE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False,
               ownership_closed=True, scope='The exact primary build plus the three cold-host observer inputs.',
               retention='Keep for its 118785-token observer-cost and cold-phase comparison. Review at closure.')
    check(row, committed=False)
    save(row)


def main(action):
    require(action in ('prepare', 'configure', 'compile'), 'select one observer build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, primary.TARGETS
        builder.check, builder.save = check, save
        if action == 'prepare':
            prepare()
        else:
            builder.execute(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
