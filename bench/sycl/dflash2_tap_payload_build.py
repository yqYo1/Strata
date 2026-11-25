"""Add the five-layer tap observer to the measured target-snapshot engine."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_target_snapshot_build as snapshot
from official_model_sources import merge_delta

base, bank, builder, require = snapshot.base, snapshot.bank, snapshot.builder, snapshot.require
KEY = 'dflash2_tap_payload_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TAPS = 'src/prefill/sycl/dflash2_taps.cpp'
OBSERVER = '57191d9f489f30efe09290b972e20e709d23f07f'
SPEED = 'ad8652973c8f025e73bd07bc791a9976a55e20a9'
SPEED_KEY = 'dflash2_snapshot_direct_prefill_v1'


def project(expected=None):
    native, foundation, files, recipe, proof = snapshot.project(expected)
    parent = bank.previous.read(snapshot.ROOT / 'record.json')
    compact = builder.compact(parent)
    require(parent['stage'] == 'built' and parent['complete'] and parent['passed']
            and parent['ownership_closed'] and not parent['active'] and parent['host_control']['passed']
            and compact == bank.basis.validation.committed(bank.RESULT, (snapshot.KEY,))
            and parent['source_files'] == bank.stage.hashed(files) and parent['recipe'] == recipe
            and parent['projection'] == proof and parent['foundation'] == foundation
            and parent['external_inputs'] == native['external_inputs'],
            'recover the original measured snapshot source and recipe')

    speed = bank.previous.historic(SPEED, SPEED_KEY)
    selected = speed['snapshot_reference']
    require(speed['stage'] == 'closed' and speed['complete'] and speed['passed']
            and speed['ownership_closed'] and not speed['active']
            and speed['decision']['all_warm_requests_1000']['snapshot']
            and selected['binary'] == parent['binaries']['strata']
            and selected['source'] == parent['native_source_commit']
            and selected['native_build_sha256'] == bank.signature(compact),
            'bind the exact snapshot generation used for the measured long prefill')

    before = files[TAPS]
    after = merge_delta(before, TAPS, OBSERVER + '^', OBSERVER)
    require(after != before and merge_delta(after, TAPS, OBSERVER, OBSERVER + '^') == before,
            'the observer must preserve the existing tap producer and every unrelated source change')
    files[TAPS] = after
    old, new = str(snapshot.ROOT), str(ROOT)
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={name: value.replace(old, new) for name, value in recipe['environment'].items()},
                  targets=recipe['targets'])
    proof = dict(proof, tap_observer_source=OBSERVER, tap_parent_prefill_evidence=SPEED,
                 tap_parent_build_sha256=bank.signature(compact),
                 overlays=proof['overlays'] | bank.stage.hashed({TAPS: after}))
    return native, foundation, files, recipe, proof


def context():
    ctx = base.context('dflash2-kv-stage')
    check_build = ctx.check
    ctx.KEY, ctx.ROOT, ctx.project = KEY, ROOT, project
    ctx.scope = ('Rebuild the measured target-snapshot engine with only the independent DFlash2 tap observer. '
                 'Preserve the same SYCL recipe and require the observer source in the actual compiler census.')
    ctx.retention = ('Use this comparison build for the bounded tap smoke and the long writer parity pair. '
                     'Preserve the independent width-four and physical-context references. '
                     'Review this build at the first failed gate or pair closure.')

    def check(row, committed=True):
        if not committed and row['stage'] == 'prepared':
            controls = {THIS, snapshot.THIS}
            for module in tuple(sys.modules.values()):
                filename = getattr(module, '__file__', None)
                if filename and (path := Path(filename).resolve()).is_relative_to(bank.P / 'bench' / 'sycl'):
                    controls.add(path.relative_to(bank.P).as_posix())
            pins = bank.stage.source.pin(bank.P, sorted(controls))
            require(pins['revision'] == row['source_commit'], 'commit the observer build controller first')
            row['controller_sources'].update(pins['source_sha256'])
        require({THIS, snapshot.THIS} <= row['controller_sources'].keys(), 'missing observer build controller pins')
        return check_build(row, committed)

    ctx.check = check
    return ctx


def main(action):
    require(action in ('prepare', 'configure', 'compile', 'host'), 'select a tap observer build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        ctx = context()
        builder.ROOT, builder.KEY, builder.check, builder.save = ROOT, KEY, ctx.check, ctx.save
        if action == 'prepare':
            base.prepare(ctx)
        elif action == 'host':
            bank.stage.host(root_path=ROOT, check_build=ctx.check, save_record=ctx.save)
        else:
            row = bank.previous.read(ROOT / 'record.json')
            builder.TARGETS = row['recipe']['targets']
            builder.execute(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
