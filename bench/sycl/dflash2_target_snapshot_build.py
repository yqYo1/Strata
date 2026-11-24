"""Project the target-only snapshot fix onto the measured DFlash2 engine."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_build as base

bank, builder, require = base.bank, base.builder, base.require
KEY = 'dflash2_target_snapshot_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
BEFORE = '0795b5ada220f49afc4e7b456340b9bc019c9462'
FIX = '3a93dafd744f4becec7c6781602f142ac790ee6c'
HOST = 'c7a34a69bee636015a60805882232858f2e251cc'
HOST_KEY = 'dflash2_target_snapshot_host_v2'
PARENT_KEY = base.DFLASH2_KV_KEY
PARENT_ROOT = base.DFLASH2_KV_ROOT
COMPILED = ('include/strata/core/conversation_snapshot.hpp',
            'src/core/conversation_state.cpp', 'src/program/generate.cpp')
FIXTURES = ('src/core/conversation_validation_test.cpp',
            'src/core/conversation_snapshot_test.cpp', 'cmake/sycl_backend.cmake')


def recover_v2(expected):
    native, foundation, files, recipe, proof = base.project(expected)
    previous = bank.previous.read(base.ROOT / 'record.json')
    require(previous['stage'] == 'built' and previous['passed'] and previous['complete']
            and previous['ownership_closed'] and not previous['active']
            and builder.compact(previous) == bank.basis.validation.committed(bank.RESULT, (base.KEY,))
            and previous['source_files'] == bank.stage.hashed(files)
            and previous['recipe'] == recipe and previous['projection'] == proof
            and previous['external_inputs'] == native['external_inputs'],
            'recover the previously measured V2 source and recipe from Git')
    return native, foundation, files, recipe, proof, previous


def project(expected=None):
    # Construct from the older recipe even after its executable is retired.
    # The ordinary live-baseline reader retains its binary checks.
    source = bank.basis.core.source.device.load_module('target_snapshot_parent', base.__file__)
    source.checked_upload_project = recover_v2
    native, foundation, files, recipe, proof = source.dflash2_kv_project(expected)
    parent = bank.previous.read(PARENT_ROOT / 'record.json')
    original = bank.stage.hashed(files)
    require(parent['stage'] == 'built' and parent['complete'] and parent['passed']
            and parent['ownership_closed'] and not parent['active']
            and builder.compact(parent) == bank.basis.validation.committed(bank.RESULT, (PARENT_KEY,))
            and parent['source_files'] == original and parent['recipe'] == recipe
            and parent['projection'] == proof and parent['foundation'] == foundation,
            'recover the measured DFlash2 engine source and build recipe')
    host = bank.previous.historic(HOST, HOST_KEY)
    require(host['stage'] == 'closed' and host['passed'] and host['ownership_closed']
            and not host['active'] and host['checks'] == 1656 and host['source_commit'] == FIX,
            'recover the passing target-only host fixture')
    for path in (*COMPILED, *FIXTURES):
        before = files[path]
        files[path] = base.merge_delta(before, path, BEFORE, FIX)
        require(base.merge_delta(files[path], path, FIX, BEFORE) == before,
                'snapshot change would overwrite unrelated input: ' + path)
    hashes = bank.stage.hashed(files)
    require({p for p in hashes if hashes[p] != original[p]} == set(COMPILED + FIXTURES),
            'the target-only conversation fix is the only native change')
    for path in (*COMPILED[:2], FIXTURES[0]):
        data = bank.stage.source.git(bank.P, 'show', FIX + ':' + path)
        require(files[path] == data, 'the target-only host fixture source differs: ' + path)
    old, new = str(PARENT_ROOT), str(ROOT)
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                  targets=recipe['targets'])
    proof = dict(proof, target_snapshot_fix=FIX, target_snapshot_host=HOST,
                 target_snapshot_parent_hash=parent['binaries']['strata']['sha256'],
                 overlays={p: hashes[p] for p in proof['overlays'].keys() | set(COMPILED)},
                 auxiliary_sources={p: hashes[p] for p in FIXTURES})
    return native, foundation, files, recipe, proof


def context():
    ctx = base.context('dflash2-kv-stage')
    check_build = ctx.check
    ctx.KEY, ctx.ROOT, ctx.project = KEY, ROOT, project
    ctx.scope = ('Target-only conversation snapshots for the same DFlash2 long-prefill and decode routes. '
                 'Check same-device A/B/A prompt reuse, cache payloads, a nonaligned resumed suffix and '
                 'near-capacity conversation switching. Rebuild all engine units with the retained recipe.')
    ctx.retention = ('Keep the incumbent DFlash2 prefill control and the independent coherent decode '
                     'reference for this comparison. Reconcile the earlier V2 executable consumer '
                     'before the new generation. Retire the extra generation at qualification closure.')

    def check(row, committed=True):
        if not committed:
            pin = bank.stage.source.pin(bank.P, [THIS])
            require(pin['revision'] == row['source_commit'], 'commit the snapshot build controller first')
            row['controller_sources'].update(pin['source_sha256'])
        require(THIS in row['controller_sources'], 'pin the snapshot build controller')
        return check_build(row, committed)

    ctx.check = check
    return ctx


def main(action):
    require(action in ('projection', 'prepare', 'configure', 'compile', 'host'), 'select one build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        ctx = context()
        builder.ROOT, builder.KEY, builder.check, builder.save = ctx.ROOT, ctx.KEY, ctx.check, ctx.save
        if action == 'projection':
            _, _, files, _, proof = ctx.project()
            print({'files': len(files), 'source': proof['target_snapshot_fix']})
        elif action == 'prepare':
            require(not (base.ROOT / 'build' / 'strata').exists(),
                    'retire the superseded V2 executable after checking its consumers and source recovery')
            base.prepare(ctx)
        elif action == 'host':
            bank.stage.host(root_path=ctx.ROOT, check_build=ctx.check, save_record=ctx.save)
        else:
            row = bank.previous.read(ctx.ROOT / 'record.json')
            builder.TARGETS = row['recipe']['targets']
            builder.execute(action)


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
