"""Rebuild the measured early PLE engine from its committed source projection."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_cold_ple_measure as ple
import dflash2_decode_ranked_refill_build as retained_inputs

window, bank, wait, require = ple.window, ple.bank, ple.wait, ple.require
base_build = window.build
builder, source = base_build.builder, base_build.source
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_early_native_v3'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
SOURCE = '8f539f82e38e1ec2c3e4285a29928aa1a32f4ea3'
PATCH = '8a2a6c4b2fc70b3465bee13555b2f83e961dfb57'
ARCHIVE = '6819afd49'
ARCHIVE_KEY = 'dflash2_prefill_ple_early_native_v2'
ARCHIVE_ROOT = bank.B / ARCHIVE_KEY.replace('_', '-')
TIMING_KEY = 'dflash2_prefill_ple_early_speed_v1'
CAPACITY_KEY = 'dflash2_prefill_ple_early_capacity_v1'
RESTORE_KEY = 'dflash2_prefill_ple_early_restore_v1'
RECIPE_SHA256 = 'b1cc47c1b8888ea4a807f8a7bfa9cfed5dc1c28ac07a456a8b667085267c70ef'
OVERLAYS = ('include/strata/prefill/prefill.hpp', 'src/prefill/layer_major.inc', 'src/prefill/prefill.cpp')


def archived_build():
    row = bank.previous.read(ARCHIVE_ROOT / 'record.json')
    compact = bank.previous.historic(ARCHIVE, ARCHIVE_KEY)
    speed, capacity, restore = (bank.previous.historic(ARCHIVE, key)
                                for key in (TIMING_KEY, CAPACITY_KEY, RESTORE_KEY))
    identity = bank.signature(compact)
    require(wait.retained.closed(row, 'built') and builder.compact(row) == compact
            and row['overlay_commit'] == PATCH and bank.signature(row['recipe']) == RECIPE_SHA256
            and all(wait.retained.closed(p, 'closed') and p['native_build_sha256'] == identity
                    and p['native_source_commit'] == PATCH for p in (speed, capacity, restore))
            and restore['binary'] == speed['binary'] == row['binaries']['strata']
            and capacity['binary'] == row['binaries']['strata_full_context_diagnostic']
            and speed['decision']['advance'] and capacity['outcomes']['full-copy']['final_slot_consumed']
            and restore['physical_sha256'] == bank.signature(capacity)
            and restore['timing_sha256'] == capacity['timing_sha256'] == bank.signature(speed),
            'recover the original source, build recipe and qualified PLE results')
    return row, identity


def inputs(revision):
    archive, identity = archived_build()
    controllers = (*archive['controller_sources'], retained_inputs.THIS)
    reference, base, original, _, controls = retained_inputs.retained_build_inputs(revision, controllers)
    require(archive['foundation'] == base['foundation'] and archive['driver'] == reference['driver']
            and archive['build_tools'] == base['build_tools'] and archive['external_inputs'] == base['external_inputs']
            and archive['base_binary'] == reference['binary']
            and archive['base_build_sha256'] == reference['native_build_sha256']
            and archive['native_base_commit'] == reference['native_source_commit']
            and controls.keys() == set(controllers), 'keep the measured compiler, runtime and fallback')
    files = archive['source_files']
    payload = dict(source.blobs(bank.P, SOURCE, files))
    require(bank.stage.hashed(payload) == files and archive['overlays'] == {p: files[p] for p in OVERLAYS},
            'the projected Git source must recover all original native inputs')

    old, new = str(ARCHIVE_ROOT), str(ROOT)
    recipe = archive['recipe']
    args = [arg.replace(old, new) for arg in recipe['configure']]
    require(args[2:5] == [str(TREE), '-B', str(ROOT / 'build')]
            and recipe['targets'] == base_build.primary.TARGETS, 'relocate only the observed build paths')
    objects = sorted(str(Path(p).relative_to(ARCHIVE_ROOT / 'build')) for p in archive['actual_inputs']['objects'])
    preserved = ('native_base_commit', 'overlay_commit', 'source_files', 'overlays', 'foundation', 'external_inputs',
                 'build_tools', 'driver', 'base_binary', 'base_build_sha256')
    fields = {k: archive[k] for k in preserved}
    fields.update(source_commit=revision, native_source_commit=SOURCE, controller_sources=controls,
                  original_build_sha256=identity, fresh_object_paths_sha256=bank.signature(objects),
                  recipe=dict(configure=args, targets=recipe['targets'],
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}))
    return fields, payload, original, archive


def prepare(ctx):
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing candidate generation before preparing')
    require({p.name for p in ARCHIVE_ROOT.iterdir()} == {'record.json'},
            'the archived generation should contain only its sealed build receipt')
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
               ownership_closed=True, scope='Rebuild every original early PLE translation unit. Check the exact '
                   'new executable through STOP, consumption, paired prefill, physical capacity and live RESTORE.',
               retention='Keep through the same-build checks and sustained generation comparison. '
                   'Review at the first failed gate or comparison closure.')
    ctx.check(row, committed=False)
    ctx.save(row)


def context():
    ctx = bank.basis.core.source.device.load_module('early_ple_build', base_build.__file__)
    ctx.KEY, ctx.ROOT, ctx.TREE, ctx.OVERLAYS = KEY, ROOT, TREE, OVERLAYS
    ctx.inputs = inputs
    inherited_check = ctx.check

    def check(row, committed=True):
        original = inherited_check(row, committed=committed)
        if row['stage'] == 'built':
            observed = sorted(str(Path(p).relative_to(ROOT / 'build')) for p in row['actual_inputs']['objects'])
            require(bank.signature(observed) == row['fresh_object_paths_sha256'],
                    'rebuild every original translation unit')
        return original

    ctx.check = check
    ctx.prepare = lambda: prepare(ctx)
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned build controller interpreter')
    context().main(action)
