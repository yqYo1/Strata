"""Build the optional aligned KV retention through the typed DFlash2 tap."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_build as base

bank, builder, require = base.bank, base.builder, base.require
KEY = 'dflash2_tapped_kv_stage_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
UNIT = base.upload.UNIT
SELECTOR = 'STRATA_SYCL_PREFILL_KEEP_DFLASH2_TAPPED_KV'
PREVIOUS = b'''    // General callbacks and final layers reload. The typed DFlash2 tap has a fixed capture set.
    const bool untapped_dflash2_layer = prefill_keep_kv_stage_requested &&
        on_layer.target<Dflash2PrefillTaps::LayerTap>() && !Dflash2PrefillTaps::captures_layer(LB);
    const bool keep_kv_stage = prefill_keep_kv_stage_requested && m.layer_major_active && m.native16_active &&
        LE == LB + 1 && LE < g.n_layers && (!on_layer || untapped_dflash2_layer) && !m.pp && !next_ &&
        same_queue && m.copy == m.cs && stage_own();
'''
REPLACEMENT = f'''    static const bool retain_dflash2_tapped_kv = [] {{
        const char* value = std::getenv("{SELECTOR}");
        if (!value || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        throw std::invalid_argument("{SELECTOR} requires 0 or 1");
    }}();
    // The typed tap reads the residual and writes its separately owned tail
    // on this compute queue. Aligned staging keeps its own allocation.
    const bool keep_kv_stage = prefill_keep_kv_stage_requested && m.layer_major_active && m.native16_active &&
        LE == LB + 1 && LE < g.n_layers && !m.pp && !next_ &&
        same_queue && m.copy == m.cs && stage_own() &&
        (!on_layer || (on_layer.target<Dflash2PrefillTaps::LayerTap>() &&
            (!Dflash2PrefillTaps::captures_layer(LB) ||
             (retain_dflash2_tapped_kv && m.cs && sycl_backend::queue_for(m.cs).is_in_order()))));
'''.encode()


def project(expected=None):
    native, foundation, files, recipe, proof = base.dflash2_kv_project(expected)
    parent = bank.previous.read(base.DFLASH2_KV_ROOT / 'record.json')
    recorded = bank.basis.validation.committed(bank.RESULT, (base.DFLASH2_KV_KEY,))
    original_hashes = bank.stage.hashed(files)
    require(recorded == builder.compact(parent) and parent['stage'] == 'built'
            and parent['passed'] and parent['complete'] and parent['ownership_closed']
            and original_hashes == parent['source_files'] and recipe == parent['recipe']
            and proof == parent['projection'] and foundation == parent['foundation'],
            'recover the exact measured DFlash2 stage build')
    before = files[UNIT]
    files[UNIT] = base.once(before, PREVIOUS, REPLACEMENT)
    require(base.once(files[UNIT], REPLACEMENT, PREVIOUS) == before,
            'the selected guard must reverse to the original source')
    hashes = bank.stage.hashed(files)
    require({path for path in hashes if hashes[path] != original_hashes[path]} == {UNIT},
            'only the aligned staging guard may change')
    overlays = dict(proof['overlays'])
    overlays[UNIT] = hashes[UNIT]
    proof = dict(proof, overlays=overlays, tap_stage_base_binary=parent['binaries']['strata'],
                 tap_stage_base_source=parent['native_source_commit'], tap_stage_selector=SELECTOR)
    old, new = str(base.DFLASH2_KV_ROOT), str(ROOT)
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={key: val.replace(old, new) for key, val in recipe['environment'].items()},
                  targets=recipe['targets'])
    return native, foundation, files, recipe, proof


def context():
    ctx = base.context('dflash2-kv-stage')
    old_check = ctx.check
    ctx.KEY, ctx.ROOT, ctx.project = KEY, ROOT, project
    ctx.scope = ('One optional prefill guard for the known DFlash2 tap. Keep the recovered owner, queue, '
                 'layout, toolchain and decode. Compare the 15/35 tap layers separately at fixed aligned KV '
                 'retention, with resumed and ragged prompts, cancellation and physical capacity checks.')
    ctx.retention = ('Keep the locked DFlash2 reference through the next same-build guard comparison. '
                     'Reconcile the active build consumers before preparing the new generation and '
                     'retire superseded native or CPU-only comparison artifacts after closure.')

    def check(row, committed=True):
        if not committed:
            pin = bank.stage.source.pin(bank.P, [THIS])
            require(row['stage'] == 'prepared' and pin['revision'] == row['source_commit'],
                    'commit the exact tapped-layer controller first')
            row['controller_sources'].update(pin['source_sha256'])
        require(THIS in row['controller_sources'], 'include the tapped-layer controller in the build inputs')
        return old_check(row, committed)

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
            print({'source_files': len(files), 'source_unit': UNIT, 'selector': proof['tap_stage_selector']})
        elif action == 'prepare':
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
