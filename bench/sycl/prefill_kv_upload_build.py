"""Build the measured 128-slot split prefill and its shared Gate comparison."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_large_gather_model_source as gather
import prefill_split_upload_build as upload
from official_model_sources import merge_delta, once

bank, builder, require = upload.bank, upload.builder, upload.require
KEY = 'prefill_kv_upload_native_v2'
ROOT = bank.B / KEY.replace('_', '-')
PREVIOUS_ROOT = bank.B / 'prefill-kv-upload-native-v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PARENT_EVIDENCE = '83ae2717d51f5ab54f50787cbcae5acf1a945cff'
UPLOAD_EVIDENCE = '982211f5c91b4251ff89b8ad93f1f369a46fbad0'
UPLOAD_BUILD = 'e4dfad786347e5adf0f913d69df50454e3decbaa'
SCOPE = 'd92f63fa7e92e31324d27062f8ac5f1702a71bab'
REQUEST_SCOPE = 'b8aa2d3cba417f2c1d2d92381dad1aad4004b341'
OVERLAYS = ((upload.UPLOAD, (upload.UNIT, upload.WINDOW, upload.LAYER)),
            (upload.HELPER, (upload.WINDOW, upload.HEADER, upload.FIXTURE)),
            (SCOPE, (upload.WINDOW,)),
            (REQUEST_SCOPE, (upload.UNIT, upload.LAYER, upload.WINDOW)))
GATE_KEY = 'prefill_shared_gate_native_v1'
GATE_ROOT = bank.B / GATE_KEY.replace('_', '-')
GATE_SOURCE = '2cebe8a15e21c17fb345653fce90c4e3cf44c922'
GATE_PATHS = (upload.UNIT, upload.WINDOW, upload.LAYER)
DFLASH2_KV_KEY = 'dflash2_prefill_kv_stage_native_v1'
DFLASH2_KV_ROOT = bank.B / DFLASH2_KV_KEY.replace('_', '-')
DFLASH2_KV_SOURCE = '44a2926a6fd4a53cd26499b8415e53f9be47d50e'
DFLASH2_KV_PATHS = ('include/strata/prefill/dflash2_taps.hpp',
                    'src/prefill/sycl/dflash2_taps.cpp', 'src/program/generate.cpp')


def apply_upload(source):
    original = bank.stage.hashed(source)
    for revision, paths in OVERLAYS:
        for path in paths:
            before = source[path]
            source[path] = merge_delta(before, path, revision + '^', revision)
            require(merge_delta(source[path], path, revision, revision + '^') == before,
                    'upload would replace unrelated source: ' + path)
    changed = {path: h for path, h in bank.stage.hashed(source).items() if original.get(path) != h}
    require(set(changed) == {upload.UNIT, upload.WINDOW, upload.LAYER, upload.HEADER, upload.FIXTURE},
            'unexpected native change in upload projection')
    return changed


def project(expected=None):
    native, foundation, source, recipe, original = gather.build_inputs(expected)
    parent = bank.previous.historic(PARENT_EVIDENCE, gather.KEY)
    require(parent['stage'] == 'built' and parent['passed'] and parent['complete']
            and parent['ownership_closed'] and not parent['active'] and parent['host_control']['passed']
            and parent['native_source_commit'] == gather.DISPATCH_COUNTS
            and parent['projection'] == original and parent['recipe'] == recipe
            and parent['external_inputs'] == native['external_inputs']
            and parent['source_files'] == dict(files=len(source), sha256=original['source_sha256']),
            'recover the exact measured KV/gather source and recipe')
    earlier = bank.previous.historic(UPLOAD_EVIDENCE, 'prefill_split_upload_native_v1')
    timing = bank.previous.historic(UPLOAD_EVIDENCE, 'prefill_split_upload_speed_v1')
    require(earlier['native_source_commit'] == UPLOAD_BUILD and earlier['passed']
            and earlier['recipe']['targets'] == recipe['targets']
            and earlier['binaries']['strata'] == timing['binary']
            and timing['complete'] and timing['ownership_closed'] and timing['passed']
            and timing['decision']['repeated_warm_gain']['batched/demand'],
            'recover the earlier same-queue scheduling evidence')
    changed = apply_upload(source)
    old, new = str(gather.MODEL_ROOT), str(ROOT)
    require(recipe['configure'][2:5] == [old + '/source', '-B', old + '/build'],
            'unexpected parent configure paths')
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                  targets=recipe['targets'])
    proof = dict(parent_evidence=PARENT_EVIDENCE, parent_source=parent['native_source_commit'],
                 parent_binary=parent['binaries']['strata'], parent_source_files=parent['source_files'],
                 upload_evidence=UPLOAD_EVIDENCE, overlay_commits=[rev for rev, _ in OVERLAYS],
                 overlays=changed, object_count=earlier['projection']['object_count'],
                 object_set_sha256=earlier['projection']['object_set_sha256'])
    return native, foundation, source, recipe, proof


def checked_upload_project(expected=None):
    native, foundation, source, recipe, proof = project(expected)
    baseline = bank.previous.read(ROOT / 'record.json')
    base_hashes = bank.stage.hashed(source)
    require(baseline['stage'] == 'built' and baseline['complete'] and baseline['passed']
            and baseline['ownership_closed'] and not baseline['active']
            and builder.compact(baseline) == bank.basis.validation.committed(bank.RESULT, (KEY,))
            and baseline['source_files'] == base_hashes and baseline['recipe'] == recipe
            and baseline['projection'] == proof
            and baseline['external_inputs'] == native['external_inputs']
            and bank.sha(baseline['binaries']['strata']['path']) == baseline['binaries']['strata']['sha256'],
            'recover the qualified split-prefill baseline and preserve its executable')
    return native, foundation, source, recipe, proof, baseline


def shared_gate_project(expected=None):
    native, foundation, source, recipe, proof, baseline = checked_upload_project(expected)
    base_hashes = baseline['source_files']
    for path in GATE_PATHS:
        before = source[path]
        source[path] = merge_delta(before, path, GATE_SOURCE + '^', GATE_SOURCE)
        require(merge_delta(source[path], path, GATE_SOURCE, GATE_SOURCE + '^') == before,
                'shared Gate change would replace unrelated source: ' + path)
    hashes = bank.stage.hashed(source)
    require({path for path in hashes if hashes[path] != base_hashes[path]} == set(GATE_PATHS),
            'unexpected native changes in the shared Gate projection')
    proof = dict(proof, gate_source=GATE_SOURCE, gate_parent_binary=baseline['binaries']['strata'],
                 gate_parent_source_sha256=bank.signature(base_hashes),
                 overlays={path: hashes[path] for path in proof['overlays'].keys() | set(GATE_PATHS)})
    old, new = str(ROOT), str(GATE_ROOT)
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={key: val.replace(old, new) for key, val in recipe['environment'].items()},
                  targets=recipe['targets'])
    return native, foundation, source, recipe, proof


def dflash2_kv_project(expected=None):
    native, foundation, source, recipe, proof, baseline = checked_upload_project(expected)
    base_hashes = baseline['source_files']
    for path in DFLASH2_KV_PATHS:
        before = source[path]
        source[path] = merge_delta(before, path, DFLASH2_KV_SOURCE + '^', DFLASH2_KV_SOURCE)
        require(merge_delta(source[path], path, DFLASH2_KV_SOURCE, DFLASH2_KV_SOURCE + '^') == before,
                'DFlash2 tap binding would replace unrelated source: ' + path)

    # V2 implements aligned retention with a boolean. The later branch also
    # implements partial-page and packed suffix modes. Map only the aligned guard.
    include = b'#include "strata/prefill/native16.hpp"\n'
    include_tap = include + b'#include "strata/prefill/dflash2_taps.hpp"\n'
    reload_guard = (
        b'    // External layer taps and the final output callback keep the original reloads.\n'
        b'    const bool keep_kv_stage = prefill_keep_kv_stage_requested && m.layer_major_active && m.native16_active &&\n'
        b'        LE == LB + 1 && LE < g.n_layers && !on_layer && !m.pp && !next_ &&\n')
    tap_guard = (
        b'    // General callbacks and final layers reload. The typed DFlash2 tap has a fixed capture set.\n'
        b'    const bool untapped_dflash2_layer = prefill_keep_kv_stage_requested &&\n'
        b'        on_layer.target<Dflash2PrefillTaps::LayerTap>() && !Dflash2PrefillTaps::captures_layer(LB);\n'
        b'    const bool keep_kv_stage = prefill_keep_kv_stage_requested && m.layer_major_active && m.native16_active &&\n'
        b'        LE == LB + 1 && LE < g.n_layers && (!on_layer || untapped_dflash2_layer) && !m.pp && !next_ &&\n')
    before = source[upload.UNIT]
    selected = once(once(before, include, include_tap), reload_guard, tap_guard)
    require(once(once(selected, tap_guard, reload_guard), include_tap, include) == before,
            'the aligned guard must reverse to the measured source')
    source[upload.UNIT] = selected
    hashes = bank.stage.hashed(source)
    changed = {name for name in hashes if hashes[name] != base_hashes[name]}
    require(changed == {*DFLASH2_KV_PATHS, upload.UNIT}, 'unexpected DFlash2 KV comparison inputs')
    proof = dict(proof, dflash2_kv_source=DFLASH2_KV_SOURCE, dflash2_kv_base_binary=baseline['binaries']['strata'],
                 dflash2_kv_base_source_sha256=bank.signature(base_hashes),
                 overlays={name: hashes[name] for name in proof['overlays'].keys() | changed})
    old, new = str(ROOT), str(DFLASH2_KV_ROOT)
    recipe = dict(configure=[arg.replace(old, new) for arg in recipe['configure']],
                  environment={key: val.replace(old, new) for key, val in recipe['environment'].items()},
                  targets=recipe['targets'])
    return native, foundation, source, recipe, proof


def context(variant='upload'):
    ctx = bank.basis.core.source.device.load_module('kv_upload_build', upload.__file__)
    if variant == 'shared-gate':
        ctx.ROOT, ctx.KEY, ctx.project = GATE_ROOT, GATE_KEY, shared_gate_project
        ctx.scope = ('Shared Gate only, in the exact V2 split-prefill source. Compare original '
                     'and reload to measure placement, reload and reuse to measure removal of '
                     'expansions, and original and reuse for the total effect. Keep KV reuse, '
                     'scalar gather, batched uploads, attention placement and decode fixed.')
        ctx.retention = ('Retain the measured V2 prefill and coherent decode references. '
                         'Retire the closed earlier KV/gather build before preparing this comparison.')
    elif variant == 'dflash2-kv-stage':
        ctx.ROOT, ctx.KEY, ctx.project = DFLASH2_KV_ROOT, DFLASH2_KV_KEY, dflash2_kv_project
        ctx.scope = ('DFlash2 aligned KV retention on QSA layers outside the fixed tap set. '
                     'Keep the measured V2 engine, draft settings, prefill ownership and compiler recipe. '
                     'Compare reload and retention in independent processes of the resulting engine.')
        ctx.retention = ('Preserve the measured V2 and coherent decode references during this comparison. '
                         'Review the additional build after the native route and timing decision.')
    else:
        ctx.ROOT, ctx.KEY, ctx.project = ROOT, KEY, project
        ctx.scope = ('Compare demand and batched issue in the original 128-slot split workspace. '
                     'Keep KV reuse, scalar gather and attention copy fixed.')
        ctx.retention = ('Keep the measured KV/gather and independent decode references. '
                         'Review the third generation on comparison closure or failed admission.')
    return ctx


def prepare(ctx):
    root = ctx.ROOT
    require(not root.exists() and ctx.KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing KV/upload generation first')
    require(not upload.ROOT.exists(), 'verify retirement of the previous upload build')
    require(not PREVIOUS_ROOT.exists(), 'verify retirement of the one-layer upload generation')
    if ctx.KEY == GATE_KEY:
        require(not (gather.MODEL_ROOT / 'build').exists(),
                'close and retire the previous ordinary KV/gather build before adding a generation')
    native, foundation, source, recipe, proof = ctx.project()
    modules = (gather, gather.fixture, gather.kv, upload, upload.boundary, upload.boundary.stock)
    controls = [THIS, *(str(Path(m.__file__).resolve().relative_to(bank.P)) for m in modules)]
    pins = bank.stage.source.pin(bank.P, [*controls, 'tools/sycl', *bank.basis.POLICIES])
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=pins['revision'],
               controller_sources=pins['source_sha256'], source_files=bank.stage.hashed(source),
               foundation=foundation, projection=proof, recipe=recipe, external_inputs=native['external_inputs'],
               build_tools={name: bank.sha(name) for name in ('/usr/bin/cmake', '/usr/bin/ninja')},
               driver=builder.guards.driver_guard(), retained_decode_reference=upload.decode_reference(recover=True),
               active=False, passed=False, complete=False, ownership_closed=True, commands=[],
               scope=ctx.scope, retention=ctx.retention)
    root.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (root / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = root / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    ctx.check(row, committed=False)
    ctx.save(row)


def main(action, variant='upload'):
    require(action in ('projection', 'prepare', 'configure', 'compile', 'host'), 'select one build stage')
    require(variant in ('upload', 'shared-gate', 'dflash2-kv-stage'), 'unknown native comparison')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        ctx = context(variant)
        builder.ROOT, builder.KEY, builder.check, builder.save = ctx.ROOT, ctx.KEY, ctx.check, ctx.save
        if action == 'projection':
            _, _, source, _, proof = ctx.project()
            print({'source_files': len(source), 'projection': proof})
        elif action == 'prepare':
            prepare(ctx)
        elif action == 'host':
            bank.stage.host(root_path=ctx.ROOT, check_build=ctx.check, save_record=ctx.save)
        else:
            row = bank.previous.read(ctx.ROOT / 'record.json')
            builder.TARGETS = row['recipe']['targets']
            builder.execute(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
