"""Build the measured whole-horizon PLE engine with one decode generator overlay."""

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_record_window as grouped
import dflash2_decode_layer_quota_replay as quota
from make_profile import write_profile

bank, early, require = grouped.bank, grouped.early, grouped.require
validate = grouped.validate
builder, source = early.builder, early.source
load = bank.basis.core.source.device.load_module
THIS = 'bench/dflash2_decode_layer_placement_build.py'
KEY = 'dflash2_decode_layer_placement_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
PREIMAGE = 'be65a4811e953e3299dea61e802d09c59071f025'
NATIVE_BASE = '968e71988953dca3d98037d3b3fec9f903bdfd6f'
CANDIDATE = '398d6480ba395ec9cfb1eede5daa1303ea974470'
GENERATOR = 'src/program/generate.cpp'
PROFILES_FILE = 'data/expert-profile.bin'
PROFILE_SLOTS = 'tests/data/dflash2_decode_layer_placement_slots.json'
EXPERT_LAYOUT = 'tests/data/native_experts/iq3_s.txt'
PRIOR_TIMING = 'ac3638fb3708cccaa6e73bc8629cb048af148e7d'
PRIOR_PLACEMENT = 'dflash2_decode_phase_placement_speed_v1'


def object_paths(row, root):
    directory = root / 'build'
    return sorted(Path(path).relative_to(directory).as_posix()
                  for path in row['actual_inputs']['objects'])


def placement_rankings():
    profile = bank.P / PROFILES_FILE
    formats = quota.replay.manifest_formats(bank.P / EXPERT_LAYOUT)
    order, stock, sizes, _ = quota.stock_geometry(profile, quota.replay.expected_membership(profile), formats)
    selections = json.loads((bank.P / PROFILE_SLOTS).read_bytes())
    require(isinstance(selections, list) and len(selections) == quota.replay.LAYERS
            and all(isinstance(layer, list) for layer in selections)
            and all(type(expert) is int and 0 <= expert < quota.replay.EXPERTS
                    for layer in selections for expert in layer), 'invalid committed placement slots')
    blend = [(layer, expert) for layer, experts in enumerate(selections) for expert in experts]
    require(len(blend) == len(stock) == len(set(blend)) and blend[:quota.PROTECTED] == stock[:quota.PROTECTED]
            and [layer for layer, _ in blend] == [layer for layer, _ in stock],
            'the blend must keep the protected slots and per-layer capacities')
    rankings = {}
    for name, slots in (('stock', stock), ('blend', blend)):
        ranking, capacity = quota.global_profile(slots, order, sizes)
        require(capacity == 968, 'the recovered profile changed its original byte budget')
        rankings[name] = ranking
    return rankings


def build_inputs(revision):
    archived = bank.previous.read(grouped.BUILD_ROOT / 'record.json')
    checked = validate.committed(bank.RESULT, (grouped.BUILD_KEY,))
    require(grouped.wait.retained.closed(archived, 'built') and builder.compact(archived) == checked
            and archived['native_source_commit'] == PREIMAGE
            and archived['native_base_commit'] == NATIVE_BASE
            and archived['overlay_commit'] == PREIMAGE, 'require the live measured whole-window build')
    foundation = grouped.build_context().check(archived)
    fields, payload, _, _ = grouped.build_inputs(PREIMAGE)
    source_hashes = archived['source_files']
    require(fields['source_files'] == bank.stage.hashed(payload) == source_hashes,
            'recover the original source projection from Git')
    preimage = dict(source.blobs(bank.P, NATIVE_BASE, (GENERATOR,)))
    overlay = dict(source.blobs(bank.P, CANDIDATE, (GENERATOR,)))
    native_compiled = archived['actual_inputs']['compiled']
    measured_path = str(grouped.TREE / GENERATOR)
    require(preimage[GENERATOR] == payload[GENERATOR] != overlay[GENERATOR]
            and native_compiled[measured_path] == source_hashes[GENERATOR],
            'replace only the generator actually compiled in the measured build')
    payload.update(overlay)
    projected = bank.stage.hashed(payload)
    require(projected.keys() == source_hashes.keys()
            and {p for p, h in projected.items() if h != source_hashes[p]} == {GENERATOR},
            'the candidate may change only the measured generator')

    paths = (*archived['controller_sources'], THIS, PROFILES_FILE, PROFILE_SLOTS, EXPERT_LAYOUT,
             quota.THIS, 'bench/dflash2_decode_residency_replay.py',
             'tools/make_profile.py', 'tools/sycl/cpu_work_census.py',
             'bench/dflash2_decode_layer_placement_check.py',
             'bench/dflash2_decode_layer_placement_speed.py',
             'bench/dflash2_prefill_ple_horizon_continuation.py')
    controls = source.pin(bank.P, paths)['source_sha256']
    require(controls.keys() == set(paths)
            and bank.stage.hashed(dict(source.blobs(bank.P, revision, controls))) == controls,
            'commit the exact build controller, profiles and inherited helpers')
    timing = bank.previous.historic(PRIOR_TIMING, PRIOR_PLACEMENT)
    prior_profiles = timing['qualified_profiles']
    require(grouped.wait.retained.closed(timing, 'closed') and set(prior_profiles) == {'stock', 'blend'}
            and len(timing['processes']) == 6
            and not timing['decision']['eligible_for_qualification']
            and all(s['tg_wins'] == 3 and s['tg_geomean'] > 1 for s in timing['decision']['summary'].values()),
            'require the owned placement evidence and original profile identities')
    profiles = {name: dict(asset, path=str(ROOT / 'profiles' / (name + '.bin')))
                for name, asset in prior_profiles.items()}

    old_root, new_root = str(grouped.BUILD_ROOT), str(ROOT)
    recipe = archived['recipe']
    configure = [arg.replace(old_root, new_root) for arg in recipe['configure']]
    require(configure[2:5] == [str(TREE), '-B', str(ROOT / 'build')]
            and recipe['targets'] == early.base_build.primary.TARGETS,
            'relocate only the measured compiler and build paths')
    fields = {key: archived[key] for key in ('native_base_commit', 'foundation', 'external_inputs',
              'build_tools', 'driver', 'base_binary', 'base_build_sha256')}
    fields.update(source_commit=revision, native_source_commit=CANDIDATE, overlay_commit=CANDIDATE,
                  overlays=bank.stage.hashed(overlay), source_files=projected, controller_sources=controls,
                  prefill_build_sha256=bank.signature(checked), prior_placement_sha256=bank.signature(timing),
                  profiles=profiles, object_paths_sha256=bank.signature(object_paths(archived, grouped.BUILD_ROOT)),
                  recipe=dict(configure=configure, targets=recipe['targets'],
                              environment={k: v.replace(old_root, new_root) for k, v in recipe['environment'].items()}))
    return fields, payload, foundation, archived


def context():
    ctx = load('decode_layer_placement_build', early.base_build.__file__)
    ctx.KEY, ctx.ROOT, ctx.TREE, ctx.OVERLAYS = KEY, ROOT, TREE, (GENERATOR,)
    ctx.inputs = build_inputs
    check = ctx.check

    def checked(row, committed=True):
        foundation = check(row, committed=committed)
        require(all(bank.sha(profile['path']) == profile['sha256'] for profile in row['profiles'].values()),
                'a generated expert profile differs from the original comparison')
        if row['stage'] == 'built':
            require(bank.signature(object_paths(row, ROOT)) == row['object_paths_sha256'],
                    'require the complete measured translation-unit set')
        return foundation

    def prepare():
        require(not ROOT.exists() and KEY not in validate.committed(bank.RESULT),
                'inspect the existing placement generation before preparing')
        revision = source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
        fields, payload, _, _ = build_inputs(revision)
        ROOT.mkdir(mode=0o700)
        for name in ('owned', 'tmp', 'cache', 'profiles'):
            (ROOT / name).mkdir(mode=0o700)
        for name, ranking in placement_rankings().items():
            write_profile(fields['profiles'][name]['path'], ranking)
        for name, contents in payload.items():
            target = TREE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        row = dict(fields, stage='prepared', active=False, commands=[], complete=False, passed=False,
                   ownership_closed=True, scope='Same prefill and decode by default. An opt-in decode placement '
                   'change uses the existing per-layer stock cache.',
                   retention='Keep for the named model admission, cancellation and paired placement timings.')
        checked(row, committed=False)
        ctx.save(row)

    ctx.check, ctx.prepare = checked, prepare
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned build controller interpreter')
    context().main(action)
