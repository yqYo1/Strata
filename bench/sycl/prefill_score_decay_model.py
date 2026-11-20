"""Build one coherent model for independent prefill score-key and GDN comparisons."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import prefill_gdn_decay as decay
import prefill_retained_score_keys as keys
import prefill_large_gather_model_source as parent
from official_model_sources import merge_delta

bank, require = parent.bank, parent.bank.require
ROOT = bank.B / 'prefill-score-decay-model-v1'
KEY = 'prefill_score_decay_model_v1'
PARENT_BUILD = '83ae2717d51f5ab54f50787cbcae5acf1a945cff'
TERMINAL = '1919720d900ede6c9b55d25d09bb82b503777fd2'
SCORE_COUNTS = 'e88604bca4535fac04d4f8f61a3dc4469634ddaf'
DECAY_COUNTS = '063f2e392394b2d465ee82a76aabcd7aa2421958'
SCORE_FILES = (keys.HEADER, keys.ADMISSION, keys.CALLER, keys.PACKER)
DECAY_FILES = (decay.HEADER, *(name for name in decay.UNITS if name != decay.FIXTURE))
TERMINAL_FILES = ('src/program/generate.cpp', 'include/strata/program/full_context_diagnostic.hpp',
                  'tests/core/full_context_diagnostic_test.cpp')
CHANGED = tuple(sorted(set((*SCORE_FILES, *DECAY_FILES, *TERMINAL_FILES))))
BEFORE_COUNTS = '5f4aa020ac81f0f49d17680a2d8d738c9a5095fc2c47d617b6eaebbe4f36af73'
IMAGE_HASH = '9d10b4918adef780b6890080e94aec7ee47d3c02b5464ad1e208917e857beef0'
RECIPE_HASH = '1c5523147696377269762da5c25ec5b09e83fd75a8a3333b1c08eb2f632019e7'


def project(source, recipe):
    original = bank.stage.hashed(source)
    require(bank.signature(original) == 'f35b6b1a3f0c4f6317bb132caf43d01d19bc06acb26eb50ea4a8f2234e570ca2',
            'recover the complete measured KV/gather source before composing the model')
    for paths, before, after in ((SCORE_FILES, keys.BASE, keys.SOURCE), (DECAY_FILES, decay.BASE, decay.SOURCE),
                                 (TERMINAL_FILES, TERMINAL + '^', TERMINAL)):
        for name in paths:
            previous = source[name]
            source[name] = merge_delta(previous, name, before, after)
            require(merge_delta(source[name], name, after, before) == previous, 'irreversible source change: ' + name)
    require(bank.signature(bank.stage.hashed(source)) == BEFORE_COUNTS, 'engine before dispatch counters changed')
    for revision in (SCORE_COUNTS, DECAY_COUNTS):
        previous = source[keys.CALLER]
        source[keys.CALLER] = merge_delta(previous, keys.CALLER, revision + '^', revision)
        require(merge_delta(source[keys.CALLER], keys.CALLER, revision, revision + '^') == previous,
                'dispatch counters change the underlying implementation')
    hashes = bank.stage.hashed(source)
    require(len(hashes) == 1171 and bank.signature(hashes) == IMAGE_HASH
            and {p for p in hashes if original[p] != hashes[p]} == set(CHANGED), 'unexpected model source')
    old, new = str(parent.MODEL_ROOT), str(ROOT)
    relocated = dict(configure=[x.replace(old, new) for x in recipe['configure']],
                     environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                     targets=recipe['targets'])
    require(bank.signature(relocated) == RECIPE_HASH, 'model tools, options or target set changed')
    return relocated, dict(source_sha256=IMAGE_HASH, recipe_sha256=RECIPE_HASH,
                           changed={p: hashes[p] for p in CHANGED}, parent_source=parent.DISPATCH_COUNTS,
                           score_keys=keys.SOURCE, gdn_decay=decay.SOURCE, terminal_diagnostic=TERMINAL,
                           score_counts=SCORE_COUNTS, decay_counts=DECAY_COUNTS)


def component_gates(parent_hash):
    gates = {}
    for module in (keys, decay):
        row = bank.basis.validation.committed(bank.RESULT, (module.KEY,))
        stages = row['stages']
        require(row['source'] == module.SOURCE and row['parent_build'] == parent_hash
                and row['component_performance_gate'] and [s['stage'] for s in stages] == list(module.STAGES)
                and all(s['passed'] and s['ownership_closed'] for s in stages)
                and all(s['native_exit_code'] == 0 and not s['fresh_faults'] for s in stages[1:])
                and all(s['clean_timing'] and s['median_paired_time_ratio'] < 1 for s in stages[-3:]),
                'component state or performance gate failed: ' + module.KEY)
        gates[module.KEY] = bank.signature(stages)
    boundary = bank.basis.validation.committed(bank.RESULT, (keys.BOUNDARY_KEY,))
    stages = boundary['stages']
    require(boundary['source'] == keys.BOUNDARY_SOURCE and boundary['parent_build'] == parent_hash
            and [s['stage'] for s in stages] == ['build', 'boundary']
            and all(s['passed'] and s['ownership_closed'] for s in stages)
            and stages[-1]['stale_extent_control_detected'] and stages[-1]['host_probe_accepted']
            and stages[-1]['native_exit_code'] == 0 and not stages[-1]['fresh_faults'],
            'finish the same-parent indexer append/restore/alias device check')
    gates[keys.BOUNDARY_KEY] = bank.signature(stages)
    return gates


def check_dispatch(frame, start, rows, *, retain_keys, store_decay):
    """One completed layer-major segment on this model, excluding the final verifier row."""
    require(0 <= start < 262144 and 0 < rows <= 262144 - start, 'invalid prefill extent')
    slices = full_rows = reused_rows = 0
    for chunk in range(start, start + rows, 8192):
        limit = min(chunk + 8192, start + rows)
        previous = 0
        for begin in range(chunk, limit, 256):
            # qsa_step_fill at the last query in this slice, including its dead row.
            active = min(begin + 256, limit) // 4 + 1
            slices += 1
            full_rows += active
            reused_rows += previous
            previous = active if retain_keys else 0
    chunks = (rows + 8191) // 8192
    require(frame.count('strata prefill score packs:') == 12 and frame.count('strata prefill GDN decay:') == 36,
            'expected one complete dispatch record per model layer')
    score_observed = re.findall(r'^strata prefill score packs: layer=(\d+) slices=(\d+) '
                                r'packed_live_rows=(\d+) reused_live_rows=(\d+)$', frame, re.M)
    gdn_observed = re.findall(r'^strata prefill GDN decay: layer=(\d+) chunks=(\d+) '
                              r'reference_rows=(\d+) stored_rows=(\d+)$', frame, re.M)
    require([tuple(map(int, v)) for v in score_observed] ==
            [(layer, slices, full_rows - reused_rows, reused_rows) for layer in range(3, 48, 4)],
            'score slices, actual packing or retained rows differ from request positions')
    require([tuple(map(int, v)) for v in gdn_observed] ==
            [(layer, chunks, 0 if store_decay else rows, rows if store_decay else 0)
             for layer in range(48) if layer % 4 != 3],
            'missing GDN recurrence or unexpected decay producer/consumer')
    return dict(score_layers=12, gdn_layers=36, score_slices_per_layer=slices,
                packed_live_rows_per_layer=full_rows-reused_rows, reused_live_rows_per_layer=reused_rows,
                gdn_rows_per_layer=rows, chunks_per_layer=chunks)


def inputs(expected=None):
    native, foundation, source, recipe, original = parent.build_inputs(expected)
    measured = bank.previous.historic(PARENT_BUILD, parent.KEY)
    require(measured['passed'] and measured['ownership_closed'] and measured['host_control']['passed']
            and measured['projection'] == original and measured['recipe'] == recipe,
            'wrong original same-model build')
    recipe, projection = project(source, recipe)
    projection['component_gates'] = component_gates(bank.signature(measured))
    return native, foundation, source, recipe, projection


def context(revision):
    # Reuse the full-build source, header dependency, AOT and linker check.
    # Changes to any class are compiled in every consuming translation unit.
    ctx = bank.basis.core.source.device.load_module('score_decay_model_build', parent.__file__)
    ctx.KEY, ctx.MODEL_ROOT, ctx.CHANGED = KEY, ROOT, CHANGED
    ctx.DISPATCH_COUNTS, ctx.EVIDENCE, ctx.build_inputs = revision, PARENT_BUILD, inputs
    return ctx


def prepare(pins, ctx):
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing generation before preparing another')
    native, foundation, source, recipe, proof = inputs()
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=pins['revision'],
               controller_sources=pins['source_sha256'], source_files=bank.stage.hashed(source),
               parent_evidence=PARENT_BUILD, projection=proof, recipe=recipe, foundation=foundation,
               driver=bank.builder.guards.driver_guard(), external_inputs=native['external_inputs'],
               retained_decode_reference=bank.builder.decode_reference(recover=True), commands=[],
               active=False, complete=False, passed=False, ownership_closed=True,
               plan=dict(unit='Independently select retained FP16 score keys and stored GDN decay. Keep the '
                              'KV/gather reference, attention arithmetic and decoder placement fixed across four arms.',
                         validation='Check same-build off against the recorded parent. Validate component dispatch, '
                                    'fresh and restored state, checked final slot and decode handoff before speed.'))
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, payload in source.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(payload)
    ctx.check_build(row, False)
    ctx.save_build(row)


def check_host(ctx):
    row = bank.previous.read(ROOT / 'record.json')
    ctx.check_build(row)
    require(row['stage'] == 'built' and row['passed'] and 'host_control' not in row,
            'use the newly built diagnostic host fixture')
    root = ROOT / 'host-control'
    root.mkdir(mode=0o700)
    owner = bank.builder.ownership.Owner(root / 'owned', lambda: bank.stage.write(root / 'ownership.json', owner.commands))
    binary = row['binaries']['full_context_diagnostic_host_test']
    check = dict(binary=binary, passed=False)
    try:
        entry, out, _ = owner.run('bounds', [binary['path']], row['recipe']['environment'], bank.P,
                                  wall=30, cpu=20, rss_cap=256 << 20, text_cap=64 << 10,
                                  file_cap=1 << 20, total_cap=4 << 20)
        require(bank.builder.ownership.closed(entry) and entry['exit_code'] == 0, 'host bounds fixture failed')
        check.update(bank.device.cpu_cases('full_context_diagnostic_host_test', out.read_text()), passed=True)
    finally:
        check['ownership_closed'] = owner.active is None and all(bank.builder.ownership.closed(e) for e in owner.commands)
        row['host_control'] = check
        if not check['passed'] or not check['ownership_closed']:
            row.update(stage='failed', passed=False, complete=False)
        ctx.save_build(row)
    require(check['ownership_closed'], 'host fixture owner still active')


def main(action):
    modules = (__file__, keys.__file__, decay.__file__, keys.device.__file__, parent.__file__,
               parent.kv.__file__, parent.fixture.__file__, keys.device.model.__file__)
    paths = [str(Path(name).resolve().relative_to(bank.P)) for name in modules]
    pins = bank.stage.source.pin(bank.P, [*paths, 'tools/sycl', *bank.basis.POLICIES])
    require(set(paths) <= pins['source_sha256'].keys(), 'commit the controller and its inputs first')
    if action == 'recipe':
        source, recipe, _ = parent.project()
        recipe, proof = project(source, recipe)
        print(f"{len(source)} source files, {len(proof['changed'])} changed, {len(recipe['targets'])} targets; "
              f"image {proof['source_sha256']}; recipe {proof['recipe_sha256']}")
        return
    revision = pins['revision'] if action == 'prepare' else bank.previous.read(ROOT / 'record.json')['source_commit']
    ctx = context(revision)
    bank.builder.ROOT, bank.builder.KEY = ROOT, KEY
    bank.builder.TARGETS = [*bank.TARGETS, parent.fixture.TARGET]
    bank.builder.check, bank.builder.save = ctx.check_build, ctx.save_build
    if action == 'prepare':
        prepare(pins, ctx)
    elif action == 'host':
        check_host(ctx)
    else:
        bank.builder.execute(action)


if __name__ == '__main__':
    if sys.argv[1:2] == ['dispatch']:
        _, _, name, start, rows, keys_arm, decay_arm = sys.argv
        path = Path(name)
        require(keys_arm in ('0','1') and decay_arm in ('0','1') and path.stat().st_size <= 16 << 20,
                'use one bounded completed segment and explicit 0/1 arms')
        print(check_dispatch(path.read_text(), int(start), int(rows), retain_keys=keys_arm == '1',
                             store_decay=decay_arm == '1'))
        sys.exit(0)
    action, = sys.argv[1:]
    require(sys.executable == bank.basis.reference.model.PYTHON and action in ('recipe', 'prepare', 'configure', 'compile', 'host'),
            'use the pinned interpreter and a planned stage')
    if action == 'recipe':
        main(action)
    else:
        with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bank.builder.guards.no_model_owners([])
            main(action)
