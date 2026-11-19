"""Compose the large-job prefill gather with the measured resident-input engine."""

import fcntl
import json
import sys

sys.dont_write_bytecode = True
import build_prefill_mixed_gather as fixture

bank = fixture.bank
MODEL_ROOT = bank.B / 'prefill-large-gather-model-v1'
EVIDENCE = '2b4cbdf245221275bec51f70902793a6fe280a20'
CHANGED = (fixture.HEADER, fixture.KERNEL, fixture.FIXTURE, 'src/prefill/prefill.cpp',
           'src/prefill/moe_window.inc', 'src/prefill/layer_major.inc')


def project():
    # The parent ELF proves build continuity while present. Its complete source
    # image and recipe must remain reconstructible from Git after its consumers end.
    history = fixture.git_file(EVIDENCE, str(bank.RESULT.relative_to(bank.P)))
    records = json.loads(history)
    parent = records[bank.KEY]
    if bank.ROOT.exists():
        live = bank.previous.read(bank.ROOT / 'record.json')
        native = bank.check(live)
        bank.require(bank.builder.compact(live) == parent, 'live engine differs from measured parent')
    else:
        retained = bank.basis.validation.committed(bank.RESULT, (bank.KEY,))
        retired = retained.get('artifact_retirement', {})
        bank.require({k: v for k, v in retained.items() if k != 'artifact_retirement'} == parent
                     and retired.get('complete') and retired.get('all_actual_inputs_checked')
                     and retired.get('checked_ELF') == parent['binaries']['strata']['sha256']
                     and retired.get('source_sha256') == parent['source_files']['sha256'],
                     'missing parent requires its verified original-source and ELF retirement')
        native, _, _ = bank.basis.basis()
    gate = records['prefill_mixed_gather_dispatch_v1']
    built = records['prefill_mixed_gather_dispatch_build_v1']
    current = bank.basis.validation.committed(bank.RESULT, ('prefill_mixed_gather_dispatch_v1',))
    bank.require({k: v for k, v in current.items() if k != 'retirement'} == gate
                 and bank.basis.validation.committed(bank.RESULT, ('prefill_mixed_gather_dispatch_build_v1',)) == built,
                 'component evidence was superseded')
    bank.require(built['source_commit'] == fixture.CANDIDATE and built['build_sha256'] == gate['build_sha256']
                 and built['parent_sha256'] == bank.signature(parent)
                 and gate['faster_each_process'] == [True] * 4
                 and [s['stage'] for s in gate['stages']] == ['contiguous', 'ragged', '0', '1', '2']
                 and all(s['passed'] and s['ownership_closed'] and s['fresh_faults'] == []
                         and s['native_exit_code'] == 0 for s in gate['stages']),
                 'all shapes must pass the production-dispatch comparison')
    for stage in gate['stages'][2:]:
        bank.require([(s['rows'], s['cluster']) for s in stage['samples']] ==
                     [(2334, 1), (2334, 16), (8192, 1), (8192, 16)]
                     and all(s['median_time_ratio'] < 1 for s in stage['samples']),
                     'ineligible production shape')
    source, recipe = bank.image(native)
    original_hashes = bank.stage.hashed(source)
    bank.require(dict(files=len(original_hashes), sha256=bank.signature(original_hashes)) == parent['source_files']
                 and recipe == parent['recipe'] and parent['external_inputs'] == native['external_inputs'],
                 'recover the whole measured engine before applying the gather')
    for path in CHANGED:
        bank.require(source[path] == fixture.git_file(fixture.BASE, path),
                     'gather would replace unrelated compiled changes in ' + path)
        source[path] = fixture.git_file(fixture.CANDIDATE, path)
    hashes = bank.stage.hashed(source)
    bank.require(set(hashes) == set(original_hashes)
                 and {p for p, digest in hashes.items() if digest != original_hashes[p]} == set(CHANGED),
                 'unexpected engine source change')
    before, after = str(bank.ROOT), str(MODEL_ROOT)
    bank.require(recipe['targets'] == bank.TARGETS and fixture.TARGET not in recipe['targets']
                 and recipe['configure'][2:5] == [before + '/source', '-B', before + '/build'],
                 'unexpected parent targets or build location')
    child_recipe = dict(configure=[s.replace(before, after) for s in recipe['configure']],
                        environment={k: v.replace(before, after) for k, v in recipe['environment'].items()},
                        targets=[*recipe['targets'], fixture.TARGET])
    return source, child_recipe, dict(native_source=fixture.CANDIDATE, parent_binary=parent['binaries']['strata'],
        parent_source_sha256=parent['source_files']['sha256'], source_sha256=bank.signature(hashes),
        files=len(hashes), changed={p: hashes[p] for p in CHANGED}, recipe_sha256=bank.signature(child_recipe),
        component_sha256=bank.signature(gate))


if __name__ == '__main__':
    bank.require(sys.argv[1:] == ['check'], 'use check to verify the composed source and recipe')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        _, recipe, proof = project()
        print(f"{proof['files']} source files, {len(proof['changed'])} gather inputs, "
              f"{len(recipe['targets'])} targets, image {proof['source_sha256']}, recipe {proof['recipe_sha256']}")
