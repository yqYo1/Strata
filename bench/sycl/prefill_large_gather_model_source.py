"""Compose the large-job prefill gather with the measured resident-input engine."""

import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_prefill_mixed_gather as fixture

bank = fixture.bank
MODEL_ROOT = bank.B / 'prefill-large-gather-model-v1'
KEY = 'prefill_large_gather_model_v1'
EVIDENCE = '2b4cbdf245221275bec51f70902793a6fe280a20'
CHANGED = (fixture.HEADER, fixture.KERNEL, fixture.FIXTURE, 'src/prefill/prefill.cpp',
           'src/prefill/moe_window.inc', 'src/prefill/layer_major.inc')


def project(foundation_native=None):
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
        native = foundation_native
        if native is None:
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


def build_inputs(expected=None):
    bank.require(not bank.ROOT.exists(), 'finish the resident-input consumers and verify its retirement first')
    native, foundation, _ = bank.basis.basis(expected=expected)
    source, recipe, projection = project(native)
    bank.require(projection['source_sha256'] == 'b5d2c3bc755bc6ae53062fd74679a1c0eaa7b19ec24496352206d6165fb0e599'
                 and projection['recipe_sha256'] == '4eeb51b69215bc2d2e0bea04e1676e7007e69fb119367d8532b7717714a817dc',
                 'the selected component and original model build recipe are required')
    return native, foundation, source, recipe, projection


def check_build(row, committed=True):
    if committed:
        bank.require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == bank.builder.compact(row),
                     'commit the exact preceding build stage')
    native, foundation, source, recipe, projection = build_inputs(row['foundation'])
    pins = bank.stage.source.pin(bank.P, list(row['controller_sources']))
    bank.require(row['controller_sources'] == pins['source_sha256'] and row['projection'] == projection
                 and row['native_source_commit'] == fixture.CANDIDATE and row['parent_evidence'] == EVIDENCE
                 and row['foundation'] == foundation and row['external_inputs'] == native['external_inputs']
                 and row['driver'] == bank.builder.guards.driver_guard() and row['recipe'] == recipe
                 and row['retained_decode_reference'] == bank.builder.decode_reference()
                 and row['source_files'] == bank.stage.hashed(source), 'changed coherent gather build inputs')
    materialized = MODEL_ROOT / 'source'
    bank.require({p.relative_to(materialized).as_posix() for p in materialized.rglob('*') if p.is_file()}
                 == row['source_files'].keys() and all(bank.sha(materialized / p) == h for p, h in row['source_files'].items())
                 and all(bank.sha(Path(dep['root']) / p) == h for dep in native['dependencies']
                         for p, h in dep['source_sha256'].items())
                 and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
                 'actual source tree, dependency or configured input changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        bank.require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                     and all(str(materialized / p) in actual['compiled'] for p in CHANGED)
                     and set(row['binaries']) == set(recipe['targets'])
                     and all(bank.sha(v['path']) == v['sha256'] for v in row['binaries'].values())
                     and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items()),
                     'missing production gather, caller, fixture or final executable evidence')
    return native


def prepare():
    bank.require(not MODEL_ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
                 'fresh coherent candidate required')
    native, foundation, source, recipe, projection = build_inputs()
    controllers = [str(Path(p).resolve().relative_to(bank.P)) for p in (__file__, fixture.__file__)]
    pins = bank.stage.source.pin(bank.P, [*controllers, 'tools/sycl', *bank.basis.POLICIES])
    bank.require(all(p in pins['source_sha256'] for p in controllers), 'commit both gather build controllers')
    row = dict(stage='prepared', source_commit=pins['revision'], controller_sources=pins['source_sha256'],
               native_source_commit=fixture.CANDIDATE, parent_evidence=EVIDENCE, projection=projection,
               source_files=bank.stage.hashed(source), recipe=recipe, foundation=foundation,
               external_inputs=native['external_inputs'], driver=bank.builder.guards.driver_guard(),
               retained_decode_reference=bank.builder.decode_reference(recover=True),
               active=False, passed=False, complete=False, ownership_closed=True, commands=[],
               plan=dict(unit='Only the prefill gather selector changes. Its scalar and vec8 arms share this ELF.',
                         validation='Same-build host and traced gather fixtures before model state, RESTORE, '
                                    'actual large-job dispatch counts and repeated clean 64K PP/TTFT. '
                                    'Independent sustained decode and physical capacity remain open.',
                         retention='Keep the original prefill and decode references. Retire this comparison '
                                   'generation after verifying its source and closing its model consumers.'))
    MODEL_ROOT.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (MODEL_ROOT / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = MODEL_ROOT / 'source' / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(content)
    check_build(row, False)
    bank.builder.save(row)


def save_build(row):
    if row['stage'] == 'configured':
        build = MODEL_ROOT / 'build'
        files = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend((build / 'CMakeFiles').glob('*.ninja'))
        row['configured_files'] = {str(path): bank.sha(path) for path in files}
    bank.stage.save_build(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    bank.require(action in ('check', 'prepare', 'configure', 'compile', 'host'), 'select a coherent build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'check':
            _, recipe, proof = project()
            print(f"{proof['files']} source files, {len(proof['changed'])} gather inputs, "
                  f"{len(recipe['targets'])} targets, image {proof['source_sha256']}, recipe {proof['recipe_sha256']}")
        else:
            bank.builder.ROOT, bank.builder.KEY = MODEL_ROOT, KEY
            bank.builder.TARGETS = [*bank.TARGETS, fixture.TARGET]
            bank.builder.check, bank.builder.save = check_build, save_build
            if action == 'prepare':
                prepare()
            elif action == 'host':
                bank.stage.host(root_path=MODEL_ROOT, check_build=check_build, save_record=save_build)
            else:
                bank.builder.execute(action)
