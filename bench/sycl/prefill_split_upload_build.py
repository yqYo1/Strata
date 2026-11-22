"""Build the split-prefill upload experiment from the qualified boundary source."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import decode_kv_fast_prefill_build as boundary

bank, builder, require = boundary.bank, boundary.builder, boundary.require
ROOT = bank.B / 'prefill-split-upload-native-v1'
KEY = 'prefill_split_upload_native_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PARENT_EVIDENCE = '04b27731118e10a68ab6954c88218b2cde5e6306'
FIXTURE_EVIDENCE = 'b1392e966e1ae576fe71ad8ff47767ee48a9427e'
UPLOAD = 'b17cced7887a2a989482e18f2281357b220f7402'
HELPER = '99cc428a60e0e629a4c4ec58d48b204d75e82d1e'
UNIT = 'src/prefill/prefill.cpp'
WINDOW = 'src/prefill/moe_window.inc'
LAYER = 'src/prefill/layer_major.inc'
HEADER = 'include/strata/prefill/expert_bank_upload.hpp'
FIXTURE = 'tests/sycl/expert_bank.cpp'
VERIFIED = {
    UNIT: 'b9d49202dbdd4f1d0135af0faf4c6038d12a4f1d2f58c79181bfdac54cc6ed9b',
    WINDOW: '3d1df9055f289de60563b52162ce8af7bb167d8aa4442d0f82d2b1397ba7db5a',
    LAYER: '69ebdcfac4e799dddb27cd208dcde998b3d76fa8e14cac66ae57851625701277',
    HEADER: '44238ca87239cfbbf47308c1275693747c97509a04a86fd1aa02aa489b3f5a46',
}


def project(expected=None):
    native, foundation, source, recipe, parent_projection = boundary.project(expected, boundary_only=True)
    parent = bank.previous.historic(PARENT_EVIDENCE, boundary.BOUNDARY_KEY)
    safety = bank.previous.historic(PARENT_EVIDENCE, boundary.BOUNDARY_CAPACITY_KEY)
    hashes = bank.stage.hashed(source)
    require(parent['recipe'] == recipe and parent['projection'] == parent_projection
            and parent['source_files'] == dict(files=len(hashes), sha256=bank.signature(hashes))
            and parent['foundation_sha256'] == bank.signature(foundation)
            and parent['external_inputs'] == native['external_inputs']
            and parent['native_source_commit'] == safety['native_source_commit']
            and parent['stage'] == 'built' and parent['passed'] and parent['complete'] and parent['ownership_closed']
            and safety['stage'] == 'closed' and safety['passed'] and safety['complete'] and safety['ownership_closed']
            and safety['build_sha256'] == bank.signature(parent)
            and safety['outcomes']['full']['final_slot_consumed'],
            'recover the exact full-context-qualified boundary composition')
    source.update(boundary.source_patch(source, UPLOAD, (UNIT, WINDOW, LAYER), context=1))
    source.update(boundary.source_patch(source, HELPER, (WINDOW, HEADER, FIXTURE), context=1))
    overlays = bank.stage.hashed({p: source[p] for p in (*VERIFIED, FIXTURE)})
    require(all(overlays[p] == h for p, h in VERIFIED.items())
            and {p for p, h in bank.stage.hashed(source).items() if h != hashes.get(p)} == overlays.keys(),
            'the verified upload and device fixture must be the only native change')
    old, new = str(boundary.BOUNDARY_ROOT), str(ROOT)
    recipe = dict(configure=[v.replace(old, new) for v in recipe['configure']],
                  environment={k: v.replace(old, new) for k, v in recipe['environment'].items()},
                  targets=recipe['targets'])
    component_receipt = boundary.stock.fixture.git_file(FIXTURE_EVIDENCE, 'docs/SYCL_NATIVE16_PREFILL.md')
    require(b'Both `99cc428a6` fixture processes closed normally' in component_receipt
            and b'dd69762b225b612ba88ae1cf3542b0e9e0dcefcb6ab4f4b9db1c165c270b1326' in component_receipt,
            'recover the original checked device fixture receipt')
    proof = dict(parent_evidence=PARENT_EVIDENCE, parent_source=parent['native_source_commit'],
                 parent_binary=parent['binaries']['strata'], parent_capacity_sha256=bank.signature(safety),
                 component_evidence=FIXTURE_EVIDENCE, overlay_commits=[UPLOAD, HELPER],
                 overlays=overlays, object_count=parent_projection['object_count'],
                 object_set_sha256=parent_projection['object_set_sha256'])
    return native, foundation, source, recipe, proof


def check(row, committed=True):
    if committed:
        require(bank.basis.validation.committed(bank.RESULT, (KEY,)) == builder.compact(row),
                'commit the preceding upload build stage')
    controls = row['controller_sources']
    if row['stage'] == 'built':
        checked = bank.stage.hashed(dict(bank.stage.source.blobs(bank.P, row['source_commit'], controls)))
    else:
        checked = bank.stage.source.pin(bank.P, list(controls))['source_sha256']
        require(row['retained_decode_reference'] == builder.decode_reference(),
                'the independent decode reference changed')
    native, foundation, source, recipe, proof = project(row['foundation'])
    require(checked == controls and row['source_commit'] == row['native_source_commit']
            and row['source_files'] == bank.stage.hashed(source) and row['recipe'] == recipe
            and row['projection'] == proof and row['foundation'] == foundation
            and row['external_inputs'] == native['external_inputs']
            and row['driver'] == builder.guards.driver_guard()
            and all(bank.sha(p) == h for p, h in row['build_tools'].items()),
            'frozen upload build sources, recipe or runtime changed')
    tree = ROOT / 'source'
    require({p.relative_to(tree).as_posix() for p in tree.rglob('*') if p.is_file()} == row['source_files'].keys()
            and all(bank.sha(tree / p) == h for p, h in row['source_files'].items())
            and all(bank.sha(Path(d['root']) / p) == h
                    for d in native['dependencies'] for p, h in d['source_sha256'].items())
            and all(bank.sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'materialized source, dependency or generated input changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        objects = sorted(Path(p).relative_to(ROOT / 'build').as_posix() for p in actual['objects'])
        require(not actual['partial_archive_reuse'] and actual['uncovered'] == 0
                and actual['fresh_objects'] == len(objects) == proof['object_count']
                and bank.signature(objects) == proof['object_set_sha256']
                and all(actual['compiled'].get(str(tree / p)) == h for p, h in proof['overlays'].items())
                and all(bank.sha(p) == h for group in bank.previous.ACTUAL for p, h in actual[group].items())
                and set(row['binaries']) == set(recipe['targets'])
                and all(bank.sha(b['path']) == b['sha256'] for b in row['binaries'].values()),
                'the new engine and actual expert-bank fixture must be coherently rebuilt')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / n for n in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): bank.sha(p) for p in files}
    boundary.save_build(row)


def prepare():
    require(not ROOT.exists() and not boundary.BOUNDARY_ROOT.exists()
            and KEY not in bank.basis.validation.committed(bank.RESULT),
            'close and verify retirement of the boundary comparison build first')
    timing = bank.basis.validation.committed(bank.RESULT, ('decode_kv_boundary_preservation_117489_v1',))
    require(timing['stage'] == 'closed' and timing['complete'] and timing['ownership_closed']
            and not timing['active'] and len(timing['processes']) == 6
            and all(p['passed'] and p['ownership_closed'] and not p['fresh_faults'] for p in timing['processes']),
            'require the complete independent boundary measurements')
    native, foundation, source, recipe, proof = project()
    controls = [THIS, boundary.THIS, *(str(Path(m.__file__).resolve().relative_to(bank.P))
                 for m in (boundary.stock, boundary.stock.fixture, boundary.stock.kv))]
    pins = bank.stage.source.pin(bank.P, [*controls, 'tools/sycl', *bank.basis.POLICIES])
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=pins['revision'],
               controller_sources=pins['source_sha256'], source_files=bank.stage.hashed(source),
               foundation=foundation, projection=proof, recipe=recipe, external_inputs=native['external_inputs'],
               build_tools={p: bank.sha(p) for p in ('/usr/bin/cmake', '/usr/bin/ninja')},
               driver=builder.guards.driver_guard(), retained_decode_reference=builder.decode_reference(recover=True),
               active=False, passed=False, complete=False, ownership_closed=True, commands=[],
               scope='Demand, batched and coalesced on the same split-prefill engine and scalar decode.',
               retention='Preserve the independent phase references. Review after the upload comparisons or first failed gate.')
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, content in source.items():
        path = ROOT / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    check(row, committed=False)
    save(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('projection', 'verify-parent', 'prepare', 'configure', 'compile', 'host'),
            'select one build stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        builder.ROOT, builder.KEY, builder.check, builder.save = ROOT, KEY, check, save
        if action == 'projection':
            _, _, source, _, proof = project()
            print({'source_files': len(source), 'projection': proof})
        elif action == 'verify-parent':
            parent, checked = boundary.qualified_boundary()
            _, _, _, _, proof = project(parent['foundation'])
            require(proof['parent_binary'] == checked['binary'], 'the retained boundary engine differs')
            print({'checked_parent': checked, 'source_files': len(parent['source_files']),
                   'coherent_objects': parent['actual_inputs']['fresh_objects']})
        elif action == 'prepare':
            prepare()
        elif action == 'host':
            bank.stage.host(root_path=ROOT, check_build=check, save_record=save)
        else:
            row = bank.previous.read(ROOT / 'record.json')
            builder.TARGETS = row['recipe']['targets']
            builder.execute(action)
