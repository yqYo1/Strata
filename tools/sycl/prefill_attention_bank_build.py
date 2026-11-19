#!/usr/bin/env python3
"""Build a coherent stage-all image with one attention-store change."""

import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_prefix_expert_build as stage

P, B, RESULT, builder = stage.P, stage.B, stage.RESULT, stage.builder
basis, require, sha, signature = stage.basis, stage.require, stage.sha, stage.signature
ROOT = B / 'prefill-attention-bank-native-v1'
KEY = 'prefill_attention_bank_native_v1'
COMPONENT = 'prefill_attention_bank_component_v1'
COMPONENT_EVIDENCE = 'fe515ce995169764eb19316abf987b4426ffa9b4'
STAGE_EVIDENCE = 'fe6d810020758c2759891c1ac75c2df12ed2dca2'
OVERLAYS = ('include/strata/prefill/native16.hpp', 'src/prefill/sycl/native16.cpp',
            'src/prefill/prefill.cpp', 'src/prefill/layer_major.inc')
ACTUAL = ('compiled', 'generated', 'links', 'objects', 'produced')


def read(path):
    return json.loads(path.read_bytes())


def historic(evidence, key):
    return json.loads(stage.source.git(P, 'show', evidence + ':' + str(RESULT.relative_to(P))))[key]


def stage_image(native, component):
    stage.ROOT, stage.KEY = stage.STAGE_ROOT, stage.STAGE_BUILD
    original = historic(STAGE_EVIDENCE, stage.STAGE_BUILD)
    prior = stage.archive()
    require(original['archive'] == {key: signature(value) for key, value in prior.items()}
            and original['recipe'] == stage.recipe(prior),
            'exact old stage-all history and recipe required')
    payload = stage.recover(native, prior, original['source_commit'], verify_live=False)
    hashes = stage.hashed(payload)
    require(original['source_files'] == dict(files=len(hashes), sha256=signature(hashes)),
            'exact old stage-all source composition required')
    revision = component['source_commit']
    parent = dict(stage.source.blobs(P, revision + '^', OVERLAYS))
    require(all(parent[path] == payload[path] for path in OVERLAYS),
            'attention store patch must start at the exact old stage-all bytes')
    return original, payload


def inputs(native, *, retired=True):
    component = historic(COMPONENT_EVIDENCE, COMPONENT)
    require(component['stage'] == 'closed' and component['ownership_closed'] and not component['active']
            and component['processes'][0]['cases'] == 4 and component['decision']['model_eligible']
            and len(component['processes']) == 7
            and all(p['passed'] and p['ownership_closed'] and not p['fresh_faults']
                    for p in component['processes']), 'exact successful device component required')
    original, payload = stage_image(native, component)
    if retired:
        final = basis.validation.committed(RESULT, (stage.STAGE_BUILD,))
        proof = final['artifact_retirement']
        require({k: v for k, v in final.items() if k != 'artifact_retirement'} == original
                and proof['complete'] and proof['all_actual_inputs_checked']
                and proof['checked_ELF'] == original['binaries']['strata']['sha256']
                and proof['source_sha256'] == original['source_files']['sha256']
                and not stage.STAGE_ROOT.exists(), 'finish the stage-all build handoff before a new generation')
    overlay = dict(stage.source.blobs(P, component['source_commit'], OVERLAYS))
    require(all((P / p).read_bytes() == raw and payload[p] != raw for p, raw in overlay.items()),
            'exact four tested attention-store files required')
    payload.update(overlay)
    origin, target = str(stage.STAGE_ROOT), str(ROOT)
    recipe = dict(original['recipe'],
                  configure=[v.replace(origin, target) for v in original['recipe']['configure']],
                  environment={k: v.replace(origin, target)
                               for k, v in original['recipe']['environment'].items()})
    return stage.hashed(payload), payload, recipe, {
        'stage_all_evidence': STAGE_EVIDENCE, 'stage_all_source': original['source_commit'],
        'stage_all_ELF': original['binaries']['strata']['sha256'],
        'component_evidence': COMPONENT_EVIDENCE, 'component_ELF': component['binary']['sha256'],
        'attention_store_source': component['source_commit'], 'overlays': stage.hashed(overlay),
    }


def check(row, committed=True):
    if committed:
        require(basis.validation.committed(RESULT, (KEY,)) == builder.compact(row),
                'commit the exact completed build stage first')
    native, foundation, _ = basis.basis(expected=row['foundation'])
    hashes, _, recipe, history = inputs(native)
    require(row['source_files'] == hashes and row['recipe'] == recipe and row['history'] == history
            and source_pins() == row['controller_sources'] and row['foundation'] == foundation
            and row['driver'] == builder.guards.driver_guard()
            and row['retained_decode_reference'] == builder.decode_reference(),
            'committed source, recipe, driver or phase reference changed')
    require(all(sha(ROOT / 'source' / p) == h for p, h in hashes.items())
            and all(sha(Path(dep['root']) / p) == h for dep in native['dependencies']
                    for p, h in dep['source_sha256'].items())
            and all(sha(p) == h for p, h in row.get('configured_files', {}).items()),
            'materialized native inputs changed')
    if row['stage'] == 'built':
        require(row['actual_inputs']['uncovered'] == 0
                and all(sha(p) == h for field in ACTUAL for p, h in row['actual_inputs'][field].items())
                and all(sha(v['path']) == v['sha256'] for v in row['binaries'].values()),
                'actual compiled inputs or executables changed')
    return native


def source_pins():
    return stage.source.pin(P, [*builder.PREFIXES, *OVERLAYS])['source_sha256']


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        paths = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        paths.extend(sorted((build / 'CMakeFiles').glob('*.ninja')))
        row['configured_files'] = {str(p): sha(p) for p in paths}
    stage.save_build(row)


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(RESULT), 'fresh build root only')
    native, foundation, _ = basis.basis()
    hashes, payload, recipe, history = inputs(native)
    controls, decode = source_pins(), builder.decode_reference(recover=True)
    revision = stage.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache', 'owned'):
        (ROOT / name).mkdir(mode=0o700)
    for path, raw in payload.items():
        target = ROOT / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    row = dict(stage='prepared', source_commit=revision, native_base_commit=basis.original.SOURCE,
               foundation=foundation, source_files=hashes, controller_sources=controls, history=history,
               recipe=recipe, external_inputs=native['external_inputs'], driver=builder.guards.driver_guard(),
               retained_decode_reference=decode, commands=[], passed=False, complete=False,
               active=False, ownership_closed=True,
               plan=dict(unit='Exact stage-all composition plus the independently timed attention-store files.',
                         build='Rebuild all callers/fixtures and native device objects. No object reuse.',
                         qualification='Use the resulting same ELF for copy/direct. No inherited model gate.',
                         retention='Third build only during this comparison. Review at closure or first failure.'))
    check(row, False)
    save(row)


def native():
    row = read(ROOT / 'record.json')
    original = check(row)
    require(row['stage'] == 'built' and row['passed'] and row['complete'] and not row['active']
            and row['ownership_closed'] and row['host_control']['passed'],
            'same-build CPU fixture and closed native build required')
    return row, original


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'configure', 'compile', 'host'), 'prepare|configure|compile|host')
    builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, stage.TARGETS
    builder.check, builder.save = check, save
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        elif action == 'host':
            stage.host(root_path=ROOT, check_build=check, save_record=save)
        else:
            builder.execute(action)
