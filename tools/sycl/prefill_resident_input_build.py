"""Coherently rebuild the verified attention-store image with the input-borrow candidate."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prefill_attention_bank_build as previous

basis, stage, builder = previous.basis, previous.stage, previous.builder
require, sha, signature = previous.require, previous.sha, previous.signature
P, B, RESULT = previous.P, previous.B, previous.RESULT
ROOT = B / 'prefill-resident-input-native-v1'
KEY = 'prefill_resident_input_native_v1'
BASE_EVIDENCE = 'a3ad7713654cf9dbc61a786d02912d7b7ee56e57'
PATCH_BASE = '3d263da77f548c5252cd2864fd3e3588519bab1c'
INPUT_SOURCE = '45f2ba9dda476c9d0966859d1424d83e88168085'
FIXTURE = 'tests/sycl/prefill_attention_bank.cpp'
CMAKE = 'cmake/sycl_vllm_xe2_native16.cmake'
TARGET = 'sycl_prefill_attention_bank_test'
TARGETS = [*stage.TARGETS, TARGET]
COMPARISON = 'prefill_ordinary_bank_comparison_v1'


def image(native):
    """Recover every source byte from Git, including the superseded store image."""
    old = previous.historic(BASE_EVIDENCE, previous.KEY)
    component = previous.historic(previous.COMPONENT_EVIDENCE, previous.COMPONENT)
    _, payload = previous.stage_image(native, component)
    outputs = dict(stage.source.blobs(P, component['source_commit'], previous.OVERLAYS))
    require(stage.hashed(outputs) == old['history']['overlays'], 'output-store source differs')
    payload.update(outputs)
    require(old['source_files'] == dict(files=len(payload), sha256=signature(stage.hashed(payload))),
            'recovered parent is not the measured coherent store image')
    paths = (*previous.OVERLAYS, CMAKE)
    base = dict(stage.source.blobs(P, PATCH_BASE, paths))
    require(all(base[k] == payload[k] for k in paths) and FIXTURE not in payload,
            'resident-input patch base differs from the original compiled inputs')
    changes = dict(stage.source.blobs(P, INPUT_SOURCE, (*paths, FIXTURE)))
    payload.update(changes)
    origin, target = str(previous.ROOT), str(ROOT)
    original = old['recipe']
    require(original['targets'] == stage.TARGETS and original['configure'][2] == origin + '/source'
            and original['configure'][4] == origin + '/build', 'parent configure paths or targets differ')
    recipe = dict(configure=[v.replace(origin, target) for v in original['configure']],
                  environment={k: v.replace(origin, target) for k, v in original['environment'].items()},
                  targets=TARGETS)
    return payload, recipe


def comparison_closed():
    value = basis.validation.committed(RESULT, (COMPARISON,))
    require(value['stage'] == 'closed' and value['passed'] and value['complete']
            and not value['active'] and value['ownership_closed'] and len(value['processes']) == 9
            and all(p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in value['processes']), 'complete the ordinary store comparison')
    return signature({name: value[name] for name in ('source_commit', 'binary', 'plan', 'processes', 'decision')})


def prepare():
    require(not ROOT.exists() and KEY not in basis.validation.committed(RESULT), 'fresh native build required')
    cohort = comparison_closed()
    old = previous.historic(BASE_EVIDENCE, previous.KEY)
    retained = basis.validation.committed(RESULT, (previous.KEY,))
    retirement = retained.get('artifact_retirement', {})
    require({k: v for k, v in retained.items() if k != 'artifact_retirement'} == old
            and retirement.get('complete') and retirement.get('all_actual_inputs_checked')
            and retirement.get('checked_ELF') == old['binaries']['strata']['sha256']
            and retirement.get('source_sha256') == old['source_files']['sha256']
            and not previous.ROOT.exists(), 'close the prior build consumers and retire that generation first')
    native, foundation, _ = basis.basis()
    payload, recipe = image(native)
    pins = stage.source.pin(P, ['tools/sycl', *basis.POLICIES])
    decode = builder.decode_reference(recover=True)
    row = dict(stage='prepared', source_commit=pins['revision'], native_source_commit=INPUT_SOURCE,
               native_base_commit=basis.original.SOURCE, parent_evidence=BASE_EVIDENCE,
               comparison_sha256=cohort, foundation=foundation, controller_sources=pins['source_sha256'],
               source_files=stage.hashed(payload), recipe=recipe, external_inputs=native['external_inputs'],
               driver=builder.guards.driver_guard(), retained_decode_reference=decode, commands=[],
               active=False, ownership_closed=True, passed=False, complete=False,
               plan=dict(unit='Only the four input-borrow engine overlays and the same-build test target.',
                         build='New directory, no object reuse, dependencies and all actual link inputs checked.',
                         validation='Host control, traced input-correctness device fixture, model numerical and '
                                    'route cases before timing the four independent input/output combinations.',
                         retention='Third generation alongside the independently retained prefill and decode '
                                   'references. Review at first failed gate or the completed four-arm decision.'))
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache', 'owned'):
        (ROOT / name).mkdir(mode=0o700)
    for name, content in payload.items():
        target = ROOT / 'source' / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(content)
    check(row, False)
    builder.save(row)


def check(row, committed=True):
    if committed:
        require(basis.validation.committed(RESULT, (KEY,)) == builder.compact(row), 'commit the exact build stage')
    native, foundation, _ = basis.basis(expected=row['foundation'])
    payload, recipe = image(native)
    require(row['native_source_commit'] == INPUT_SOURCE and row['parent_evidence'] == BASE_EVIDENCE
            and row['comparison_sha256'] == comparison_closed() and row['foundation'] == foundation
            and row['controller_sources'] == stage.source.pin(P, list(row['controller_sources']))['source_sha256']
            and row['source_files'] == stage.hashed(payload) and row['recipe'] == recipe
            and row['external_inputs'] == native['external_inputs'] and row['driver'] == builder.guards.driver_guard()
            and row['retained_decode_reference'] == builder.decode_reference(), 'pinned build inputs differ')
    require(all(sha(ROOT / 'source' / n) == h for n, h in row['source_files'].items())
            and all(sha(Path(dep['root']) / n) == h for dep in native['dependencies']
                    for n, h in dep['source_sha256'].items())
            and all(sha(p) == h for p, h in row.get('configured_files', {}).items()), 'source or dependency changed')
    if row['stage'] == 'built':
        actual = row['actual_inputs']
        require(actual['uncovered'] == 0 and not actual['partial_archive_reuse']
                and str(ROOT / 'source' / FIXTURE) in actual['compiled']
                and set(row['binaries']) == set(TARGETS)
                and all(sha(v['path']) == v['sha256'] for v in row['binaries'].values())
                and all(sha(p) == h for group in previous.ACTUAL for p, h in actual[group].items()),
                'fresh objects, fixture, link inputs or ELF differ')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        files = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        files.extend((build / 'CMakeFiles').glob('*.ninja'))
        row['configured_files'] = {str(path): sha(path) for path in files}
    stage.save_build(row)


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'configure', 'compile', 'host'), 'prepare|configure|compile|host')
    builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
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
