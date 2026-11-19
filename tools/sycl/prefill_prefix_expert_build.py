#!/usr/bin/env python3
"""Build the matched prefill copy scheduling controls; repository LICENSE."""

import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import coherent_model_basis as basis
import prefill_window_build as builder

P, B, RESULT = basis.P, basis.B, basis.RESULT
ROOT = B / 'prefill-prefix-expert-schedule-native-v1'
KEY = 'prefill_prefix_expert_schedule_native_v1'
HOST = 'prefill_fullfit_upload_plan_test'
TARGETS = [*builder.TARGETS, 'sycl_moe_upload_census_host_test', HOST]
OVERLAYS = (
    'cmake/sycl_vllm_xe2_native16.cmake',
    'include/strata/prefill/expert_bank_upload.hpp',
    'include/strata/prefill/fullfit_upload_plan.hpp', 'include/strata/sycl/native_copy_list.hpp',
    'src/prefill/prefill.cpp', 'src/prefill/layer_major.inc', 'src/prefill/moe_window.inc',
    'tests/sycl/expert_bank.cpp',
)
SOURCE_BUILD = 'prefill_prefix_expert_host_usm_native_v1'
SOURCE_RESULT = 'prefill_prefix_expert_host_usm_timing_v1'
CONFIGURE_FAILURE = 'prefill_prefix_expert_transport_native_v1'
BUILD_HISTORY = ('prefill_window_native_v2', 'prefill_first_demand_native_v1',
                 'prefill_prefix_router_native_v1', 'prefill_prefix_expert_native_v1', SOURCE_BUILD)
TRANSPORT_BUILD = 'prefill_prefix_expert_transport_native_v2'
ORDINAL_RESULT = 'prefill_prefix_expert_transport_ordinal_v1'
COMPONENT_OVERLAP = 'fullfit_attention_copy_ordinal_v1'
HISTORY = (*BUILD_HISTORY, SOURCE_RESULT, CONFIGURE_FAILURE, TRANSPORT_BUILD, ORDINAL_RESULT, COMPONENT_OVERLAP)
source, write = builder.source, builder.write
require, sha, signature = basis.require, basis.sha, basis.signature
save_build = builder.save


def archive():
    rows = {key: basis.validation.committed(RESULT, (key,)) for key in HISTORY}
    preceding, ordinal, component = (rows[key] for key in (TRANSPORT_BUILD, ORDINAL_RESULT, COMPONENT_OVERLAP))
    require(preceding['artifact_retirement']['complete']
            and preceding['artifact_retirement']['checked_ELF'] == preceding['binaries']['strata']['sha256']
            and not Path(preceding['artifact_retirement']['capture']['root']).exists()
            and ordinal['stage'] == 'closed' and ordinal['passed'] and ordinal['ownership_closed']
            and ordinal['decision']['matched_work'] and ordinal['binary'] == preceding['binaries']['strata']
            and component['stage'] == 'closed' and component['ownership_closed']
            and component['decision']['eligible_native_ordinal_overlap'],
            'closed model ordinal control and independent component overlap required')
    return rows


def hashed(payload):
    return {path: hashlib.sha256(raw).hexdigest() for path, raw in payload.items()}


def recover(native, prior, revision):
    payload = dict(source.blobs(P, basis.original.SOURCE, native['project']['source_sha256']))
    require(hashed(payload) == native['project']['source_sha256'], 'coherent base differs')
    for key in BUILD_HISTORY:
        build = prior[key]
        overlay = dict(source.blobs(P, build['source_commit'], build['overlays']))
        require(hashed(overlay) == build['overlays'], 'archived overlays differ: ' + key)
        payload.update(overlay)
        hashes = hashed(payload)
        require(dict(files=len(hashes), sha256=signature(hashes)) == build['source_files'],
                'recovered native composition differs: ' + key)
    image, timing = prior[SOURCE_BUILD], prior[SOURCE_RESULT]
    require(timing['stage'] == 'closed' and timing['ownership_closed'] and not timing['active']
            and timing['native_source_commit'] == image['source_commit']
            and timing['binary'] == image['binaries']['strata']
            and timing['artifact_retirement']['complete']
            and timing['artifact_retirement']['build']['source_sha256'] == signature(hashed(payload))
            and timing['disposition']['selected_source'] == 'arena',
            'exact closed source comparison and recovered retired build required')
    failure = prior[CONFIGURE_FAILURE]
    require(failure['stage'] == 'failed' and failure['ownership_closed'] and not failure['active']
            and failure['failure']['stage'] == 'cmake' and not failure['failure']['gpu_executed'],
            'preserve the closed original configure failure')
    preceding = prior[TRANSPORT_BUILD]
    previous_overlay = dict(source.blobs(P, preceding['source_commit'], preceding['overlays']))
    previous_files = {**payload, **previous_overlay}
    require(hashed(previous_overlay) == preceding['overlays']
            and dict(files=len(previous_files), sha256=signature(hashed(previous_files))) == preceding['source_files'],
            'recover the original transport image before applying scheduling overlays')
    changes = dict(source.blobs(P, revision, OVERLAYS))
    require(set(changes) == set(OVERLAYS)
            and all((P / path).read_bytes() == raw for path, raw in changes.items()),
            'commit exact transport and test overlays')
    payload.update(changes)
    return payload


def recipe(prior):
    image = prior[SOURCE_BUILD]
    root = str(Path(image['binaries']['strata']['path']).parent.parent)
    original = image['recipe']
    configure = [v.replace(root, str(ROOT)) for v in original['configure']]
    return dict(configure=[*configure, '-DSTRATA_SYCL_PREFILL_NATIVE_COPY=ON'],
                environment={k: v.replace(root, str(ROOT)) for k, v in original['environment'].items()},
                targets=TARGETS)


def check(row, committed=True):
    if committed:
        require(basis.validation.committed(RESULT, (KEY,)) == builder.compact(row),
                'commit the exact completed build stage first')
    native, proof, _ = basis.basis(expected=row['foundation'])
    prior = archive()
    require(row['archive'] == {key: signature(value) for key, value in prior.items()}
            and row['foundation'] == proof and row['recipe'] == recipe(prior)
            and row['driver'] == builder.guards.driver_guard()
            and row['retained_decode_reference'] == builder.decode_reference()
            and source.pin(P, list(row['controller_sources']))['source_sha256'] == row['controller_sources'],
            'pinned phase references, platform or controller changed')
    require(all(sha(ROOT / 'source' / path) == digest for path, digest in row['source_files'].items())
            and all(sha(Path(dep['root']) / path) == digest for dep in native['dependencies']
                    for path, digest in dep['source_sha256'].items())
            and all(sha(path) == digest for path, digest in row.get('configured_files', {}).items()),
            'actual build inputs changed')
    if row['stage'] == 'built':
        require(all(sha(path) == digest for field in ('compiled', 'generated', 'links', 'objects', 'produced')
                    for path, digest in row['actual_inputs'][field].items())
                and all(sha(binary['path']) == binary['sha256'] for binary in row['binaries'].values()),
                'actual compiled inputs or executable changed')
    return native


def save(row):
    if row['stage'] == 'configured':
        build = ROOT / 'build'
        paths = [build / name for name in ('CMakeCache.txt', 'build.ninja', 'compile_commands.json')]
        paths += sorted((build / 'CMakeFiles').glob('*.ninja'))
        row['configured_files'] = {str(path): sha(path) for path in paths}
    save_build(row)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh build root only')
    native, proof, _ = basis.basis()
    prior = archive()
    pins = source.pin(P, [*builder.PREFIXES, *OVERLAYS, 'docs/SYCL_NATIVE16_PREFILL.md'])
    revision = pins['revision']
    payload = recover(native, prior, revision)
    decode = builder.decode_reference(recover=True)
    ROOT.mkdir(mode=0o700)
    for directory in ('source', 'tmp', 'cache', 'owned'):
        (ROOT / directory).mkdir(mode=0o700)
    for path, raw in payload.items():
        target = ROOT / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    hashes = hashed(payload)
    row = dict(stage='prepared', source_commit=revision, native_base_commit=basis.original.SOURCE,
               archive={key: signature(value) for key, value in prior.items()}, foundation=proof,
               overlays={path: hashes[path] for path in OVERLAYS}, source_files=hashes,
               controller_sources=pins['source_sha256'], recipe=recipe(prior),
               external_inputs=native['external_inputs'], driver=builder.guards.driver_guard(),
               retained_decode_reference=decode, commands=[], active=False, ownership_closed=True,
               passed=False, complete=False,
               plan=dict(unit='Fixed copy-only list, private bank and request-owned host USM. Isolate '
                         'earlier issue, first-reader completion and their interaction.',
                         host=[HOST], retention='Third build only for this comparison after preserving '
                         'both independent phase references. Review at closure or first failed gate.'))
    check(row, False)
    save(row)


def host():
    row = json.loads((ROOT / 'record.json').read_text())
    check(row)
    require(row['stage'] == 'built' and row['passed'] and row['ownership_closed']
            and not row['active'] and 'host_control' not in row, 'one same-build host control')
    root = ROOT / 'host-control'
    root.mkdir(mode=0o700)
    owner = builder.ownership.Owner(root / 'owned', lambda: write(root / 'ownership.json', owner.commands))
    binary = row['binaries'][HOST]
    result, stdout, _ = owner.run('plan', [binary['path']], row['recipe']['environment'], P,
        wall=30, cpu=20, rss_cap=256 << 20, text_cap=64 << 10, file_cap=1 << 20, total_cap=4 << 20)
    closed = builder.ownership.closed(result)
    expected = 'prefill prefix expert IDs: sparse_and_repeated=checked refusals=7 checked_copy=pass'
    passed = closed and result['exit_code'] == 0 and expected in stdout.read_text().splitlines()
    row['host_control'] = dict(passed=passed, binary=binary, ownership_closed=closed,
                               exit_code=result['exit_code'], error=result.get('error'))
    if not passed:
        row.update(stage='failed', passed=False, complete=False)
    save(row)
    require(passed, 'prefix expert host check failed')


if __name__ == '__main__':
    require(sys.argv[1:] in [['prepare'], ['configure'], ['compile'], ['host']],
            'expected prepare, configure, compile or host')
    builder.ROOT, builder.KEY, builder.TARGETS = ROOT, KEY, TARGETS
    builder.check, builder.save = check, save
    action = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        elif action == 'host':
            host()
        else:
            builder.execute(action)
