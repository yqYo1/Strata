#!/usr/bin/env python3
"""Main review decision after actual source, build and dependency inspection.
Repository LICENSE. Host-only receipt production; does not execute a native test.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = A / 'own-metric-query-tiny-v4'
PACKAGE = '1816ff99b174030b6586283fde10f28f49afb3eca008a72de7a5b3e118167cff'


def pin(p):
    p = Path(p)
    with p.open('rb') as stream:
        h = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=h)


def write(name, data):
    p = A / name
    assert not p.exists()
    p.write_text(json.dumps(data, indent=2) + '\n')
    return pin(p)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    assert pin(D / 'SHA256SUMS')['sha256'] == PACKAGE
    for line in (D / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert pin(D / name)['sha256'] == digest
    candidate = A / 'own-metric-collector-build-candidate-v5.json'
    proof = A / 'own-metric-collector-compile-proof-v1.json'
    assert pin(candidate)['sha256'] == '725f4d066d72097e4f394f0b11a03a4b6021e911fe179f4e4822325527eecdd1'
    assert pin(proof)['sha256'] == '4504acbf5fc1029f365fbf37373a49759ed7201fef9be684b8028a481370f899'
    r, inspection = json.loads(candidate.read_text()), json.loads(proof.read_text())
    assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
    assert inspection['passed'] and inspection['complete'] and inspection['build_candidate'] == pin(candidate)
    assert r['binary_candidate'] == inspection['binary'] == pin(r['binary_candidate']['path'])
    assert pin(A / 'own-metric-status-offline-v3.json')['sha256'] == '60147b2258b99010910dd31e6dbde70a59bf507839dd26857ee84a55491c1b76'
    prep_path = A / 'own-metric-collector-source-preparation-v1.json'
    prep = json.loads(prep_path.read_text())
    original_path = B / 'main-primary-unitrace-build-root-v1/record.json'
    original = json.loads(original_path.read_text())
    base_dir = A / 'own-vllm-resource-timeline-v3'
    manifest = json.loads((base_dir / 'manifest.json').read_text())
    manifest['_sha256'] = pin(base_dir / 'manifest.json')['sha256']
    base = load(base_dir / 'controller.py', 'sealed_collector_base')
    owner = load(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py', 'sealed_collector_owner')
    identity = load(D / 'collector_identity.py', 'sealed_collector_identity')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert base.boot_id() == r['boot_id']
        for c in r['commands']:
            assert owner.closed(c) and c['exit_code'] == 0
        base.absence(r['commands'])
        assert r['reviewed_package_sha256'] == 'a9de0e10438e7249224aacc185b8dec4678596f16b9b92a6efb652c83636980d'
        assert pin(D / 'pti-query-status.patch')['sha256'] == pin(A / 'own-metric-query-tiny-v3/pti-query-status.patch')['sha256']
        old_source = json.loads(Path(prep['patched_inventory']['path']).read_text())
        for name, digest in old_source['files'].items():
            assert pin(Path(r['source_root']) / name)['sha256'] == digest
        tool_paths = ['/usr/bin/cmake', '/usr/bin/ninja', '/usr/bin/python3', '/usr/bin/ld',
            r['recipe'][0][r['recipe'][0].index('-DCMAKE_BUILD_TYPE=Release') + 1].split('=', 1)[1]]
        tools = [pin(p) for p in tool_paths]
        # Reuse the independently frozen original compiler/linker/runtime inputs;
        # all original hashes are checked again, never refreshed in place.
        original_tools = [dict(path=path, **value) if 'path' not in value else value for path, value in manifest['file_pins'].items()
            if path.endswith(('/bin/clang++', '/bin/cmake', '/bin/ninja', '/bin/ld'))]
        for value in original_tools:
            base.check_pin(value['path'], value)
        generated = [*inspection['generated_inputs'], *[item['copied'] for item in r['sdk_inputs']]]
        build = Path(r['build_root'])
        inputs = write('own-metric-collector-toolchain-inputs-v1.json', dict(
            schema='strata-pti-query-build-inputs-v1', complete=True,
            consumer='Main patched unitrace build', files=[*tools, *original_tools,
                pin(build / 'CMakeCache.txt'), pin(build / 'build.ninja'), pin(candidate), pin(proof),
                pin(prep_path), *r['offline_header_cache']],
            dependencies=inspection['compiler_consumed_inputs'], generated_inputs=generated,
            recipe=r['recipe'], environment=r['environment'],
            complete_official_source_inventory=prep['patched_inventory'],
            original_compiler_and_sdk_reference=pin(original_path),
            generated_capture=inspection['generated_capture'],
            generated_input_scope='Actual files captured during compile before official Release cleanup; archive hashes and original mtimes reconcile with Ninja output records. Post-build -M uses byte-identical archive includes for the deleted files.',
            SDK_revisions=dict(level_zero=original['level_zero_headers_tag'], compute_runtime=original['compute_runtime_headers_revision']),
            launcher_executed=False, GPU_executed=False))
        source = write('own-metric-collector-source-seal-v1.json', dict(
            schema='strata-pti-query-source-seal-v2', status='MAIN_REVIEWED_SEALED', reviewer='Main',
            official_pti_commit='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033', clean_baseline=True,
            reviewed_package_sha256=PACKAGE, baseline_inventory=prep['baseline_inventory'],
            patched_inventory=prep['patched_inventory'], toolchain_inputs=inputs,
            original_build_receipt=pin(original_path), patch=pin(D / 'pti-query-status.patch'),
            patch_apply_fuzz=0, patch_apply_offset=0, actual_application=pin(prep_path),
            actual_applied_v2_patch=prep['patch'], reviewed_v3_patch_bytes_equal=True,
            actual_commit_macro='Empty: private complete verified source snapshot is not a Git checkout',
            review='Main read the full audit header/patch, status reconciliation and v3 catalog/report-validity changes; all 28 offline cases passed. Exact three modified/additional source hashes and complete baseline were independently checked.'))
        binary_review = write('own-metric-collector-binary-inspection-v1.json', dict(
            schema='strata-main-patched-collector-binary-review-v1', reviewed=True, reviewer='Main',
            binary=r['binary_candidate'], source_receipt=source, fresh_objects=True,
            actual_compile_proof=True, proof=pin(proof), DT_NEEDED=inspection['DT_NEEDED'],
            source_header_dependency_proof=True, fresh_ninja_object_and_link_proof=True,
            generated_archive_proof=True, no_compiler_cache=True,
            limits='Host-only binary review. No GPU collection, provider internal correctness or full-model qualification follows from this inspection.'))
        build_receipt = write('own-metric-collector-build-seal-v1.json', dict(
            schema='strata-pti-query-build-v2', mode='build', passed=True, complete=True,
            active=False, ownership_closed=True, boot_id=r['boot_id'], reviewed_package_sha256=PACKAGE,
            binary_candidate=r['binary_candidate'], source_receipt=source, toolchain_inputs=inputs,
            recipe=r['recipe'], commands=r['commands'], compile_commands=r['compile_commands'],
            build_log=pin(build.parent / 'owned/collector-build.stdout'),
            dependency_inventory=pin(proof), binary_inspection=binary_review,
            actual_candidate=pin(candidate), actual_build_reviewed_package_sha256=r['reviewed_package_sha256'], host_admission_only_reviewed_v4=True, generated_inputs_captured_during_build=True,
            GPU_executed=False, launcher_executed=False))
        # No new runtime library is enrolled. Read-only inspection found exactly
        # the existing XPTI/C++/C/loader dependencies; lazy metric/provider libraries
        # remain the separately pinned originals and actual maps are checked later.
        extra_files = {}
        for value in [*tools, *inspection['compiler_consumed_inputs'], *generated, pin(proof), binary_review]:
            if value['path'] not in manifest['file_pins']:
                extra_files[value['path']] = value
        review = write('own-metric-collector-dependency-review-v1.json', dict(
            schema='strata-main-patched-collector-dependency-review-v1', approved=True, reviewer='Main',
            elf_pins={}, file_pins=extra_files, replacement_collector=r['binary_candidate'],
            replacement_build_receipt=build_receipt, original_reference_collector=manifest['collector'],
            all_actual_dependencies_reviewed=True, DT_NEEDED=inspection['DT_NEEDED'],
            original_runtime_allowlist_unchanged=True, new_runtime_elf_enrollment=False,
            source_provenance='Official Intel PTI complete verified source with its original LICENSE; Main-reviewed local diagnostic-only patch, ordinary system C/C++ headers and existing compiler/XPTI SDK. Excluded engine source is never an input.',
            runtime_scope='The tiny run must capture actual mapped ELF identities; any unsealed dependency holds. Enumeration alone did not qualify collection.'))
        health_path = A / 'own-prompt-softmax-long-clean-0-r0-v1.json'
        health = pin(health_path)
        base.health_gate(str(health_path), health['sha256'], owner, r['boot_id'], manifest)
        catalog = {key: pin(B / rel) for key, rel in dict(
            receipt='main-metric-enum-query-v2/record.json', build_receipt='main-metric-enum-build-v2/record.json',
            binary='main-metric-enum-build-v2/metric-enumeration',
            stdout='main-metric-enum-query-v2/owned/native-enumeration.stdout').items()}
        seal_data = dict(status='MAIN_REVIEWED_SEALED', run_id='v1', reviewer='Main',
            boot_id=r['boot_id'], reviewed_package_sha256=PACKAGE,
            review_scope='ONE_TINY_PATCHED_QUERY_EXACT_API_STATUS',
            retention_review_utc='2026-10-17T03:00:00Z', health_receipt=health,
            additional_dependency_review=dict(report=review, elf_pins={}, file_pins=extra_files),
            catalog=catalog, replacement_collector=dict(binary=r['binary_candidate'],
                source_receipt=source, build_receipt=build_receipt),
            offline_suite=pin(A / 'own-metric-status-offline-v4.json'),
            latest_native_health_context=pin(A / 'own-metric-full-catalog-evidence-v2.json'))
        identity.sealed_replacement(seal_data, manifest, base, owner, PACKAGE)
        seal = write('own-metric-query-tiny-main-seal-v1.json', seal_data)
        print(json.dumps(dict(seal=seal, binary=r['binary_candidate'],
            additional_files=len(extra_files), native_test_executed=False)))


if __name__ == '__main__':
    main()
