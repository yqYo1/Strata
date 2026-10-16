#!/usr/bin/env python3
"""Main builds a fresh host collector only; no launcher/device execution.
Repository LICENSE. Actual source/build inspection and native admission are separate.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = A / 'own-metric-query-tiny-v3'
LLVM = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(p):
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p))


def main():
    root = B / 'main-metric-query-collector-build-v3'
    target = A / 'own-metric-collector-build-candidate-v3.json'
    assert not root.exists() and not target.exists()
    package = 'a9de0e10438e7249224aacc185b8dec4678596f16b9b92a6efb652c83636980d'
    assert sha(D / 'SHA256SUMS') == package
    for line in (D / 'SHA256SUMS').read_text().splitlines():
        h, n = line.split('  ', 1)
        assert sha(D / n) == h
    offline = A / 'own-metric-status-offline-v3.json'
    assert sha(offline) == '60147b2258b99010910dd31e6dbde70a59bf507839dd26857ee84a55491c1b76'
    assert json.loads(offline.read_text())['passed']
    prep = A / 'own-metric-collector-source-preparation-v1.json'
    assert sha(prep) == 'ca6a681c771e5b376f64b4eb39007a57c16fc5e05e80be124fe0603e80f6da03'
    preparation = json.loads(prep.read_text())
    assert preparation['complete'] and preparation['ownership_closed']
    source = Path(preparation['private_source'])
    inventory = preparation['patched_inventory']
    assert sha(Path(inventory['path'])) == inventory['sha256']
    source_pins = json.loads(Path(inventory['path']).read_text())['files']
    # Actual application used the completed v2 patch. V3 preserves those exact
    # bytes; both paths and the equivalence are recorded, never relabeling history.
    assert preparation['patch']['sha256'] == sha(D / 'pti-query-status.patch')
    old_path = B / 'main-primary-unitrace-build-root-v1/record.json'
    assert sha(old_path) == '692da9f9d706fea659e78378a397c71d86b08cb9fba0c994e390de29b6a20647'
    old = json.loads(old_path.read_text())
    original_build = Path(old['build_path'])
    original_binary = original_build / 'libunitrace_tool.so'
    assert sha(original_binary) == '8a8fe37441606bf66ab42fc39cc447b5257f48d09f04fbcc60703e30826671eb'
    owner_source = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(owner_source) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('metric_collector_build_owner', owner_source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        build = root / 'build'
        build.mkdir()
        sdk_pins = []
        for name, digest in old['generated_header_pins'].items():
            original = original_build / name
            assert sha(original) == digest
            copy = build / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(original.read_bytes())
            sdk_pins.append(dict(original=pin(original), copied=pin(copy)))
        options = next(c['argv'] for c in old['commands'] if c['label'] == 'configure-unitrace')
        configure = ['/usr/bin/cmake', *options[1:]]
        configure[configure.index('-S') + 1] = str(source / 'tools/unitrace')
        configure[configure.index('-B') + 1] = str(build)
        configure.append('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON')
        recipe = [configure, ['/usr/bin/cmake', '--build', str(build), '--parallel', '2']]
        env = dict(PATH=str(LLVM / 'bin') + ':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8',
            LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=str(LLVM / 'lib'),
            TMPDIR=str(root))
        record = dict(schema='strata-metric-collector-build-candidate-v3', status='INCOMPLETE', mode='build',
            passed=False, complete=False, active=False, commands=[], recipe=recipe, environment=env,
            reviewed_package_sha256=package, source_preparation=pin(prep), patched_inventory=inventory,
            actual_applied_patch=preparation['patch'], reviewed_v3_same_patch=pin(D / 'pti-query-status.patch'),
            patch_bytes_equal=True, original_build_receipt=pin(old_path), original_collector=pin(original_binary),
            source_root=str(source), build_root=str(build), sdk_inputs=sdk_pins,
            GPU_executed=False, launcher_executed=False, collection_qualified=False,
            source_receipt_finalization_required=True, binary_review_required=True,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            budgets=dict(whole_seconds=1200, child_seconds=900, raw_bytes=64 << 20, private_bytes=1 << 30),
            retention=dict(owner='Main', consumer='Main source/compile/dependency inspection before one bounded metric-query smoke',
                review_utc='2026-10-17T03:00:00Z'))
        owner = None
        started = time.monotonic()

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = helper.Owner(root / 'owned', persist)
        record['commands'] = owner.commands

        def frozen():
            assert time.monotonic() - started < 1200
            assert sha(D / 'SHA256SUMS') == package
            for name, h in source_pins.items():
                assert sha(source / name) == h
            for sdk in sdk_pins:
                assert sha(Path(sdk['copied']['path'])) == sdk['copied']['sha256']
            assert sha(original_binary) == record['original_collector']['sha256']

        try:
            frozen()
            for label, argv in zip(('collector-configure', 'collector-build'), recipe):
                c, _, _ = owner.run(label, argv, env, source / 'tools/unitrace', wall=900, cpu=1200,
                    rss_cap=8 << 30, text_cap=64 << 20, total_cap=1 << 30, file_cap=64 << 20)
                assert helper.closed(c) and c['exit_code'] == 0
                frozen()
            binary = build / 'libunitrace_tool.so'
            record['binary_candidate'] = pin(binary)
            record['compile_commands'] = pin(build / 'compile_commands.json')
            record.update(passed=True, complete=True, status='BUILD_CANDIDATE_MAIN_INSPECTION_REQUIRED')
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(helper.closed(c) and c['exit_code'] == 0 for c in owner.commands)
            for c in owner.commands:
                for old in c['owners'] + [c['direct_identity']]:
                    current = helper.identity(old['pid'])
                    assert current is None or current['start_ticks'] != old['start_ticks']
            record['elapsed_seconds'] = time.monotonic() - started
            persist()
            target.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=pin(target), status=record['status'], complete=record['complete'], GPU_executed=False)), flush=True)
        return 0 if record['passed'] and record['ownership_closed'] else 1


if __name__ == '__main__':
    sys.exit(main())
