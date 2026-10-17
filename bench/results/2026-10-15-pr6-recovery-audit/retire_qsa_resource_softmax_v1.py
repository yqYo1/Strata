#!/usr/bin/env python3
"""Retire only closed successful raw diagnostics after compact evidence checks."""
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    target = A / 'qsa-resource-softmax-retention-v1.json'
    assert not target.exists()
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(owner_path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owner = load_module('resource_retention_owner', owner_path)
    package = A / 'own-vllm-xe2-qualification-v1'
    assert sha(package / 'manifest.json') == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    metadata = json.loads((package / 'manifest.json').read_text())
    helper = package / 'main_controller.py'
    assert sha(helper) == metadata['package_files']['main_controller.py']['sha256']
    guard = load_module('resource_retention_guard', helper)
    pairs = [(f'own-qsa-gemm-unitrace-direct-{mode}-v1.json', f'main-qsa-gemm-unitrace-direct-{mode}-v1') for mode in ['tiny', 'model']]
    pairs += [('own-qsa-softmax-actual-entry-build-v1.json', 'main-qsa-softmax-actual-entry-build-v1')]
    pairs += [(f'own-qsa-softmax-actual-entry-{arm}-{mode}-v1.json', f'main-qsa-softmax-actual-entry-{arm}-{mode}-v1') for arm in ['serial', 'subgroup'] for mode in ['smoke', 'component']]
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.no_model_owners(metadata['known_model_files'])
        guard.driver_guard()
        records, candidates, diagnostics = [], [], []
        for receipt_name, root_name in pairs:
            receipt, root = A / receipt_name, B / root_name
            assert root.resolve() == root and root.parent == B
            data = json.loads(receipt.read_text())
            assert data['passed'] and data['complete'] and data['ownership_closed'] and not data['active']
            if 'fresh_faults' in data:
                assert data['fresh_faults'] == [] and data['boot_unchanged']
            assert not data.get('driver_dump_exists') and not data.get('driver_dump_entries')
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == data['boot_id']
            for command in data['commands']:
                assert owner.closed(command) and command['exit_code'] == 0
                for old in command['owners'] + [command['direct_identity']]:
                    current = owner.identity(old['pid'])
                    assert current is None or current['start_ticks'] != old['start_ticks']
            records.append(dict(path=str(receipt), sha256=sha(receipt), original_passed=True))
            if 'collection' in data:
                collection = data['collection']
                assert collection['proved']
                for pin in collection['files']:
                    raw = Path(pin['path'])
                    assert raw.is_relative_to(root) and sha(raw) == pin['sha256']
                    assert raw.read_text() == collection['raw_summary']
            native = root / 'owned/engine.stderr'
            if native.exists():
                lines = native.read_text().splitlines()
                useful = [line for line in lines if
                    line.startswith(('strata prefill:', 'strata QSA score prefill:', 'strata MoE', 'strata mtp:', 'strata serve: prompt')) or
                    (line.startswith('strata trace: layer-major layer ') and ': completed ' in line)]
                diagnostics.append(dict(root=str(root), retained_native_lines=useful))
            for path in root.rglob('*'):
                if path.is_symlink() or not path.is_file():
                    continue
                reason = None
                if path.suffix in ('.stdout', '.stderr', '.log', '.jsonl'):
                    reason = 'Successful raw API/protocol/progress or aggregate duplicate; compact oracle/profile/request/fault/ownership evidence verified in immutable receipt and this manifest.'
                elif path == root / 'record.json' and json.loads(path.read_text()) == data:
                    reason = 'Exact semantic duplicate of closed canonical receipt.'
                elif path == root / 'result.json' and json.loads(path.read_text()) == data.get('result'):
                    reason = 'Exact semantic duplicate of result embedded in canonical receipt.'
                elif path.name == 'results.json' and path.is_relative_to(root / 'stock-results'):
                    assert json.loads(path.read_text()) == data['model_evidence']['suite']
                    reason = 'Exact semantic duplicate of official suite embedded in canonical receipt.'
                if reason:
                    st = path.stat()
                    candidates.append(dict(path=str(path), bytes=st.st_size, allocated_bytes=st.st_blocks * 512,
                        inode=st.st_ino, device=st.st_dev, sha256=sha(path), retained_in=str(receipt), reason=reason))
        summary = A / 'qsa-gemm-current-resource-profile-v1.json'
        compact = json.loads(summary.read_text())
        assert compact['passed'] and compact['source_receipt']['sha256'] == sha(A / 'own-qsa-gemm-unitrace-direct-model-v1.json')
        manifest = dict(schema=1, passed=False, complete=False, receipts=records, diagnostics=diagnostics,
            summary=dict(path=str(summary), sha256=sha(summary)), deletions=candidates,
            retained='Canonical receipt bytes unchanged; current full engine/archive tree, actual-entry fixture ELF, source pins, configuration and small launch proofs retained for the explicit next softmax model comparison. No tensor capture was made.',
            original_statuses_preserved=True, available_bytes_before=shutil.disk_usage(B).free)
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        for item in candidates:
            path = Path(item['path']); st = path.stat()
            assert not path.is_symlink() and st.st_ino == item['inode'] and st.st_dev == item['device'] and sha(path) == item['sha256']
            path.unlink()
        assert all(sha(Path(item['path'])) == item['sha256'] for item in records)
        manifest.update(passed=True, complete=True, removed_files=len(candidates),
            removed_logical_bytes=sum(item['bytes'] for item in candidates),
            removed_file_allocation_bytes=sum(item['allocated_bytes'] for item in candidates),
            available_bytes_after=shutil.disk_usage(B).free)
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({key: manifest[key] for key in ['passed', 'complete', 'removed_files', 'removed_logical_bytes']}))


if __name__ == '__main__':
    main()
