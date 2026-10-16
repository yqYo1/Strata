#!/usr/bin/env python3
"""Retire integrated proposal mirrors after proving durable Git recoverability."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S = A / 'own-dense-prefill-nll-v3'
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import require, sha


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = A / 'integrated-nll-source-mirrors-retention-v1.json'
        require(not target.exists(), 'fresh mirror retirement')
        commit = subprocess.check_output(['git','rev-parse','dfca0e14^{commit}'], cwd=P, text=True).strip()
        remote = subprocess.check_output(['git','rev-parse','origin/feature/sycl'], cwd=P, text=True).strip()
        require(commit == remote, 'integrated source pushed')
        m = json.loads((S / 'manifest.json').read_text())
        require(sha(S / 'manifest.json') == 'e917011fdb210265df32125ed35fec8d6beb46bc6e1f1f472dd7b24d94fc3464', 'immutable proposal manifest')
        require(json.loads((A / 'own-dense-prefill-nll-smoke-v4.json').read_text())['passed'], 'integration comparison completed')
        require(json.loads((A / 'closed-own-dense-prefill-nll-retention-v1.json').read_text())['status'] == 'complete', 'all integration jobs closed and compacted')
        reachable = {v.split()[0] for v in subprocess.check_output(['git','rev-list','--objects',commit], cwd=P, text=True).splitlines()}
        extras = [
            'add_executable(sycl_host_transfer_probe EXCLUDE_FROM_ALL tools/sycl/host_transfer_probe.cpp)\ntarget_link_libraries(sycl_host_transfer_probe PRIVATE strata_sycl_runtime)\n',
            '  add_executable(sycl_gemm_device_probe EXCLUDE_FROM_ALL tools/sycl/gemm_device_probe.cpp)\n  target_link_libraries(sycl_gemm_device_probe PRIVATE strata_prefill)\n  target_compile_options(sycl_gemm_device_probe PRIVATE -fno-fast-math -ffp-contract=off)\n']
        deletions = []
        for v in m['production_source_snapshot']:
            path = S / 'base' / v['path']
            data = path.read_bytes()
            require(sha(path) == v['sha256'] and len(data) == v['bytes'], 'original base snapshot pin')
            oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            require(oid in reachable, 'exact original base blob is reachable from integrated commit')
            require(subprocess.check_output(['git','cat-file','blob',oid], cwd=P) == data, 'Git object content proof')
            deletions.append(dict(path=str(path), bytes=len(data), sha256=v['sha256'], git_blob=oid,
                                  recovery=['git','cat-file','blob',oid], status='planned'))
        for v in m['changes']:
            path = S / 'files' / v['path']
            data = path.read_bytes()
            require(sha(path) == v['proposed_sha256'], 'original proposed source pin')
            actual = subprocess.check_output(['git','show',commit + ':' + v['path']], cwd=P)
            if v['path'] == 'cmake/sycl_backend.cmake':
                for extra in extras:
                    require(actual.count(extra.encode()) == 1, 'one independent target block')
                    actual = actual.replace(extra.encode(), b'')
            require(actual == data, 'exact proposed source recoverable from integrated Git source')
            deletions.append(dict(path=str(path), bytes=len(data), sha256=v['proposed_sha256'],
                                  recovery_commit=commit, recovery_path=v['path'],
                                  remove_exact_blocks=extras if v['path'] == 'cmake/sycl_backend.cmake' else [], status='planned'))
        manifest = dict(status='planned', owner='Main', recipe_sha256=sha(Path(__file__)),
                        integrated_commit=commit, manifest_sha256=sha(S / 'manifest.json'),
                        original_proposal_status_unchanged=True, physical_free_space_claim=False,
                        reason='Source integrated, built and mechanically tested; no remaining full-mirror consumer. All333 base blobs reachable and byte verified; all6 proposed files reconstruct exactly from pushed source. Patch/provenance/manifest/original statuses retained.',
                        deletions=deletions)
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        for item in deletions:
            path = Path(item['path'])
            require(not path.is_symlink() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'identity before unlink')
            path.unlink()
            item['status'] = 'removed'
        for root in (S / 'base', S / 'files'):
            for path in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p:len(p.parts), reverse=True):
                path.rmdir()
            root.rmdir()
        manifest.update(status='complete', files=len(deletions), removed_logical_bytes=sum(v['bytes'] for v in deletions))
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({k:manifest[k] for k in ('status','files','removed_logical_bytes')}))


if __name__ == '__main__':
    main()
