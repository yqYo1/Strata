#!/usr/bin/env python3
"""Main-owned offline tests of the completed exact-status source proposal.
Repository LICENSE. No collector build, runtime initialization or device work.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source = A / 'own-metric-query-tiny-v2'
    package = sha(source / 'SHA256SUMS')
    for line in (source / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(source / name) == digest
    owner_source = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(owner_source) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    target = A / 'own-metric-status-offline-v2.json'
    root = B / 'main-metric-status-offline-v2'
    assert not target.exists() and not root.exists()
    spec = importlib.util.spec_from_file_location('main_metric_offline_owner', owner_source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        record = dict(schema='strata-metric-status-offline-v2', status='INCOMPLETE', passed=False,
            active=False, commands=[], GPU_executed=False, collector_built=False, collection_qualified=False,
            source_package_sha256=package, source_review='Main read complete C++ audit/header/diff, controller base+delta, status parser, offline fixtures and tests. Catalog identity and report-validity defects separately recorded; v3 source correction dispatched.',
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = helper.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        command, out, err = owner.run('offline-source-status-tests',
            ['/usr/bin/python3', '-I', '-B', str(source / 'test_offline.py'), '-v'], env, A.parents[2],
            wall=60, cpu=45, rss_cap=1 << 30, text_cap=8 << 20, total_cap=32 << 20, file_cap=8 << 20)
        assert helper.closed(command)
        for old in command['owners'] + [command['direct_identity']]:
            current = helper.identity(old['pid'])
            assert current is None or current['start_ticks'] != old['start_ticks']
        assert package == sha(source / 'SHA256SUMS')
        for line in (source / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            assert sha(source / name) == digest
        record.update(passed=command['exit_code'] == 0, complete=True, ownership_closed=True,
            status='OFFLINE_TESTS_PASS_COLLECTION_HELD' if command['exit_code'] == 0 else 'OFFLINE_TESTS_FAIL',
            stdout=out.read_text(), stderr=err.read_text())
        persist()
        target.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(path=str(target), sha256=sha(target), status=record['status'],
            exit_code=command['exit_code'], collection_qualified=False)))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
