#!/usr/bin/env python3
"""Retire consumed8192-query frames and successful diagnostic duplicates."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent; P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def committed(path):
    assert subprocess.check_output(['git', 'show', 'HEAD:' + str(path.relative_to(P))], cwd=P) == path.read_bytes()


def main():
    out = A / 'qsa-large-resource-raw-retirement-v1.json'; assert not out.exists()
    ownerpath = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(ownerpath) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('qsa_large_retirement_owner', ownerpath)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    names = [f'own-qsa-large-resource-first{x}-v1.json' for x in 'ABC']
    names += [f'own-qsa-large-resource-clean{x}{k}-v1.json' for x in 'ABC' for k in range(3)]
    names += [f'own-qsa-large-resource-compare-{x}-v1.json' for x in
              ['firstB', 'firstC'] + [f'clean{x}{k}' for x in 'ABC' for k in range(3)]]
    names += ['own-qsa-frame-comparator-check-v1.json']
    planned = {}
    def add(path, reason):
        path = Path(path); st = path.lstat()
        assert path.parent.resolve().is_relative_to(B.resolve())
        row = dict(path=str(path), bytes=st.st_size, inode=st.st_ino, device=st.st_dev,
                   reason=reason, removed=False)
        if stat.S_ISLNK(st.st_mode):
            row.update(type='symlink', link_target=os.readlink(path))
        else:
            assert stat.S_ISREG(st.st_mode)
            row.update(type='regular', sha256=sha(path))
        planned[str(path)] = row
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = A / 'hardware-resource-followup-v8.json'; committed(summary)
        s = json.loads(summary.read_text()); assert s['passed'] and len(s['processes']) == 9
        receipts = {}
        for name in names:
            path = A / name; committed(path); j = json.loads(path.read_text()); receipts[name] = j
            assert j['passed'] and j['complete'] and j['ownership_closed'] and not j['active']
            for c in j['commands']:
                assert m.closed(c)
                for old in c['owners'] + [c['direct_identity']]:
                    now = m.identity(old['pid']); assert now is None or now['start_ticks'] != old['start_ticks']
            root = B / name.removesuffix('.json').replace('own-', 'main-', 1)
            assert root.parent == B
            duplicate = root / 'record.json'
            assert duplicate.read_bytes() == path.read_bytes()
            add(duplicate, 'Byte-identical duplicate of committed receipt; samples/status remain unchanged.')
            expected_logs = {k: v for c in j['commands'] for k, v in c['logs'].items()}
            for raw in (root / 'owned').iterdir():
                if raw.suffix in ['.stdout', '.stderr']:
                    assert raw.stat().st_size == expected_logs[raw.name]['bytes']
                    assert sha(raw) == expected_logs[raw.name]['sha256']
                    add(raw, 'Successful closed API/comparator/journal detail consumed; progress, '
                             'checks, timings, counts, fault outcomes and ownership retained in Git.')
        first = receipts['own-qsa-large-resource-firstA-v1.json']['cases'][0]['frame']
        assert sha(first['path']) == first['sha256']
        add(first['path'], 'Canonical consumer complete: all11 candidate frames independently finite/byte-compared; '
                           'all90timed samples retained. No mismatch or current RESTORE consumer.')
        check = receipts['own-qsa-frame-comparator-check-v1.json']
        for v in check['fixture_pins'].values():
            path = Path(v['path']); assert path.stat().st_size == v['bytes'] and sha(path) == v['sha256']
            add(path, 'Nine comparator contract cases consumed; exact fixture identities, outcomes '
                      'and reproducible fixture source retained in Git.')
        fixtures = B / 'main-qsa-frame-comparator-check-v1/fixtures'
        hard = fixtures / 'hardlink.bin'; good = fixtures / 'good-a.bin'
        assert hard.stat().st_ino == good.stat().st_ino and hard.stat().st_dev == good.stat().st_dev
        add(hard, 'Consumed same-inode rejection fixture; not a separate physical payload.')
        link = fixtures / 'symlink.bin'; assert link.is_symlink() and link.resolve() == good
        add(link, 'Consumed symlink rejection fixture; remove link itself without following it.')
        assert {str(p) for p in fixtures.iterdir()} <= planned.keys()
        record = dict(schema='strata-qsa-large-resource-raw-retirement-v1', passed=False,
            complete=False, GPU_executed=False, source_sha256=sha(__file__),
            original_receipt_statuses_unchanged=True, all_receipts_verified_in_Git=True,
            all_native_ownership_closed=True, canonical_comparison_consumer_closed=True,
            retained='Current source, v4binary and frozen archives; all individual timings, '
                     'fullframe comparison receipts and fault/lifetime/source/environment evidence in Git.',
            files=list(planned.values()), total_logical_bytes=sum(x['bytes'] for x in planned.values()),
            accounting='Logical paths include sparse files and a hardlink; not physical disk blocks freed.')
        def persist(): out.write_text(json.dumps(record, indent=2) + '\n')
        persist()
        for row in record['files']:
            path = Path(row['path']); st = path.lstat()
            assert (st.st_dev, st.st_ino, st.st_size) == (row['device'], row['inode'], row['bytes'])
            if row['type'] == 'regular': assert sha(path) == row['sha256']
            else: assert path.is_symlink() and os.readlink(path) == row['link_target']
            path.unlink(); row['removed'] = True; persist()
        record.update(passed=True, complete=True); persist()
        print(json.dumps(dict(receipt=str(out), passed=True, files=len(record['files']),
                              logical_bytes=record['total_logical_bytes'])))


if __name__ == '__main__':
    main()
