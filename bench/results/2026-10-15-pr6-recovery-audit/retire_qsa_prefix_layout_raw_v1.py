#!/usr/bin/env python3
"""Retire successful layout-experiment raw logs; keep the current control fixture."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def committed(p):
    data = subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(P))],cwd=P)
    assert hashlib.sha256(data).hexdigest() == sha(p)


def main():
    dst = A/'retention-qsa-prefix-layout-raw-v1.json'
    assert not dst.exists()
    owner_path = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(owner_path) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('layout_retention_owner',owner_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    stems = ['qsa-bf16-prefix-coalesced-'+s for s in ['build','smoke','component']]
    stems += ['qsa-bf16-prefix-layout-bench-'+s for s in ['build','smoke-base','smoke-coalesced']]
    stems += [f'qsa-bf16-prefix-layout-bench-{arm}-32768-{n}' for arm in ['base','coalesced'] for n in range(3)]
    rec = dict(schema='strata-layout-raw-retention-v1', deleted=[], receipts={},
               logical_bytes=0, allocated_file_bytes=0, original_statuses_unchanged=True,
               scope='Successful full UR/protocol/build/journal logs and byte-identical private record duplicates only.',
               retained_current_fixture=dict(consumer='Main next same-input parallel-probability component baseline and frozen-input validation',
                   path=str(B/'main-qsa-bf16-prefix-layout-bench-build-v1/sycl_qsa_bf16_prefix_layout_bench'),
                   next_review='When replacement same-input component fixture is qualified, retire superseded private source/object/fixture captures; preserve source recipe and result hashes in Git.'),
               actual_free_space_reclaimed='Not inferred; filesystem snapshots may retain old extents.')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = A/'qsa-bf16-prefix-layout-cohort-v1.json'
        committed(summary)
        cohort = json.loads(summary.read_text())
        assert cohort['passed'] and cohort['complete'] and cohort['active_child_pid'] is None
        rec['summary'] = dict(path=str(summary),sha256=sha(summary))
        paths = {}
        protected = set()
        for stem in stems:
            p = A/f'own-{stem}-v1.json'
            committed(p)
            value = json.loads(p.read_text())
            assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
            assert not value.get('fresh_faults') and not value.get('driver_dump_entries')
            rec['receipts'][str(p)] = sha(p)
            protected.update(value.get('frozen_files',{}))
            for c in value['commands']:
                assert m.closed(c) and c['exit_code'] == 0
                for old in c['owners'] + [c['direct_identity']]:
                    now = m.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            root = B/f'main-{stem}-v1'
            assert sha(root/'record.json') == sha(p)
            paths[root/'record.json'] = dict(replacement=str(p),reason='Byte-identical closed receipt; committed Git report is canonical.')
            for f in (root/'owned').iterdir():
                if f.is_file() and f.suffix in ['.stdout','.stderr']:
                    paths[f] = dict(replacement=str(p),reason='Successful full API/build/journal/protocol output; all individual timings, oracle/guard/input results, exact source/env/runtime pins, owner outcomes and fault findings recorded in committed receipt.')
        for f,why in paths.items():
            assert not f.is_symlink() and f.is_relative_to(B) and str(f) not in protected
            st = f.stat()
            rec['deleted'].append(dict(path=str(f),bytes=st.st_size,allocated_file_bytes=st.st_blocks*512,sha256=sha(f),**why))
            rec['logical_bytes'] += st.st_size
            rec['allocated_file_bytes'] += st.st_blocks*512
        for row in rec['deleted']:
            f = Path(row['path'])
            assert sha(f) == row['sha256'] and f.stat().st_size == row['bytes']
            f.unlink()
        assert all(sha(p) == digest for p,digest in rec['receipts'].items())
        dst.write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(manifest=str(dst),files=len(rec['deleted']),logical_bytes=rec['logical_bytes'],allocated_file_bytes=rec['allocated_file_bytes'])))


if __name__ == '__main__':
    main()
