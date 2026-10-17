#!/usr/bin/env python3
"""Retire closed QSA64 success traces and verified Git-restorable source mirrors."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = A / 'qsa-prefix64-retention-v1.json'
    assert not out.exists()
    assert json.loads((A / 'qsa-prefix64-decision-v1.json').read_text())['complete']
    names = ['qsa-prefix64-build-v1', 'qsa-prefix64-baseline32-v1', 'qsa-prefix64-smoke-v1',
             'qsa-prefix64-component-v1', 'qsa-prefix64-bench-smoke-base-v1', 'qsa-prefix64-bench-smoke-tile64-v1']
    names += [f'qsa-prefix64-bench-{arm}-{cells}-{rep}-v1' for cells in (32768, 131072)
              for arm in ('base', 'tile64') for rep in range(3)]
    helper = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(helper) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('qsa64_retention_owner', helper)
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        originals, deletions = {}, []

        def select(path, reason, replacement, restore=None):
            assert path.is_file() and not path.is_symlink()
            item = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                        reason=reason, replacement=replacement)
            if restore:
                item['restore_from_git'] = restore
            deletions.append(item)

        for name in names:
            receipt = A / (name + '.json')
            value = json.loads(receipt.read_text())
            assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
            assert not value.get('fresh_faults') and not value.get('driver_dump_entries')
            originals[str(receipt)] = sha(receipt)
            for command in value['commands']:
                assert owned.closed(command) and command['exit_code'] == 0
                for old in command['owners'] + [command['direct_identity']]:
                    now = owned.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            root = B / ('main-' + name)
            for log in (root / 'owned').rglob('*'):
                if log.is_file() and log.suffix in ('.stdout', '.stderr', '.log', '.jsonl'):
                    select(log, 'Closed successful raw API/build text. Original receipt retains stdout/cases, samples, runtime maps, kernel counts, fault and ownership results.', str(receipt))
            duplicate = root / 'record.json'
            if duplicate.exists():
                assert json.loads(duplicate.read_text()) == value
                select(duplicate, 'Exact structured duplicate of immutable receipt.', str(receipt))
        source_commit = subprocess.check_output(['git', 'rev-parse', '14d8904a'], text=True).strip()
        for path in (A / 'qsa-prefix64-candidate-v1/files').rglob('*'):
            if not path.is_file():
                continue
            relative = str(path.relative_to(A.parents[2]))
            restored = subprocess.check_output(['git', 'show', source_commit + ':' + relative])
            assert hashlib.sha256(restored).hexdigest() == sha(path)
            select(path, 'Rejected64cell source mirror no longer consumed. Independent16cell candidate is complete in its own directory; original proposal source remains reachable in Git.', str(A / 'qsa-prefix64-decision-v1.json'), dict(commit=source_commit, path=relative))
        review = json.loads((A / 'research-qsa-xmx-source-review-pin-v1.json').read_text())
        assert (A / 'research-round1023-index-proposal-v1.json').is_file()
        for relative, expected in review['sha256'].items():
            path = Path(review['snapshot']) / relative
            restored = subprocess.check_output(['git', 'show', review['source_commit'] + ':' + relative])
            assert hashlib.sha256(restored).hexdigest() == sha(path) == expected
            select(path, 'Native review closed; source restored and hash-verified from pushed Git, so the reviewed snapshot is redundant.', str(A / 'research-round1023-index-proposal-v1.json'), dict(commit=review['source_commit'], path=relative))
        record = dict(schema='strata-qsa-prefix64-retention-v1', complete=False,
                      recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      original_receipts=originals, files=deletions, deleted=0, logical_bytes_removed=0,
                      scope='Closed success text and verified source duplicates only. Numerical PASS/performance rejection preserved; no active work or last failure fixture removed. Binaries/build objects remain available. Keep current16cell candidate inputs.')

        def save():
            out.write_text(json.dumps(record, indent=2) + '\n')

        save()
        try:
            for row in deletions:
                path = Path(row['path'])
                assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
                path.unlink()
                record['deleted'] += 1
                record['logical_bytes_removed'] += row['bytes']
                save()
            for path, expected in originals.items():
                assert sha(path) == expected
            record['complete'] = True
        finally:
            save()
        print(json.dumps({k: record[k] for k in ('complete', 'deleted', 'logical_bytes_removed')}), flush=True)


if __name__ == '__main__':
    main()
