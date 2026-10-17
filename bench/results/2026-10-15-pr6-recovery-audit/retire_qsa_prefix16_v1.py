#!/usr/bin/env python3
"""Retire closed QSA16 success/diagnostic API traces; preserve numerical failure fixtures."""
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
    out = A / 'qsa-prefix16-retention-v1.json'
    assert not out.exists()
    assert json.loads((A / 'qsa-prefix16-decision-v1.json').read_text())['complete']
    names = ['qsa-prefix16-build-v1', 'qsa-prefix16-baseline32-v1', 'qsa-prefix16-smoke-v1',
             'qsa-prefix16-component-v1', 'qsa-prefix16-bench-smoke-base-v1', 'qsa-prefix16-bench-smoke-tile16-v1']
    names += [f'qsa-prefix16-bench-{arm}-{cells}-{rep}-v1' for cells in (32768, 131072)
              for arm in ('base', 'tile16') for rep in range(3)]
    helper = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(helper) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('qsa16_retention_owner', helper)
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
        diagnostic_path = A / 'qsa16-oracle-observer-base-262144-0-v1.json'
        diagnostic = json.loads(diagnostic_path.read_text())
        assert diagnostic['diagnostic_complete'] and diagnostic['complete'] and diagnostic['ownership_closed']
        assert not diagnostic['numerical_passed'] and not diagnostic['passed'] and diagnostic['native_exit_code'] == 1
        assert diagnostic['fresh_faults'] == [] and not diagnostic['driver_dump_entries']
        originals[str(diagnostic_path)] = sha(diagnostic_path)
        for command in diagnostic['commands']:
            assert owned.closed(command)
            for old in command['owners'] + [command['direct_identity']]:
                now = owned.identity(old['pid'])
                assert now is None or now['start_ticks'] != old['start_ticks']
        diagnostic_root = B / 'main-qsa16-oracle-observer-base-262144-0-v1'
        for log in (diagnostic_root / 'owned').rglob('*'):
            if log.is_file() and log.suffix in ('.stdout', '.stderr', '.log', '.jsonl'):
                select(log, 'Diagnostic numericalfailure is preserved in receipt/decision and prior representative. Full API history has no current consumer; exact divergence, failed criterion,3snapshots,2kernelcount and normalownedexit1 retained.', str(diagnostic_path))
        # Preserve the original 596-byte stdout and 41-byte stderr failure
        # representative, all original failures and every required binary/source.
        record = dict(schema='strata-qsa-prefix16-retention-v1', complete=False,
                      recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      original_receipts=originals, files=deletions, deleted=0, logical_bytes_removed=0,
                      scope='Closed success text and redundant diagnostic API text only. Preserve originalfailed262144smalllogs, alloriginalresultstatuses, numericcoordinate/values and current16cell+observer source/object/binary fixtures. No activejob or lastneededfailurefixture removed. Numericalfailure remainsFAIL; original16cell scope hasmixed32Kregression/128Ksmallgain.')

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
