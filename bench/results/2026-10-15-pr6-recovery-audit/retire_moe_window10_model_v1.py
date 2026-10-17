#!/usr/bin/env python3
"""Retire reviewed successful build/model text after both four-request cohorts close."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = A / 'moe-window10-model-retention-v1.json'
    excerpt_path = A / 'moe-window10-model-excerpts-v1.json'
    assert not out.exists() and not excerpt_path.exists()
    analysis = A / 'moe-window10-first-subsequent-analysis-v1.json'
    assert json.loads(analysis.read_text())['passed']
    helper = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(helper) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('window10_retire_owner', helper)
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    jobs = [('moe-accum-engine-build-v2.json', 'main-moe-accum-engine-build-v2', 'build')]
    for suffix in ('v4', 'warm4-v5'):
        for mode in (8, 10):
            stem = f'moe-accum-model-profile-window{mode}-tokens65536-{suffix}'
            jobs.append((stem + '.json', 'main-' + stem, 'model'))
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = {str(analysis): sha(analysis)}
        excerpts = {}
        deletions = []

        def select(path, replacement, reason):
            assert path.is_file() and not path.is_symlink()
            deletions.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                                  replacement=replacement, reason=reason))

        for name, directory, kind in jobs:
            path = A / name
            value = json.loads(path.read_text())
            assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
            assert value['fresh_faults'] == [] and value['boot_unchanged']
            assert not value.get('driver_dump_exists') and not value.get('driver_dump_entries')
            receipts[str(path)] = sha(path)
            for command in value['commands']:
                assert owned.closed(command) and command['exit_code'] == 0
                for old in command['owners'] + [command['direct_identity']]:
                    now = owned.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            root = B / directory
            saved = []
            for log in sorted((root / 'owned').rglob('*')):
                if not log.is_file() or log.suffix not in ('.stdout', '.stderr', '.log', '.jsonl'):
                    continue
                lines = log.read_text(errors='replace').splitlines()
                if log.stat().st_size < 4096:
                    useful = lines
                else:
                    prefixes = ('strata prefill:', 'strata MoE', 'strata serve: prompt',
                                'strata lm profile:', 'strata lm qsa profile:', 'strata prefill cpu ',
                                'strata mtp: prompt expert ', 'strata trace:', '-- cursor:')
                    useful = [line for line in lines if line.startswith(prefixes)]
                    useful += [line for line in lines if re.search(r'(?i)(warning:|error:|exception|failed)', line)]
                saved.append(dict(path=str(log), bytes=log.stat().st_size, sha256=sha(log), selected_lines=useful))
                select(log, str(excerpt_path), 'Closed successful text/API/protocol/build output; immutable receipt and reviewed excerpts retain useful evidence.')
            excerpts[str(path)] = dict(receipt_sha256=sha(path), original_status='PASS', logs=saved)
            duplicates = [(root / 'record.json', value)]
            if kind == 'model':
                duplicates += [(root / 'result.json', value['result'])]
                duplicates += [(p, value['benchmark_result']) for p in root.glob('stock-results/*/results.json')]
            for duplicate, expected in duplicates:
                if duplicate.exists():
                    assert json.loads(duplicate.read_text()) == expected, str(duplicate)
                    select(duplicate, str(path), 'Exact structured duplicate preserved in the original committed receipt.')
        excerpt_path.write_text(json.dumps(dict(schema='strata-moe-window10-model-excerpts-v1', jobs=excerpts),
                                           indent=2, ensure_ascii=False) + '\n')
        manifest = dict(schema='strata-moe-window10-model-retention-v1', complete=False,
                        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        original_receipt_hashes=receipts, excerpts=dict(path=str(excerpt_path), sha256=sha(excerpt_path)),
                        files=deletions, deleted=0, logical_bytes_removed=0,
                        scope='Only closed successful owned text and verified exact JSON duplicates. Keep engine/build products, libraries, wrappers, exclusive claims/start-exit proof, configs, profiles and all fixture/model inputs. No active run, code, original result, GPU or service changes.')

        def save():
            out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')

        save()
        try:
            for row in deletions:
                path = Path(row['path'])
                assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
                path.unlink()
                manifest['deleted'] += 1
                manifest['logical_bytes_removed'] += row['bytes']
                save()
            for path, expected in receipts.items():
                assert sha(path) == expected
            manifest['complete'] = True
        finally:
            save()
        print(json.dumps({k: manifest[k] for k in ('complete', 'deleted', 'logical_bytes_removed')}), flush=True)


if __name__ == '__main__':
    main()
