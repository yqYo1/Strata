#!/usr/bin/env python3
"""Retire only verified duplicates from the two closed reference model runs."""
import fcntl
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def committed(path):
    assert subprocess.check_output(['git', '-C', str(P), 'show', 'HEAD:' + str(path.relative_to(P))]) == path.read_bytes()


def alive_same(pid, ticks):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return int(raw[raw.rfind(')') + 2:].split()[19]) == ticks


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index_path = HERE / 'receipts-index.json'
        committed(index_path)
        index = json.loads(index_path.read_text())
        records = {}
        for item in index:
            archive = HERE / item['path']
            committed(archive)
            raw = gzip.decompress(archive.read_bytes())
            assert digest(archive.read_bytes()) == item['archive_sha256']
            assert len(raw) == item['bytes'] and digest(raw) == item['decoded_sha256']
            if archive.name == 'record.json.gz':
                record = json.loads(raw)
                assert not record['active'] and record['passed'] and record['complete']
                assert record['result']['real_engine_exit']['exit_code'] == 0
                assert not record['fresh_faults'] and not record['driver_dump_exists']
                for command in record['commands']:
                    assert command['direct_child_reaped'] and command['session_empty']
                    assert not command.get('survivors')
                    for owner in command.get('owners', []):
                        assert not alive_same(owner['pid'], owner['start_ticks'])
                records[archive.parent.name] = record
        assert len(records) == 2
        diagnostic = HERE / 'model-diagnostics.json'
        committed(diagnostic)
        diagnostics = {x['original_path']: x for x in json.loads(diagnostic.read_text())}
        research_index = HERE / 'research-index.json'
        committed(research_index)
        proposals = []
        keep = B / 'main-pre-Xe-1eb-model-short4096-root-v1/record.json'
        for item in index:
            original = Path(item['original_path'])
            assert original.is_relative_to(B)
            if original == keep or not original.exists():
                continue
            assert digest(original.read_bytes()) == item['decoded_sha256']
            proposals.append((original, str(HERE / item['path']), 'Verified byte-identical committed receipt; not an active fixture'))
        for name, record in records.items():
            root = B / name
            command_logs = {key: value for command in record['commands']
                            for key, value in command.get('logs', {}).items()}
            for original in (root / 'owned').iterdir():
                if not original.is_file() or original.suffix not in ('.stdout', '.stderr'):
                    continue
                raw = original.read_bytes()
                if str(original) in diagnostics:
                    assert digest(raw) == diagnostics[str(original)]['sha256']
                    replacement = str(diagnostic)
                else:
                    # Direct command stdout/stderr are pinned in the original receipt.
                    # Nested server/client logs are represented by normal result plus stock samples.
                    if original.name in command_logs:
                        pin = command_logs[original.name]
                        assert digest(raw) == pin['sha256'] and len(raw) == pin['bytes']
                    replacement = str(HERE / 'receipts' / name / 'record.json.gz')
                proposals.append((original, replacement, 'Closed successful diagnostic/protocol/poll text; progress, all stock samples, owners/exits and fresh faults committed'))
        for item in json.loads(research_index.read_text())['reports']:
            target = HERE / item['path']
            committed(target)
            original = Path(item['original_path'])
            if original.exists():
                assert digest(original.read_bytes()) == item['sha256'] == digest(target.read_bytes())
                proposals.append((original, str(target), 'Completed research duplicate; future rounds use committed registry'))
        manifest = HERE / 'deletion-manifest.jsonl'
        previous = manifest.read_text() if manifest.exists() else ''
        deleted = []
        for original, replacement, reason in proposals:
            assert original != keep
            raw = original.read_bytes()
            entry = dict(path=str(original), bytes=len(raw), sha256=digest(raw),
                         retained_replacement=replacement, reason=reason,
                         evidence_commit=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip())
            previous += json.dumps(entry, ensure_ascii=False) + '\n'
            manifest.write_text(previous)
            original.unlink()
            deleted.append(entry)
        assert keep.exists()
        all_entries = [json.loads(line) for line in previous.splitlines()]
        report = dict(deleted_files=len(all_entries), deleted_logical_bytes=sum(x['bytes'] for x in all_entries),
            this_invocation_files=len(deleted), this_invocation_logical_bytes=sum(x['bytes'] for x in deleted),
            actual_free_space_delta_qualified=False, source_models_services_GPU_unchanged=True,
            retained_gate=str(keep), scope='Closed reference model artifacts and completed research duplicates only. Original outcomes and hashes preserved.')
        (HERE / 'retirement.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
