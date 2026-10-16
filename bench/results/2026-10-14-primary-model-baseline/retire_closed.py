#!/usr/bin/env python3
"""Retire duplicate closed logs only after exact evidence is committed."""
import fcntl
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def committed(path):
    data = subprocess.check_output(['git', '-C', str(P), 'show', 'HEAD:' + str(path.relative_to(P))])
    assert data == path.read_bytes(), str(path)
    return data


def alive_same(pid, ticks):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    fields = text[text.rfind(')') + 2:].split()
    return int(fields[19]) == ticks


def main():
    index = json.loads(committed(D / 'receipts-index.json'))
    committed(D / 'README.ja.md')
    committed(D / 'model-diagnostic-windows.json')
    committed(D / 'research-index.json')
    records = [x for x in index if x['original_path'].endswith('/record.json')]
    manifest = []
    kept = []
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for entry in records:
            archive = D / entry['path']
            data = gzip.decompress(committed(archive))
            assert sha(data) == entry['decoded_sha256']
            record = json.loads(data)
            assert not record['active']
            incomplete_observation = any(not command.get('observation_complete') for command in record['commands'])
            for command in record['commands']:
                assert command.get('direct_child_reaped') and command.get('session_empty') and not command.get('survivors')
                for owner in command.get('owners', []):
                    assert not alive_same(owner['pid'], owner['start_ticks']), owner
            root = Path(entry['original_path']).parent
            assert root.parent == B and root.name.startswith('main-primary-') and not root.is_symlink()
            paths = sorted((root / 'owned').glob('*.stdout')) + sorted((root / 'owned').glob('*.stderr'))
            if root.name in {'main-primary-cold-jit-short-root-v8', 'main-primary-cold-jit-short-root-v9'}:
                committed(D / 'cold-jit-evidence.json')
                paths += [path for name in ('native-maps.txt', 'thread-samples.jsonl', 'engine-stderr-tail.txt')
                          if (path := root / 'owned' / name).exists()]
            if root.name in {'main-primary-watchdog-short-root-v10', 'main-primary-watchdog-default-short-root-v11'}:
                committed(D / 'cold-jit-default-evidence.json')
                paths += [path for name in ('native-maps.txt', 'thread-samples.jsonl', 'engine-stderr-tail.txt')
                          if (path := root / 'owned' / name).exists()]
            for path in paths:
                assert path.is_file() and not path.is_symlink()
                data = path.read_bytes()
                if root.name == 'main-primary-model-accepted-diagnostic-root-v7' and path.name == 'engine.stderr':
                    kept.append(dict(path=str(path), bytes=len(data), sha256=sha(data), owner='Main',
                        consumer='one representative initial-prompt watchdog/host-stack failure; next current-compiler cold-JIT or queue-lifecycle diagnosis',
                        byte_budget=256 << 10, review_point='after host stack/cold-JIT countermeasure resolves or supersedes this distinct failure',
                        original_observation_complete=not incomplete_observation))
                    assert len(data) <= 256 << 10
                    continue
                manifest.append(dict(path=str(path), bytes=len(data), sha256=sha(data),
                    retained=[entry['path'], 'model-diagnostic-windows.json', 'cold-jit-evidence.json', 'cold-jit-default-evidence.json'],
                    reason='closed duplicate full API/build/progress log; exact receipts and compact mechanism evidence committed'))
            for item in index:
                source = Path(item['original_path'])
                if source.parent != root or not source.exists():
                    continue
                archive = D / item['path']
                decoded = gzip.decompress(committed(archive))
                assert source.read_bytes() == decoded
                manifest.append(dict(path=str(source), bytes=len(decoded), sha256=sha(decoded),
                    retained=[item['path']], reason='byte-identical closed receipt now canonical in Git'))
        research = json.loads((D / 'research-index.json').read_text())
        for report in research['reports']:
            source = Path(report['original_path'])
            if not source.exists():
                continue
            target = D / report['path']
            data = committed(target)
            assert source.read_bytes() == data and sha(data) == report['sha256']
            manifest.append(dict(path=str(source), bytes=len(data), sha256=sha(data), retained=[report['path']],
                reason='finished read-only report; exact original committed, use project registry for next round'))
        # Persist identities before unlink; never rewrite original pass/failure.
        output = D / 'deletion-manifest.jsonl.gz'
        previous = [json.loads(line) for line in gzip.decompress(output.read_bytes()).decode().splitlines()] if output.exists() else []
        previous_paths = {item['path']: item for item in previous}
        assert not any(item['path'] in previous_paths for item in manifest), 'retired path unexpectedly recreated'
        combined = previous + manifest
        raw = ''.join(json.dumps(item) + '\n' for item in combined).encode()
        output.write_bytes(gzip.compress(raw, mtime=0))
        assert gzip.decompress(output.read_bytes()) == raw
        for item in manifest:
            path = Path(item['path'])
            assert sha(path.read_bytes()) == item['sha256']
            path.unlink()
        result = dict(files=len(combined), logical_bytes=sum(x['bytes'] for x in combined),
            snapshots_or_actual_free_bytes_not_inferred=True,
            unchanged_original_result_status=True,
            kept_failure_diagnostics=kept,
            kept_build_consumer='next unchanged-Niko-client primary speed baseline and native function adaptation',
            build_budget_bytes=4 << 30,
            kept_observer_consumer='bounded next host-failure diagnosis, replace with host-abort observer or retire after recipe supersession',
            next_review='native model countermeasure or binary/recipe change')
        (D / 'retirement.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
