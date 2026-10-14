#!/usr/bin/env python3
"""Retire represented, committed evidence only; keep active replay inputs."""
import fcntl
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
Q = Path(__file__).parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def committed(path):
    data = subprocess.check_output(['git', '-C', str(P), 'show', 'HEAD:' + str(path.relative_to(P))])
    assert data == path.read_bytes(), str(path)
    return data


def alive_same(pid, ticks):
    try:
        data = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return int(data[data.rfind(')') + 2:].split()[19]) == ticks


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = json.loads(committed(Q / 'receipts-index.json'))
        committed(Q / 'source-closure.json')
        committed(Q / 'stock-comparison.json')
        committed(Q / 'README.ja.md')
        committed(Q / 'export-failure.json')
        refusal = json.loads(committed(Q / 'qsa-test-refusal.json'))
        budget = json.loads(committed(Q / 'kv-log-budget.json'))
        assert budget['resolved_trace_question']
        assert not budget['post_budget_health']['fault_entries'] and not budget['post_budget_health']['driver_dump_exists']
        manifest = []
        for item in index:
            if not item['original_path'].endswith('/record.json'):
                continue
            decoded = gzip.decompress(committed(Q / item['path']))
            assert sha(decoded) == item['decoded_sha256']
            record = json.loads(decoded)
            assert not record['active']
            for cmd in record['commands']:
                assert cmd['direct_child_reaped'] and cmd['session_empty'] and not cmd.get('survivors')
                for identity in cmd.get('owners', []):
                    assert not alive_same(identity['pid'], identity['start_ticks'])
            root = Path(item['original_path']).parent
            assert root.parent == B and not root.is_symlink()
            if not record['passed'] and 'gpu-parity' in root.name:
                # Two classified harness failures have separate exact receipts and compact evidence.
                # Do not group other incomplete runs with them by exit code/signal alone.
                assert root.name in {'main-pre-Xe-1eb-gpu-parity-root-v1', 'main-pre-Xe-1eb-gpu-parity-root-v2'}
                if root.name.endswith('-v1'):
                    assert record['source_revision'] == refusal['source_revision']
                else:
                    assert 'aggregate owner log cap' in record['error']
            paths = list((root / 'owned').glob('*.stdout')) + list((root / 'owned').glob('*.stderr'))
            paths += [x for x in (root / 'owned').glob('*') if x.name in
                {'engine-stderr-tail.txt', 'native-maps.txt', 'thread-samples.jsonl'}]
            for path in paths:
                assert path.is_file() and not path.is_symlink()
                raw = path.read_bytes()
                manifest.append(dict(path=str(path), bytes=len(raw), sha256=sha(raw),
                    retained=[item['path'], 'source-closure.json', 'stock-comparison.json', 'export-failure.json'],
                    reason='Closed successful API/build/progress output, or resolved missing-source configure error; exact receipt and necessary evidence committed'))
            archive = root / 'source.tar'
            if archive.exists():
                raw = archive.read_bytes()
                assert sha(raw) == record['source_archive_sha256']
                manifest.append(dict(path=str(archive), bytes=len(raw), sha256=sha(raw),
                    retained=[item['path'], 'source-closure.json'],
                    reason='Duplicate Git archive; verified original-commit source manifest and frozen extracted tree remain the active replay inputs'))
            # These are reproducibility inputs to the pinned check_current_fp16_stock recipe.
            # Keep its short gate and historical build receipts until source replay decision.
            keep_gate = root.name in {'main-primary-fp16-prefill-short-root-v1',
                                     'main-pre-Xe-1eb-build-root-v1', 'main-pre-Xe-1eb-build-root-v2'}
            for entry in index:
                path = Path(entry['original_path'])
                if path.parent != root or not path.exists() or keep_gate and path.name == 'record.json':
                    continue
                raw = path.read_bytes()
                assert raw == gzip.decompress(committed(Q / entry['path']))
                manifest.append(dict(path=str(path), bytes=len(raw), sha256=sha(raw),
                    retained=[entry['path']], reason='Closed duplicate receipt, canonical byte-identical archive committed'))
            if record.get('stock_result_path'):
                path = Path(record['stock_result_path'])
                assert path.is_relative_to(root / 'stock-results')
                if path.exists():
                    raw = path.read_bytes()
                    assert sha(raw) == record['stock_result_sha256']
                    assert json.loads(raw) == record['stock_result']
                    manifest.append(dict(path=str(path), bytes=len(raw), sha256=sha(raw),
                        retained=[item['path'], 'stock-comparison.json'],
                        reason='Exact full structured stock results already inside committed original receipt'))
                # Human table duplicates the individual measurements in results.json/receipt.
                for path in (root / 'stock-results').glob('*/README.md'):
                    raw = path.read_bytes()
                    manifest.append(dict(path=str(path), bytes=len(raw), sha256=sha(raw),
                        retained=[item['path'], 'stock-comparison.json'], reason='Duplicate successful stock console table'))
        output = Q / 'deletion-manifest.jsonl.gz'
        previous = [json.loads(x) for x in gzip.decompress(output.read_bytes()).decode().splitlines()] if output.exists() else []
        old_paths = {x['path'] for x in previous}
        assert not any(x['path'] in old_paths for x in manifest)
        combined = previous + manifest
        data = ''.join(json.dumps(x) + '\n' for x in combined).encode()
        output.write_bytes(gzip.compress(data, mtime=0))
        assert gzip.decompress(output.read_bytes()) == data
        for item in manifest:
            path = Path(item['path'])
            assert sha(path.read_bytes()) == item['sha256']
            path.unlink()
        result = dict(files=len(combined), logical_bytes=sum(x['bytes'] for x in combined),
            actual_free_space_or_snapshot_release_not_inferred=True, original_status_unchanged=True,
            retained_replay_consumer='Frozen1eb parity and matched stock32K; current FP16 short gate of pinned comparison recipe',
            source_budget_bytes=64 << 20, build_budget_bytes=4 << 30,
            review_point='After source-component decision or superseded comparison fixtures')
        (Q / 'retirement.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
