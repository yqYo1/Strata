#!/usr/bin/env python3
"""Retain a closed stock arm, preserving the original receipt and status."""
import fcntl
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path


B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    arm = sys.argv[1]
    assert arm in ['0', '1']
    root = B / ('main-primary-stock-compact' + arm + '-root-v1')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads((root / 'record.json').read_text())
        assert record['active'] is False
        assert all(x.get('observation_complete') and x.get('direct_child_reaped') and
                   x.get('session_empty') and not x.get('survivors')
                   for x in record['commands'])
        stock_files = list(root.glob('stock-results/*/results.json'))
        assert len(stock_files) <= 1
        dest = HERE / 'stock-evidence' / ('compact' + arm)
        assert not dest.exists()
        dest.mkdir(parents=True)
        paths = [root / name for name in ['record.json', 'result.json',
                                         'engine-start.json', 'engine-exit.json']]
        if stock_files:
            paths.append(stock_files[0])
        archives = []
        for source in paths:
            if not source.exists():
                continue
            data = source.read_bytes()
            # These are the specific owned, whitelisted-environment receipts,
            # never private environment dumps or unrelated contract records.
            assert b'"HF_TOKEN"' not in data and b'"HUGGING_FACE_HUB_TOKEN"' not in data
            label = 'stock-results.json' if source in stock_files else source.name
            archive = dest / (label + '.gz')
            packed = gzip.compress(data, mtime=0)
            archive.write_bytes(packed)
            assert gzip.decompress(archive.read_bytes()) == data
            archives.append(dict(original_path=str(source), path=str(archive.relative_to(HERE)),
                                 original_bytes=len(data), decoded_sha256=sha(data),
                                 archive_sha256=sha(packed)))
        samples = record.get('long_phase_samples', [])
        summary = {k: record.get(k) for k in ['active', 'passed', 'complete', 'error', 'compact',
            'source_revision', 'binary_sha256', 'build_receipt_sha256', 'source_diff_sha256',
            'prefill_median_tps', 'decode_median_tps', 'effective_allocation',
            'fresh_faults', 'driver_dump_exists', 'boot_unchanged', 'performance_qualified',
            'quality_qualified', 'physical_262144_qualified', 'adopted']}
        summary.update(long_phase_samples=samples, archives=archives,
            source_receipt_unmodified=True, cleanup_performed=False,
            recipe_sha256=sha(Path(__file__).read_bytes()),
            scope='One terminal stock arm; partial comparison, not default adoption or quality/full-input/RESTORE qualification.')
        if record.get('passed'):
            assert record['complete'] and len(samples) == 3
            assert not record['fresh_faults'] and not record['driver_dump_exists'] and record['boot_unchanged']
            assert all(x['ok'] and x['prompt_tokens'] == x['read_tokens'] == 32049 and
                       x['generated_tokens'] == 24 and x['reused_tokens'] == 0 for x in samples)
            summary['prefill_pooled_tps'] = 1000 * sum(x['read_tokens'] for x in samples) / sum(x['prompt_ms'] for x in samples)
            summary['decode_pooled_tps'] = 1000 * sum(x['generated_tokens'] for x in samples) / sum(x['decode_ms'] for x in samples)
            assert statistics.median(x['prompt_tps'] for x in samples) == record['prefill_median_tps']
        (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k: summary.get(k) for k in ['compact', 'passed', 'complete',
            'prefill_median_tps', 'decode_median_tps', 'prefill_pooled_tps', 'decode_pooled_tps']}), flush=True)


if __name__ == '__main__':
    main()
