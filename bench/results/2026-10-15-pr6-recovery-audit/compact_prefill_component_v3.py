#!/usr/bin/env python3
"""Verify and compact per-operation flow categories; retain raw trace for analysis."""
import collections
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def bucket(category):
    if category and category.startswith('Flow_H2D_'):
        return 'Flow_H2D'
    if category and category.startswith('Flow_D2H_'):
        return 'Flow_D2H'
    return category


def main():
    module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = A/'prefill-window-profile-component-v3.json'
        raw_copy = B/'main-prefill-window-profile-component-root-v3/record.json'
        target = A/'prefill-window-profile-component-v3-compact.json'
        manifest = A/'prefill-window-component-v3-retention-v1.json'
        require(not target.exists() and not manifest.exists(), 'fresh compact evidence')
        digest, size = sha(original), original.stat().st_size
        require(digest == sha(raw_copy), 'byte-identical retained original receipt')
        d = json.loads(original.read_text())
        require(d['passed'] and d['complete'] and not d['active'] and not d['fresh_faults'] and
                not d['driver_dump_exists'] and d['boot_unchanged'] and d['session_released'], 'actual component qualification')
        for c in d['commands']+d['control_commands']:
            require(module.closed(c), 'original command not closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'component still active')
        trace = Path(d['native_trace']['path'])
        require(sha(trace) == d['native_trace']['sha256'], 'complete trace identity')
        counts = collections.Counter()
        for row in d['native_trace']['event_counts']:
            counts[(row['phase'], bucket(row['category']))] += row['count']
        require(sum(counts.values()) == d['native_trace']['events'] and len(counts) <= 16,
                'event total conserved in compact summary')
        d['native_trace']['event_counts'] = [dict(phase=k[0], category=k[1], count=v) for k, v in counts.items()]
        d['original_receipt'] = dict(sha256=digest, bytes=size, retained_path=str(raw_copy),
            status_unchanged=True, compaction='Collapse per-command flow categories; raw trace retains causal IDs for Main analysis.')
        with target.open('x') as f:
            json.dump(d, f, indent=2)
            f.write('\n')
        require(target.stat().st_size < 1 << 20, 'compact receipt byte budget')
        m = dict(GPU_executed=False, recipe_sha256=sha(Path(__file__)),
            original_status=dict(passed=True, complete=True, active=False),
            deleted=[dict(path=str(original), bytes=size, sha256=digest,
                reason='Duplicate 40MB per-operation category receipt; exact original remains in private B and compact Git receipt preserves totals, samples and ownership.')],
            replacements=[dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target))],
            retained=[dict(path=str(raw_copy), bytes=raw_copy.stat().st_size, sha256=sha(raw_copy),
                consumer='Main long-profile/trace summary crosscheck', next_review='After bounded long trace and summary close; retire redundant operation-category receipt.'),
                dict(path=str(trace), bytes=trace.stat().st_size, sha256=sha(trace),
                consumer='Main interval/correlation analysis', next_review='After verified component/long interval summaries')])
        # One exact duplicate only; no active run, raw trace, fixture or status changes.
        require(sha(original) == digest and sha(raw_copy) == digest, 'deletion identities')
        original.unlink()
        with manifest.open('x') as f:
            json.dump(m, f, indent=2)
            f.write('\n')
        print(json.dumps(dict(passed=True, events=d['native_trace']['events'],
            compact_groups=len(counts), compact_bytes=target.stat().st_size,
            duplicate_receipt_bytes_retired=size)), flush=True)


if __name__ == '__main__':
    main()
