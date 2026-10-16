#!/usr/bin/env python3
"""Offline, trace-local append/start relations; no CPU/GPU causality claim.
Repository LICENSE. The sole input is the closed canonical command table.
"""
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def intersect(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def main():
    deadline = time.monotonic() + 300
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = A / 'submission-gap-profile-v1.json'
        assert not target.exists()
        reference = A / 'hardware-resource-profile-v1.json'
        assert sha(reference) == 'ae8140e87bd4ebfa059f9864d6fbc7e8c390093adfd9c0de7dcc0295ec36c410'
        measured = json.loads(reference.read_text())
        sample = Path(measured['sample_pin']['path'])
        assert sha(sample) == measured['sample_pin']['sha256']
        table = json.loads(sample.read_text())
        assert table['columns'] == ['raw_line', 'name_index', 'thread', 'device', 'append_ns', 'submit_ns', 'start_ns', 'end_ns']
        sys.path.insert(0, str(A / 'own-vllm-resource-timeline-v3'))
        import timeline as t
        assert sha(Path(t.__file__)) == '2121482d114f1de1fb087a6bc1425314022af1198a3242986b571a2664ed2c12'
        window = measured['windows']['long_attention_envelope']['measurement']
        lo, hi = window['window_ns']
        rows = sorted((r for r in table['rows'] if r[6] < hi and r[7] > lo), key=lambda r: (r[6], r[7], r[0]))
        busy = t.union((max(lo, r[6]), min(hi, r[7])) for r in rows)
        gaps = t.complement(busy, [lo, hi])
        assert len(gaps) == window['command_gap_count'] and t.length(gaps) == window['no_observed_command_ns']
        # Include every observed command whose append-to-start interval intersects
        # the window, including a command that starts beyond its upper boundary.
        before_start = t.union((max(lo, r[4]), min(hi, r[6])) for r in table['rows'] if r[4] < hi and r[6] > lo)
        overlap = intersect(gaps, before_start)
        # Independent interval identity checks the intersection implementation.
        assert overlap == t.length(gaps) + t.length(before_start) - t.length(t.union(gaps + before_start))
        groups = defaultdict(lambda: dict(gap_count=0, gap_ns=0, next_append_after_gap_start_ns=0))
        representative = []
        cursor = 0
        for n, (start, end) in enumerate(gaps):
            if n % 10000 == 0:
                assert time.monotonic() < deadline
            while cursor < len(rows) and rows[cursor][6] < end:
                cursor += 1
            assert cursor < len(rows) and rows[cursor][6] == end
            row = rows[cursor]
            name = table['names'][row[1]]
            normalized = t.summary_name(name)
            value = groups[normalized]
            value['gap_count'] += 1
            value['gap_ns'] += end - start
            value['next_append_after_gap_start_ns'] += max(0, min(end, row[4]) - start)
            if end - start >= 80000:
                representative.append(dict(gap_start_ns=start, gap_end_ns=end,
                    next_timeline_line=row[0], next_kernel=name, next_thread=row[2],
                    next_append_ns=row[4], next_submit_ns=row[5], next_start_ns=row[6]))
        durations = sorted(b - a for a, b in gaps)
        assert sum(g['gap_ns'] for g in groups.values()) == sum(durations)
        assert time.monotonic() < deadline
        source = Path('/home/yayoi/ghq/github.com/intel/pti-gpu/tools/unitrace/src/levelzero/ze_collector.h')
        source_pin = dict(path=str(source), bytes=source.stat().st_size, sha256=sha(source))
        result = dict(schema='strata-submission-gap-profile-v1', GPU_executed=False,
            reference_sha256=sha(reference), canonical_samples=measured['sample_pin'],
            source_sha256=sha(Path(__file__)), source_clock_definition=source_pin,
            official_pti_commit='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033',
            source_lines='ze_collector.h:3505-3607,3713-3724,4828-4871,2745-2780',
            window_ns=[lo, hi], full_prefill_window=False, gap_count=len(gaps), gap_sum_ns=sum(durations),
            duration_order_statistics_ns={str(q): durations[(len(durations) - 1) * q // 100] for q in (0, 25, 50, 75, 90, 95, 99, 100)},
            gap_ns_with_any_observed_append_to_start_interval=overlap,
            gap_ns_without_any_observed_append_to_start_interval=sum(durations) - overlap,
            next_kernel_gap_groups=dict(sorted(groups.items(), key=lambda kv: kv[1]['gap_ns'], reverse=True)),
            gap_rows_at_least_80us=sorted(representative, key=lambda r: r['gap_end_ns'] - r['gap_start_ns'], reverse=True),
            all_individual_samples='Original canonical command table deterministically reconstructs every gap; no duplicate timeline retained.',
            interpretation=[
                'All times are within the same PTI-correlated trace; no external host CPU clock bridge.',
                'Append timestamp is sampled in the OnEnter callback before the original append API returns. Append-to-start means an observed API attempt interval, not proof of accepted or runnable queued work.',
                'Gaps are uncovered by observed timed commands, not proof the device is physically idle.',
                'Group attribution names the first command starting after a gap; it does not say that kernel caused the gap.',
                'These are instrumented durations; collector/provider/callback overhead may contribute. No clean speed or resource-utilization claim.',
                'Controlled reductions of submissions and counters are needed to identify host issue, driver dispatch, dependencies or other-device activity causally.'],
            numerical_or_performance_adoption=False)
        target.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(path=str(target), sha256=sha(target), gaps=len(gaps), gap_ns=sum(durations),
            with_append_interval_ns=overlap, without_append_interval_ns=sum(durations) - overlap,
            largest_groups=[dict(kernel=k, **v) for k, v in list(result['next_kernel_gap_groups'].items())[:8]])))


if __name__ == '__main__':
    main()
