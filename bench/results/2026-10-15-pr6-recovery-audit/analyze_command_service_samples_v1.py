#!/usr/bin/env python3
"""Validate actual event/group receipts; describe bounded sampled command service."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def fields(line, prefix):
    assert line.startswith(prefix)
    return dict(item.split('=', 1) for item in line[len(prefix):].split())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(path):
    path = Path(path).resolve()
    receipt = json.loads(path.read_text())
    assert receipt['passed'] and receipt['complete'] and not receipt['active']
    assert receipt['ownership_closed'] and receipt['fresh_faults'] == []
    assert not receipt.get('driver_dump_exists', False)
    assert not receipt.get('driver_dump_entries', [])
    summary = fields(receipt['command_profile_summary'], 'strata command profile: ')
    assert summary['complete'] == '1' and summary['error'] == 'none'
    assert summary['scope'] == 'selected_first_windows'
    assert summary['hardware_busy'] == 'UNMEASURED'
    assert int(summary['windows']) == int(summary['expected'])
    assert 0 < int(summary['peak_events']) <= int(summary['cap']) == 160
    resolution = int(summary['timer_resolution_ns'])
    assert resolution > 0 and int(summary['max_query_wall_ns']) < 1_000_000_000
    samples = []
    grouped = defaultdict(list)
    for line in receipt['command_profile_records']:
        row = fields(line, 'strata command sample: ')
        row = {key: (value if key == 'kind' else int(value)) for key, value in row.items()}
        assert 0 <= row['window'] < int(summary['windows'])
        assert row['submit'] <= row['start'] < row['end']
        row['service_ns'] = row['end'] - row['start']
        row['queue_delay_ns'] = row['start'] - row['submit']
        assert row['service_ns'] >= resolution
        assert 0 <= row['bin'] < 8
        samples.append(row)
        grouped[row['window'], row['kind'], row['bin']].append(row)
    assert len(samples) == int(summary['records']) <= 640
    groups = []
    seen = set()
    by_window = defaultdict(lambda: defaultdict(float))
    for line in receipt['command_profile_groups']:
        group = fields(line, 'strata command group: ')
        group = {key: (value if key == 'kind' else int(value)) for key, value in group.items()}
        key = group['window'], group['kind'], group['bin']
        assert key not in seen
        seen.add(key)
        rows = grouped[key]
        assert 0 < len(rows) == group['selected'] <= group['eligible']
        assert sum(row['service_ns'] for row in rows) == group['sample_service_ns']
        assert sum(row['queue_delay_ns'] for row in rows) == group['sample_queue_delay_ns']
        values = [row['service_ns'] for row in rows]
        expanded = sum(values) * group['eligible'] / group['selected']
        group.update(
            sampled_service_ms=sum(values) / 1e6,
            expanded_service_ms_estimate=expanded / 1e6,
            sample_service_ns_min=min(values),
            sample_service_ns_median=statistics.median(values),
            sample_service_ns_max=max(values),
            sampled_rows_min=min(row['m'] for row in rows),
            sampled_rows_max=max(row['m'] for row in rows),
            observed_formats=sorted({row['format'] for row in rows}),
        )
        by_window[group['window']][group['kind']] += expanded / 1e6
        groups.append(group)
    assert seen == set(grouped)
    windows = []
    # These edges occur inside one selected job, without another job in
    # between, in the pinned moe_window.inc path. Never use down->gather:
    # unselected jobs can be present between those retained events.
    job_edges = {('gather', 'gu'), ('gu', 'gu'), ('gu', 'swiglu'),
                 ('swiglu', 'down'), ('down', 'down')}
    for window in sorted(by_window):
        own = [g for g in groups if g['window'] == window]
        layers = {g['layer'] for g in own}
        assert len(layers) == 1 and all(g['first_row'] == 0 for g in own)
        events = [r for r in samples if r['window'] == window]
        first_upload = next(r for r in events if r['kind'] == 'upload')
        last_write = next(r for r in reversed(events) if r['kind'] == 'write')
        span = last_write['end'] - first_upload['start']
        assert span > 0
        edges = []
        for previous, following in zip(events, events[1:]):
            if ((previous['kind'], following['kind']) not in job_edges or
                    any(previous[k] != following[k] for k in ('expert', 'm', 'bin'))):
                continue
            edges.append(dict(
                previous=previous['kind'], following=following['kind'],
                gap_ns=following['start'] - previous['end'],
                late_submit_ns=max(0, following['submit'] - previous['end']),
            ))
        gaps = [e['gap_ns'] for e in edges]
        late = [e['late_submit_ns'] for e in edges]
        neighbors = dict(
            count=len(edges),
            gap_ns_sum=sum(gaps),
            gap_ns_min=min(gaps, default=0),
            gap_ns_median=statistics.median(gaps) if gaps else 0,
            gap_ns_max=max(gaps, default=0),
            negative_gap_count=sum(g < 0 for g in gaps),
            late_submit_ns_sum=sum(late),
            late_submit_ns_median=statistics.median(late) if late else 0,
            late_submit_ns_max=max(late, default=0),
            edges_by_kind={f'{a}->{b}': sum(e['previous'] == a and e['following'] == b for e in edges)
                           for a, b in sorted(job_edges)},
        )
        windows.append(dict(window=window, layer=layers.pop(), first_row=0,
                            sampled_commands=sum(g['selected'] for g in own),
                            eligible_tagged_commands=sum(g['eligible'] for g in own),
                            first_upload_start_to_last_write_end_ms=span / 1e6,
                            known_within_job_neighbor_intervals=neighbors,
                            expanded_service_ms_estimate_by_kind=dict(by_window[window])))
    return dict(
        schema='strata-command-service-sample-analysis-v1',
        source_receipt=dict(path=str(path), sha256=sha(path)),
        analyzer_sha256=sha(__file__), validated=True,
        summary=summary, sample_count=len(samples), group_count=len(groups),
        windows=windows, groups=groups,
        interpretation={
            'measured': 'Backend command end-start and start-submit for retained actual events only.',
            'expanded_estimate': 'Per kind/row-bin sample mean times eligible tagged commands, in the selected first windows only.',
            'selection': 'Four deterministic rank-selected jobs per row bin and four experts, not a random-sample confidence guarantee.',
            'limitations': [
                'Expanded totals assume selected commands represent their stratum; quantization formats and shapes may differ within it.',
                'Queue delay is neither host CPU time nor GPU arithmetic time.',
                'No CPU/device clock subtraction, hardware busy fraction, PCIe saturation, or full-model extrapolation.',
                'Gaps between sampled events may contain unselected commands; they are not evidence of idle time.',
                'Within-job neighbor intervals use only the listed adjacent tag edges and matching expert/M/bin in the pinned source; down-to-gather is excluded.',
                'Late-submit intervals indicate delay before the next command submission in this queue/backend timebase; they do not identify CPU work, runtime work, or global GPU idleness.',
                'The window span excludes router computation, router readback/grouping, layer uploads and other model phases.',
                'Synthetic fixture dimensions are tags for validation, not actual GEMM throughput.',
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.receipt)
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(validated=True, samples=result['sample_count'],
                          groups=result['group_count'], output=str(args.output))))


if __name__ == '__main__':
    main()
