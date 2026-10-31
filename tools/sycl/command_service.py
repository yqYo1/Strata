"""Validate sparse GPU service samples, including actual fused engine hooks.

Repository LICENSE. This reader establishes sample conservation and clocks,
not physical traffic, saturation, a critical path or unobserved operator work.
"""
from collections import defaultdict
from measurement_guards import require

PREFIX = 'strata command '
JOB = {'gather', 'gu', 'swiglu', 'down', 'accumulate', 'gu_swiglu', 'down_scatter'}
KINDS = JOB | {'upload', 'expert_upload', 'dequant_gu', 'dequant_down', 'combine', 'write', 'job_weights', 'status'}


def row_bin(rows, version):
    if version == 3 and rows >= 2334:
        return 8
    if rows < 16:
        return 0
    value = 1
    while value < 7 and rows >= 16 << value:
        value += 1
    return value


def fields(line, prefix):
    pairs = [p.split('=', 1) for p in line[len(prefix):].split()]
    require(pairs and all(len(p) == 2 for p in pairs) and len({p[0] for p in pairs}) == len(pairs),
            'malformed/duplicate command fields')
    return dict(pairs)


def validate(rows, expected_windows):
    require(1 <= expected_windows <= 4, 'invalid expected window count')
    summary = rows['profile'][0]
    version = int(summary['version'])
    require(version in (2, 3), 'unsupported command profile version')
    cap = {2: 192, 3: 224}[version]
    require(summary['service_complete'] == '1' and summary['error'] == 'none' and
            summary['invalid'] == summary['invalid_saved'] == '0' and not rows['invalid'],
            'incomplete command service profile: ' + summary['error'])
    require(int(summary['windows']) == int(summary['expected']) == expected_windows and
            int(summary['cap']) == cap and 0 < int(summary['peak_events']) <= cap and
            summary['scope'] == 'selected_first_windows' and summary['hardware_busy'] == 'UNMEASURED',
            'command window/cap/scope differs')
    resolution = int(summary['timer_resolution_ns'])
    require(resolution > 0 and 0 <= int(summary['max_query_wall_ns']) < 1000000000,
            'command profiling resolution/query bound')
    samples, groups = rows['sample'], rows['group']
    require(0 < len(samples) == int(summary['records']) <= expected_windows * cap,
            'command record count/cap differs')
    keys = [(int(g['window']), g['kind'], int(g['bin'])) for g in groups]
    require(len(set(keys)) == len(keys) and {k[0] for k in keys} == set(range(expected_windows)),
            'duplicate/missing command stratum/window')
    by_group = defaultdict(list)
    last_end = {}
    for sample in samples:
        window, kind, bin_ = int(sample['window']), sample['kind'], int(sample['bin'])
        require(0 <= window < expected_windows and kind in KINDS and 0 <= bin_ < (9 if version == 3 else 8),
                'unknown command sample/window/bin')
        if kind in JOB:
            m = int(sample['m'])
            require(1 <= m <= 8192 and row_bin(m, version) == bin_, 'command shape/bin mismatch')
        else:
            require(bin_ == 0, 'non-job command bin differs')
        if kind in {'gu_swiglu', 'down_scatter'}:
            require(version == 3 and int(sample['m']) >= 2334, 'fused sample version/threshold differs')
        submit, start, end = (int(sample[k]) for k in ('submit', 'start', 'end'))
        require(min(submit, start, end) >= 0 and end > start and end - start >= resolution and
                start >= last_end.get(window, 0), 'unordered or invalid command GPU clocks')
        last_end[window] = end
        by_group[(window, kind, bin_)].append((end - start, start - submit))
    require(set(by_group) == set(keys), 'command sample/group identities differ')
    for key, group in zip(keys, groups):
        require(group['kind'] in KINDS and int(group['layer']) == (0, 15, 31, 47)[key[0]] and
                int(group['first_row']) == 0, 'command group layer/scope differs')
        values = by_group[key]
        eligible, selected, service, delay = (int(group[k]) for k in
            ('eligible', 'selected', 'service_samples', 'queue_delay_samples'))
        require(eligible >= selected == service == len(values) > 0 and
                delay == sum(q >= 0 for _, q in values) and
                int(group['sample_service_ns']) == sum(s for s, _ in values) and
                int(group['sample_queue_delay_ns']) == sum(q for _, q in values if q >= 0),
                'command stratum count/service conservation differs')
    inversions = sum(q < 0 for values in by_group.values() for _, q in values)
    require(int(summary['submit_clock_inversions']) == inversions and
            int(summary['submit_clock_qualified']) == int(not inversions), 'command submit clock qualification differs')
    compact = []
    for sample in samples:
        sample = dict(sample)
        submit, start, end = (int(sample.pop(k)) for k in ('submit', 'start', 'end'))
        compact.append(dict(sample, service_ns=end - start, start_minus_submit_ns=start - submit))
    return dict(version=version, summaries=[summary], groups=groups, samples=compact, invalid=[],
                scope='Selected first windows only;sampled GPU service is not device saturation or a phase critical path.')


def parse_profiles(text, expected_windows=4):
    rows = {k: [] for k in ('sample', 'group', 'invalid', 'profile')}
    result = []
    for line in text.splitlines():
        if not line.startswith(PREFIX):
            continue
        kind, sep, _ = line[len(PREFIX):].partition(': ')
        require(sep and kind in rows, 'unknown command profile record')
        rows[kind].append(fields(line, PREFIX + kind + ': '))
        if kind == 'profile':
            result.append(validate(rows, expected_windows))
            rows = {k: [] for k in rows}
    require(result and not any(rows.values()), 'missing command profile footer')
    return result
