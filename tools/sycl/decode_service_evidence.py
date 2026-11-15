"""Read the schema2 caller ledger; independently authored, repository LICENSE.

Native waits are nested within caller regions. Worker CPU and physical device
traffic require separate evidence; these values do not establish saturation.
Only complete, clean, exactly conserved model records are accepted here.
"""
import re

STEM = 'strata decode service'
PREFIX = STEM + ': '
REGIONS = ('capture', 'stage', 'layer_wait', 'pool', 'copy_retire', 'submit',
    'final_compute_wait', 'final_copy_wait', 'sampling_wait')
COUNTER_FIELDS = ('calls', 'throws', 'wall_ns', 'cpu_ns')
WAIT_FIELDS = tuple('wait_' + key for key in COUNTER_FIELDS)
FIELDS = ('version', 'request', 'available', 'complete', 'open_parts', 'open_waits',
    'wall_ns', 'cpu_ns', 'unclassified_wall_ns', 'unclassified_cpu_ns') + tuple(
        region + '_' + key for region in REGIONS for key in (*COUNTER_FIELDS, *WAIT_FIELDS)
    ) + tuple('unclassified_' + key for key in WAIT_FIELDS)
U64_MAX = (1 << 64) - 1


def require(condition, message):
    if not condition:
        raise RuntimeError('decode service evidence: ' + message)


def _counter(row, stem):
    calls, throws, wall, cpu = (row[stem + '_' + key] for key in COUNTER_FIELDS)
    require(throws <= calls, 'exception/count mismatch in ' + stem)
    require(throws == 0, 'model region observed an exception: ' + stem)
    require(calls != 0 or (wall == 0 and cpu == 0), 'time without calls in ' + stem)


def parse_line(line):
    require(line.startswith(PREFIX) and '\n' not in line and '\r' not in line
        and len(line) <= 16384, 'malformed or oversized record')
    row = {}
    for token in line[len(PREFIX):].split(' '):
        match = re.fullmatch(r'([a-z][a-z0-9_]*)=(0|[1-9][0-9]*)', token)
        require(match is not None, 'noncanonical field')
        key, number = match.groups()
        require(key not in row and len(number) <= 20, 'duplicate field or oversized integer')
        value = int(number)
        require(value <= U64_MAX, 'unsigned counter overflow')
        row[key] = value
    require(set(row) == set(FIELDS), 'missing or unknown schema2 fields')
    require(row['version'] == 2 and row['request'] > 0 and row['available'] == 1
        and row['complete'] == 1 and row['open_parts'] == row['open_waits'] == 0
        and row['wall_ns'] > 0, 'unavailable, incomplete or open accounting')
    for region in REGIONS:
        _counter(row, region)
        _counter(row, region + '_wait')
        require(row[region + '_calls'] != 0 or row[region + '_wait_calls'] == 0,
                'native wait without its containing caller region')
        for clock in ('wall_ns', 'cpu_ns'):
            require(row[region + '_wait_' + clock] <= row[region + '_' + clock],
                    'native wait exceeds containing region: ' + region)
    _counter(row, 'unclassified_wait')
    for clock in ('wall_ns', 'cpu_ns'):
        require(sum(row[r + '_' + clock] for r in REGIONS)
            + row['unclassified_' + clock] == row[clock], 'caller partition is not conserved')
        require(row['unclassified_wait_' + clock] <= row['unclassified_' + clock],
                'unclassified native wait exceeds its caller remainder')
        require(sum(row[r + '_wait_' + clock] for r in (*REGIONS, 'unclassified')) <= row[clock],
                'nested native waits exceed the request interval')
    return row


def extract(text, expected_requests):
    """Join exact request IDs; an empty list requires observation to be absent."""
    require(type(expected_requests) in (list, tuple) and all(type(n) is int
        and 0 < n <= U64_MAX for n in expected_requests)
        and len(set(expected_requests)) == len(expected_requests), 'invalid expected request IDs')
    rows = []
    for line in text.splitlines():
        partial = len(line) >= len('strata decode') and PREFIX.startswith(line)
        if STEM in line or partial:
            rows.append(parse_line(line))
    require([row['request'] for row in rows] == list(expected_requests),
            'missing, repeated, reordered or unexpected request records')
    return rows


def compact(row):
    """Retain useful region totals; omit zero exception/open-state fields."""
    waits = (*REGIONS, 'unclassified')
    return dict(request=row['request'], wall_ns=row['wall_ns'], caller_cpu_ns=row['cpu_ns'],
        accounting_passed=True,
        native_wait_wall_ns=sum(row[r + '_wait_wall_ns'] for r in waits),
        native_wait_caller_cpu_ns=sum(row[r + '_wait_cpu_ns'] for r in waits),
        unclassified=dict(wall_ns=row['unclassified_wall_ns'],
            caller_cpu_ns=row['unclassified_cpu_ns'],
            native_wait_calls=row['unclassified_wait_calls'],
            native_wait_wall_ns=row['unclassified_wait_wall_ns'],
            native_wait_caller_cpu_ns=row['unclassified_wait_cpu_ns']),
        regions={r: dict(calls=row[r + '_calls'], wall_ns=row[r + '_wall_ns'],
            caller_cpu_ns=row[r + '_cpu_ns'], native_wait_calls=row[r + '_wait_calls'],
            native_wait_wall_ns=row[r + '_wait_wall_ns'],
            native_wait_caller_cpu_ns=row[r + '_wait_cpu_ns']) for r in REGIONS})
