"""Bounded production profiling probe evidence parser; no GPU operations."""
import math
import re


def require(value, message):
    if not value:
        raise ValueError(message)


def fields(text):
    result = {}
    for term in text.split():
        key, value = term.split('=', 1)
        require(key not in result, 'duplicate field ' + key)
        result[key] = value
    return result


def validate(stdout, stderr, mode):
    lines = stdout.splitlines()
    headers = [x for x in lines if x.startswith('PROBE ')]
    require(len(headers) == 1 and headers[0].startswith('PROBE mode=' + mode + ' STRATA_NO_BLAS=' + ('1' if mode == 'free' else '0') + ' samples=8 '), 'explicit diagnostic mode and samples')
    controls = [x for x in lines if x.startswith('CONTROL ')]
    expected_controls = [
        'CONTROL name=unrecorded pair result=0 sentinel_unchanged=1 error=event_elapsed_ms: an event was not recorded',
        'CONTROL name=completed non-timing pair result=0 sentinel_unchanged=1 error=event_elapsed_ms: an event was not created for timing']
    require(controls == expected_controls, 'both exact negative controls')
    begins = [x for x in lines if x.startswith('SAMPLE_BEGIN ')]
    require(begins == ['SAMPLE_BEGIN index=%d planned_copies=2 planned_transfer_bytes=33554432' % i for i in range(8)], 'finite planned ledger')
    samples = []
    expected_api = [
        'API operation=unrecorded pair result=0 error=event_elapsed_ms: an event was not recorded',
        'API operation=completed non-timing pair result=0 error=event_elapsed_ms: an event was not created for timing']
    for line in lines:
        if not line.startswith('SAMPLE '):
            continue
        head, tail = line[7:].split(' reversed_error=', 1)
        reverse, rest = tail.split(' same_result=', 1)
        values = fields(head + ' same_result=' + rest)
        index = len(samples)
        required = dict(index=index, copies=2, transfer_bytes=33554432, records=2, compute_waits=1, elapsed_queries=3, api_calls=8, checked_words=4194304, normal_result=1, reversed_result=0, sentinel_unchanged=1, raw_endpoints_available=1, same_result=1, plausible=1)
        for key, number in required.items():
            require(int(values[key]) == number, 'sample exact field ' + key)
        require(values['normal_error'] == 'none' and values['same_error'] == 'none' and float(values['same_ms']) == 0, 'successful and equal queries')
        start, end = int(values['raw_start_ns']), int(values['raw_end_ns'])
        require(0 < start < end, 'ordered distinct completed raw endpoints')
        expected_reverse = 'event_elapsed_ms: end < start (start=%d, end=%d)' % (end, start)
        require(reverse == expected_reverse, 'actual reverse diagnostic preserves raw endpoints')
        elapsed, wall = float(values['elapsed_ms']), float(values['host_submit_to_completion_wall_ms'])
        require(math.isfinite(elapsed) and math.isfinite(wall) and elapsed > 0 and wall > 0 and elapsed <= wall * 1.25 + 10, 'positive finite gross wall sanity')
        # This checks the returned interval's unit conversion only; raw tags do
        # not establish calibration, timer domain or exclusive DMA execution.
        raw_ms = (end - start) * 1e-6
        require(abs(elapsed - raw_ms) <= max(2e-7, raw_ms * 1e-6), 'returned interval agrees with ordered raw tags')
        values['reversed_error'] = reverse
        samples.append(values)
        expected_api.append('API operation=sample reversed event_elapsed_ms result=0 error=' + reverse)
    require(len(samples) == 8, 'all eight complete samples')
    totals = [x for x in lines if x.startswith('RESULT ')]
    require(len(totals) == 1, 'single terminal result')
    expected_total = dict(status='PASS', samples='8', planned_samples='8', positive_reversals='8', api_calls='75', records='18', copies='16', compute_waits='9', cleanup_waits='1', elapsed_queries='26', checked_words='33554432')
    require(fields(totals[0][7:]) == expected_total, 'exact terminal API/word ledger')
    actual_api = [x for x in stderr.splitlines() if x.startswith('API ')]
    require(actual_api == expected_api, 'only ten explicitly expected API refusals')
    require(not re.search(r'^FAIL|^DATA |exception=|UR_RESULT_ERROR|ZE_RESULT_ERROR', stderr, re.M), 'no unclassified API/runtime/data failure')
    diagnostic_lines = [x for x in lines if x.startswith(('PROBE ', 'CONTROL ', 'SAMPLE_BEGIN ', 'SAMPLE ', 'RESULT '))]
    require(len(diagnostic_lines) == 20, 'finite exact diagnostic line count')
    return dict(samples=samples, controls=controls, totals=expected_total, expected_false_api=actual_api, sample_count=8, complete_word_comparison=True, raw_endpoint_unit_conversion=True, scope='queue timeline only; not exclusive kernel/DMA service or timestamp-domain calibration')
