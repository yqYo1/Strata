"""Read verifier graph envelopes with separate validity for estimated submit clocks."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from decode_segment_clocks import compare_windows, layer_kind, window_walls

PREFIX = 'strata decode segment: '
FIELDS = ('request', 'window', 'pos', 'qsa_interval', 'segment', 'status', 'fields',
          'submit', 'start', 'end', 'resolution_ns', 'host_wait_ns', 'pool_callback_ns')
WINDOWS = 4
SEGMENTS = 49


def segment_values(line):
    pairs = [part.split('=', 1) for part in line[len(PREFIX):].split()]
    if any(len(pair) != 2 for pair in pairs) or tuple(pair[0] for pair in pairs) != FIELDS:
        raise ValueError('malformed verifier segment')
    return {name: int(value) for name, value in pairs}


def summarize(lines, request):
    result = []
    previous_end = 0
    count = 0
    for line in lines:
        if not line.startswith(PREFIX):
            continue
        sample = segment_values(line)
        window, segment = divmod(count, SEGMENTS)
        if window >= WINDOWS or (sample['request'], sample['window'], sample['segment']) != (
                request, 9 + window, segment):
            raise ValueError('missing, repeated, foreign or out-of-order verifier segment')
        submit, start, end = (sample[name] for name in ('submit', 'start', 'end'))
        if (sample['status'] or sample['fields'] != 7 or sample['qsa_interval'] != 4 or
                not 0 < start < end or start < previous_end or
                not 0 < sample['resolution_ns'] <= end - start or
                min(submit, sample['host_wait_ns'], sample['pool_callback_ns'], sample['pos']) < 0):
            raise ValueError('incomplete or invalid verifier start/end evidence')
        if segment == 0:
            if result and sample['pos'] <= result[-1]['position']:
                raise ValueError('sampled position did not advance')
            result.append(dict(position=sample['pos'], first_graph_start_ns=start,
                               boundaries={}, submit_clock_exceptions=[]))
        current = result[-1]
        if sample['pos'] != current['position'] or (segment == SEGMENTS - 1 and sample['pool_callback_ns']):
            raise ValueError('segment position or final callback differs')

        previous = layer_kind(segment - 1) if segment else 'input'
        following = layer_kind(segment) if segment < SEGMENTS - 1 else 'head'
        bucket = current['boundaries'].setdefault(f'{previous}_to_{following}', dict(
            segments=0, graph_ns=0, inter_graph_ns=0, queue_delay_sum_ns=0,
            queue_delay_samples=0, host_wait_ns=0, pool_callback_ns=0))
        bucket['segments'] += 1
        bucket['graph_ns'] += end - start
        if segment:
            bucket['inter_graph_ns'] += start - previous_end
        if 0 < submit <= start:
            bucket['queue_delay_sum_ns'] += start - submit
            bucket['queue_delay_samples'] += 1
        else:
            current['submit_clock_exceptions'].append(dict(segment=segment, submit=submit, start=start))
        for field in ('host_wait_ns', 'pool_callback_ns'):
            bucket[field] += sample[field]
        previous_end = end
        count += 1

        if segment == SEGMENTS - 1:
            boundaries = current['boundaries'].values()
            current['graph_ns'] = sum(bucket['graph_ns'] for bucket in boundaries)
            current['inter_graph_ns'] = sum(bucket['inter_graph_ns'] for bucket in boundaries)
            current['span_ns'] = end - current.pop('first_graph_start_ns')
            if current['span_ns'] != current['graph_ns'] + current['inter_graph_ns']:
                raise ValueError('the layer graph intervals do not account for the sampled span')
            eligible = sum(bucket['queue_delay_samples'] for bucket in boundaries)
            current['queue_delay_samples'] = eligible
            current['queue_delay_ns'] = (sum(bucket['queue_delay_sum_ns'] for bucket in boundaries)
                                         if eligible == SEGMENTS else None)
    if count != WINDOWS * SEGMENTS:
        raise ValueError(f'expected 196 completed verifier segments, got {count}')
    return result


def paired_envelopes(profile, control, request):
    """Bind four same-position wall samples before reporting device envelopes."""
    with profile.open() as stream:
        windows = summarize(stream, request)
        stream.seek(0)
        observed = window_walls(stream, request, True)
    with control.open() as stream:
        wall = window_walls(stream, request, False)
    if list(observed) != [sample['position'] for sample in windows]:
        raise ValueError('device and wall evidence use different positions')
    return dict(windows=windows, profiled_window_ns=observed, wall_control_ns=wall,
                local_overhead=compare_windows(observed, wall))


def check_reader():
    lines = []
    for i in range(WINDOWS * SEGMENTS):
        window, segment = divmod(i, SEGMENTS)
        submit = 3 + 10 * i if segment == 1 else 0 if segment == 2 else 1 + 10 * i
        lines.append(f'{PREFIX}request=5 window={9 + window} pos={90 + window} qsa_interval=4 '
                     f'segment={segment} status=0 fields=7 submit={submit} start={2 + 10 * i} '
                     f'end={6 + 10 * i} resolution_ns=1 host_wait_ns=7 '
                     f'pool_callback_ns={0 if segment == SEGMENTS - 1 else 3}')
    for window in summarize(lines, 5):
        if (window['graph_ns'], window['inter_graph_ns'], window['span_ns'], window['queue_delay_samples'],
                window['queue_delay_ns'], [row['segment'] for row in window['submit_clock_exceptions']]) != (
                    196, 288, 484, 47, None, [1, 2]):
            raise ValueError('envelope, gap or invalid submit accounting')
    good = [line.replace(f"submit={segment_values(line)['submit']} ", f'submit={1 + i * 10} ', 1)
            for i, line in enumerate(lines)]
    if any(window['queue_delay_ns'] != SEGMENTS for window in summarize(good, 5)):
        raise ValueError('valid submit coverage did not qualify')
    for damaged in (lines[:-1], [lines[0].replace('end=6', 'end=1'), *lines[1:]],
                    [lines[0].replace('fields=7', 'fields=3'), *lines[1:]],
                    [lines[0].replace('status=0', 'status=1'), *lines[1:]],
                    [lines[0], lines[1].replace('segment=1', 'segment=0'), *lines[2:]]):
        try:
            summarize(damaged, 5)
        except ValueError:
            continue
        raise ValueError('damaged start/end evidence passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-reader', action='store_true')
    parser.add_argument('profile', nargs='?', type=Path)
    parser.add_argument('control', nargs='?', type=Path)
    parser.add_argument('request', nargs='?', type=int)
    args = parser.parse_args()
    if args.check_reader and args.profile is None:
        check_reader()
        print('checked envelope, submit validity, gaps and damaged segment records')
    elif not args.check_reader and args.profile and args.control and args.request and args.request > 0:
        print(json.dumps(paired_envelopes(args.profile, args.control, args.request), indent=2))
    else:
        parser.error('use --check-reader or PROFILE CONTROL REQUEST')


if __name__ == '__main__':
    main()
