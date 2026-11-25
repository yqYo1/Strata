"""Check the request-local DFlash2 census against an independent routing dump."""

from collections import Counter
from pathlib import Path
import json
import re
import struct
import sys

import dflash2_decode_residency_replay as replay


LIMIT = 64 << 20
MAX_WINDOWS = 256
HEADER = struct.Struct('<QQQ')
LAYER_COUNTS = struct.Struct(f'<{replay.EXPERTS}I')
RECORD_BYTES = HEADER.size + replay.LAYERS * LAYER_COUNTS.size
MARKER = 'strata DFlash2 route census: '
WINDOW = 'strata trace: window '


def requests(stderr, routes):
    """Return checked byte spans and ordered window widths for each GEN."""
    pending = None
    previous_request, previous_end = 0, 0
    for line in stderr.splitlines():
        if line.startswith(MARKER + 'begin '):
            m = re.fullmatch(MARKER + r'begin ([0-9]+) ([0-9]+) ([0-9]+)', line)
            replay.require(m is not None and pending is None, 'nested or malformed census begin')
            request, start, prompt = map(int, m.groups())
            replay.require(request > previous_request and 1 <= prompt <= 262144
                           and previous_end <= start <= len(routes) and start % replay.ROUTE.size == 0,
                           'invalid census request or trace beginning')
            pending = request, start, prompt, []
        elif line.startswith(WINDOW):
            m = re.fullmatch(WINDOW + r'([0-9]+) ([0-9]+)', line)
            replay.require(m is not None and pending is not None, 'window outside a census request')
            p, width = map(int, m.groups())
            positions = pending[3]
            replay.require(1 <= width <= 4 and 0 <= p <= 262144 - width,
                           'invalid decode window extent')
            if positions:
                prior, prior_width = positions[-1]
                replay.require(1 <= p - prior <= prior_width, 'decode windows skip or repeat positions')
            else:
                replay.require(p == pending[2] - 1 and width == 1, 'missing last-prompt-token head')
            positions.append((p, width))
        elif line.startswith(MARKER + 'end '):
            m = re.fullmatch(MARKER + r'end ([0-9]+) ([0-9]+)', line)
            replay.require(m is not None and pending is not None, 'unmatched census end')
            request, end = map(int, m.groups())
            current, start, _, positions = pending
            widths = [width for _, width in positions]
            expected_bytes = sum(widths) * replay.LAYERS * replay.ROUTE.size
            replay.require(current == request and end <= len(routes) and end == start + expected_bytes,
                           'trace span does not contain exactly the offered rows')
            yield request, start, end, widths
            previous_request, previous_end = request, end
            pending = None
        elif line.startswith(MARKER):
            raise ValueError('unrecognized census marker')
    replay.require(pending is None, 'unterminated census request')


def verify(stderr, routes, counts):
    replay.require(0 < len(counts) <= LIMIT and len(counts) % RECORD_BYTES == 0
                   and len(routes) <= LIMIT and len(routes) % replay.ROUTE.size == 0,
                   'bad census or routing dump extent')
    verified = []
    for i, (request, start, end, widths) in enumerate(requests(stderr, routes)):
        offset = i * RECORD_BYTES
        replay.require(offset + RECORD_BYTES <= len(counts), 'missing request counts')
        windows = min(len(widths), MAX_WINDOWS)
        rows = sum(widths[:windows])
        replay.require(HEADER.unpack_from(counts, offset) == (request, windows, rows),
                       'census header differs from the observed requests')
        payload = counts[offset + HEADER.size:offset + RECORD_BYTES]
        # The independent raw writer emits an entry per routed row. Only
        # decode's first 256 windows may contribute to the native count map.
        prefix_bytes = rows * replay.LAYERS * replay.ROUTE.size
        prefix = memoryview(routes)[start:start + prefix_bytes]
        totals = [Counter() for _ in range(replay.LAYERS)]
        for frame in replay.window_counts(prefix, widths[:windows]):
            for total, entries in zip(totals, frame, strict=True):
                total.update(entries)
        expected = b''.join(LAYER_COUNTS.pack(*(total[e] for e in range(replay.EXPERTS))) for total in totals)
        replay.require(payload == expected, 'per-expert routed-row counts differ')
        verified.append(dict(request=request, windows=windows, offered_rows=rows, request_windows=len(widths),
                             trace_begin=start, trace_end=end, counts_sha256=replay.sha(payload)))
    replay.require(bool(verified) and len(verified) * RECORD_BYTES == len(counts), 'extra or absent request record')
    return verified


def read_capture(paths):
    files = [Path(path) for path in paths]
    sizes = [file.stat().st_size for file in files]
    replay.require(sum(sizes) <= LIMIT, 'diagnostic capture exceeds its byte budget')
    data = [file.read_bytes() for file in files]
    replay.require([len(blob) for blob in data] == sizes, 'diagnostic capture changed while reading')
    return data


if __name__ == '__main__':
    stderr, routes, counts = read_capture(sys.argv[1:])
    print(json.dumps(verify(stderr.decode(), routes, counts), indent=2))
