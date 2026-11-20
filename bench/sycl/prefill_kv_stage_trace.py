"""Check Q8 host publication and prefix reloads in one traced layer-major model request."""

import re
from pathlib import Path


LAYER_START = re.compile(r'strata trace: layer-major layer (\d+) of 48: loading 512 experts')
LAYER_END = re.compile(r'strata trace: layer-major layer (\d+) of 48: completed (\d+) chunks')
MIRROR = re.compile(r'strata QSA mirror bulk: tokens=(\d+) pos0=(\d+) segments=(\d+) '
                    r'copies=(\d+) requested_D2H_bytes=(\d+) queue=same accepted=1 host_quantization=0')
STAGE = re.compile(r'strata prefill KV stage: layer=(\d+) reloaded_pages=(\d+) reused_pages=(\d+)')
PAGE = 4
CELL_BYTES = 1056  # Two KV heads, each with INT8 K/V and eight FP16 group scales.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def expected_prefix(positions, layer, reuse):
    loaded = kept = calls = 0
    for index, position in enumerate(positions):
        pages = (position + PAGE - 1) // PAGE
        if reuse and layer < 47 and index and position % PAGE == 0:
            kept += pages
        else:
            loaded += pages
            calls += 4 * bool(pages)
    return loaded, kept, calls


def check_clean_request(text, start, rows, chunk, *, reuse):
    """Validate the native per-layer counters without enabling the full request trace."""
    require(0 <= start < 262144 and 0 < rows <= 262144 - start and chunk > 0, 'invalid request range')
    require('strata trace:' not in text, 'clean request contains the diagnostic trace')
    positions = range(start, start + rows, chunk)
    actual = []
    for line in text.splitlines():
        if line.startswith('strata prefill KV stage:'):
            match = STAGE.fullmatch(line)
            require(match, 'malformed clean KV stage counters')
            actual.append(tuple(map(int, match.groups())))
    expected = []
    for layer in range(3, 48, 4):
        loaded, kept, _ = expected_prefix(positions, layer, reuse)
        if loaded or kept:
            expected.append((layer, loaded, kept))
    require(actual == expected, 'clean prefix reload/reuse differs from the planned layer dispatch')
    return dict(reloaded_pages=sum(r[1] for r in actual), reused_pages=sum(r[2] for r in actual))


def check_request(text, start, rows, chunk, *, reuse, external_tap=False, historical=False):
    """Use the separately observed append positions to check the stage's own counters."""
    require(0 <= start < 262144 and 0 < rows <= 262144 - start and chunk > 0, 'invalid request range')
    require(not (historical and reuse), 'historical engine has no KV reuse selector')
    positions = range(start, start + rows, chunk)
    layer = None
    completed = 0
    receipts = []
    totals = dict(reloaded_pages=0, reused_pages=0, prefix_H2D_calls=0, mirror_D2H_calls=0, mirror_D2H_bytes=0)
    for line in text.splitlines():
        match = LAYER_START.fullmatch(line)
        if match:
            require(layer is None and int(match[1]) == completed, 'layer started out of order')
            layer = completed
            receipts = []
            stage = None
        elif line.startswith('strata QSA mirror bulk:'):
            match = MIRROR.fullmatch(line)
            require(match and layer is not None and layer % 4 == 3 and len(receipts) < len(positions),
                    'unexpected host mirror writer')
            count, pos, segments, copies, host_bytes = map(int, match.groups())
            require(pos == positions[len(receipts)] and count == min(chunk, start + rows - pos),
                    'host mirror extent differs from the request')
            head = min(count, (-pos) % PAGE)
            full, tail = divmod(count - head, PAGE)
            require(segments == 2 * bool(head) + bool(full) + 2 * bool(tail)
                    and copies == 4 * segments and host_bytes == CELL_BYTES * count,
                    'bulk mirror missed a head, page, or tail')
            receipts.append(pos)
            totals['mirror_D2H_calls'] += copies
            totals['mirror_D2H_bytes'] += host_bytes
        elif line.startswith('strata prefill KV stage:'):
            match = STAGE.fullmatch(line)
            require(match and layer is not None and layer % 4 == 3 and int(match[1]) == layer
                    and stage is None and len(receipts) == len(positions), 'invalid stage summary or owner')
            stage = tuple(map(int, match.groups()[1:]))
        elif (match := LAYER_END.fullmatch(line)):
            require(layer is not None and int(match[1]) == layer and int(match[2]) == len(positions),
                    'incomplete layer or chunk count')
            if layer % 4 == 3:
                require(len(receipts) == len(positions), 'missing QSA application or host publication')
                loaded, kept, calls = expected_prefix(receipts, layer, reuse and not external_tap)
                expected = (loaded, kept) if not historical and loaded + kept else None
                require(stage == expected, f'layer {layer} stage counters {stage} differ from {expected}')
                totals['reloaded_pages'] += loaded
                totals['reused_pages'] += kept
                totals['prefix_H2D_calls'] += calls
            layer = None
            completed += 1
    require(completed == 48 and layer is None, 'all 48 layer passes are required')
    totals['requested_prefix_H2D_bytes'] = totals['reloaded_pages'] * PAGE * CELL_BYTES
    return totals


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frame', type=Path, help='trace covering exactly one completed request')
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--rows', required=True, type=int, help='number of batched input rows')
    parser.add_argument('--chunk', default=8192, type=int)
    parser.add_argument('--mode', choices=('historical', 'reload', 'reuse', 'external-tap'), required=True)
    args = parser.parse_args()
    print(json.dumps(check_request(args.frame.read_text(), args.start, args.rows, args.chunk,
                                   reuse=args.mode in ('reuse', 'external-tap'),
                                   historical=args.mode == 'historical', external_tap=args.mode == 'external-tap')))
