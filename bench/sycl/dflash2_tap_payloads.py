"""Read the DFlash2 tap digests from two owned fresh-prompt diagnostics."""

import argparse
import json
from pathlib import Path
import re

LAYERS = (5, 15, 25, 35, 45)
WIDTH = 2560
TAIL = 2047
CAPACITY = 262144
PROMPTS = (82, 122881, 82)
REQUEST = re.compile(r'^strata trace: request (\d+) (\d+)$', re.M)
AUDIT_PREFIX = 'strata DFlash2 tap payload: '
AUDIT = re.compile(re.escape(AUDIT_PREFIX) + r'first=(\d+) rows=(\d+) width=(\d+) ' +
                   ' '.join(f'layer{layer}=([0-9a-f]{{16}})' for layer in LAYERS))
INGEST = re.compile(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=0 '
                    r'vram_bytes=([1-9]\d*)$', re.M)


def read_fresh_tap_payloads(log, prompts=PROMPTS):
    """Bind the five layer receipts to fresh requests and checked ingestion."""
    markers = list(REQUEST.finditer(log))
    if (not markers or any(not 2 <= n <= CAPACITY for n in prompts) or
            [tuple(map(int, m.groups())) for m in markers] != [(n, 0) for n in prompts] or
            log.count(AUDIT_PREFIX) != len(prompts)):
        raise ValueError('expected one DFlash2 tap audit per planned fresh request')
    starts = [m.start() for m in markers] + [len(log)]
    records = []
    for prompt, start, end in zip(prompts, starts[:-1], starts[1:], strict=True):
        frame = log[start:end]
        payloads = [line for line in frame.splitlines() if line.startswith(AUDIT_PREFIX)]
        activations = list(INGEST.finditer(frame))
        payload = AUDIT.fullmatch(payloads[0]) if len(payloads) == 1 else None
        first = max(0, prompt - TAIL)
        rows = prompt - 1 - first
        if (payload is None or len(activations) != 1 or int(activations[0][1]) != rows or
                frame.index(payloads[0]) >= activations[0].start() or
                tuple(map(int, payload.groups()[:3])) != (first, rows, WIDTH)):
            raise ValueError(f'invalid DFlash2 tap range or ingestion for {prompt} input tokens')
        records.append(dict(prompt=prompt, first=first, rows=rows, width=WIDTH,
                            readback_bytes=rows * WIDTH * 4 * len(LAYERS),
                            layer_digests=dict(zip(LAYERS, payload.groups()[3:], strict=True))))
    return records


def compare(copy_log, direct_log, prompts=PROMPTS):
    copy = read_fresh_tap_payloads(copy_log, prompts)
    direct = read_fresh_tap_payloads(direct_log, prompts)
    differences = [
        dict(index=index, prompt=ref['prompt'], layers=[
            layer for layer in LAYERS if ref['layer_digests'][layer] != actual['layer_digests'][layer]])
        for index, (ref, actual) in enumerate(zip(copy, direct, strict=True)) if ref != actual
    ]
    return dict(copy=copy, direct=direct, differences=differences, tap_payloads_match=not differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('copy_stderr', type=Path)
    parser.add_argument('direct_stderr', type=Path)
    args = parser.parse_args()
    for path in (args.copy_stderr, args.direct_stderr):
        if not 0 < path.stat().st_size <= 4 << 20:
            parser.error('expected a bounded native diagnostic log')
    try:
        result = compare(args.copy_stderr.read_text(), args.direct_stderr.read_text())
    except ValueError as e:
        parser.error(str(e))
    print(json.dumps(result, indent=2))
    return int(not result['tap_payloads_match'])


if __name__ == '__main__':
    raise SystemExit(main())
