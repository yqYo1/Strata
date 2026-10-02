#!/usr/bin/env python3
"""Compare complete SYCL verify windows, including rejected draft rows.

STRATA_SYCL_VERIFY_LOGITS appends each completed window: little-endian int64
position, int32 row count/vocabulary, row-count int32 input ids, then row-major
float32 logits. Use fresh paths for each engine process. This checks a kernel
contract, without making a model-quality claim.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read(path):
    size = Path(path).stat().st_size
    windows = []
    with Path(path).open('rb') as f:
        while f.tell() < size:
            header = f.read(16)
            if len(header) != 16:
                raise ValueError(f'{path}: truncated window header')
            pos, rows, vocab = struct.unpack('<qii', header)
            if pos < 0 or not 1 <= rows <= 8 or vocab < 1:
                raise ValueError(f'{path}: invalid window header')
            if size - f.tell() < rows * (1 + vocab) * 4:
                raise ValueError(f'{path}: truncated window payload')
            tokens = np.fromfile(f, dtype='<i4', count=rows)
            values = np.fromfile(f, dtype='<f4', count=rows * vocab).reshape(rows, vocab)
            if np.any(tokens < 0) or np.any(tokens >= vocab) or not np.isfinite(values).all():
                raise ValueError(f'{path}: invalid input ids or nonfinite logits')
            windows.append((pos, tokens, values))
    if not windows:
        raise ValueError(f'{path}: empty window dump')
    return windows


def compare(a, b):
    if len(a) != len(b):
        raise ValueError('window counts differ')
    rows = matches = 0
    unequal = []
    maximum = 0.0
    for i, ((pa, ta, va), (pb, tb, vb)) in enumerate(zip(a, b)):
        if pa != pb or va.shape != vb.shape or not np.array_equal(ta, tb):
            raise ValueError(f'window {i}: positions, shapes or input ids differ')
        rows += va.shape[0]
        matches += int((va.argmax(axis=1) == vb.argmax(axis=1)).sum())
        if not np.array_equal(va.view('<u4'), vb.view('<u4')):
            unequal.append(i)
            maximum = max(maximum, float(np.max(np.abs(va.astype('float64') - vb))))
    return dict(windows=len(a), rows=rows, bitwise_equal=not unequal,
                unequal_windows=unequal, top1_matches=matches, absolute_error_max=maximum,
                positions=[p for p, _, _ in a], window_rows=[v.shape[0] for _, _, v in a])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--require-exact', action='store_true')
    args = parser.parse_args()
    try:
        result = compare(read(args.reference), read(args.candidate))
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(args.require_exact and not result['bitwise_equal'])
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
