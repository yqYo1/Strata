#!/usr/bin/env python3
"""Compare Strata/llama_logits binary dumps without claiming a quality gate."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    with Path(path).open('rb') as f:
        header = np.fromfile(f, dtype='<i4', count=2)
        if header.size != 2 or np.any(header <= 0):
            raise ValueError(f'{path}: invalid vocabulary/row header')
        vocab, rows = map(int, header)
        if Path(path).stat().st_size != 8 + vocab * rows * 4:
            raise ValueError(f'{path}: incomplete or oversized dump')
        values = np.fromfile(f, dtype='<f4', count=vocab * rows).reshape(rows, vocab)
    if not np.isfinite(values).all():
        raise ValueError(f'{path}: nonfinite logits')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--rows', type=int, help='explicitly compare only this common prefix')
    parser.add_argument('--require-exact', action='store_true')
    args = parser.parse_args()
    try:
        a, b = read(args.reference), read(args.candidate)
        if a.shape[1] != b.shape[1]:
            raise ValueError('vocabulary sizes differ')
        if args.rows is not None:
            if args.rows <= 0 or args.rows > min(a.shape[0], b.shape[0]):
                raise ValueError('requested prefix is outside the dumps')
            a, b = a[:args.rows], b[:args.rows]
        if a.shape != b.shape:
            raise ValueError('row counts differ; use --rows for an explicit prefix')
        exact = np.array_equal(a.view('<u4'), b.view('<u4'))
        pred_a, pred_b = a.argmax(axis=1), b.argmax(axis=1)
        a, b = a.astype('float64'), b.astype('float64')
        norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        if np.any(norm == 0):
            raise ValueError('cosine is undefined for an all-zero row')
        cosine = (a * b).sum(axis=1) / norm
        rmse = np.sqrt(((a - b) ** 2).mean(axis=1))
        print(json.dumps({
            'rows': a.shape[0], 'vocabulary': a.shape[1], 'bitwise_equal': exact,
            'top1_matches': int((pred_a == pred_b).sum()),
            'top1_mismatch_positions': np.flatnonzero(pred_a != pred_b).tolist(),
            'cosine_min': float(cosine.min()), 'cosine_mean': float(cosine.mean()),
            'rmse_max': float(rmse.max()), 'absolute_error_max': float(np.abs(a - b).max()),
            'reference_top1': pred_a.tolist(), 'candidate_top1': pred_b.tolist(),
        }, indent=2, allow_nan=False))
        return int(args.require_exact and not exact)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
