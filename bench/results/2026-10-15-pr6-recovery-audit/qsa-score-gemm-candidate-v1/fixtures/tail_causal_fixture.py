#!/usr/bin/env python3
"""Independent CPU causal/tail/cell-ranking and address oracle; standard library.
Does not import candidate code, create a queue, allocate full 256K tensors, or
claim to validate oneMKL/device floating point. Main is the sole executor.
"""
from pathlib import Path
import json
import math
import struct

def cell_oracle(case):
    n, width = case['n'], case['width']
    nb = n // 4
    scores = [case['pooled_scores'][b] for b in range(nb)]
    scores.append(case['dead_score'] + (1e9 if n % 4 else 0))
    # Enumerate only existing CELLS. This automatically gives a completed dead
    # slot weight zero and is independent of the GPU's block-radix algorithm.
    cells = sorted(range(n), key=lambda c: (-scores[c // 4], c))[:width]
    return sorted(cells), scores

def fp16(value):
    try:
        return struct.unpack('e', struct.pack('e', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)

def bf16(value):
    bits = struct.unpack('I', struct.pack('f', value))[0]
    # Round-to-nearest-even finite BF16 reference; preserve NaN explicitly.
    if bits & 0x7fffffff > 0x7f800000:
        bits = (bits & 0xffff0000) | 0x00400000
    else:
        bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('f', struct.pack('I', bits))[0]

def main():
    data = json.loads(Path(__file__).with_name('tail_causal_cases.json').read_text())
    for case in data['cases']:
        ids, scores = cell_oracle(case)
        assert ids == case['expected_ids'], (case['name'], ids)
        if 'expected_tail_score' in case:
            assert scores[-1] == case['expected_tail_score']
        assert len(ids) == case['width'] and ids == sorted(set(ids))
        assert all(0 <= cell < case['n'] for cell in ids)
    shapes = 0
    for t in range(1, 257):
        for last in (256, 257, 2051, 2052, 4095, 4096, 4097, 262141, 262142, 262143, 262144):
            if last < t:
                continue
            live = last // 4
            for qi in (0, t - 1):
                n = last - t + qi + 1
                nb = n // 4
                assert 0 <= nb <= live <= 65536
                # All heads preserve physical token stride; no K=128 wrapper.
                for h in range(4):
                    assert qi * 512 + h * 128 + 127 < t * 512
                # dead is the extra converted row after all finalized rows.
                for b in (0, nb):
                    col = live if b == nb else b
                    assert 0 <= col < live + 1 <= 65537
                    assert qi * 65539 + col < t * 65539
                    assert qi * 65538 + b < t * 65538
            shapes += 1
    scratch = 65539 * 128 * 2 + 256 * 512 * 2 + 256 * 65539 * 4
    assert scratch == 84152064
    assert math.isinf(fp16(65536)) and bf16(65536) == 65536
    assert fp16(2 ** -24) == 2 ** -24
    assert bf16(1) == bf16(1 + 1/4096) == 1
    # Four head-specific ReLUs are required; ReLU(sum(dot)) would return zero.
    heads = (-128.0, 128.0, -256.0, 256.0)
    assert sum(max(0.0, x) for x in heads) == 384 and max(0.0, sum(heads)) == 0
    print(json.dumps({'status':'PASS independent CPU expectations',
                      'tail_cases':len(data['cases']), 'shape_endpoints':shapes,
                      'explicit_scratch_max_bytes':scratch,
                      'device_numeric_validation':False},sort_keys=True))
if __name__ == '__main__':
    main()
