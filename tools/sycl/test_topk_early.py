"""Host selection proof checks, against independent per-cell numeric ranking.

These checks cannot qualify the generated kernel, collective convergence,
allocation/lifetime behavior, runtime, performance or model execution.
"""
import math
import random
import struct
import unittest

import topk_early_sources as source


def value(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def key(x):
    b = struct.unpack('<I', struct.pack('<f', x))[0]
    if b & 0x7fffffff > 0x7f800000:
        return 0
    if b & 0x7fffffff == 0:
        b = 0
    return (~b & 0xffffffff) if b & 0x80000000 else b | 0x80000000


def ranked_cells(scores, n, width):
    # Uses numerical ordering, rather than the kernel's bit key/radix tree.
    def rank(i):
        x = scores[i // 4]
        return math.isnan(x), -x if not math.isnan(x) else 0, i
    return sorted(sorted(range(n), key=rank)[:width])


def radix_model(scores, n, width, early):
    if not 0 <= width <= n or len(scores) != n // 4 + 1:
        raise ValueError('invalid model fixture')
    if width == n:
        return list(range(n)), 0
    if not width:
        return [], 0
    weighted = [(key(x), min(4, max(0, n - 4 * b))) for b, x in enumerate(scores)]
    rank, threshold, mask, passes = width, 0, 0, 0
    for shift in range(28, -1, -4):
        passes += 1
        histogram = [0] * 16
        for k, weight in weighted:
            if k & mask == threshold:
                histogram[(k >> shift) & 15] += weight
        digit = 15
        while digit > 0 and rank > histogram[digit]:
            rank -= histogram[digit]
            digit -= 1
        threshold |= digit << shift
        mask |= 15 << shift
        if early and rank == histogram[digit]:
            break
    greater = sum(weight for k, weight in weighted if k > threshold)
    budget, result = width - greater, []
    for b, (k, weight) in enumerate(weighted):
        take = weight if k > threshold else min(weight, max(0, budget)) if k == threshold else 0
        if k == threshold:
            budget -= weight
        result.extend(range(4 * b, 4 * b + take))
    return result, passes


class EarlyCompletion(unittest.TestCase):
    def test_constructor_reverses(self):
        proof = source.proof()
        self.assertEqual(proof['reversible_selection_edits'], 1)
        self.assertEqual(set(source.generated()), {'original.cpp', 'candidate.cpp', 'topk-early.hpp'})

    def test_numeric_rank_and_weighted_tails(self):
        rng = random.Random(0x7148ad)
        special = [0xff800000, 0x7f800000, 0x80000000, 0, 0x7fc00001,
                   0xffc12345, 0xbf800000, 0x3f800000, 1, 0x80000001]
        cases, early, complete = 0, 0, 0
        for n in range(1, 97):
            scores = [value(rng.getrandbits(32) if b % 3 else special[b % len(special)])
                      for b in range(n // 4 + 1)]
            for width in range(n + 1):
                expected = ranked_cells(scores, n, width)
                candidate, passes = radix_model(scores, n, width, True)
                self.assertEqual(candidate, expected, (n, width))
                self.assertEqual(radix_model(scores, n, width, False)[0], expected)
                early += 0 < passes < 8
                complete += passes == 8
                cases += 1
        self.assertGreater(early, 0)
        self.assertGreater(complete, 0)
        print('MODEL cases=%d early=%d full8=%d' % (cases, early, complete), flush=True)

    def test_uniform_ties_and_prefix_completion(self):
        for n in [1, 3, 4, 5, 31, 32, 33, 63, 64, 65, 2051, 2052]:
            for width in sorted({0, min(n, 1), min(n, 3), min(n, 2048), min(n, 2051), n}):
                scores = [1.] * (n // 4 + 1)
                self.assertEqual(radix_model(scores, n, width, True)[0], list(range(width)))
        self.assertLess(radix_model([1., 2., 3., 4., float('inf')], 16, 4, True)[1], 8)

    def test_physical_context_end_and_tail(self):
        rng = random.Random(262144)
        for n in [262141, 262142, 262143, 262144]:
            scores = [float(rng.randrange(-1000, 1001)) for _ in range(n // 4 + 1)]
            if n % 4 == 0:
                scores[-1] = float('inf')  # Zero-weight spare cannot affect selection.
            for width in [1, 3, 2048, 2051]:
                self.assertEqual(radix_model(scores, n, width, True)[0], ranked_cells(scores, n, width))


if __name__ == '__main__':
    unittest.main()
