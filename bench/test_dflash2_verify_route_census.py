import unittest

import dflash2_verify_route_census as census


def frame(width, expert_start):
    route = census.replay.ROUTE
    return b''.join(route.pack(layer, 10, *range(expert_start, expert_start + 10), bytes(40))
                    for layer in range(48) for _ in range(width))


def counts(request, windows, rows, expert_start):
    layer = [0] * 512
    layer[expert_start:expert_start + 10] = [rows] * 10
    return census.HEADER.pack(request, windows, rows) + census.LAYER_COUNTS.pack(*layer) * 48


class RouteCensusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        routes = bytearray(b'\xff' * census.replay.ROUTE.size)
        start = len(routes)
        logs = [f'{census.MARKER}begin 1 {start} 33217']
        for i in range(257):
            width = 1 if i == 0 else 2
            logs.append(f'{census.WINDOW}{33216 + i} {width}')
            routes.extend(frame(width, 23 if i < 256 else 71))
        logs.append(f'{census.MARKER}end 1 {len(routes)}')
        # The next prompt appends raw trace rows outside the decode span.
        routes.extend(b'\xff' * census.replay.ROUTE.size)
        logs += [f'{census.MARKER}begin 3 {len(routes)} 262143', f'{census.WINDOW}262142 1']
        routes.extend(frame(1, 2))
        logs += [f'{census.MARKER}end 3 {len(routes)}',
                 f'{census.MARKER}begin 4 {len(routes)} 262144', f'{census.MARKER}end 4 {len(routes)}']
        cls.stderr = '\n'.join(logs)
        cls.routes = bytes(routes)
        cls.counts = counts(1, 256, 511, 23) + counts(3, 1, 1, 2) + counts(4, 0, 0, 0)

    def test_boundaries(self):
        result = census.verify(self.stderr, self.routes, self.counts)
        self.assertEqual([(r['request'], r['windows'], r['request_windows'], r['offered_rows']) for r in result],
                         [(1, 256, 257, 511), (3, 1, 1, 1), (4, 0, 0, 0)])

    def test_cross_request_counts(self):
        swapped = self.counts[:census.RECORD_BYTES] + counts(3, 1, 1, 23) + self.counts[-census.RECORD_BYTES:]
        with self.assertRaisesRegex(ValueError, 'routed-row counts differ'):
            census.verify(self.stderr, self.routes, swapped)

    def test_truncated_or_missing(self):
        with self.assertRaisesRegex(ValueError, 'bad census or routing dump extent'):
            census.verify(self.stderr, self.routes[:-1], self.counts)
        with self.assertRaisesRegex(ValueError, 'unterminated census'):
            census.verify(self.stderr.rsplit('\n', 1)[0], self.routes, self.counts)

    def test_first_anchor(self):
        changed = self.stderr.replace(f'{census.WINDOW}33216 1', f'{census.WINDOW}33215 1', 1)
        with self.assertRaisesRegex(ValueError, 'last-prompt-token'):
            census.verify(changed, self.routes, self.counts)

    def test_shifted_trace_end(self):
        changed = self.stderr.replace(f'{census.MARKER}end 3 {len(self.routes)}',
                                      f'{census.MARKER}end 3 {len(self.routes) - census.replay.ROUTE.size}', 1)
        with self.assertRaisesRegex(ValueError, 'trace span'):
            census.verify(changed, self.routes, self.counts)


if __name__ == '__main__':
    unittest.main()
