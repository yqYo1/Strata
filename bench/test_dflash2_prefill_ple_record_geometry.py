"""Host controls for PLE read geometry and exposed host-phase accounting."""

import unittest

import dflash2_prefill_ple_record_geometry as geometry


PIECES = ((0, 8192), (8192, 4096))


def host(tokens, allocate, issue, collect, expand, end):
    times = (allocate, issue, collect, expand, end)
    fields = ' '.join(f'{label}={ms * 1_000_000}'
                      for label, ms in zip(geometry.HOST_STAMPS, times, strict=True))
    return f'{geometry.HOST_PREFIX}tokens={tokens} {fields}\n'


def join(offset, tokens, begin, end):
    return (f'{geometry.JOIN_PREFIX}offset={offset} tokens={tokens} '
            f'begin_ns={begin * 1_000_000} end_ns={end * 1_000_000}\n')


class PleRecordGeometryTest(unittest.TestCase):
    def test_expansion_is_partly_exposed_for_first_ticket_and_hidden_for_tail(self):
        text = (host(8192, 10, 12, 15, 24, 32) + join(0, 8192, 28, 34)
                + host(4096, 40, 41, 42, 43, 47) + join(8192, 4096, 48, 49))
        first, tail = geometry.host_timeline(text, PIECES, 7.0)
        self.assertEqual(first['phases']['expand'], dict(wall_ns=8_000_000, exposed_ns=4_000_000))
        self.assertEqual(first['unattributed_join_ns'], 2_000_000)
        self.assertEqual(tail['phases']['expand'], dict(wall_ns=4_000_000, exposed_ns=0))
        self.assertEqual(tail['unattributed_join_ns'], 1_000_000)

    def test_join_that_starts_before_the_worker(self):
        text = host(8192, 10, 12, 15, 24, 32) + join(0, 8192, 9, 34)
        result, = geometry.host_timeline(text, PIECES[:1], 25.0)
        self.assertEqual(result['phases']['collect'], dict(wall_ns=9_000_000, exposed_ns=9_000_000))
        self.assertEqual(result['unattributed_join_ns'], 3_000_000)

    def test_incomplete_or_misaligned_receipts_fail(self):
        worker = host(8192, 10, 12, 15, 24, 32)
        cases = (
            (worker, 7.0, 'coverage'),
            (worker + join(8192, 8192, 28, 34), 6.0, 'order or extent'),
            (worker + join(0, 8192, 28, 31), 3.0, 'order or extent'),
            (worker + join(0, 8192, 28, 34), 6.1, 'reconcile'),
        )
        for text, duration, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, reason):
                geometry.host_timeline(text, PIECES[:1], duration)
        with self.assertRaisesRegex(RuntimeError, 'coverage'):
            geometry.host_timeline(worker, (), 0.0)

    def test_record_duplicates_and_a_fully_cached_tail(self):
        first = (geometry.PREFIX + 'rows=131072 jobs=3 requested_bytes=16384 in_file_bytes=16370 '
                 'page_touches=4 unique_pages=3 record_bytes=131072 record_touches=3 unique_records=2\n')
        cached = (geometry.PREFIX + 'rows=65536 jobs=0 requested_bytes=0 in_file_bytes=0 '
                  'page_touches=0 unique_pages=0 record_bytes=131072 record_touches=0 unique_records=0\n')
        records = geometry.read_geometry(first + cached, PIECES)
        self.assertEqual(records[0]['record_touches'] - records[0]['unique_records'], 1)
        self.assertEqual(records[1]['jobs'], 0)
        with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
            geometry.read_geometry(first + cached.replace('unique_records=0', 'unique_records=1'), PIECES)


if __name__ == '__main__':
    unittest.main()
