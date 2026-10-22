"""Qualification of dump rejection and the logical union interpretation."""
from array import array
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
from qsa_selection_capture import analyze


def capture(mutate=None):
    # 32785 prompt tokens leaves sixteen prefill rows after four8192 chunks.
    chunks = []
    for layer in range(12):
        ids = array('i', range(2051)) * 16
        if mutate and layer == 0:
            mutate(ids)
        chunks.append(struct.pack('<4i', layer, 32768, 16, 2051) + ids.tobytes())
    return b''.join(chunks)


class CaptureTests(unittest.TestCase):
    def check(self, raw):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dump'; path.write_bytes(raw)
            return analyze(path, 32785, 12)

    def test_identical_sets_are_logical_reuse(self):
        result = self.check(capture())
        self.assertEqual(result['captured_queries'], 16)
        self.assertEqual(len(result['long_layers']), 12)
        union = result['long_layers'][0]['logical_16query_union']
        self.assertEqual(union['union_cells'], 2051)
        self.assertEqual(union['separate_cells'], 2051 * 16)
        self.assertEqual(union['union_fraction'], 1 / 16)

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.check(capture(lambda a: a.__setitem__(1, 0)))

    def test_future_rejected(self):
        with self.assertRaisesRegex(ValueError, 'causal'):
            self.check(capture(lambda a: a.__setitem__(2050, 40000)))

    def test_negative_rejected(self):
        with self.assertRaisesRegex(ValueError, 'causal'):
            self.check(capture(lambda a: a.__setitem__(0, -1)))

    def test_truncated_body_rejected(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            self.check(capture()[:-4])

    def test_missing_layer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'twelve'):
            self.check(capture()[:-(16 + 16 * 2051 * 4)])

    def test_reordered_layer_rejected(self):
        raw = bytearray(capture()); raw[:4] = struct.pack('<i', 1)
        with self.assertRaisesRegex(ValueError, 'twelve'):
            self.check(raw)

    def test_partial_header_rejected(self):
        with self.assertRaisesRegex(ValueError, 'truncated'):
            self.check(capture() + b'X')

    def test_unused_warmup_padding_is_not_selected(self):
        records = []
        for layer in range(12):
            ids = array('i', [-1]) * (11 * 2051)
            for row in range(11):
                ids[row * 2051:row * 2051 + row + 1] = array('i', range(row + 1))
            records.append(struct.pack('<4i', layer, 0, 11, 2051) + ids.tobytes())
        result = self.check(b''.join(records) + capture())
        self.assertEqual(result['warmup_records'], 12)


if __name__ == '__main__': unittest.main()
