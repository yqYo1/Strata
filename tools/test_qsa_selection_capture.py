"""Qualification of dump rejection and the logical union interpretation."""
from array import array
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
from qsa_selection_capture import analyze
from capture_qsa_selection import route_frame


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
        self.assertFalse(result['replay']['retained'])
        self.assertIsNone(result['replay']['consumer'])

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


class RequestFramingTests(unittest.TestCase):
    def test_warmup_markers_cannot_satisfy_long_route(self):
        text = ('strata prefill: layer-major candidate, tokens=32767 chunk=8192 '
                'residual_dtype=bf16 residual_element_bytes=2 residual_total=671068160 '
                'residual_gpu_rows=32767 residual_gpu=671068160 residual_host=0 residual_staging=0 x\n'
                'strata serve: prompt warmup\n'
                'strata serve: prompt long\n')
        with self.assertRaisesRegex(RuntimeError, 'long-request GPU'):
            route_frame(text, 32767)

    def test_wrong_long_residual_extent_rejected(self):
        text = ('strata serve: prompt warmup\n'
                'strata prefill: layer-major candidate, tokens=32767 chunk=8192 '
                'residual_dtype=bf16 residual_element_bytes=2 residual_total=671068159 '
                'residual_gpu_rows=32767 residual_gpu=671068159 residual_host=0 residual_staging=0 x\n'
                'strata MoE window configuration: requested_mode=10 effective_mode=10 '
                'maximum_rows=32767 mixed_fp16_bytes=0\n'
                'strata serve: prompt long\n')
        with self.assertRaisesRegex(RuntimeError, 'long-request GPU'):
            route_frame(text, 32767)


if __name__ == '__main__': unittest.main()
