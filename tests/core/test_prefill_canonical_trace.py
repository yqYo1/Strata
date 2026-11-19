import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "sycl"))
import prefill_canonical_trace as trace


class CanonicalTraceTests(unittest.TestCase):
    rows = [("a", 64, 256), ("b", 320, 64), ("c", 384, 256)]

    def test_join_can_end_at_partial_stride(self):
        copies = [(1064, 2000, 320), (1384, 2512, 256)]
        trace.compare_layer(self.rows, copies, 1000, 2000, True)
        with self.assertRaises(ValueError):
            trace.compare_layer(self.rows, copies, 1000, 2000, False)

    def test_counts_and_bytes_cannot_hide_wrong_offsets_or_padding(self):
        for copies in ([(1064, 2000, 320), (1384, 2320, 256)],
                       [(1064, 2000, 256), (1320, 2256, 320)],
                       [(1064, 2000, 257), (1321, 2257, 319)],
                       [(2000, 1064, 320), (2512, 1384, 256)]):
            with self.subTest(copies=copies), self.assertRaises(ValueError):
                trace.compare_layer(self.rows, copies, 1000, 2000, True)

    def test_source_gap_splits_even_when_device_payloads_touch(self):
        with self.assertRaises(ValueError):
            trace.compare_layer([("a", 64, 256), ("b", 384, 256)],
                                [(1064, 2000, 512)], 1000, 2000, True)

    def test_actual_return_status_and_queue_are_required(self):
        begin = "prefill canonical layer=0 begin contiguous=0 src=0x1000"
        def returned(queue, status):
            return (f"<--- urEnqueueUSMMemcpy(.hQueue = {queue}, .blocking = false, "
                    f".pDst = 0x2000, .pSrc = 0x1040, .size = 256, .numEventsInWaitList = 0) -> {status};")
        first = returned("0x21", "UR_RESULT_SUCCESS")
        with self.assertRaisesRegex(ValueError, "submission queue"):
            trace.verify([begin, first, returned("0x22", "UR_RESULT_SUCCESS")], [self.rows], False)
        with self.assertRaisesRegex(ValueError, "failed or malformed"):
            trace.verify([begin, returned("0x21", "UR_RESULT_ERROR_INVALID_VALUE")], [self.rows], False)
