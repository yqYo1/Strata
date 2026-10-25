"""Host-only rejection controls for the one-call discovery envelope."""
import sys
import unittest

sys.dont_write_bytecode = True
import observe_onemkl as o

TEXT = ('BEGIN type=bf16 role=HCDown rows=64 N=320 K=10240 ld=323\n'
        'OP_BEGIN invocation=1 pid=42 tid=7 raw_ns=2000\n'
        'OP_RETURN invocation=1 raw_ns=3000\nOP_DRAIN invocation=1 raw_ns=4000 checked=1\n'
        'PASS type=bf16 role=HCDown rows=64 values=20480 analytic=FULL_EXACT guards=PASS inputs=PASS device_bytes=7947264\n'
        'DONE cases=1 scope=synthetic_location_probe_not_model_performance\n')
EVENT = dict(ph='X', cat='gpu_op', name='opaque_BLAS', ts=2.5, dur=1.)


class Envelope(unittest.TestCase):
    def test_valid_is_only_a_charged_envelope(self):
        x = o.analyze(TEXT, [], [EVENT], 42, 1000, 5000)
        self.assertEqual(x['wrapper_invocations'], 1)
        self.assertEqual(x['source_FMA_operations'], 209715200)
        self.assertEqual(x['device_records_inside_charged_envelope'], 1)
        self.assertIn('no internal operation association', x['limits'])

    def test_copy_outside_operator_does_not_prove_GPU_math(self):
        x = o.analyze(TEXT, [], [dict(EVENT, ts=1.1, dur=.5)], 42, 1000, 5000)
        self.assertEqual(x['PTI_device_records'], 1)
        self.assertEqual(x['device_records_inside_charged_envelope'], 0)

    def test_missing_duplicate_and_reordered_boundary(self):
        line = 'OP_RETURN invocation=1 raw_ns=3000\n'
        for text in (TEXT.replace(line, ''), TEXT.replace(line, line * 2),
                     TEXT.replace('raw_ns=3000', 'raw_ns=1999'),
                     TEXT.replace('pid=42', 'pid=43'), TEXT.replace('tid=7', 'tid=0')):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                o.analyze(text, [], [EVENT], 42, 1000, 5000)

    def test_bad_oracle_footer(self):
        for text in (TEXT.replace('values=20480', 'values=20479'), TEXT.replace('guards=PASS', 'guards=FAIL'),
                     TEXT.replace('DONE cases=1', 'DONE cases=2')):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                o.analyze(text, [], [EVENT], 42, 1000, 5000)

    def test_invalid_device_clock(self):
        for events in ([], [dict(EVENT, ts=float('nan'))], [dict(EVENT, dur=-1.)],
                       [dict(EVENT, ts=.9)], [dict(EVENT, dur=3.)]):
            with self.subTest(events=events), self.assertRaises(RuntimeError):
                o.analyze(TEXT, [], events, 42, 1000, 5000)

    def test_rejected_enqueue_is_preserved(self):
        spans = [dict(name='urEnqueueNativeCommandExp', status=46, tid=8)]
        x = o.analyze(TEXT, spans, [EVENT], 42, 1000, 5000)
        self.assertEqual(x['enqueue_outcomes'], [dict(name='urEnqueueNativeCommandExp', status=46, calls=1)])
        self.assertEqual(x['enqueue_threads'], 1)


if __name__ == '__main__': unittest.main()
