"""Offline regression checks for independently bound decode clock/work evidence."""
import copy
import unittest

import decode_footer as footer


def fixture(cap):
    rows, generated, offered, accepted = (2, 2, 0, 0) if cap else (5, 5, 3, 3)
    entries = 96 if cap else 144
    sample = dict(prompt_tokens=1000, reused_tokens=0, read_tokens=1000, generated_tokens=generated,
                  drafts_accepted=accepted, drafts_offered=offered, decode_ms=200.)
    text = ('strata trace: window 999 1\nstrata trace: window 1000 ' + ('1' if cap else '4') + '\n' +
        'strata decode timing: 2 windows, avg T ' + ('1.00, 1.00' if cap else '2.50, 2.50') +
        ' tokens/window, 100.00 ms/window = verify ' + ('98.00' if cap else '80.00') +
        ' (GPU-reach wait 20.00 + per-layer host 30.00 [plan 1.00 actq 1.00 jobs 1.00 CPU 26.00]' +
        ' + stage 5.00) + commit/emit 1.00 + draft ' + ('0.00' if cap else '18.00') +
        '; per layer-window: CPU experts 1.00 (' + ('1.00' if cap else '1.50') +
        ' entries), VRAM hits ' + ('9.00' if cap else '23.50') + ', PCIe 0.00\n' +
        f'strata serve: prompt 1000 tokens = 0 reused + 1000 read in 20 ms (50 tok/s), {generated} generated' +
        f' in 200 ms (25 tok/s), drafts accepted {accepted} of {offered}, 0 checkpoints\n' +
        'strata serve: SYCL_EXPERT_COPY submitted_blobs=0 submitted_bytes=0 completed_blobs=0 completed_bytes=0' +
        f' host_view_groups=0 gpu_miss_groups=0 cpu_unique_misses=96 cpu_entries={entries}\n')
    return text, [sample], [dict(target_windows=2, target_rows=rows)]


class DecodeFooter(unittest.TestCase):
    def test_actual_t1_and_t4_work_and_rounded_clocks(self):
        for cap in (0, 1):
            text, samples, routes = fixture(cap)
            out, = footer.analyze(text, samples, routes)
            self.assertEqual(out['maximum_T'], 1 if cap else 4)
            self.assertEqual(out['target_rows'], 2 if cap else 5)
            self.assertEqual(out['outer_residual_ms_per_window'], 1.)
            self.assertEqual(out['host_nested_residual_ms_per_window'], 1.)
            self.assertEqual(out['approximate_ms_per_emitted']['draft'], 0 if cap else 7.2)
            self.assertEqual(out['copy']['submitted_bytes'], 0)

    def test_request_boundaries_reject_moved_or_duplicate_footers(self):
        text, samples, routes = fixture(0)
        footer.analyze(text + text, samples * 2, routes * 2)
        timing = next(l for l in text.splitlines(True) if l.startswith('strata decode timing:'))
        copy_line = next(l for l in text.splitlines(True) if l.startswith('strata serve: SYCL_EXPERT_COPY'))
        for invalid in (text + timing, text.replace(timing, '') + timing,
                        text + copy_line, copy_line + text.replace(copy_line, '')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                footer.analyze(invalid, samples, routes)

    def test_counter_route_and_clock_corruptions_fail(self):
        text, samples, routes = fixture(1)
        for old, new in [('cpu_entries=96', 'cpu_entries=97'),
                         ('submitted_bytes=0', 'submitted_bytes=1'),
                         ('avg T 1.00', 'avg T 4.00'),
                         ('window 1000 1', 'window 1000 4'),
                         ('100.00 ms/window', '100.04 ms/window'),
                         ('verify 98.00', 'verify 101.00'),
                         ('CPU 26.00', 'CPU 31.00'),
                         ('draft 0.00', 'draft nan')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                footer.analyze(text.replace(old, new), samples, routes)
        bad = copy.deepcopy(samples); bad[0]['generated_tokens'] += 1
        with self.assertRaises(ValueError): footer.analyze(text, bad, routes)


if __name__ == '__main__':
    unittest.main()
