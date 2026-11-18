"""Host regression controls for QSA source projection and sample admission."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent/'sycl'))
import qsa_v_tile_trial as trial


class VTileTrialTests(unittest.TestCase):
    def test_only_projected_v_gather_moves_between_arms(self):
        inputs = trial.proposal.generated(trial.P/trial.proposal.PREFIX)
        late = inputs['separate_late_v.cpp'].decode().replace('separate_late_v', 'experiment')
        early = inputs['separate_early_v.cpp'].decode().replace('separate_early_v', 'experiment')
        start = late.index('        if constexpr (VectorV4) {')
        end = late.index('        item.barrier', start)
        block = late[start:end]
        anchor = '        if (tid < Cells) local_live |= rows[tid] >= 0;'
        self.assertEqual(early, (late[:start]+late[end:]).replace(anchor, block+anchor, 1))
        self.assertIn('MatrixTile<2 * Cells * Dim>', early)
        self.assertIn('const auto value_tile = kv + Cells * Dim;', early)
        with self.assertRaises(ValueError):
            trial.proposal.project(b'corrupt original', False)

    def test_real_sample_order_and_timing_floor(self):
        lines = []
        for j, order in enumerate(trial.ORDERS):
            for k, arm in enumerate(order):
                lines.append(f'sample process=0 cells=78453 queries=8192 cap=2051 pattern=gather '
                             f'round={j} order={k} variant={arm} implementation={trial.ARMS[arm]} '
                             f'milliseconds={15-arm} compared=50331648 exact_bits=PASS guards=PASS status=PASS')
        lines.append('case samples=18 full_bits=PASS guards=PASS inputs=PASS status=PASS')
        sample = '\n'.join(lines)
        cases = [(78453,8192,2051,'gather')]
        result = trial.compare_samples(sample, 0, cases)
        self.assertEqual(result[0]['medians_ms'], dict(zip(trial.ARMS, (15,14,13))))
        for bad in (sample.replace('milliseconds=15','milliseconds=9',1),
                    sample.replace('variant=0','variant=2',1),
                    '\n'.join(lines[1:])):
            with self.assertRaises(RuntimeError):
                trial.compare_samples(bad, 0, cases)


if __name__ == '__main__':
    unittest.main()
