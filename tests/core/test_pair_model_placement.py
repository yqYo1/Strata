"""CPU rejection controls for the measured model's placement evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from pair_model_placement import validate

ENV = {'STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS': '384',
       'STRATA_SYCL_PREFILL_MIXED_SCRATCH': '1',
       'STRATA_SYCL_PREFILL_NATIVE16': 'bf16'}
PREFIX = ('strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
          'KV=active-logical-prefix transient_bytes=241555328; '
          'source=resident-Mode0 score_pitch=33 operand_layout=row-major '
          'softmax=parallel-exp-ascending-sum')
MODE = ('strata MoE window configuration: requested_mode=10 effective_mode=10 '
        'maximum_rows=81920 mixed_fp16_bytes=0')
BANK = ('strata MoE expert bank: requested_slots=384 effective_slots=384 staged=1 '
        'host_slab_bytes=1363148800 device_bytes=1022361600 residual_gpu_rows=117488 '
        'allocation_limit_bytes=5368709120 general_map=private_scratch_never_published')
PLACEMENT = ('strata prefill: layer-major candidate, tokens=117488 chunk=8192 '
             'residual_dtype=bf16 residual_element_bytes=2 residual_total=2406154240 '
             'residual_gpu_rows=117488 residual_gpu=2406154240 residual_host=0 '
             'residual_staging=0 layer_cache=1022361600; '
             'extra allocations, GPU headroom=536870912, RAM headroom=8589934592')


def fixture(arm='original'):
    prefix = PREFIX if arm == 'original' else PREFIX.replace('241555328', '325580672')
    return '\n'.join([prefix, MODE, BANK, PLACEMENT])


class PlacementTest(unittest.TestCase):
    def rejected(self, log, arm='original'):
        with self.assertRaises(ValueError):
            validate(log, [117488, 117488], arm, ENV)

    def test_both_complete_arms(self):
        for arm in ('original', 'candidate'):
            rows = validate(fixture(arm) + '\n' + fixture(arm), [117488] * 2, arm, ENV)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['bank']['requested_slots'], 384)
            self.assertEqual(rows[0]['placement']['residual_host'], 0)

    def test_second_request_mode_fallback_or_missing(self):
        for mode in (MODE.replace('effective_mode=10', 'effective_mode=2'), ''):
            self.rejected(fixture() + '\n' + fixture().replace(MODE, mode))

    def test_bank_missing_or_decode_slots(self):
        self.rejected((fixture() + '\n' + fixture()).replace(BANK, ''))
        self.rejected(fixture() + '\n' + fixture().replace('requested_slots=384', 'requested_slots=1256'))

    def test_exact_prefix_bytes_and_route(self):
        for bad in ('241555329', '325580672'):
            self.rejected(fixture() + '\n' + fixture().replace('241555328', bad))
        self.rejected(fixture() + '\n' + fixture().replace('operand_layout=row-major', 'operand_layout=dense16-vnni'))

    def test_residual_and_bank_conservation(self):
        for before, after in [('residual_gpu=2406154240', 'residual_gpu=2406154241'),
                              ('device_bytes=1022361600', 'device_bytes=1022361601'),
                              ('residual_total=2406154240', 'residual_total=2406154241'),
                              ('host_slab_bytes=1363148800', 'host_slab_bytes=1363148801')]:
            self.rejected(fixture() + '\n' + fixture().replace(before, after))

    def test_duplicate_or_cross_request_records(self):
        self.rejected(fixture() + '\n' + MODE + '\n' + fixture())
        self.rejected('\n'.join([PREFIX, MODE, PREFIX, MODE, BANK, PLACEMENT, BANK, PLACEMENT]))

    def test_malformed_or_allocation_limit(self):
        self.rejected(fixture() + '\n' + fixture().replace('staged=1', 'staged=-1'))
        self.rejected(fixture() + '\n' + fixture().replace('allocation_limit_bytes=5368709120', 'allocation_limit_bytes=1'))

    def test_coherent_fallback_keeps_actual_placement(self):
        log = fixture().replace('effective_slots=384 staged=1', 'effective_slots=512 staged=0')
        log = log.replace('1022361600', '1363148800')
        log = log.replace('residual_gpu_rows=117488', 'residual_gpu_rows=98304')
        log = log.replace('residual_gpu=2406154240', 'residual_gpu=2013265920')
        log = log.replace('residual_host=0', 'residual_host=392888320')
        row = validate(log, [117488], 'original', ENV)[0]
        self.assertEqual(row['bank']['staged'], 0)
        self.assertEqual(row['placement']['residual_host'], 392888320)


if __name__ == '__main__':
    unittest.main()
