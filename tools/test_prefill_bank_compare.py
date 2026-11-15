"""Refuse silent capacity fallback and isolate the declared bank setting."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent/'sycl'))
import prefill_bank_compare as bank


class PrefillBankTests(unittest.TestCase):
    def test_actual_slot_selection_and_capacity_are_required(self):
        raw=(bank.P/bank.prior.FIXTURE).read_text()
        p=dict(index=0,arm='clean',read=32769,heads=1,expected_bank_slots=512)
        good=raw.replace('requested_slots=384','requested_slots=512').replace('effective_slots=384','effective_slots=512').replace('prefill_slots=384','prefill_slots=512')
        good=good.replace('total_slots=384','total_slots=512')
        good=good.replace('device_bytes=1022361600','device_bytes=1363148800').replace('layer_cache=1022361600','layer_cache=1363148800')
        self.assertEqual(bank.request_route(good,p)['fast_window']['prefill_slots'],512)
        for text in (raw,good.replace('residual_gpu_rows=32768','residual_gpu_rows=24576'),
                     good.replace('decode_changed=0','decode_changed=1')):
            with self.assertRaises((ValueError,RuntimeError)):bank.request_route(text,p)

    def test_per_arm_environment_changes_only_bank(self):
        plans=[dict(index=i,arm='clean',read=32769,heads=1,expected_bank_slots=n) for i,n in enumerate(bank.BANKS)]
        row={k:{} for k in ('binary','native_validation_probe')};row.update(native_source_commit='1'*40,boot='boot')
        from unittest.mock import patch
        with patch.object(bank.prior,'environment',return_value={'fixed':'yes','STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS':'384'}):
            for p in plans:
                self.assertEqual(bank.environment(row,p),dict(fixed='yes',STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS=str(p['expected_bank_slots'])))

    def test_three_pairs_are_required_for_advancement(self):
        samples=[dict(prompt_ms=40.,ttft_s=41.)]*3
        route=lambda n:dict(fast_window=dict(requested_bank_H2D_bytes=n))
        rows=[dict(passed=True,complete=True,ownership_closed=True,native_exit_code=0,fresh_faults=[],
            native_identity=dict(pid=i+1,start_ticks=i+100),
            samples=copy.deepcopy(samples),subsequent_median_ms=39. if n==512 else 40.,
            routes=[route(90 if n==512 else 100)]*3) for i,n in enumerate(bank.BANKS)]
        self.assertTrue(bank.decision(rows)['advance_to_length_and_decode_checks'])
        rows[3]['subsequent_median_ms']=41.
        self.assertFalse(bank.decision(rows)['advance_to_length_and_decode_checks'])


if __name__=='__main__':unittest.main()
