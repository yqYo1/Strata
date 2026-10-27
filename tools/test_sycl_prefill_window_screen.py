#!/usr/bin/env python3
"""Host-only corrupted workload/route/clock rejection; repository LICENSE."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import prefill_window_screen as screen


class ScreenEvidence(unittest.TestCase):
    def setUp(self):
        self.row=dict(plan=dict(request_hashes=screen.request_hashes(),timing_floor_ms=10000))
        self.samples=[dict(run=i,ok=True,eligible=True,stream_complete=True,thinking_requested=True,reasoning_observed=True,
            request_sha256=s,actual_prompt_tokens=78451,actual_read_tokens=78451,actual_reused_tokens=0,
            generated_tokens=24,prompt_tps_source='engine',prompt_ms=78000.2) for i,s in enumerate(self.row['plan']['request_hashes'])]
        # Production marker grammar,including the historically mislabeled router-ID D2H field.
        part='\n'.join(['strata prefill: layer-major candidate, tokens=78450 chunk=8192 residual_dtype=bf16',
            'strata MoE window configuration: requested_mode=10 effective_mode=10',
            'strata MoE accumulator window: windows=48 dequants=20000 jobs=30000 source_H2D_bytes=150624000 GPU_resident_windows_only=1 operator_max=8192 maximum_rows=78450',
            'strata MoE GU fusion: fused_jobs=1000 fused_rows=2334000 separate_jobs=29000 min_rows=2334',
            'strata MoE Down fusion: fused_jobs=0 fused_rows=0 separate_jobs=30000 min_rows=2334',
            'strata MoE expert bank complete: uploads=20000 requested_H2D_bytes=1000000 permanent_hits=0 permanent_slots=200 total_slots=384 final_drain=checked',
            'strata serve: prompt 78451 tokens = 0 reused + 78451 read in 78000 ms '])
        self.text=(part+'\n')*4

    def test_production_grammar_and_router_direction(self):
        _,rates,work=screen.parse_samples(self.row,self.samples,self.text)
        self.assertEqual(len(rates),4)
        self.assertEqual(work[0]['requested_router_ID_D2H_bytes'],150624000)
        self.assertNotIn('source_H2D_bytes',work[0])

    def test_cache_output_thinking_payload_and_clock_mutations(self):
        changes={'actual_reused_tokens':1,'actual_read_tokens':78450,'generated_tokens':23,
            'thinking_requested':False,'reasoning_observed':False,'stream_complete':False,
            'request_sha256':'wrong','prompt_tps_source':'wall','prompt_ms':9999}
        for field,value in changes.items():
            with self.subTest(field=field):
                samples=copy.deepcopy(self.samples);samples[1][field]=value
                with self.assertRaises(RuntimeError):screen.parse_samples(self.row,samples,self.text)

    def test_route_work_and_clock_mutations(self):
        for old,new in [('windows=48','windows=96'),('maximum_rows=78450','maximum_rows=81920'),
            ('dequants=20000','dequants=20001'),('separate_jobs=29000','separate_jobs=28999'),
            ('final_drain=checked','final_drain=uncertain'),('78000 ms','78001 ms'),
            ('requested_mode=10 effective_mode=10','requested_mode=10 effective_mode=8'),
            ('source_H2D_bytes=150624000','source_H2D_bytes=150624001')]:
            with self.subTest(field=old),self.assertRaises(RuntimeError):
                screen.parse_samples(self.row,self.samples,self.text.replace(old,new,1))


class PrelaunchFailure(unittest.TestCase):
    def test_journal_and_late_runtime_failures_are_terminal_before_owner_creation(self):
        with tempfile.TemporaryDirectory(prefix='single-window-host-',dir=screen.B) as name:
            for failure in ('journal','python'):
                with self.subTest(failure=failure):
                    row=dict(stage='prepared',processes=[],plan={},inputs=[dict(root=name)])
                    review=dict(status='reviewed_model_control_eligible',model_trial_allowed=True,
                        source_sha256=screen.digest(screen.__file__),plan_identity=screen.identity(row['plan']))
                    with patch.object(screen,'current',return_value=({},{})) as admission, \
                            patch.object(screen.prior.runtime.validation,'committed',return_value=review), \
                            patch.object(screen.subprocess,'check_output',return_value='-- cursor: fixture\n') as journal, \
                            patch.object(screen.prior.python_runtime,'admitted') as python, \
                            patch.object(screen.prior.baseline,'Owner') as owner, patch.object(screen,'save') as saved:
                        (journal if failure=='journal' else python).side_effect=RuntimeError('injected '+failure+' failure')
                        with self.assertRaisesRegex(RuntimeError,'injected'):screen.run_one(row)
                        saved.assert_called_once_with(row);owner.assert_not_called()
                        self.assertEqual(row['stage'],'admission-failed')
                        self.assertEqual(row['processes'][0]['ownership'],'not_launched')
                        self.assertFalse(row['processes'][0]['native_launched'])
                        self.assertFalse(row['active']);self.assertTrue(row['ownership_closed'])
                        admission.reset_mock()
                        with self.assertRaisesRegex(RuntimeError,'stop on failure'):screen.run_one(row)
                        admission.assert_not_called();owner.assert_not_called();self.assertEqual(saved.call_count,1)


if __name__=='__main__':unittest.main(failfast=True)
