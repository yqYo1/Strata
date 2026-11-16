"""Predictive scaling and cohort-separation controls; repository LICENSE."""
import copy
import importlib.util
from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))

spec = importlib.util.spec_from_file_location('prefill_scaling',
    Path(__file__).resolve().parents[2]/'tools/prefill_scaling.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class PrefillScaling(unittest.TestCase):
    @staticmethod
    def samples(lengths=(8192, 16384, 32768, 49152, 65536, 98304)):
        return [dict(source_commit='a'*40, configuration_id='fixed', route='GPU-throughput',
            scope='subsequent-request', process=str(p), input_tokens=n, reused_tokens=0,
            context_before=0, prefill_s=2+n*.0007+(p-1)*.02,
            ttft_s=2.2+n*.0007+(p-1)*.02) for n in lengths for p in range(3)]

    def test_linear_held_out_lengths_recover_predictions_not_just_training_points(self):
        group, = t.analyze(self.samples())['groups']
        model = group['analysis']['prefill_s']['candidates'][0]
        self.assertTrue(model['eligible'])
        self.assertLess(model['held_out_RMSE_s'],1e-10)
        for held in model['held_out']:
            self.assertAlmostEqual(held['predicted_s'],2+held['tokens']*.0007,places=9)
        self.assertFalse(model['physical_fixed_cost_established'])

    def test_curvature_is_visible_in_held_out_error(self):
        rows = self.samples()
        for row in rows:
            row['prefill_s'] += 1e-9*row['input_tokens']**2
        candidates = t.analyze(rows)['groups'][0]['analysis']['prefill_s']['candidates']
        linear, quadratic = candidates[0], candidates[1]
        self.assertGreater(linear['held_out_RMSE_s'],.1)
        self.assertLess(quadratic['held_out_RMSE_s'],1e-9)

    def test_two_lengths_preserve_samples_and_slope_without_a_fitted_law(self):
        result = t.analyze(self.samples((32769,65537)))
        group, = result['groups']
        self.assertFalse(group['analysis']['prefill_s']['models_eligible'])
        self.assertEqual(group['analysis']['prefill_s']['candidates'],[])
        self.assertEqual(group['points'][0]['processes'],3)
        self.assertAlmostEqual(group['analysis']['prefill_s']['adjacent_slopes_ms_per_token'][0]['slope'],.7)
        self.assertFalse(result['physical_law_established'])

    def test_routes_sources_configurations_and_first_use_stay_separate(self):
        rows = self.samples()
        for field, value in [('route','host-tail'),('scope','first-use'),
                ('source_commit','b'*40),('configuration_id','other')]:
            other = copy.deepcopy(rows)
            for row in other:
                row[field] = value
                row['prefill_s'] *= 2
            result = t.analyze(rows+other)
            self.assertEqual(len(result['groups']),2,field)

    def test_one_process_with_many_requests_does_not_dominate_other_processes(self):
        rows = self.samples((32768,))
        rows[0]['prefill_s'] = 100
        rows += [copy.deepcopy(rows[0]) for _ in range(40)]
        point = t.analyze(rows)['groups'][0]['points'][0]
        self.assertAlmostEqual(point['prefill_s']['median'],2+32768*.0007+.02)
        self.assertEqual(point['processes'],3)

    def test_reuse_context_invalid_counts_and_nonfinite_times_are_refused(self):
        for field, value in [('reused_tokens',1),('reused_tokens',False),('context_before',32),
                ('input_tokens',True),('input_tokens',0),('prefill_s',float('nan')),
                ('ttft_s',-1),('scope','mixed')]:
            rows = self.samples();rows[0][field] = value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                t.analyze(rows)

    def test_coherent_adapter_requires_actual_routes_same_binary_and_fresh_cache_policy(self):
        route=dict(actual_chunk=8192,residual_dtype='bf16',configured_context=262144,
            configured_resident=20480,actual_typed_QSA=True,residual_gpu_rows=16,rows=16,
            fast_window=dict(staged=True,split=False,prefill_slots=384))
        binary=dict(path='/exact/strata',sha256='b'*64)
        process=dict(passed=True,clean_speed=True,native_exit_code=0,ownership_closed=True,
            native_source_commit='a'*40,binary=binary,fresh_faults=[],index=0,read=17,
            routes=[route,route],samples=[dict(actual_read_tokens=17,actual_reused_tokens=0,
                prompt_ms=10,ttft_s=.012),dict(actual_read_tokens=17,actual_reused_tokens=0,
                prompt_ms=8,ttft_s=.01)])
        record=dict(stage='closed',passed=True,complete=True,ownership_closed=True,active=False,
            native_source_commit='a'*40,binary=binary,environment_template_sha256='c'*64,
            model_assets_sha256='d'*64,configuration=dict(args=['--conversation-cache-mib','0',
                '--prompt-cache','0']),processes=[process])
        normalized=t.coherent_samples(record)
        self.assertEqual([r['scope'] for r in normalized],['first-use','subsequent-request'])
        self.assertEqual([r['prefill_s'] for r in normalized],[.01,.008])
        self.assertEqual(normalized[0]['route']['prefill_slots'],384)
        for mutation in ['binary','cache','route','typed_time']:
            changed=copy.deepcopy(record)
            if mutation=='binary':changed['processes'][0]['binary']=dict(binary,sha256='f'*64)
            elif mutation=='cache':changed['configuration']['args'][-1]='1'
            elif mutation=='route':changed['processes'][0]['routes'].pop()
            else:changed['processes'][0]['samples'][0]['prompt_ms']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                t.coherent_samples(changed)


if __name__ == '__main__':
    unittest.main()
