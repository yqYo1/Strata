"""Host regressions for length selection and genuinely reserved validation.

Main-authored,repository LICENSE. Analytic synthetic functions exercise model
recovery and held-out independence;these tests execute no GPU or model.
"""
import copy
from contextlib import redirect_stderr
import io
import math
from pathlib import Path
import sys
import unittest

import prefill_curve_fit as fit
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import eager_prefill_curve as campaign
import strata_bench as bench


def points(function):
    result=[]
    for n in (*fit.TRAIN_READS,fit.HELD_OUT_READ):
        t=function(n/65536)
        result.append(dict(read=n,first_use=dict(prompt_ms=t+2000,ttft_s=(t+2200)/1000),
            subsequent_prompt_ms=[t,t],subsequent_TTFT_s=[(t+200)/1000]*2))
    return result


class CurveFits(unittest.TestCase):
    def test_every_controller_command_passes_the_real_cli_scope(self):
        for read in campaign.READS:
            with self.subTest(read=read):
                command=campaign.command(dict(root='/tmp/prefill-curve-cli-test',port=18765,read=read))
                args=bench.parse_arguments(command[2:])
                self.assertEqual(args.prefill_exact_tokens,read)
                self.assertTrue(args.prefill_only and args.think)
                self.assertEqual(args.prefill_short_diagnostic,read<32768)
                if read<32768:
                    command.remove('--prefill-short-diagnostic')
                    with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                        bench.parse_arguments(command[2:])

    def test_affine_and_curved_functions_recover_coefficients(self):
        for name,function,coefficients in [
            ('affine',lambda x:10000+50000*x,dict(a=10000,b=50000)),
            ('quadratic',lambda x:10000+20000*x+25000*x*x,dict(a=10000,b=20000,c=25000)),
            ('n_log_n',lambda x:20000+50000*x+25000*x*math.log2(x),dict(a=20000,b=50000,c=25000))]:
            with self.subTest(name=name):
                value=fit.characterize(points(function))['channels']['warmed_prompt_ms']
                self.assertEqual(value['selected_model'],name)
                for key,expected in coefficients.items():
                    self.assertAlmostEqual(value['candidates'][name]['coefficients_ms'][key],expected,places=6)
                self.assertLess(value['candidates'][name]['independent56K']['relative_error'],1e-10)

    def test_reserved_point_cannot_choose_model_or_coefficients(self):
        data=points(lambda x:10000+20000*x+25000*x*x);before=fit.characterize(data)
        reserved=next(p for p in data if p['read']==fit.HELD_OUT_READ)
        reserved['first_use']['prompt_ms']*=10;reserved['first_use']['ttft_s']*=10
        reserved['subsequent_prompt_ms']=[v*10 for v in reserved['subsequent_prompt_ms']]
        reserved['subsequent_TTFT_s']=[v*10 for v in reserved['subsequent_TTFT_s']]
        after=fit.characterize(data)
        for channel in fit.CHANNELS:
            a,b=before['channels'][channel],after['channels'][channel]
            self.assertEqual(a['selected_model'],b['selected_model'])
            self.assertEqual(a['minimum_LOO_model'],b['minimum_LOO_model'])
            self.assertFalse(b['held_out_used_for_fit_or_selection'])
            for name in a['candidates']:
                self.assertEqual(a['candidates'][name]['coefficients_ms'],b['candidates'][name]['coefficients_ms'])
            self.assertGreater(b['candidates'][b['selected_model']]['independent56K']['relative_error'],.8)

    def test_warmed_spread_may_prefer_simple_model_without_hiding_others(self):
        data=points(lambda x:10000+50000*x+500*x*x)
        for p in data:
            t=p['subsequent_prompt_ms'][0];p['subsequent_prompt_ms']=[t*.9,t*1.1]
        result=fit.characterize(data)['channels']['warmed_prompt_ms']
        self.assertEqual(result['minimum_LOO_model'],'quadratic')
        self.assertEqual(result['selected_model'],'affine')
        self.assertAlmostEqual(result['training_relative_half_range_envelope'],.1)
        self.assertIn('affine',result['candidates_within_training_envelope'])
        self.assertIn('quadratic',result['candidates_within_training_envelope'])
        self.assertEqual(len(result['candidates']),3)

    def test_wrong_lengths_duplicates_and_invalid_clocks_fail(self):
        mutations=[lambda p:p.pop(),lambda p:p[0].update(read=32769),
            lambda p:p[0].update(read=16384),lambda p:p[0]['subsequent_prompt_ms'].__setitem__(0,float('nan')),
            lambda p:p[0]['first_use'].update(ttft_s=0),
            lambda p:p[0].update(subsequent_TTFT_s=[1.]),lambda p:p[0].update(read=True)]
        for mutate in mutations:
            p=points(lambda x:10000+50000*x);mutate(p)
            with self.assertRaises(ValueError):fit.characterize(p)

    def test_input_order_does_not_change_fits_or_mutate_samples(self):
        p=points(lambda x:10000+50000*x);original=copy.deepcopy(p)
        self.assertEqual(fit.characterize(p),fit.characterize(list(reversed(p))))
        self.assertEqual(p,original)


if __name__=='__main__':unittest.main()
