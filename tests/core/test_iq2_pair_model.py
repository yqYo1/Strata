"""Comparison decisions use independent processes, real eligibility and separate phases.

The retained pilot supplies the schema/real request fixture through its pinned
Git revision. Rate perturbations below are synthetic and are never measurements.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import iq2_pair_model as subject


class Comparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observed = subject.pilot()

    def processes(self, decode=1.03, prefill=1.0):
        rows=[]
        for i,arm in enumerate(subject.ORDER):
            d=self.observed
            r={k:copy.deepcopy(d[k]) for k in ('requests','iq2_dispatch','census','routes','decode_timing','warmups')}
            r.update(cohort=i,arm=arm,passed=True,complete=True,active=False,ownership_closed=True,
                native_exit_code=0,boot_unchanged=True,fault_probe_successful=True,fresh_faults=[],effective_environment_verified=True,
                native_identity=dict(pid=100+i,start_ticks=1000+i),binary_sha256=d['binary']['sha256'],
                build_source_commit=d['build_source_commit'],benchmark_fingerprint='same-client',qualification_sha256='same-proof',
                machine_identity_sha256='same-machine',model_assets_fingerprint='same-model',observer_environment_sha256='same-observers')
            for audit in r['iq2_dispatch']:audit['pair_rows']=audit['nt1_rows']*arm
            for j,s in enumerate(r['requests']):
                factor=decode[j] if isinstance(decode,tuple) else decode
                if arm:
                    s['decode_ms']/=factor;s['decode_tps']*=factor
                    s['prompt_ms']/=prefill;s['prompt_tps']*=prefill
            rows.append(r)
        return rows

    def test_repeated_decode_gain_and_prefill_nonregression(self):
        decision=subject.summarize(self.processes())
        self.assertTrue(decision['eligible_for_clean_comparison'])
        self.assertTrue(decision['MTP_pair_confirmed'])
        self.assertFalse(decision['clean_speed_qualified'])
        self.assertFalse(decision['model_adoption_qualified'])
        for case in decision['cases']:
            self.assertEqual(case['decode']['decision'],'faster_screen')
            self.assertAlmostEqual(case['decode']['geomean'],1.03)
            self.assertEqual(case['prefill']['decision'],'inconclusive')

    def test_faster_prefill_cannot_hide_slower_decode(self):
        decision=subject.summarize(self.processes(decode=.96,prefill=1.10))
        self.assertFalse(decision['eligible_for_clean_comparison'])
        for case in decision['cases']:
            self.assertEqual(case['prefill']['decision'],'faster_screen')
            self.assertEqual(case['decode']['decision'],'slower_screen')
            self.assertFalse(case['decode']['nonregression'])

    def test_workloads_are_judged_independently(self):
        decision=subject.summarize(self.processes(decode=(1.04,.96)))
        self.assertFalse(decision['eligible_for_clean_comparison'])
        self.assertEqual([x['decode']['decision'] for x in decision['cases']],['faster_screen','slower_screen'])

    def test_variation_is_not_dismissed_as_noise(self):
        rows=self.processes()
        for sample in rows[1]['requests']:
            sample['decode_ms']*=1.03/.99
        decision=subject.summarize(rows)
        self.assertFalse(decision['eligible_for_clean_comparison'])
        self.assertTrue(all(c['decode']['decision']=='inconclusive' for c in decision['cases']))

    def test_refuse_incomplete_order_reused_process_and_changed_identity(self):
        rows=self.processes()
        for bad in [rows[:-1],list(reversed(rows))]:
            with self.assertRaises(RuntimeError):subject.summarize(bad)
        for key in ['native_identity','binary_sha256','benchmark_fingerprint','qualification_sha256',
                    'machine_identity_sha256','model_assets_fingerprint','observer_environment_sha256']:
            bad=copy.deepcopy(rows)
            bad[1][key]=copy.deepcopy(rows[0][key]) if key=='native_identity' else 'changed'
            with self.subTest(key=key),self.assertRaises(RuntimeError):subject.summarize(bad)
        bad=copy.deepcopy(rows);bad[0]['arm']=False
        with self.assertRaises(RuntimeError):subject.summarize(bad)

    def test_refuse_fault_bad_dispatch_missing_work_and_wrong_route(self):
        rows=self.processes()
        for key,value in [('fresh_faults',['fault']),('ownership_closed',False),('complete',False),
                          ('passed',1),('native_exit_code',1),('active',True),('effective_environment_verified',False)]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):subject.summarize(bad)
        for key in ['census','routes','decode_timing','iq2_dispatch']:
            bad=copy.deepcopy(rows);bad[0][key].pop()
            with self.subTest(key=key),self.assertRaises(RuntimeError):subject.summarize(bad)
        for key,value in [('pair_rows',1),('row_ops',1),('nt1_rows',1),('cpus',[0,1,2,3,4,6])]:
            bad=copy.deepcopy(rows);bad[0]['iq2_dispatch'][0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):subject.summarize(bad)
        bad=copy.deepcopy(rows);bad[0]['routes'][0]['restored_dense_layers']=47
        with self.assertRaises(RuntimeError):subject.summarize(bad)

    def test_request_eligibility_and_body_binding(self):
        for key,value in [('read_tokens',100),('generated_tokens',1),('reasoning_chars',0),
                          ('output_chars',1),('prompt_ms',0),('prompt_ms',float('nan')),('prompt_tps_source','client')]:
            bad=self.processes();bad[0]['requests'][0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):subject.summarize(bad)
        bad=self.processes();bad[1]['requests'][0]['request_sha256']='changed'
        with self.assertRaises(RuntimeError):subject.summarize(bad)

    def test_output_hashes_are_not_an_equality_requirement(self):
        rows=self.processes()
        for s in rows[1]['requests']:s.update(output_sha256='different-answer',reasoning_sha256='different-thought')
        self.assertTrue(subject.summarize(rows)['eligible_for_clean_comparison'])

    def test_observed_mtp_classes_are_required_in_each_arm(self):
        rows=self.processes();s=rows[1]['requests'][1]
        s['drafts_accepted']=s['drafts_offered']//2;s['acceptance_rate']=s['drafts_accepted']/s['drafts_offered']
        decision=subject.summarize(rows)
        self.assertFalse(decision['MTP_pair_confirmed'])
        self.assertFalse(decision['eligible_for_clean_comparison'])


if __name__=='__main__':unittest.main()
