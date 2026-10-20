#!/usr/bin/env python3
"""False-adoption guards for independent117K retained-engine process pairs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import cache_long_compare as c
from strata_bench import RequestJournal, Suite, client_fingerprint


def rows(factor=1.04):
    result=[]
    for cohort,arm in enumerate(c.ORDER):
        samples=[]
        for run in range(4):
            speed=(900 if run==0 else 990)*(factor if arm else 1)
            samples.append(dict(run=run,ok=True,eligible=True,stream_complete=True,thinking_requested=True,
                reasoning_observed=True,actual_prompt_tokens=117489,actual_read_tokens=117489,
                actual_reused_tokens=0,generated_tokens=24,prompt_tps_source='engine',
                prompt_ms=117489000/speed,request_sha256=str(run)*64))
        result.append(dict(cohort=cohort,arm=arm,native_identity=dict(pid=100+cohort,start_ticks=10),
            benchmark_fingerprint='same',machine_identity={'boot':'same'},model_assets_fingerprint='same',
            binary_sha256=str(arm)*64,build_source_commit=str(arm)*40,passed=True,ownership_closed=True,
            fresh_faults=[],observer_records_absent=True,actual_route_checked=True,requests=samples))
    return result


class Comparison(unittest.TestCase):
    def test_warm_target_separate_from_first_use_and_decode(self):
        result=c.summarize(rows())
        self.assertEqual(result['prefill_decision'],'candidate_faster_screen')
        self.assertTrue(result['candidate1000tok_s_all_process_medians'])
        self.assertLess(result['first_use_tps'][1],1000)
        self.assertFalse(result['sustained_decode_qualified'])
        self.assertFalse(result['adoption_qualified'])

    def test_small_gain_and_regression_keep_original_decisions(self):
        self.assertEqual(c.summarize(rows(1.01))['prefill_decision'],'inconclusive_variability_or_small_effect')
        result=c.summarize(rows(.95))
        self.assertEqual(result['prefill_decision'],'candidate_slower_screen')
        self.assertFalse(result['prefill_nonregression'])

    def test_one_reversed_pair_prevents_direction_claim(self):
        cases=rows()
        for s in cases[1]['requests']: s['prompt_ms']*=1.08
        self.assertEqual(c.summarize(cases)['prefill_decision'],'inconclusive_variability_or_small_effect')

    def test_reject_partial_duplicate_or_changed_source(self):
        original=rows()
        variants=[original[:-1]]
        for key,value in [('native_identity',original[0]['native_identity']),('benchmark_fingerprint','different'),
            ('machine_identity',{}),('binary_sha256','different'),('build_source_commit','different'),
            ('ownership_closed',False),('fresh_faults',['fault']),('actual_route_checked',False)]:
            changed=copy.deepcopy(original);changed[1][key]=value;variants.append(changed)
        for changed in variants:
            with self.subTest(change=changed[1].get('binary_sha256')),self.assertRaises(RuntimeError): c.summarize(changed)

    def test_reject_bad_clock_reused_or_mismatched_work(self):
        for key,value in [('request_sha256','different'),('actual_reused_tokens',1),('actual_read_tokens',117488),
            ('prompt_ms',float('nan')),('prompt_ms',0),('prompt_ms',True),('eligible',False),('generated_tokens',25)]:
            cases=rows();cases[1]['requests'][2][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(RuntimeError): c.summarize(cases)


class Journal(unittest.TestCase):
    def evidence(self,root,count,wrong_identity=False):
        directory=root/'bench'/'one';directory.mkdir(parents=True)
        suite=Suite('http://127.0.0.1:1',None,131072,count,True,prefill_only=True,prefill_tokens=98304,
            workload_id='native_gdn_20261018')
        plan=suite.prefill_plan();model=dict(model='fixture',quant='IQ3_S',context=131072)
        samples=copy.deepcopy(rows()[0]['requests'][:count])
        result=dict(params=dict(mode='prefill-only',runs=count,think=True,prefill_tokens=98304,
            workload=suite.workload_identity()),complete=True,machine=model,
            comparison_identity=dict(**model,mode='prefill-only',thinking=True,workload=suite.workload_identity(),
                client=client_fingerprint(),prefill_plan=plan),
            results=dict(failures=[],prefill_only=dict(**plan,requests=samples,complete=True,eligible_runs=count)))
        journal=RequestJournal(directory,lambda value:value)
        journal.append('identity','suite',**model,mode='prefill-only',thinking=True,requested_runs=count,
            workload=suite.workload_identity(),source=client_fingerprint(),prefill_plan=plan)
        for phase,run in [('warm-up/chat',None),*(('prefill-only',i) for i in range(count))]:
            sample={k:v for k,v in samples[run or 0].items() if k != 'run'}
            start=journal.append('start',phase,run,max_tokens=24,thinking=True,request_sha256=sample['request_sha256'])
            journal.append('result',phase,run,request_index=start,sample=sample)
        if wrong_identity: journal.append('identity','suite')
        journal.state(complete=True,terminal='complete')
        return result

    def test_one_and_four_request_journals(self):
        for count in (1,4):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);result=self.evidence(root,count)
                samples,proof=c.official(root,result,count)
                self.assertEqual(len(samples),count)
                self.assertEqual(proof['successful_long_requests'],count)

    def test_reject_hash_valid_extra_identity_or_changed_body(self):
        for extra in (True,False):
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);result=self.evidence(root,4,extra)
                if not extra: result['results']['prefill_only']['requests'][2]['generated_tokens']=23
                with self.assertRaises(RuntimeError): c.official(root,result,4)


class Placement(unittest.TestCase):
    loan=dict(first=132,lent=1124,chunk=8192,capacity=2321664000,need=2319958528)
    borrowed='strata generate: expert cache 1256 slots, fixture\nstrata serve: the prompt path borrows 1124 CUDA0 cache slots (2.16 GiB)\n'
    owned='strata generate: expert cache 128 slots, fixture\nstrata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n'

    def test_exact_admitted_suffix_and_owned_route(self):
        self.assertEqual(c.cache_placement(self.borrowed,1,self.loan)['loan_first'],132)
        self.assertEqual(c.cache_placement(self.owned,0,self.loan)['loan_slots'],0)

    def test_reject_changed_or_ambiguous_loan(self):
        variants=[(self.borrowed.replace('1124','1123'),self.loan),
            (self.borrowed+self.borrowed.splitlines()[1]+'\n',self.loan),
            (self.borrowed,self.loan|dict(first=131)),
            (self.borrowed.replace('1256 slots','1255 slots'),self.loan),
            (self.borrowed+self.owned.splitlines()[1]+'\n',self.loan)]
        for text,loan in variants:
            with self.subTest(loan=loan,text=text),self.assertRaises(RuntimeError): c.cache_placement(text,1,loan)


class TerminalReceipt(unittest.TestCase):
    def check(self,root,probe,*,boot='same',persist_error=False):
        record=dict(complete=True,passed=True,stage='check',active=False,
            route_evidence=dict(checked_direct_owner_retirements=96),native_identity=dict(pid=1,start_ticks=1))
        owner=SimpleNamespace(active=None,commands=[])
        def persist():
            if persist_error: raise OSError('owner receipt unavailable')
            (root/'owner.json').write_text(json.dumps(record))
        with patch.object(c.subprocess,'check_output',side_effect=probe),patch.object(c,'read_boot',return_value=boot),\
             patch.object(c.guards,'driver_guard',return_value={'checked':True}):
            c.finalize(record,owner,'cursor','same',persist,root/'result.json')
        return record,json.loads((root/'result.json').read_text())

    def test_failed_or_timed_out_probe_keeps_unknown_faults_and_terminal_result(self):
        for failure in (c.subprocess.CalledProcessError(1,['journalctl']),c.subprocess.TimeoutExpired(['journalctl'],15)):
            with self.subTest(failure=type(failure).__name__),tempfile.TemporaryDirectory() as directory:
                record,saved=self.check(Path(directory),failure)
                self.assertFalse(record['passed']);self.assertTrue(saved['ownership_closed'])
                self.assertIsNone(saved['fresh_faults']);self.assertFalse(saved['fault_probe_successful'])
                self.assertIn('kernel_journal_probe_error',saved)
                self.assertTrue(saved['complete'])

    def test_normal_probe_and_changed_boot(self):
        for boot in ('same','changed'):
            with self.subTest(boot=boot),tempfile.TemporaryDirectory() as directory:
                record,saved=self.check(Path(directory),lambda *a,**k:'',boot=boot)
                self.assertEqual(saved['passed'],boot=='same')
                self.assertEqual(saved['fresh_faults'],[])
                self.assertTrue(saved['fault_probe_successful'])

    def test_result_still_written_if_owner_receipt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(OSError): self.check(root,lambda *a,**k:'',persist_error=True)
            saved=json.loads((root/'result.json').read_text())
            self.assertFalse(saved['passed']);self.assertIn('persistence_error',saved)


if __name__=='__main__':unittest.main()
