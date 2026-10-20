#!/usr/bin/env python3
"""Offline admission/reducer regressions for independent cache capacity arms."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import cache_compare as c


def sample():
    return dict(ok=True,eligible=True,speed_eligible=True,stream_complete=True,finish_reason='stop',
        prompt_tokens=33211,read_tokens=33211,reused_tokens=0,generated_tokens=2932,usage_generated_tokens=2932,
        generated_tokens_source='usage',output_chars=5578,reasoning_chars=2677,completion_includes_thinking=True,
        drafts_offered=2500,drafts_accepted=2125,acceptance_rate=.85,decode_ms=200000.,decode_tps=14.66,
        decode_tps_source='engine',total_s=250.,prompt_ms=50000.,prompt_tps_source='engine',request_sha256='a'*64)


def rows():
    result=[]
    for i,arm in enumerate(c.ORDER):
        s=sample()
        if arm:s['decode_ms']/=1.10
        result.append(dict(arm=arm,source_commit='source',binary_sha256='binary',invariant_signature='signature',
            machine_identity={'boot':'boot'},model_assets_fingerprint='model',native_identity=dict(pid=100+i,start_ticks=1000+i),
            passed=True,ownership_closed=True,fresh_faults=[],observer_records_absent=True,successful_warmups=4,
            route_evidence=dict(actual_mode=10,expert_cache_slots=1280 if arm else 128,gpu_residual_rows=33210),request=s))
    return result


def native_log(arm,check=False):
    start=(f'strata generate: expert cache {1280 if arm else 128} slots, 2.46 GiB of VRAM; policy is\n'+
        ('strata serve: the prompt path borrows 1115 CUDA0 cache slots (2.16 GiB)\n' if arm else
         'strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n'))
    for n,g in ((74,24),(263,1),(909,1),(1068,8)):
        start+=f'strata serve: prompt {n} tokens = 0 reused + {n} read in 100 ms, {g} generated in 100 ms\n'
    start+=('strata prefill: layer-major candidate, tokens=33210 chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=680140800 residual_gpu_rows=33210 residual_gpu=680140800 residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592\n'
        'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=33210 mixed_fp16_bytes=0\n')
    if check:
        for layer in range(48):
            start+=(f'strata MoE mixed owners: phase=entry layer={layer} first=0 rows=33210 primary=0x1 direct_owners=1 previous_attention=checked_drain\n'
                    f'strata MoE mixed owners: phase=retired layer={layer} first=0 rows=33210 route_status=0 shared_HC=checked_drain callbacks=after_return\n')
    return start+('strata serve: prompt 33211 tokens = 0 reused + 33211 read in 50000 ms, 2932 generated in 200000 ms\n'
        'strata serve: SYCL_EXPERT_COPY submitted_blobs=0 submitted_bytes=0 completed_blobs=0 completed_bytes=0 host_view_groups=0 gpu_miss_groups=0 cpu_unique_misses=100 cpu_entries=110\n'
        'strata serve: decode expert cache hit rate: 10.2% (162365 hits / 1586400 lookups)\n')


class CacheComparison(unittest.TestCase):
    def test_clean_source_review_requires_closed_worker_and_exact_source(self):
        sha=c.hashlib.sha256(b'committed-source').hexdigest()
        row=dict(clean_comparison_accepted=True,worker_closed=True,source_commit='review-source',
            frozen_inputs={name:sha for name in c.REVIEW_INPUTS},report_sha256='report',native_review_sha256='curator')
        with patch.object(c.gen,'committed',return_value={c.SCOPE:row}), \
             patch.object(c.gen,'git',return_value=b'committed-source'), \
             patch.object(c.gen.common,'digest',return_value=sha) as digest:
            self.assertEqual(c.source_review()['source_commit'],'review-source')
            row['worker_closed']=False
            with self.assertRaises(RuntimeError):c.source_review()
            row['worker_closed']=True
            digest.return_value='changed'
            with self.assertRaises(RuntimeError):c.source_review()
            digest.return_value=sha;row['frozen_inputs'].pop(c.REVIEW_INPUTS[0])
            with self.assertRaises(RuntimeError):c.source_review()

    def test_faster_decode_and_prefill_are_separate(self):
        values=rows();r=c.summarize(values)
        self.assertEqual(r['decode_decision'],'candidate_faster_screen')
        self.assertTrue(r['prefill_nonregression'])
        for row in values:
            if row['arm']:row['request']['prompt_ms']*=1.10
        r=c.summarize(values)
        self.assertEqual(r['decode_decision'],'candidate_faster_screen')
        self.assertEqual(r['prefill_decision'],'candidate_slower_screen')
        self.assertFalse(r['prefill_nonregression'])
        self.assertFalse(r['adoption_qualified'])

    def test_small_output_difference_is_allowed(self):
        values=rows()
        for row in values:
            if row['arm']:
                row['request']['generated_tokens']+=1;row['request']['usage_generated_tokens']+=1
        self.assertEqual(c.summarize(values)['decode_decision'],'candidate_faster_screen')

    def test_large_length_or_MTP_change_is_inconclusive(self):
        for key,value in (('generated_tokens',3500),('acceptance_rate',.70)):
            values=rows();s=values[1]['request'];s[key]=value
            if key=='generated_tokens':s['usage_generated_tokens']=value
            else:s['drafts_accepted']=1750
            with self.subTest(key=key):
                result=c.summarize(values)
                self.assertEqual(result['decode_decision'],'inconclusive_generation_or_MTP_regime')
                self.assertTrue(result['prefill_nonregression'])
                self.assertEqual(result['prefill_decision'],'inconclusive_variability_or_small_effect')

    def test_one_reversed_pair_cannot_be_called_faster(self):
        values=rows();values[1]['request']['decode_ms']*=1.20
        self.assertEqual(c.summarize(values)['decode_decision'],'inconclusive_variability_or_small_effect')

    def test_reject_incomplete_owners_wrong_route_clock_and_identity(self):
        for change in ('missing','owner','observers','warmup','route','identity','prompt','clock'):
            values=rows()
            if change=='missing':values.pop()
            elif change=='owner':values[1]['ownership_closed']=False
            elif change=='observers':values[1]['observer_records_absent']=False
            elif change=='warmup':values[1]['successful_warmups']=3
            elif change=='route':values[1]['route_evidence']['actual_mode']=2
            elif change=='identity':values[1]['native_identity']=copy.deepcopy(values[0]['native_identity'])
            elif change=='prompt':values[1]['request']['request_sha256']='b'*64
            elif change=='clock':values[1]['request']['prompt_ms']=0
            with self.subTest(change=change),self.assertRaises(RuntimeError):c.summarize(values)

    def test_signature_accepts_only_the_planned_arm_differences(self):
        config=dict(args=['--expert-cache','128','--no-prefill-borrow','--expert-profile','/a/profile'],log='/a/log')
        env=dict(STRATA_SYCL_CACHE_SCRATCH_LEASE='0',TMPDIR='/a/tmp',OTHER='same')
        original=c.invariant_signature(config,env,Path('/a'))
        changed=dict(args=['--expert-cache','1280','--expert-profile','/b/profile'],log='/b/log')
        other=dict(STRATA_SYCL_CACHE_SCRATCH_LEASE='1',TMPDIR='/b/tmp',OTHER='same')
        self.assertEqual(original,c.invariant_signature(changed,other,Path('/b')))
        other['OTHER']='different'
        self.assertNotEqual(original,c.invariant_signature(changed,other,Path('/b')))

    def test_both_clean_native_routes_and_owned_check(self):
        for arm,check in ((0,False),(1,False),(0,True)):
            with self.subTest(arm=arm,check=check):
                self.assertEqual(c.native_evidence(native_log(arm,check),sample(),arm,check)['actual_mode'],10)

    def test_reject_fallback_observers_and_wrong_measured_native_request(self):
        text=native_log(1)
        for before,after in (('effective_mode=10','effective_mode=2'),('residual_host=0','residual_host=20480'),
                ('1280 slots','1279 slots'),('2932 generated','2931 generated'),('chunk=8192','chunk=4096')):
            with self.subTest(before=before),self.assertRaises(RuntimeError):
                c.native_evidence(text.replace(before,after),sample(),1,False)
        with self.assertRaises(RuntimeError):c.native_evidence(text+'strata trace: extra observer\n',sample(),1,False)
        with self.assertRaises(RuntimeError):c.native_evidence(text,sample(),0,False)

    def test_reject_owned_ffn_retirement_before_its_entry(self):
        text=native_log(0,True)
        line=next(x for x in text.splitlines() if 'phase=retired layer=0 ' in x)
        text=text.replace(line+'\n','').replace('strata MoE mixed owners: phase=entry layer=0 ',line+'\nstrata MoE mixed owners: phase=entry layer=0 ')
        with self.assertRaises(RuntimeError):c.native_evidence(text,sample(),0,True)


if __name__=='__main__':unittest.main()
