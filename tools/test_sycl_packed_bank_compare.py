#!/usr/bin/env python3
"""Offline clean-environment,request-boundary and paired-work refusal checks."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import packed_bank_compare as p


def counters():
    return ('strata MoE expert bank complete: uploads=19631 requested_H2D_bytes=40102860800 permanent_hits=13594 permanent_slots=267 total_slots=384 final_drain=checked\n'
        'strata MoE GU fusion: fused_jobs=7485 fused_rows=49290450 separate_jobs=29868 min_rows=2334\n'
        'strata MoE Down fusion: fused_jobs=7485 fused_rows=49290450 separate_jobs=29868 min_rows=2334\n'
        'strata MoE accumulator window: windows=96 dequants=33225 jobs=37353 source_H2D_bytes=225576960 GPU_resident_windows_only=1 operator_max=8192 maximum_rows=81920\n')


def clean_log():
    text=('strata generate: expert cache 1256 slots, admitted\n'
        'strata serve: the prompt path borrows 1124 CUDA0 cache slots admitted\n'
        'strata serve: prompt 73 tokens = 0 reused + 73 read in 1 ms, 24 generated in 1 ms\n')
    for _ in range(4):
        text+=('strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920 mixed_fp16_bytes=0\n'
            'strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=2406154240 residual_gpu_rows=117488 residual_gpu=2406154240 residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592\n'
            +counters()+'strata serve: prompt 117489 tokens = 0 reused + 117489 read in 120000 ms, 24 generated in 1 ms\n')
    return text


class CleanEvidence(unittest.TestCase):
    def test_clean_environment_preserves_runtime_and_math(self):
        settings=json.loads(p.gen.SETTINGS.read_text())['environment']
        env={k:settings[k] for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys',
            'EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE')}
        env.update({k:'1' for k in p.DROP});env.update(UR_LOG_LOADER='log',
            STRATA_SYCL_PREFILL_NATIVE16='1',STRATA_MTP_BATCH_F16='1',TMPDIR='/old/tmp')
        actual=p.environment(dict(root='/old',environment=env),dict(root='/new',arm=1))
        self.assertFalse(p.DROP&actual.keys());self.assertNotIn('UR_LOG_LOADER',actual)
        self.assertEqual(actual['STRATA_SYCL_PREFILL_NATIVE16'],'1')
        self.assertEqual(actual['STRATA_MTP_BATCH_F16'],'0')
        self.assertEqual(actual['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED'],'1')
        self.assertEqual(actual['TMPDIR'],'/new/tmp')
        env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']='0'
        with self.assertRaises(RuntimeError):p.environment(dict(root='/old',environment=env),dict(root='/new',arm=1))

    def test_completed_work_refuses_missing_duplicate_or_inconsistent_counters(self):
        good=counters();self.assertEqual(p.completed_work(good)['logical_work']['accumulator']['jobs'],37353)
        for bad in (good.replace('dequants=33225','dequants=33224'),good.replace('windows=96','windows=95'),
            good.replace('source_H2D_bytes=225576960','source_H2D_bytes=0'),good.replace('min_rows=2334','min_rows=2333'),
            good.replace('operator_max=8192','operator_max=4096'),good.replace('final_drain=checked','final_drain=pending'),
            good+'\n'+good,good.replace('separate_jobs=29868','separate_jobs=29869',1)):
            with self.subTest(bad=bad[:80]),self.assertRaises(RuntimeError):p.completed_work(bad)

    def test_production_route_binding_uses_real_shared_evidence_reader(self):
        samples=[dict(generated_tokens=24,request_sha256='request') for _ in range(4)]
        admitted=dict(geometry={'requested_work':p.completed_work(counters())['expert_bank']},sample=samples[0])
        loan=dict(first=132,lent=1124)
        self.assertEqual(len(p.native_work(clean_log(),samples,42,loan,admitted)),4)
        for bad in (clean_log()+counters(),clean_log()+'strata trace: unexpected\n',
            clean_log().replace('maximum_rows=81920','maximum_rows=65536',1),
            clean_log().replace('requested_H2D_bytes=40102860800','requested_H2D_bytes=40102860801',1)):
            with self.subTest(bad=bad[-100:]),self.assertRaises(RuntimeError):p.native_work(bad,samples,42,loan,admitted)

    def rows(self,gain=1.03):
        rows=[]
        for i,arm in enumerate(p.c.ORDER):
            speed=1000*(gain if arm else 1)
            requests=[dict(run=j,ok=True,eligible=True,stream_complete=True,thinking_requested=True,
                reasoning_observed=True,actual_prompt_tokens=117489,actual_read_tokens=117489,
                actual_reused_tokens=0,generated_tokens=24,prompt_tps_source='engine',
                prompt_ms=117489000/speed,request_sha256='body'+str(j)) for j in range(4)]
            work=p.completed_work(counters())
            if arm:work['expert_bank'].update(uploads=18937,requested_H2D_bytes=38673152000,permanent_hits=14288)
            rows.append(dict(cohort=i,arm=arm,native_identity=dict(pid=i+1,start_ticks=100+i),
                benchmark_fingerprint='bench',machine_identity='machine',model_assets_fingerprint='models',
                binary_sha256='binary',build_source_commit='native',passed=True,ownership_closed=True,
                fresh_faults=[],observer_records_absent=True,actual_route_checked=True,requests=requests,
                work=[copy.deepcopy(work) for _ in range(4)]))
        return rows

    def test_frozen_pair_decision_and_target_are_separate(self):
        result=p.summarize(self.rows())
        self.assertEqual(result['prefill_decision'],'candidate_faster_screen')
        self.assertTrue(result['candidate1000tok_s_all_process_medians'])
        self.assertFalse(result['adoption_qualified'])
        self.assertEqual(p.summarize(self.rows(1.001))['prefill_decision'],'inconclusive_variability_or_small_effect')
        slower=self.rows(.97);self.assertEqual(p.summarize(slower)['prefill_decision'],'candidate_slower_screen')
        self.assertFalse(p.summarize(slower)['prefill_nonregression'])

    def test_pairing_refuses_changed_body_work_or_owner(self):
        for variant in ('body','jobs','bytes','lookups','owners','same_process','missing_work','other_native'):
            rows=self.rows()
            if variant=='body':rows[1]['requests'][1]['request_sha256']='other'
            if variant=='jobs':rows[1]['work'][1]['logical_work']['GU']['fused_rows']+=1
            if variant=='bytes':rows[1]['work'][1]['expert_bank']['requested_H2D_bytes']=40102860800
            if variant=='lookups':rows[1]['work'][1]['expert_bank']['permanent_hits']-=1
            if variant=='owners':rows[1]['ownership_closed']=False
            if variant=='same_process':rows[1]['native_identity']=rows[0]['native_identity']
            if variant=='missing_work':rows[1]['work'].pop()
            if variant=='other_native':
                for row in rows:
                    if row['arm']:row['binary_sha256']='different-binary'
            with self.subTest(variant=variant),self.assertRaises(RuntimeError):p.summarize(rows)


if __name__=='__main__':unittest.main()
