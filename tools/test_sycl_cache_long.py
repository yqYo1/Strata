#!/usr/bin/env python3
"""Reject incomplete/stale/aliased117K admission evidence without a GPU."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import cache_long as c
from strata_bench import RequestJournal, Suite, client_fingerprint

SAMPLE = dict(run=0,actual_prompt_tokens=117489,actual_read_tokens=117489,actual_reused_tokens=0,
    generated_tokens=24,ok=True,eligible=True,stream_complete=True,thinking_requested=True,
    reasoning_observed=True,prompt_tps_source='engine',request_sha256='a'*64)


def fixture():
    lines=[]
    for generation, tokens in ((1,100),(2,117489)):
        long=generation == 2
        chunk,first,capacity,need = (8192,128,2329548800,2319958528) if long else (256,1160,241280000,240304384)
        lent=1256-first
        def pub(phase,ready):
            return f'strata cache loan: phase={phase} generation={generation} chunk={chunk} first={first} lent={lent} decode_ready={ready}'
        lines.extend([f'strata loan audit: phase=layout chunk={chunk} first={first} slots=1256 bytes={capacity} need={need} lent={lent} mask_match=1',
            'strata loan audit: phase=lend table_entries=24576 table_match=1 free_bytes=400000000 total_bytes=10666115072',pub('masked',0)])
        if long:
            def dense(phase,residency,activations):
                return f'strata prefill dense: phase={phase} residency={residency} activations={activations} full_bytes=2116951040 bank_bytes=50918400 snapshot_bytes=2116951040 overlay_rows=0 session_recapture=0 verifier=1 free_bytes=400000000 total_bytes=10666115072'
            lines.extend([dense('retired','Full',0),dense('bank','Bank',0),
                'strata MoE accumulator admission: requested_mode=10 rows=81920 gpu_prefix_rows=117488 shared_arena_bytes=1180698624 free_gpu_bytes=4250000000 fixed_including_headroom_bytes=4206942336 mixed_fp16_bytes=419430400',
                'strata MoE mixed banks: storage=scratch primary_bytes=335544320 tail_bytes=83886080 tail_offset=1059131904 extra_allocation_bytes=0 phase=FFN_only_after_attention_drain',
                'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920 mixed_fp16_bytes=0',
                'strata prefill cache binding: generation=2 parent_match=1 execution_distinct=1',
                'strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=2406154240 residual_gpu_rows=117488 residual_gpu=2406154240 residual_host=0 residual_staging=0 layer_cache=1022361600; extra allocations, GPU headroom=536870912, RAM headroom=8589934592'])
            for layer in range(48):
                for firstrow,rows in c.WINDOWS:
                    tail=max(rows-65536,0)*5120
                    pointer=hex(0x2000000000+1059131904) if tail else '(nil)'
                    lines.extend([f'strata MoE mixed owners: phase=entry layer={layer} first={firstrow} rows={rows} primary=0x1000000000 primary_bytes={min(rows,65536)*5120} region=0x2000000000 region_bytes=1180698624 tail={pointer} tail_offset=1059131904 tail_bytes={tail} residual={hex(0x3000000000+firstrow*20480)} residual_bytes={rows*20480} bo=0x4000000000 queue=0x5000000000 direct_owners=0 previous_attention=checked_drain',
                        f'strata MoE cache loan: generation=2 layout_generation=4 layer={layer} first={firstrow} rows={rows} used=2311569920 capacity=2329548800',
                        f'strata MoE mixed owners: phase=retired layer={layer} first={firstrow} rows={rows} route_status=0 shared_HC=checked_drain callbacks=after_return'])
            lines.append(dense('restored','Full',48))
        lines.extend([f'strata loan audit: phase=payload rows=1256 bytes=2575206400 lent={lent} same_slots=1 payload_match=1',
            'strata loan audit: phase=refill table_entries=24576 table_match=1 free_bytes=100000000 total_bytes=10666115072',pub('ready',1),
            f'strata serve: prompt {tokens} tokens = 0 reused + {tokens} read in 100 ms (1.0 tok/s), 24 generated in 100 ms'])
    return '\n'.join(lines)+'\n'


class MixedEvidence(unittest.TestCase):
    def test_batched_warmup_config_is_explicit(self):
        c.require_batched_warmup(dict(args=['--short-read','0']))
        for flags in ([],['--short-read'],['--short-read','64'],['--short-read','0','--short-read','64']):
            with self.subTest(flags=flags),self.assertRaises(RuntimeError): c.require_batched_warmup(dict(args=flags))

    def test_short_window_route_or_oversized_warmup_is_rejected(self):
        text=fixture()
        end=text.index('strata serve: prompt 100')
        no_loan=text[end:]
        for changed in (no_loan,text.replace('prompt 100 tokens = 0 reused + 100 read','prompt 257 tokens = 0 reused + 257 read')):
            with self.assertRaises(RuntimeError): c.mixed_evidence(changed,SAMPLE)

    def test_full_two_bank_sequence(self):
        proof=c.mixed_evidence(fixture(),SAMPLE)
        self.assertEqual(proof['checked_retirements'],96)
        self.assertEqual(proof['secondary_bank_windows'],48)
        self.assertEqual(proof['residual_gpu_bytes'],2406154240)
        self.assertFalse(proof['clean_speed_qualified'])

    def test_reject_extent_generation_and_fallback_changes(self):
        edits=[('generation=2 layout_generation=4 layer=17','generation=1 layout_generation=4 layer=17'),
            ('layout_generation=4 layer=17','layout_generation=5 layer=17'),
            ('first=81920 rows=35568','first=81921 rows=35568'),
            ('tail_bytes=83886080','tail_bytes=83886078'),
            ('residual_gpu=2406154240','residual_gpu=1203077120'),
            ('effective_mode=10','effective_mode=2'),('slots=1256','slots=1280'),
            ('payload_match=1','payload_match=0'),('table_match=1','table_match=0'),
            ('parent_match=1','parent_match=0'),('direct_owners=0','direct_owners=1'),
            ('callbacks=after_return','callbacks=before_return'),('route_status=0','route_status=1'),
            ('residency=Full activations=48','residency=Full activations=47'),
            ('tail_offset=1059131904','tail_offset=1180698624'),
            ('region_bytes=1180698624','region_bytes=1059131904'),
            ('free_gpu_bytes=4250000000','free_gpu_bytes=4206942335'),
            ('used=2311569920','used=18446744073709551616')]
        for before,after in edits:
            with self.subTest(before=before),self.assertRaises(RuntimeError):
                c.mixed_evidence(fixture().replace(before,after,1),SAMPLE)

    def test_reject_missing_duplicate_and_cross_request_retirement(self):
        text=fixture(); line=next(x for x in text.splitlines() if 'phase=retired layer=47 first=81920' in x)
        for changed in (text.replace(line+'\n',''), text+line+'\n', line+'\n'+text.replace(line+'\n','')):
            with self.subTest(change=changed[:100]),self.assertRaises(RuntimeError): c.mixed_evidence(changed,SAMPLE)

    def test_reject_overlapping_and_moving_owners(self):
        for before,after in [('bo=0x4000000000','bo=0x1000000000'),
            ('region=0x2000000000','region=0x1000000000'),
            ('primary=0x1000000000','primary=0x1000000001'),
            ('queue=0x5000000000','queue=0x5000000001')]:
            with self.subTest(before=before),self.assertRaises(RuntimeError):
                c.mixed_evidence(fixture().replace(before,after,1),SAMPLE)

    def test_reject_ready_before_completed_loan(self):
        text=fixture(); ready=next(x for x in text.splitlines() if 'phase=ready generation=2' in x)
        text=text.replace(ready+'\n','').replace('strata prefill cache binding:',ready+'\nstrata prefill cache binding:')
        with self.assertRaises(RuntimeError): c.mixed_evidence(text,SAMPLE)

    def test_reject_all_long_admission_markers_outside_long_request(self):
        text=fixture()
        for prefix in c.LONG_PREFIXES:
            line=next(v for v in text.splitlines() if v.startswith(prefix))
            for changed in (line+'\n'+text,text+line+'\n'):
                with self.subTest(prefix=prefix),self.assertRaises(RuntimeError): c.mixed_evidence(changed,SAMPLE)


class OfficialJournal(unittest.TestCase):
    def evidence(self, root, change=None):
        directory=root/'bench'/'one'; directory.mkdir(parents=True)
        sample=copy.deepcopy(SAMPLE)
        result=dict(params=dict(mode='prefill-only',runs=1,think=True,prefill_tokens=98304),complete=True,
            results=dict(failures=[],prefill_only=dict(complete=True,eligible_runs=1,requests=[sample],max_tokens=24,thinking_requested=True)))
        suite=Suite('http://127.0.0.1:1',None,131072,1,True,prefill_only=True,prefill_tokens=98304,
            workload_id='native_gdn_20261018')
        plan=suite.prefill_plan(); model=dict(model='fixture-model',quant='IQ3_S',context=131072)
        result['machine']=model
        result['params']['workload']=suite.workload_identity()
        result['comparison_identity']=dict(**model,mode='prefill-only',thinking=True,
            workload=suite.workload_identity(),client=client_fingerprint(),prefill_plan=plan)
        result['results']['prefill_only'].update(plan)
        journal=RequestJournal(directory,lambda value:value)
        journal.append('identity','suite',**model,mode='prefill-only',thinking=True,requested_runs=1,
            workload=suite.workload_identity(),source=client_fingerprint(),prefill_plan=plan)
        for phase,run in (('warm-up/chat',None),('prefill-only',0)):
            start=journal.append('start',phase,run,request_sha256='a'*64,max_tokens=24,thinking=True)
            end={k:v for k,v in sample.items() if k != 'run'}
            if change: change(phase,end)
            journal.append('result',phase,run,request_index=start,sample=end)
        journal.state(complete=True,terminal='complete')
        return result

    def test_reject_extra_misplaced_or_changed_identity(self):
        for mutation in ('extra','source','model','workload'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); result=self.evidence(root)
                if mutation == 'extra':
                    # Rebuild with the real writer so the extra row's chain/state stay valid.
                    path=root/'bench'/'one'
                    writer=RequestJournal(path,lambda value:value)
                    entries=[json.loads(v) for v in (path/'requests.jsonl').read_text().splitlines()]
                    writer.index=len(entries);writer.previous=entries[-1]['sha256']
                    writer.append('identity','suite');writer.state(complete=True,terminal='complete')
                elif mutation == 'source': result['comparison_identity']['client']={}
                elif mutation == 'model': result['machine']['model']='different'
                else: result['params']['workload']={}
                with self.assertRaises(RuntimeError): c.official_evidence(root,result)

    def test_actual_hash_chained_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); result=self.evidence(root)
            sample,proof=c.official_evidence(root,result)
            self.assertEqual(sample,SAMPLE)
            self.assertEqual(proof['successful_warmups'],1)

    def test_reject_successful_but_different_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=self.evidence(root,lambda phase,row: row.update(request_sha256='b'*64))
            with self.assertRaises(RuntimeError): c.official_evidence(root,result)

    def test_reject_false_warmup_success_or_changed_suite_sample(self):
        for mode in ('warmup','suite'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                def change(phase,row):
                    if mode == 'warmup' and phase == 'warm-up/chat': row['ok']=False
                result=self.evidence(root,change)
                if mode == 'suite': result['results']['prefill_only']['requests'][0]['generated_tokens']=23
                with self.assertRaises(RuntimeError): c.official_evidence(root,result)


if __name__ == '__main__': unittest.main()
