#!/usr/bin/env python3
"""Offline negative evidence gates for the real long-generation controller."""
import copy
import json
from pathlib import Path
import tempfile
import generation_engine as g
import unittest
from generation_engine import collect_samples, require_complete_generation, sample_eligible


def sample():
    return dict(eligible=True, speed_eligible=True, ok=True, stream_complete=True,
        finish_reason='stop', reused_tokens=0, prompt_tokens=33792, read_tokens=33792,
        usage_generated_tokens=2048, generated_tokens=2048, generated_tokens_source='usage',
        output_chars=2000, reasoning_chars=1000, completion_includes_thinking=True,
        drafts_offered=1000, drafts_accepted=600, acceptance_rate=.6,
        decode_tps_source='engine', decode_ms=20000, decode_tps=102.4, total_s=60,
        run=0, order=0)


class EvidenceTests(unittest.TestCase):
    def test_short_or_thinking_only_cannot_pass_client_labels(self):
        self.assertTrue(sample_eligible(sample()))
        for patch in ({'usage_generated_tokens': 1000, 'generated_tokens': 1000},
                      {'output_chars': 0}, {'read_tokens': 100}, {'reused_tokens': 1},
                      {'completion_includes_thinking': False}):
            with self.subTest(patch=patch):
                self.assertFalse(sample_eligible(dict(sample(), **patch)))

    def test_invalid_clocks_usage_or_acceptance_are_ineligible(self):
        for patch in ({'decode_tps_source': 'client'}, {'generated_tokens_source': 'stream_pieces'},
                      {'decode_ms': float('nan')}, {'decode_tps': float('inf')},
                      {'drafts_offered': 0}, {'drafts_accepted': 1001}, {'acceptance_rate': .7},
                      {'generated_tokens': 10}, {'stream_complete': False}):
            with self.subTest(patch=patch):
                self.assertFalse(sample_eligible(dict(sample(), **patch)))

    def test_preserves_failed_rows_and_requires_both_workloads(self):
        generation = dict(workload_version=4, thinking=True, reasoning_budget_tokens=0, max_tokens=16384,
            workloads={'regular-code': {'requests': [sample()]},
                       'varied-prose': {'requests': [dict(sample(), order=1, output_chars=0)]}})
        rows = collect_samples(generation)
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[1]['controller_eligible'])
        with self.assertRaises(RuntimeError):
            require_complete_generation(generation, rows)
        generation['workloads']['varied-prose']['requests'][0]['output_chars'] = 2000
        require_complete_generation(generation, collect_samples(generation))
        del generation['workloads']['varied-prose']
        with self.assertRaises(RuntimeError):
            require_complete_generation(generation, collect_samples(generation))

    def test_changed_thinking_workload_or_duplicate_order_cannot_pass(self):
        generation = dict(workload_version=4, thinking=True, reasoning_budget_tokens=0, max_tokens=16384,
            workloads={'regular-code': {'requests': [sample()]},
                       'varied-prose': {'requests': [dict(sample(), order=1)]}})
        for key, value in [('thinking', False), ('workload_version', 3), ('max_tokens', 8192)]:
            altered = copy.deepcopy(generation)
            altered[key] = value
            with self.assertRaises(RuntimeError):
                require_complete_generation(altered, collect_samples(altered))
        generation['workloads']['varied-prose']['requests'][0]['order'] = 0
        with self.assertRaises(RuntimeError):
            require_complete_generation(generation, collect_samples(generation))


class NativeClosureEvidence(unittest.TestCase):
    def claim(self):
        return dict(binary={'sha256':'a'*64},args=['--serve','--spec','4'],boot_id='boot',
            native_validation_probe={'sha256':'b'*64},
            wrapper=dict(pid=43,start_ticks=100,sid=42),native=dict(pid=44,start_ticks=101,sid=42))

    def test_native_exit_identity_and_observer_refusals(self):
        for variant in ('valid','no_probe','failed','boolean_exit','binary','boot','args','probe','collector','profile','leader','ticks','alias','end_identity'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);claim=self.claim()
                record=dict(binary=copy.deepcopy(claim['binary']),native_validation_probe=copy.deepcopy(claim['native_validation_probe']))
                plan={'config':{'args':['--spec','4']}}
                if variant=='no_probe':claim['native_validation_probe']=record['native_validation_probe']=None
                if variant=='binary':claim['binary']['sha256']='c'*64
                if variant=='boot':claim['boot_id']='other'
                if variant=='args':claim['args'].append('--other')
                if variant=='probe':claim['native_validation_probe']={}
                if variant=='collector':claim['resource_collector']={}
                if variant=='profile':claim['native_profile']={}
                if variant=='leader':claim['native']['sid']=99
                if variant=='ticks':claim['native']['start_ticks']=True
                if variant=='alias':claim['native']['pid']=claim['wrapper']['pid']
                end=dict(copy.deepcopy(claim),exit_code=1 if variant=='failed' else 0)
                if variant=='end_identity':end['native']['pid']=55
                if variant=='boolean_exit':end['exit_code']=False
                (root/'engine-start.json').write_text(json.dumps(claim))
                (root/'engine-exit.json').write_text(json.dumps(end))
                if variant in ('valid','no_probe'):self.assertEqual(g.native_end(root,record,plan,'boot',42),end)
                else:
                    with self.assertRaises(RuntimeError):g.native_end(root,record,plan,'boot',42)



class ResourceClosureTests(unittest.TestCase):
    def test_only_declared_same_session_successful_passive_collector_is_accepted(self):
        for variant in ('valid','undeclared','plan','source','sid','alias','failed','boolean_exit','boolean_budget','profile'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);reader=root/'reader.py';reader.write_text('# passive reader\n')
                collection=dict(source=dict(path=str(reader),sha256=g.common.digest(reader)),seconds=900,interval=1)
                start=dict(binary={'sha256':'a'*64},args=['--serve','--spec','4'],boot_id='boot',native_validation_probe=None,
                    wrapper=dict(pid=43,start_ticks=100,sid=42),native=dict(pid=44,start_ticks=101,sid=42))
                end=dict(copy.deepcopy(start),exit_code=0,resource_collector=dict(pid=45,start_ticks=102,sid=42),resource_collector_exit_code=0)
                record=dict(binary=start['binary'],native_validation_probe=None,resource_collection=collection)
                if variant=='undeclared':del record['resource_collection']
                if variant=='sid':end['resource_collector']['sid']=99
                if variant=='alias':end['resource_collector']['pid']=44
                if variant=='failed':end['resource_collector_exit_code']=1
                if variant=='boolean_exit':end['resource_collector_exit_code']=False
                if variant=='boolean_budget':collection['seconds']=True
                if variant=='profile':start['native_profile']={}
                (root/'engine-start.json').write_text(json.dumps(start));(root/'engine-exit.json').write_text(json.dumps(end))
                (root/'plan.json').write_text(json.dumps({'resource_collection':{} if variant=='plan' else collection}))
                if variant=='source':reader.write_text('# changed reader\n')
                if variant=='valid':self.assertEqual(g.native_end(root,record,{'config':{'args':['--spec','4']}},'boot',42),end)
                else:
                    with self.assertRaises(RuntimeError):g.native_end(root,record,{'config':{'args':['--spec','4']}},'boot',42)


class DecodeWorkTimingTests(unittest.TestCase):
    def fixture(self):
        row=dict(prompt_tokens=33000,reused_tokens=0,read_tokens=33000,generated_tokens=3,
            drafts_accepted=1,drafts_offered=3,decode_ms=40)
        text=('strata trace: window 32999 1\nstrata trace: window 33000 4\n'
            'strata decode timing: 2 windows, avg T 2.50, 1.50 tokens/window, 20.00 ms/window = '
            'verify 12.00 (GPU-reach wait 3.00 + per-layer host 7.00 [plan 1.00 actq 1.00 jobs 1.00 '
            'CPU 4.00] + stage 2.00) + commit/emit 3.00 + draft 4.00; per layer-window: CPU experts '
            '0.62 (0.83 entries), VRAM hits 24.17, PCIe 0.00\n'
            'strata serve: prompt 33000 tokens = 0 reused + 33000 read in 10 ms (1 tok/s), '
            '3 generated in 40 ms (75 tok/s), drafts accepted 1 of 3, 0 checkpoints\n'
            'strata serve: SYCL_EXPERT_COPY submitted_blobs=0 submitted_bytes=0 completed_blobs=0 '
            'completed_bytes=0 host_view_groups=0 gpu_miss_groups=0 cpu_unique_misses=60 cpu_entries=80\n')
        return text,row

    def test_counts_actual_target_rows_and_preserves_nested_rounding_residuals(self):
        text,row=self.fixture();timing=g.decode_timing_evidence(text,[row])[0]
        self.assertEqual(timing['work']['target_rows'],5)
        self.assertEqual(timing['work']['minimum_NT1_jobs'],40)
        self.assertEqual(timing['outer_residual_ms_per_window'],1)
        self.assertEqual(timing['host_nested_residual_ms_per_window'],0)
        # A tiny negative signed residual within the explicit print-rounding allowance stays visible.
        text=text.replace('draft 4.00','draft 5.01')
        self.assertLess(g.decode_timing_evidence(text,[row])[0]['outer_residual_ms_per_window'],0)

    def test_rejects_work_cross_binding_unknown_copy_and_unconserved_clocks(self):
        text,row=self.fixture()
        edits=[('window 33000 4','window 33000 3'),('window 33000 4','window 40000 4'),
            ('cpu_unique_misses=60','cpu_unique_misses=1'),('submitted_bytes=0','submitted_bytes=1'),
            ('drafts accepted 1 of 3','drafts accepted 1 of 4'),('avg T 2.50','avg T 2.00'),
            ('CPU experts 0.62','CPU experts 0.61'),('VRAM hits 24.17','VRAM hits 24.10'),
            ('PCIe 0.00','PCIe 1.00'),('draft 4.00','draft 8.00'),('CPU 4.00','CPU 8.00')]
        for a,b in edits:
            with self.subTest(a=a),self.assertRaises(RuntimeError):g.decode_timing_evidence(text.replace(a,b),[row])
        with self.assertRaises(RuntimeError):g.decode_timing_evidence(text+text,[row])
        with self.assertRaises(RuntimeError):g.decode_timing_evidence(text.replace('strata decode timing:','missing:'),[row])

    def test_actual_one_window_DONE_and_average_print_quantization(self):
        row=dict(prompt_tokens=260,reused_tokens=0,read_tokens=260,generated_tokens=1,
            drafts_accepted=0,drafts_offered=0,decode_ms=126.0)
        text=('strata trace: window 259 1\n'
            'strata decode timing: 1 windows, avg T 1.00, 1.00 tokens/window, 126.05 ms/window = '
            'verify 125.99 (GPU-reach wait 40.85 + per-layer host 50.55 [plan 2.02 actq 0.13 jobs 0.04 '
            'CPU 48.01] + stage 6.92) + commit/emit 0.05 + draft 0.00; per layer-window: CPU experts '
            '8.25 (8.25 entries), VRAM hits 1.75, PCIe 0.00\n'
            'strata serve: prompt 260 tokens = 0 reused + 260 read in 6483 ms (40.1 tok/s), '
            '1 generated in 126 ms (7.9 tok/s), drafts accepted 0 of 0, 0 checkpoints\n'
            'strata serve: SYCL_EXPERT_COPY submitted_blobs=0 submitted_bytes=0 completed_blobs=0 '
            'completed_bytes=0 host_view_groups=0 gpu_miss_groups=0 cpu_unique_misses=396 cpu_entries=396\n')
        result=g.decode_timing_evidence(text,[row])[0]
        self.assertAlmostEqual(result['clock_binding']['absolute_total_difference_ms'],.05)
        with self.assertRaises(RuntimeError):
            g.decode_timing_evidence(text.replace('126.05 ms/window','126.06 ms/window'),[row])
        with self.assertRaises(RuntimeError):
            g.decode_timing_evidence(text,[dict(row,decode_ms=125.9)])
        # The one-decimal request quantum is not a per-window blanket tolerance.
        text,row=self.fixture()
        with self.assertRaises(RuntimeError):g.decode_timing_evidence(text,[dict(row,decode_ms=40.1)])


class ArenaScreenTests(unittest.TestCase):
    def test_actual_copy_footer_conservation_and_request_binding(self):
        from arena_copy import copy_evidence
        request=dict(prompt_tokens=66,read_tokens=66,usage_generated_tokens=16)
        log=('ordinary-Arena copy enabled mode=0\n'
             'strata serve: prompt 66 tokens = 0 reused + 66 read in 26062 ms (2.5 tok/s), '
             '16 generated in 6220 ms (2.6 tok/s), drafts accepted 12 of 12, 0 checkpoints\n'
             'strata serve: SYCL_EXPERT_COPY submitted_blobs=1309 submitted_bytes=2684518400 '
             'completed_blobs=1309 completed_bytes=2684518400 host_view_groups=1309 '
             'gpu_miss_groups=1309 cpu_unique_misses=4285 cpu_entries=6456\n')
        self.assertEqual(copy_evidence(log,request,1)['logical_copy_bytes_per_emitted_token'],167782400)
        for broken in (log+log,log.replace('completed_blobs=1309','completed_blobs=1308'),
                       log.replace('cpu_entries=6456','cpu_entries=0'),log.replace('66 read','65 read')):
            with self.assertRaises(RuntimeError):copy_evidence(broken,request,1)
        with self.assertRaises(RuntimeError):copy_evidence(log,request,0)

    def test_screen_repeats_without_requiring_identical_generated_lengths(self):
        from arena_copy import screen
        def rows(speeds):
            return [dict(arm=arm,passed=True,ownership_closed=True,source_commit='a'*40,
                binary_sha256='b'*64,request=dict(sample(),controller_eligible=True,
                    request_sha256='c'*64,decode_tps=tps,decode_ms=1000*generated/tps,
                    generated_tokens=generated,usage_generated_tokens=generated))
                for arm,tps,generated in zip((0,1,1,0,0,1),speeds,(2048,3000,2500,2048,2048,4000))]
        faster=rows((10,12,12,10,10,12))
        self.assertEqual(screen(faster)['decision'],'candidate_faster_screen')
        self.assertFalse(screen(faster)['adoption_qualified'])
        self.assertEqual(screen(rows((12,10,10,12,12,10)))['decision'],'candidate_slower_screen')
        self.assertEqual(screen(rows((10,12,9,10,10,12)))['decision'],'inconclusive')
        for field,value in (('request_sha256','d'*64),('usage_generated_tokens',1),('decode_tps',float('nan'))):
            altered=copy.deepcopy(faster);altered[1]['request'][field]=value
            with self.assertRaises(RuntimeError):screen(altered)
        altered=copy.deepcopy(faster);altered[1]['binary_sha256']='d'*64
        with self.assertRaises(RuntimeError):screen(altered)


class EarlyPcieEvidenceTests(unittest.TestCase):
    log = ('early PCIe expert graph enabled mode=0 audit=1\n'
           'strata verify: captured the 1-token window\n'
           'strata verify: captured the 4-token window\n'
           'strata serve: SYCL_PCIE_EARLY launches=2 groups=16 '
           'ready_at_pool_return=1 pending_at_pool_return=1\n'
           'strata serve: SYCL_PCIE_EARLY launches=3 groups=20 '
           'ready_at_pool_return=1 pending_at_pool_return=2\n')

    def test_early_groups_bind_to_completed_copies_and_launch_audit(self):
        from early_pcie import early_evidence
        rows=early_evidence(self.log,[dict(completed_blobs=16),dict(completed_blobs=20)])
        self.assertEqual([r['launches'] for r in rows],[2,3])
        with self.assertRaises(RuntimeError):
            early_evidence(self.log,[dict(completed_blobs=17),dict(completed_blobs=20)])

    def test_missing_duplicated_or_unaccounted_early_work_is_rejected(self):
        from early_pcie import early_evidence
        copied=[dict(completed_blobs=16),dict(completed_blobs=20)]
        for log in (self.log+self.log,self.log.replace('audit=1','audit=0'),
                    self.log.replace('captured the 4-token window','capture not reached'),
                    self.log.replace('pending_at_pool_return=2','pending_at_pool_return=0'),
                    self.log.replace('launches=2','launches=0')):
            with self.subTest(log=log):
                with self.assertRaises(RuntimeError):early_evidence(log,copied)


class EarlySchedulingScreenTests(unittest.TestCase):
    def test_measured_footer_binding_and_clean_observer_exclusion(self):
        from arena_copy import measured_early_evidence
        request=dict(prompt_tokens=33211,read_tokens=33211,usage_generated_tokens=2931)
        body=('strata serve: prompt 33211 tokens = 0 reused + 33211 read in 50 ms (1 tok/s), '
              '2931 generated in 100 ms (1 tok/s), drafts accepted 5 of 6, 0 checkpoints\n'
              'strata serve: SYCL_EXPERT_COPY submitted_blobs=16 submitted_bytes=32 completed_blobs=16 '
              'completed_bytes=32 host_view_groups=16 gpu_miss_groups=16 cpu_unique_misses=40 cpu_entries=60\n'
              'strata serve: SYCL_PCIE_EARLY launches=2 groups=16 ready_at_pool_return=0 pending_at_pool_return=0\n')
        log='early PCIe expert graph enabled mode=0 audit=0\ncaptured the 1-token window\ncaptured the 4-token window\n'+body
        self.assertEqual(measured_early_evidence(log,request,dict(completed_blobs=16),True,False)['groups'],16)
        audited=log.replace('audit=0','audit=1').replace('pending_at_pool_return=0','pending_at_pool_return=2')
        self.assertTrue(measured_early_evidence(audited,request,dict(completed_blobs=16),True,True)['audit'])
        for broken in (log+body,log.replace('2931 generated','2930 generated'),
                       log.replace('groups=16','groups=15'),log.replace('pending_at_pool_return=0','pending_at_pool_return=1'),
                       log.replace('audit=0','audit=1')):
            with self.assertRaises(RuntimeError):measured_early_evidence(broken,request,dict(completed_blobs=16),True,False)
        self.assertEqual(measured_early_evidence('',request,{},False,False),dict(enabled=False))
        with self.assertRaises(RuntimeError):measured_early_evidence(log,request,{},False,False)

    def test_screen_requires_real_early_and_copy_evidence_in_each_arm(self):
        from arena_copy import screen
        rows=[dict(arm=arm,passed=True,ownership_closed=True,source_commit='a'*40,binary_sha256='b'*64,
            request=dict(sample(),controller_eligible=True,request_sha256='c'*64,decode_tps=speed,decode_ms=2048000/speed),
            copy_evidence=dict(completed_blobs=16),early_evidence=(dict(enabled=True,audit=False,launches=2,groups=16,
                ready_at_pool_return=0,pending_at_pool_return=0) if arm else dict(enabled=False)))
            for arm,speed in zip((0,1,1,0,0,1),(10,12,12,10,10,12))]
        self.assertEqual(screen(rows,early=True)['decision'],'candidate_faster_screen')
        for patch in (dict(enabled=False),dict(enabled=True,audit=True,launches=2,groups=16,
                          ready_at_pool_return=0,pending_at_pool_return=2)):
            altered=copy.deepcopy(rows);altered[1]['early_evidence']=patch
            with self.assertRaises(RuntimeError):screen(altered,early=True)
        altered=copy.deepcopy(rows);altered[0]['copy_evidence']={}
        with self.assertRaises(RuntimeError):screen(altered,early=True)


if __name__ == '__main__':
    unittest.main()
