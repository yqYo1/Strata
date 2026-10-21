#!/usr/bin/env python3
"""Offline refusal and collection-path checks for the current model diagnostic."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent/'sycl'))
import generation_compare as c


def fixture():
    lines = ['strata generate: 5 expert-pool workers + the host thread']
    events = []
    for i,n in enumerate((74,266,912,1069,33208,33201)):
        cells=n-1
        if i >= 4:
            lines += [
                f'strata loan audit: phase=layout chunk=8192 first=128 slots=1256 bytes=2329548800 need=2319958528 lent=1128 mask_match=1',
                'strata loan audit: phase=lend table_entries=24576 table_match=1 free_bytes=400000000 total_bytes=10666115072',
                f'strata cache loan: phase=masked generation={i} chunk=8192 first=128 lent=1128 decode_ready=0',
                'strata prefill dense: phase=retired residency=Full activations=0',
                'strata prefill dense: phase=bank residency=Bank activations=0',
                f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={cells} mixed_fp16_bytes=0',
                f'strata prefill: layer-major candidate, tokens={cells} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total={cells*20480} residual_gpu_rows={cells} residual_gpu={cells*20480} residual_host=0 residual_staging=0 layer_cache=1022361600;',
                'strata prefill dense: phase=restored residency=Full activations=48',
                'strata loan audit: phase=payload rows=1256 bytes=2575206400 lent=1128 same_slots=1 payload_match=1',
                'strata loan audit: phase=refill table_entries=24576 table_match=1 free_bytes=100000000 total_bytes=10666115072',
                f'strata cache loan: phase=ready generation={i} chunk=8192 first=128 lent=1128 decode_ready=1',
                'strata MoE expert bank complete: uploads=100 requested_H2D_bytes=200 permanent_hits=300 permanent_slots=100 total_slots=384 final_drain=checked']
        for chunk in [8192]*(cells//8192)+([cells%8192] if cells%8192 else []):
            lines.append(f'strata serve: DRAFT_PREFILL path=batched mode=2 cells={chunk}')
        generated=2048 if i>=4 else (24 if i==0 else 1)
        lines.append(f'strata serve: prompt {n} tokens = 0 reused + {n} read in 100 ms (1 tok/s), {generated} generated in 100 ms')
        events.append(dict(kind='result',sample=dict(prompt_tokens=n,read_tokens=n,reused_tokens=0,generated_tokens=generated)))
    return '\n'.join(lines)+'\n', events


class RouteTests(unittest.TestCase):
    def test_binds_both_long_requests_and_restores_decode_state(self):
        text,events=fixture()
        rows=c.native_routes(text,events,0)
        self.assertEqual([r['actual_GPU_BF16_rows'] for r in rows],[33207,33200])
        self.assertEqual([r['draft_work']['draft_cells'] for r in rows],[33207,33200])
        self.assertTrue(all(r['cache_payload_and_map_match'] for r in rows))

    def test_rejects_partial_gpu_fallback_cross_request_and_unrestored_state(self):
        text,events=fixture()
        edits=[('residual_gpu_rows=33207','residual_gpu_rows=1'),('effective_mode=10','effective_mode=2'),
               ('residual_host=0','residual_host=1'),('generated in 100 ms','generated in 101 ms'),
               ('payload_match=1','payload_match=0'),('activations=48','activations=47'),
               ('final_drain=checked','final_drain=pending'),('mode=2 cells=73','mode=0 cells=73'),
               ('33208 tokens = 0 reused','33208 tokens = 1 reused')]
        # A changed elapsed clock alone is allowed; missing work/state is not.
        for old,new in [e for e in edits if e[0]!='generated in 100 ms']:
            with self.subTest(old=old),self.assertRaises(RuntimeError):
                c.native_routes(text.replace(old,new,1),events,0)
        self.assertEqual(len(c.native_routes(text.replace('generated in 100 ms','generated in 101 ms'),events,0)),2)
        with self.assertRaises(RuntimeError): c.native_routes(text,events[:-1],0)


class CollectionPathTests(unittest.TestCase):
    def test_real_native_closure_reader_is_used_and_faults_cannot_pass(self):
        for fault in (False,True):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);arm=root/'0';(arm/'bench'/'run').mkdir(parents=True)
                result=root/'result.json';binary={'path':'/not-executed','sha256':'a'*64}
                config={'args':['--spec','4']}
                plan=dict(root=str(arm),binary=binary,config=config,environment={},command=[],build_source_commit='native')
                record=dict(plans=[plan,copy.deepcopy(plan)],processes=[],stage='prepared',complete=False,
                    active=False,ownership_closed=True,budgets={})
                (root/'plan.json').write_text(json.dumps(record))
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                start=dict(binary=binary,args=['--serve','--spec','4'],boot_id=boot,native_validation_probe=None,
                    native=dict(pid=44,start_ticks=101,sid=42),wrapper=dict(pid=43,start_ticks=100,sid=42))
                (arm/'engine-start.json').write_text(json.dumps(start))
                (arm/'engine-exit.json').write_text(json.dumps(dict(start,exit_code=0)))
                text,events=fixture();(arm/'engine.stderr').write_text(text)
                (arm/'bench'/'run'/'results.json').write_text(json.dumps({'results':{}}))
                class Owner:
                    active=None
                    commands=[]
                    def __init__(self,*args): pass
                    def run(self,*args,**kwargs): return dict(exit_code=0,pid=42),None,None
                with patch.object(c,'ROOT',root),patch.object(c,'RESULT',result), \
                     patch.object(c.gen,'committed',return_value=record),patch.object(c,'current'), \
                     patch.object(c.supervision,'Owner',Owner),patch.object(c.supervision,'closed',return_value=True), \
                     patch.object(c,'official',return_value=([{'request_sha256':'b'*64}]*2,{'valid':True},events)), \
                     patch.object(c.subprocess,'check_output',side_effect=['-- cursor: x\n','xe GPU fault' if fault else '']), \
                     patch.object(c.guards,'driver_guard',return_value={}):
                    self.assertEqual(c.run(),1 if fault else 0)
                closed=json.loads(result.read_text());row=closed['processes'][0]
                self.assertTrue(row['ownership_closed'])
                self.assertEqual(row['native_exit_code'],0)
                self.assertEqual(row['passed'],not fault)
                self.assertEqual(closed['stage'],'closed' if fault else 'collecting')
                self.assertEqual(len(row['routes']),2)


if __name__=='__main__': unittest.main()
