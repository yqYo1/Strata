#!/usr/bin/env python3
"""Host corruption controls for the fixed2 qualification; no model/device work."""
import copy
import hashlib
import unittest

import mtp2_route as m
from test_mtp_resident_serve import IO


def trace():
    text = ''; samples = []
    for index, n in enumerate((33, 257, 16385, 33)):
        if index == 2:
            text += (
                'strata loan audit: phase=layout chunk=8192 first=128 slots=1256 bytes=1000 need=500 lent=1128 mask_match=1\n'
                'strata loan audit: phase=lend table_match=1 table_entries=24576 free_bytes=100 total_bytes=1000\n'
                'strata cache loan: phase=masked generation=3 chunk=8192 first=128 lent=1128 decode_ready=0\n'
                'strata prefill dense: phase=retired residency=Full activations=0\n'
                'strata prefill dense: phase=bank residency=Bank activations=0\n'
                'strata prefill: layer-major candidate, tokens=16384 chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=335544320 residual_gpu_rows=16384 residual_gpu=335544320 residual_host=0 residual_staging=0 layer_cache=1022361600\n'
                'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=16384 mixed_fp16_bytes=0\n'
                'strata prefill dense: phase=restored residency=Full activations=48\n'
                'strata loan audit: phase=payload rows=1256 bytes=2575206400 lent=1128 same_slots=1 payload_match=1\n'
                'strata loan audit: phase=refill table_match=1 table_entries=24576 free_bytes=100 total_bytes=1000\n'
                'strata cache loan: phase=ready generation=3 chunk=8192 first=128 lent=1128 decode_ready=1\n')
            for phase in ('before-detach', 'after-detach', 'after-prompt-before-restore', 'after-restore'):
                text += f'strata mtp: prompt expert phase={phase} free_bytes=100 total_bytes=1000\n'
            text += 'strata mtp: experts restored bytes=707788800, detached=0\n'
        for chunk in ([8192, 8192] if index == 2 else [n-1]):
            text += f'strata serve: DRAFT_PREFILL path={"token" if chunk<64 else "batched"} mode=2 cells={chunk}\n'
        for offset in range(16):
            text += f'strata trace: window {n-1+offset} {1 if offset == 0 else 2}\n'
        text += f'strata serve: prompt {n} tokens = 0 reused + {n} read, 16 generated\n'
        samples.append(dict(prompt_tokens=n, reused_tokens=0, read_tokens=n,
            generated_tokens=16, drafts_offered=15, drafts_accepted=0))
    return text, samples


def observer_trace():
    text = ('strata MTP_OBSERVER v=1 request=4 profile=0 request_complete=1 sample_complete=1 '
        'accounting_valid=1 skip=0 limit=2 seen=15 windows=2 completed=2 failed=0 replays=2 rows=3 T_mask=6 '
        'wall_s=1 thread_cpu_s=0.5 other_wall_s=0.7 other_cpu_s=0.35 valid_graph_clocks=0 '
        'valid_queue_clocks=0 graph_envelope_ns=0 round_clocks=0 round_envelope_ns=0 '
        'chain_clocks=0 chain_envelope_ns=0 raw_retained=0 raw_omitted=0 '
        'scope=draft_entry_through_return clocks=HOST_DISJOINT_DEVICE_NONADDITIVE '
        'arithmetic_service=UNKNOWN hardware_saturation=UNKNOWN observer_charged=1\n')
    for kind in ('round', 'chain'):
        for phase in ('capture', 'submit', 'drain', 'query'):
            active = kind == 'round' and phase != 'query'
            text += (f'strata MTP_OBSERVER_PHASE v=1 request=4 kind={kind} phase={phase} '
                f'calls={2 if active else 0} throws=0 wall_s={0.1 if active else 0} '
                f'thread_cpu_s={0.05 if active else 0}\n')
    return text


class Routes(unittest.TestCase):
    def test_mixed_prompt_and_post_restore_fixed2(self):
        text, rows = trace(); result = m.routes(text, rows)
        self.assertEqual([r['prompt_chunks'] for r in result], [[32], [256], [8192, 8192], [32]])
        self.assertEqual(result[2]['dense_layers_restored'], 48)
        self.assertEqual([r['target_rows'] for r in result], [31]*4)

    def test_handoff_and_actual_dispatch_corruption_is_rejected(self):
        text, rows = trace()
        for old, new in [('window 33 2', 'window 33 4'), ('cells=8192', 'cells=8191'),
                ('payload_match=1', 'payload_match=0'), ('activations=48', 'activations=47'),
                ('residual_gpu=335544320', 'residual_gpu=0'),
                ('experts restored bytes=707788800,', 'experts restored bytes=1,')]:
            with self.subTest(old=old), self.assertRaises(RuntimeError):
                m.routes(text.replace(old, new, 1), rows)
        bad = copy.deepcopy(rows); bad[-1]['drafts_accepted'] = 2
        with self.assertRaises(RuntimeError): m.routes(text, bad)

    def test_observer_rounds_and_host_accounting(self):
        self.assertEqual(m.observer(observer_trace())['chain_calls'], 0)

    def test_observer_bad_counts_clocks_or_duplicate_fields(self):
        text = observer_trace()
        for bad in (text.replace('T_mask=6', 'T_mask=16'),
                text.replace('kind=chain phase=submit calls=0', 'kind=chain phase=submit calls=1'),
                text.replace('other_wall_s=0.7', 'other_wall_s=0.6'),
                text.replace('thread_cpu_s=0.05', 'thread_cpu_s=nan', 1),
                text.replace('profile=0', 'profile=0 profile=0'),
                text + 'strata MTP_GRAPH_CLOCK v=1\n'):
            with self.subTest(bad=bad[:100]), self.assertRaises(RuntimeError): m.observer(bad)

    def test_four_gen16_requests_keep_original_cache_placement(self):
        cases = m.requests(); io = IO(); fixtures = []
        io.lines = ['INFO context=131072 kv=int8 kv_resident=0 expert_slots=1256 '
            'spec=4 mtp_max=2 lookup=0 pool_workers=5 pcie_frac=0.00 conversation_cache_mib=0', 'READY 131072 stop']
        for ids in cases:
            payload = m.protocol.wire(ids, 16)
            fixtures.append(dict(wire_bytes=len(payload), wire_sha256=hashlib.sha256(payload).hexdigest()))
            io.lines += [f'PP {len(ids)-1}'] + [f'T {i}' for i in range(16)] + [
                f'DONE 16 {len(ids)} 0 0 length 0 15 0 0 0 0 0 0 {len(ids)}']
        row = dict(requests=fixtures, vocab_size=248320, eos_ids=[248044, 248046]); outcome = {}
        m.protocol.interact(io, row, outcome, 2, request_ids=cases, max_new=16)
        self.assertEqual(io.sent, [*(m.protocol.wire(ids, 16) for ids in cases), b'QUIT\n'])
        self.assertTrue(io.finished)
        self.assertEqual(len(outcome['requests']), 4)
        for limit in (0, 33, True):
            with self.subTest(limit=limit), self.assertRaises(RuntimeError):
                m.protocol.interact(IO(), row, {}, 2, request_ids=cases, max_new=limit)
        with self.assertRaises(RuntimeError):
            m.protocol.interact(IO(), row, {}, 2, request_ids=cases[:2], max_new=16)


if __name__ == '__main__': unittest.main()
