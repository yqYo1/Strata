#!/usr/bin/env python3
"""Host protocol/dispatch corruption controls; no model/device execution."""
import copy
import unittest
import mtp_resident_serve as m


class IO:
    def __init__(self): self.sent = []; self.finished = False; self.lines = []
    def line(self, _): return self.lines.pop(0)
    def send(self, value): self.sent.append(value)
    def exit(self, _): self.finished = True


def sample_trace(cap):
    rows = []; text = ''; rounds = 0
    for n in (33, 257):
        # One initial token plus31 subsequent single tokens (no accepted drafts).
        row = dict(prompt_tokens=n, reused_tokens=0, read_tokens=n,
            generated_tokens=32, drafts_accepted=0, drafts_offered=0 if cap else 93)
        rows.append(row)
        text += f'strata serve: DRAFT_PREFILL path=batched mode=2 cells={n-1}\n'
        for i in range(32): text += f'strata trace: window {n-1+i} {1 if cap or not i else 4}\n'
        text += f'strata serve: prompt {n} tokens = 0 reused + {n} read, 32 generated\n'
        if not cap: rounds += 31
        text += f'strata resident MTP control: mtp_max_t={cap} cumulative_draft_rounds={rounds}\n'
    return text, rows


class Routes(unittest.TestCase):
    def test_both_valid_routes_and_round_counts(self):
        for cap in (0, 1):
            text, rows = sample_trace(cap); out = m.route_evidence(text, rows, cap)
            self.assertEqual(len(out), 2)
            self.assertEqual([r['draft_rounds_delta'] for r in out], [0, 0] if cap else [31, 31])

    def test_false_no_draft_or_missing_prompt_work_is_rejected(self):
        text, rows = sample_trace(1)
        for bad in (text.replace('window 33 1', 'window 33 4'),
                    text.replace('cumulative_draft_rounds=0', 'cumulative_draft_rounds=1', 1),
                    text.replace('mode=2 cells=32', 'mode=2 cells=31'),
                    text.replace('mode=2 cells=32', 'mode=1 cells=32')):
            with self.subTest(bad=bad[:80]), self.assertRaises(RuntimeError): m.route_evidence(bad, rows, 1)

    def test_reference_round_and_output_conservation(self):
        text, rows = sample_trace(0)
        with self.assertRaises(RuntimeError):
            m.route_evidence(text.replace('cumulative_draft_rounds=31', 'cumulative_draft_rounds=30'), rows, 0)
        bad = copy.deepcopy(rows); bad[0]['generated_tokens'] = 33
        with self.assertRaises(RuntimeError): m.route_evidence(text, bad, 0)
        bad = copy.deepcopy(rows); bad[0]['drafts_accepted'] = 4
        with self.assertRaises(RuntimeError): m.route_evidence(text, bad, 0)

    def test_protocol_wires_and_reply_extents(self):
        io = IO(); io.lines = ['INFO context=131072 kv=int8 kv_resident=0 expert_slots=1256 '
            'spec=4 mtp_max=1 lookup=0 pool_workers=5 pcie_frac=0.00 conversation_cache_mib=0', 'READY 131072 stop']
        wrong = IO(); wrong.lines = [io.lines[0].replace('kv_resident=0', 'kv_resident=131072'), io.lines[1]]
        with self.assertRaisesRegex(RuntimeError, 'actual model/cache/MTP configuration differs'):
            m.interact(wrong, {}, {}, 1)
        self.assertFalse(wrong.sent)
        fixtures = []
        for ids in m.requests():
            payload = m.wire(ids); fixtures.append(dict(wire_bytes=len(payload), wire_sha256=m.hashlib.sha256(payload).hexdigest()))
            io.lines += [f'PP {len(ids)-1}'] + [f'T {i}' for i in range(32)] + [f'DONE 32 {len(ids)} 0 0 length 0 0 0 0 0 0 0 0 {len(ids)}']
        row = dict(requests=fixtures, vocab_size=4096, eos_ids=[248044,248046]); out = {}
        m.interact(io, row, out, 1)
        self.assertEqual(io.sent, [*(m.wire(ids) for ids in m.requests()), b'QUIT\n'])
        self.assertTrue(io.finished); self.assertEqual(len(out['requests']), 2)
        self.assertFalse(io.lines)

    def test_real_early_eos_is_valid_and_false_stop_is_rejected(self):
        def fixture(finish='stop', last=248046):
            io = IO(); io.lines = ['INFO context=131072 kv=int8 kv_resident=0 expert_slots=1256 '
                'spec=4 mtp_max=1 lookup=0 pool_workers=5 pcie_frac=0.00 conversation_cache_mib=0', 'READY 131072 stop']
            cases=[]
            for ids in m.requests():
                payload=m.wire(ids);cases.append(dict(wire_bytes=len(payload),wire_sha256=m.hashlib.sha256(payload).hexdigest()))
                io.lines += [f'PP {len(ids)-1}', 'T 220', f'T {last}',
                    f'DONE 2 {len(ids)} 0 0 {finish} 0 0 0 0 0 0 0 0 {len(ids)}']
            return io,dict(requests=cases,vocab_size=248320,eos_ids=[248044,248046])
        io,row=fixture();out={};m.interact(io,row,out,1)
        self.assertEqual([s['generated_tokens']for s in out['requests']],[2,2]);self.assertTrue(io.finished)
        for finish,last in [('stop',221),('length',248046),('length',221)]:
            io,row=fixture(finish,last)
            with self.assertRaises(RuntimeError):m.interact(io,row,{},1)


if __name__ == '__main__': unittest.main()
