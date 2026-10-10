"""Root-owned CPU-only protocol/gate tests. Not executed by implementer."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_clean as c


def request(n=65536, chunk=8192, generated=64, reason='length'):
    lines=['RESUME 0','REUSED 0']
    lines += [f'PP {p} {n} 1 2' for p in c.expected_pp(n,chunk)]
    lp='LP -1 1:-1 2:-2 3:-3 4:-4 5:-5'
    for i in range(generated): lines += [f'T {i+1}',lp]
    lines += [f'DONE {generated} {n} 10 20 {reason} 0 1 0 0 0 0 0 0 {n} 0']
    return dict(protocol=lines,ids=list(range(1,generated+1)),logprobs=[lp]*generated)


class Contract(unittest.TestCase):
    def test_chunks(self):
        for chunk in (4096,8192):
            for n in (32768,65536):
                r=c.validate(request(n,chunk),n,chunk)
                self.assertEqual(r['actual_batched_positions'],n-1)
                self.assertEqual(r['prefill_chunk'],chunk)
                self.assertTrue(r['decode_performance_eligible'])
                self.assertEqual(c.expected_pp(n,chunk)[-1],n-1)
                self.assertEqual(len(c.expected_pp(n,chunk)),n//chunk)

    def test_early_stop(self):
        r=c.validate(request(chunk=4096,generated=3,reason='stop'),chunk=4096)
        self.assertTrue(r['fresh_complete_finite'])
        self.assertFalse(r['decode_performance_eligible'])
        self.assertIsNone(r['decode_tps'])
        self.assertGreater(r['prefill_tps'],0)

    def test_rejections(self):
        mutations=[lambda r:r['protocol'].__setitem__(2,'PP 8192 65536 1 2'),
                   lambda r:r['protocol'].__setitem__(0,'RESUME 1'),
                   lambda r:r['protocol'].__setitem__(-1,'DONE 64 65536 nan 20 length 0 1 0 0 0 0 0 0 65536 0'),
                   lambda r:r['protocol'].__setitem__(-1,'DONE 64 65536 10 20 length 0 1 1 0 0 0 0 0 65536 0'),
                   lambda r:r['protocol'].__setitem__(-1,'DONE 64 65536 10 20 length 0 1 0 0 0 0 0 0 65535 0')]
        for mutate in mutations:
            r=request(chunk=4096); mutate(r)
            with self.assertRaises(RuntimeError): c.validate(r,chunk=4096)
        with self.assertRaises(RuntimeError): c.validate(request(generated=3),chunk=8192)
        with self.assertRaises(RuntimeError): c.validate(request(chunk=4096),chunk=8192)

    def test_fork_closed_same_chunk_only(self):
        # Synthetic CPU receipt qualification, NOT an actual diagnostic admission.
        for chunk in (4096,8192):
            d=dict(active=False,completed=True,exit_code=0,exit_signal=None,
                   new_fault_messages=[],cleanup={},binary_sha256=c.XE_SHA,
                   diagnostic_passed=True,boot_unchanged=True,
                   source_commit='39bdadcc9e2b89b1e3c8be7bb2a603b04fa0e197',
                   controller_sha256=c.FORK_CONTROLLER_SHA[chunk],
                   argv=['binary','--prefill',str(chunk)],prefill_chunk=chunk,
                   environment={'STRATA_ARENA_HOST_USM':'1'},
                   actual_target_environment={'STRATA_ARENA_HOST_USM':'1'},
                   request=request(32768,chunk))
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'receipt.json'
                def save(): p.write_text(json.dumps(d))
                save()
                expected='a'*64
                with patch.object(c,'sha',side_effect=lambda q:expected if Path(q)==p else c.FORK_CONTROLLER_SHA[chunk]):
                    c.load_gate(p,expected,'xe',chunk)
                    for key,value in [('active',True),('completed',False),('diagnostic_passed',False),
                                      ('controller_sha256','0'*64),('exit_code',1),
                                      ('environment',{'STRATA_ARENA_HOST_USM':'0'})]:
                        old=d[key]; d[key]=value; save()
                        with self.assertRaises(RuntimeError): c.load_gate(p,expected,'xe',chunk)
                        d[key]=old
                    d['argv']=['binary','--prefill',str(12288-chunk)]; save()
                    with self.assertRaises(RuntimeError): c.load_gate(p,expected,'xe',chunk)


if __name__=='__main__': unittest.main()
