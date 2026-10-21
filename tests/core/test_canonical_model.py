#!/usr/bin/env python3
"""Reject wrong or unbound production restore routes before qualification."""
import sys,unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import canonical_model as c

class Contract(unittest.TestCase):
    def dense_text(self):
        def row(phase,free):return f'strata prefill dense: phase={phase} free_bytes={free} total_bytes=100 full_bytes=90 bank_bytes=20\n'
        return (''.join(row(p,10)for p in ('retired','cache-loan','restored'))+
            'strata serve: prompt warmup\n'+
            ''.join(row(p,30+i)for i,p in enumerate(('retired','bank','restored')))+
            'strata serve: prompt long\n')
    def test_dense_warmup_is_separate_from_long_request(self):
        rows=c.dense_memory(self.dense_text())
        self.assertEqual([r['phase']for r in rows],['retired','bank','restored'])
        self.assertEqual([r['free_bytes']for r in rows],['30','31','32'])
    def test_dense_missing_duplicate_or_outside_request_rejected(self):
        rows=self.dense_text().splitlines(keepends=True)
        for text in (''.join(rows[1:]),''.join(rows+[rows[0]]),
                ''.join(rows[1:4]+[rows[0]]+rows[4:])):
            with self.assertRaises(RuntimeError):c.dense_memory(text)
    def test_dense_wrong_request_phase_or_missing_header_rejected(self):
        text=self.dense_text()
        for bad in (text.replace('phase=cache-loan','phase=bank'),
                    text.replace('strata serve: prompt warmup\n',''),
                    text.replace('phase=bank','phase=cache-loan')):
            with self.assertRaises(RuntimeError):c.dense_memory(bad)
    def text(self,arm=1):
        return ('strata serve: prompt warmup\n'+
            f'strata canonical restore: phase=bank_full bytes=1538035200 copies={23 if arm else 1} chunk={67108864 if arm else 0}\n'+
            'strata serve: prompt long\n')
    def test_both_actual_routes_bound_to_completed_long_prompt(self):
        for arm in (0,1):self.assertEqual(c.coverage(self.text(arm),arm)['copies'],23 if arm else 1)
    def test_wrong_copy_counts_extents_and_chunk_rejected(self):
        text=self.text()
        for a,b in [('copies=23','copies=1'),('bytes=1538035200','bytes=1538035199'),
                    ('chunk=67108864','chunk=0'),('phase=bank_full','phase=prepared')]:
            with self.subTest(field=a),self.assertRaises(RuntimeError):c.coverage(text.replace(a,b),1)
    def test_missing_duplicate_or_outside_request_rejected(self):
        first,audit,last=self.text().splitlines(keepends=True)
        for text in(first+last,first+audit+audit+last,audit+first+last,first+last+audit):
            with self.assertRaises(RuntimeError):c.coverage(text,1)
    def test_immutable_arm_environments_change_only_prefill_chunk_choice(self):
        setup=c.base.setup_reader.pilot();root=c.B/'canonical-model-host-environment-not-executed'
        a,b=[c.environment(setup,root,arm)for arm in(0,1)]
        self.assertEqual({k for k in a.keys()|b.keys()if a.get(k)!=b.get(k)}, {'STRATA_SYCL_PREFILL_CANONICAL_CHUNK64'})
        self.assertEqual(a['STRATA_IQ2_PAIR'],b['STRATA_IQ2_PAIR']);self.assertEqual(a['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED'],'1')
        self.assertEqual(a['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'],'1');self.assertEqual(a['EnableDirectSubmission'],'0')
        self.assertNotIn('STRATA_DECODE_TIMING',a)
if __name__=='__main__':unittest.main()
