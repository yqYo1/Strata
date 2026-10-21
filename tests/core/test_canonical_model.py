#!/usr/bin/env python3
"""Reject wrong or unbound production restore routes before qualification."""
import sys,unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import canonical_model as c

class Contract(unittest.TestCase):
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
