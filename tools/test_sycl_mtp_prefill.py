"""Host-only admission/analysis checks for the current matched MTP comparison."""
import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent/'sycl'))
import mtp_prefill as mtp


def sample(read=74, generated=24):
    return dict(prompt_tokens=read, read_tokens=read, reused_tokens=0, generated_tokens=generated)


def log(cells=73, position=0, first=0, batch=73, read=74):
    return (f'strata draft kv: {cells} cells from {position} (first needed {first}), batch {batch}: '
            'drafter idle 0.0 ms, the batches 1.2 ms, all 1.2 ms\n'
            f'strata serve: prompt {read} tokens = 0 reused + {read} read, 24 generated\n')


class AdmissionTest(unittest.TestCase):
    def test_warmup_requires_complete_cells_not_just_positive_batch(self):
        self.assertEqual(mtp.request_evidence(log(),[sample()])[0]['draft_cells'],73)
        # These all have a positive batch, but do not prove the same request.
        for text in (log(cells=72),log(position=1),log(first=73),log(batch=63),
                     log()+log(),log().replace('0 reused','1 reused')):
            with self.subTest(text=text),self.assertRaises(RuntimeError):
                mtp.request_evidence(text,[sample()])

    def test_request_boundaries_prevent_borrowing_other_warmup_cells(self):
        text=log()+log()
        self.assertEqual(len(mtp.request_evidence(text,[sample(),sample()])),2)
        # Same aggregate cells, incorrectly split between two requests.
        with self.assertRaises(RuntimeError):
            mtp.request_evidence(log(cells=72)+log(cells=74),[sample(),sample()])

    def test_malformed_extra_records_cannot_hide_behind_conserved_cells(self):
        for extra in ('strata draft kv: malformed\n',
                      'strata draft kv: 1 cells from 73 (first needed 73), batch 64: drafter idle NaN ms, the batches 1.0 ms, all 1.0 ms\n'):
            for text in (extra+log(),log()+extra):
                with self.subTest(text=text),self.assertRaises(RuntimeError):
                    mtp.request_evidence(text,[sample()])

    def test_only_arm_and_private_root_may_change(self):
        def plan(root,arm):
            return dict(root=root,config=dict(args=['--pack','/same','--expert-profile',root+'/expert-profile.bin']),
                environment=dict(STRATA_MTP_BATCH_F16=str(arm),TMPDIR=root+'/tmp',EnableDirectSubmission='0'))
        left,right=plan('/left',0),plan('/right',1)
        self.assertEqual(mtp.invariant(left),mtp.invariant(right))
        right['environment']['EnableDirectSubmission']='1'
        self.assertNotEqual(mtp.invariant(left),mtp.invariant(right))

    def test_acceptance_change_is_an_outcome_not_an_exclusion(self):
        rows=[]
        for i,arm in enumerate(mtp.ORDER):
            request=dict(request_sha256='same',read_tokens=33000,prompt_ms=900 if arm else 1000,
                generated_tokens=2000,decode_ms=1100 if arm else 1000,acceptance_rate=.50 if arm else .80,
                prompt_tps=36666 if arm else 33000,decode_tps=1818 if arm else 2000)
            rows.append(dict(arm=arm,passed=True,ownership_closed=True,native=dict(pid=i+1,start_ticks=i+1),request=request))
        result=mtp.summarize(rows)
        self.assertEqual(result['prefill_decision'],'candidate_faster_screen')
        self.assertEqual(result['decode_decision'],'candidate_slower_screen')
        self.assertFalse(result['decode_nonregression'])
        self.assertFalse(result['adoption_qualified'])
        self.assertTrue(all(x < -.29 for x in result['acceptance_changes']))
        changed=copy.deepcopy(rows);changed[-1]['request']['generated_tokens']=3000
        self.assertEqual(mtp.summarize(changed)['decode_decision'],'inconclusive_generation_length')
        changed=copy.deepcopy(rows);changed[-1]['request']['request_sha256']='different'
        with self.assertRaises(RuntimeError):mtp.summarize(changed)


if __name__=='__main__':unittest.main()
