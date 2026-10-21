"""Host-only admission/analysis checks for the current matched MTP comparison."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent/'sycl'))
import mtp_prefill as mtp
import ur_validation


def sample(read=74, generated=24):
    return dict(prompt_tokens=read, read_tokens=read, reused_tokens=0, generated_tokens=generated)


def log(cells=73, path='batched', mode=0, read=74, chunks=None):
    return (''.join(f'strata serve: DRAFT_PREFILL path={path} mode={mode} cells={n}\n'
                   for n in (chunks if chunks is not None else [cells])) +
            f'strata serve: prompt {read} tokens = 0 reused + {read} read, 24 generated\n')


class AdmissionTest(unittest.TestCase):
    def test_unrelated_research_update_does_not_change_preload_input(self):
        original=dict(probe=dict(preload=dict(passed=True,sha256='native')),research=dict(state='ready'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'index.json'
            current=copy.deepcopy(original);current['research']['state']='running'
            path.write_text(json.dumps(current))
            with mock.patch.object(ur_validation,'P',root),mock.patch.object(ur_validation,'git',return_value=json.dumps(original).encode()):
                self.assertEqual(ur_validation.committed(path,('probe','preload')),original['probe']['preload'])
                # Whole-file consumers still reject an uncommitted input.
                with self.assertRaises(RuntimeError):ur_validation.committed(path)

    def test_scoped_preload_changes_and_missing_fields_still_fail_closed(self):
        original=dict(probe=dict(preload=dict(passed=True,sha256='native')),research={})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'index.json'
            changes=[dict(passed=True,sha256='changed'),dict(passed=1,sha256='native'),
                     dict(passed=True,sha256='native',extra=True),None]
            for changed in changes:
                current=copy.deepcopy(original)
                if changed is None:del current['probe']['preload']
                else:current['probe']['preload']=changed
                path.write_text(json.dumps(current))
                with self.subTest(changed=changed),mock.patch.object(ur_validation,'P',root),mock.patch.object(ur_validation,'git',return_value=json.dumps(original).encode()):
                    with self.assertRaises(RuntimeError):ur_validation.committed(path,('probe','preload'))

    def test_warmup_requires_complete_cells_not_just_positive_batch(self):
        self.assertEqual(mtp.request_evidence(log(),[sample()])[0]['draft_cells'],73)
        # These all have a positive batch, but do not prove the same request.
        for text in (log(cells=72),log(path='token'),log(mode=1),log(cells=0),
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
        for extra in ('strata serve: DRAFT_PREFILL malformed\n',
                      'strata serve: DRAFT_PREFILL path=batched mode=0 cells=1 trailing\n',
                      'strata draft kv: unexpected timer\n',
                      'strata serve: SNAPSHOT_VERIFY unexpected restore\n'):
            for text in (extra+log(),log()+extra):
                with self.subTest(text=text),self.assertRaises(RuntimeError):
                    mtp.request_evidence(text,[sample()])

    def test_long_request_requires_exact_chunks_and_tail(self):
        chunks=[8192]*4+[441]
        self.assertEqual(mtp.request_evidence(log(read=33210,chunks=chunks),[sample(33210)])[0]['draft_cells'],33209)
        for wrong in ([33209],[8191,8193,8192,8192,441],chunks+[1],chunks[:-1],chunks[:-1]+[440]):
            with self.subTest(chunks=wrong),self.assertRaises(RuntimeError):
                mtp.request_evidence(log(read=33210,chunks=wrong),[sample(33210)])

    def test_effective_environment_rejects_observer_reintroduction_and_safety_changes(self):
        env=dict(STRATA_MTP_BATCH_F16='0',STRATA_SNAPSHOT_VERIFY='1',STRATA_SYCL_PREFILL_SAME_QUEUE='1',
                 UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',ZES_ENABLE_SYSMAN='0')
        actual={k.encode():v.encode() for k,v in env.items()}
        mtp.require_environment(actual,env)
        for key in mtp.ABSENT_OBSERVERS:
            # Even value0 is forbidden: several observer switches use presence.
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                mtp.require_environment(actual | {key.encode():b'0'},env)
        for key in env:
            changed=dict(actual);changed.pop(key.encode())
            with self.subTest(missing=key),self.assertRaises(RuntimeError):mtp.require_environment(changed,env)

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
