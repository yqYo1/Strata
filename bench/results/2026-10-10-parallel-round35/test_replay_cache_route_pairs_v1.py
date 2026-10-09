"""SOURCE ONLY independent small-job/file/closed-gate CPU fixtures; root runs."""
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest
import replay_cache_route_pairs_v1 as replay


def admitted_gate_fixture():
    keys=('completed','healthy','math_gate_passed','pair_gate_passed','histogram_gate_passed',
          'pair_shape_gate_passed','protocol_boundary_gate_passed','text_budget_gate_passed')
    record={key:True for key in keys}
    record.update(active=False,exit_code=0,exit_signal=None,error=None,new_fault_messages=[],cleanup={},
                  controller_sha256=replay.CONTROLLER_SHA,pair_parser_sha256=replay.PARSER_SHA,
                  shape_helper_sha256=replay.SHAPE_SHA,C_original_math_gate_passed=False,D_original_healthy=False,
                  D_original_full_math_passed=False,prior_r5_original_math_passed=False,prior_r5_original_healthy=False,
                  performance_eligible=False,full_lifecycle_passed=False,adopted=False,
                  requests=[{} for unused in range(4)])
    return record


class ReplayFixtures(unittest.TestCase):
    def test_closed_all_gates_and_prior_failure_preservation(self):
        record=admitted_gate_fixture();replay.closed_gate(record)
        for key in ('completed','healthy','math_gate_passed','pair_gate_passed','histogram_gate_passed',
                    'pair_shape_gate_passed','protocol_boundary_gate_passed','text_budget_gate_passed'):
            changed=copy.deepcopy(record);changed[key]=False
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'closed success gate: '+key): replay.closed_gate(changed)
        changed=copy.deepcopy(record);changed['active']=True
        with self.assertRaisesRegex(ValueError,'receipt active/unknown'): replay.closed_gate(changed)
        changed=copy.deepcopy(record);changed['prior_r5_original_healthy']=True
        with self.assertRaisesRegex(ValueError,'preserve prior failures'): replay.closed_gate(changed)
        changed=copy.deepcopy(record);changed['cleanup']={'forced':True}
        with self.assertRaisesRegex(ValueError,'exit/fault/cleanup'): replay.closed_gate(changed)
    def test_exact_source_identity_and_unknown_receipt_pin(self):
        changed=admitted_gate_fixture();changed['controller_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source identity'): replay.closed_gate(changed)
        for value in ('',None,'UNKNOWN','a'*63):
            with self.assertRaisesRegex(ValueError,'exact SHA256 required'): replay.pin(value)
    def test_interval_hash_range_and_raw_file_sha_negatives(self):
        raw=b'prefix\nTARGET\ntail\n';literal=b'TARGET\n';sha=hashlib.sha256(literal).hexdigest()
        self.assertEqual(replay.verified_interval(raw,7,14,sha),literal)
        with self.assertRaisesRegex(ValueError,'interval SHA256 mismatch'): replay.verified_interval(raw,7,13,sha)
        with self.assertRaisesRegex(ValueError,'interval bounds/order'): replay.verified_interval(raw,7,999,sha)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stderr';path.write_bytes(raw)
            self.assertEqual(replay.verified_file(path,hashlib.sha256(raw).hexdigest()),raw)
            path.write_bytes(raw+b'extra\n')
            with self.assertRaisesRegex(ValueError,'file SHA256 mismatch'): replay.verified_file(path,hashlib.sha256(raw).hexdigest())
    def test_orphan_extra_marker_and_overlapping_interval(self):
        marker=b'CACHE_ROUTE_PAIRS_V1 BEGIN request=1\n'
        self.assertEqual(replay.owned_frames(marker,[(0,len(marker))]),1)
        with self.assertRaisesRegex(ValueError,'orphan/extra frame'): replay.owned_frames(marker+b'CACHE_ROUTE_PAIRS_V1 END\n',[(0,len(marker))])
        with self.assertRaisesRegex(ValueError,'interval bounds/order'): replay.owned_frames(marker,[(0,10),(5,len(marker))])
        with self.assertRaisesRegex(ValueError,'misplaced frame prefix'): replay.owned_frames(b'junk '+marker,[(0,len(marker)+5)])
    def test_independent_literal_canonical_shape_hash(self):
        rows=[(0,1,3,2,1,1),(1,8,2,1,0,1)]
        canonical=b'[[0,1,3,2,1,1],[1,8,2,1,0,1]]\n'
        expected=hashlib.sha256(canonical).hexdigest()
        self.assertEqual(replay.canonical_hash(rows),expected)
        self.assertNotEqual(replay.canonical_hash(list(reversed(rows))),expected)
        self.assertNotEqual(replay.canonical_hash([(0,1,3,2,0,2),(1,8,2,1,0,1)]),expected)
    def test_literal_evaluated_jobs_default_coverage_only(self):
        # Resident pair has one one-token and one two-token evaluated job;
        # nonresident pair's two-token job does not count as default coverage.
        parsed={'pairs':[dict(layer=0,expert=1,entries=3,callback_pairs=2),
                         dict(layer=1,expert=8,entries=2,callback_pairs=1)]}
        result=replay.coverage(parsed,{(0,1)},[(18,20),(22,42)]+[(21,20)]*46)
        self.assertEqual(result,dict(covered_entries=3,covered_callback_pairs=2,cells=[
            dict(phase='Down',type=20,nt=1,experts=1,tokens=1,output_rows=2560),
            dict(phase='Down',type=20,nt=2,experts=1,tokens=2,output_rows=2560),
            dict(phase='GU',type=18,nt=1,experts=1,tokens=1,output_rows=640),
            dict(phase='GU',type=18,nt=2,experts=1,tokens=2,output_rows=640)]))


if __name__=='__main__': unittest.main()
