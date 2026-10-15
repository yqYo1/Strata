# SOURCE ONLY, Main executes serially. No model/runtime required.
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import validate_wave_census as validator


def report():
    # Five chunks [3,3,3,3,1], same ten experts: independent hand expectation.
    waves=[]
    for q,unions,duplicates,maximum,bin_index,bin_count in [(2,30,20,6,2,20),(4,20,30,12,3,10)]:
        hist=[0]*16;hist[0]=10;hist[bin_index]=bin_count
        waves.append(dict(width_chunks=q,wave_count=(5+q-1)//q,partial_wave_count=1,
                          partial_chunks=1,partial_tokens=1,union_sum=unions,union_min=10,union_max=10,
                          active_incidences=50,duplicate_incidences=duplicates,expert_rows=130,
                          max_wave_ne=maximum,expert_wave_rows_hist=hist))
    return dict(version=1,complete=True,wave_schema_version=1,
                wave_grouping='request_relative_chunk_ordinal',issuer_expert_ids_verified=True,
                tokens=13,position=17,configured_chunk=3,
                layers=[dict(layer=l,chunks=5,tokens=13,gu_calls=50,down_calls=50,waves=copy.deepcopy(waves)) for l in range(48)])


class ValidatorTests(unittest.TestCase):
    def test_nonzero_position_partial_waves(self):self.assertTrue(validator.validate_wave_fields(report())['passed'])
    def test_rejects_bool_and_float_integer_fields(self):
        for key in ('version','wave_schema_version'):
            for value in (True,1.0):
                d=report();d[key]=value
                with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):validator.validate_wave_fields(d)
        for key in ('layer','chunks','tokens','gu_calls','down_calls'):
            for kind in ('bool','float'):
                d=report();value=d['layers'][0][key]
                d['layers'][0][key]=bool(value) if kind=='bool' else float(value)
                with self.subTest(key=key,kind=kind),self.assertRaises(RuntimeError):validator.validate_wave_fields(d)
    def test_rejects_wrong_union_hist_tail_schema(self):
        for key,value in [('union_sum',31),('partial_tokens',2),('max_wave_ne',32769),('expert_rows',131),('duplicate_incidences',19),('expert_wave_rows_hist',[0]*16)]:
            d=report();d['layers'][0]['waves'][0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):validator.validate_wave_fields(d)
        d=report();d['issuer_expert_ids_verified']=False
        with self.assertRaises(RuntimeError):validator.validate_wave_fields(d)
    def test_existing_model_validator_is_required(self):
        with patch.object(validator,'validate_census',return_value=dict(raw=report(),passed=True,complete=True)) as old:
            self.assertTrue(validator.validate_wave_census(stderr_path='unused',expected_tokens=13)['wave_validation']['passed'])
            old.assert_called_once_with(stderr_path='unused',expected_tokens=13)
        with patch.object(validator,'validate_census',side_effect=RuntimeError('original gate')):
            with self.assertRaises(RuntimeError):validator.validate_wave_census(stderr_path='unused',expected_tokens=13)


if __name__=='__main__':unittest.main()
