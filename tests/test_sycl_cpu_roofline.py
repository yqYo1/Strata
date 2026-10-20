"""Admission and timing eligibility checks; no GPU/model/build execution."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/sycl'))
import cpu_roofline as calibration


class TimingEligibility(unittest.TestCase):
    def test_full_count_cannot_hide_duplicate_or_missing_matrix_cell(self):
        rows=[dict(pair=list(p),workers=w,mode=m,nt=t,scan_threads=None)
              for p in calibration.PAIRS for w in (1,5) for m in ('hot','stream') for t in (1,2,3,4)]
        rows += [dict(pair=None,workers=None,mode='scan',nt=0,scan_threads=n) for n in (1,6,12)]
        calibration.require_matrix(rows)
        rows[-1]=rows[0]
        with self.assertRaises(RuntimeError):calibration.require_matrix(rows)

    def test_metadata_and_native_shards_have_separate_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata=Path(tmp)/'native_experts.txt'
            lines=['# absolute offsets in first.gguf, or in the named shard']
            for (gu,dn),count in calibration.PAIRS.items():
                n=2*calibration.SIZES[gu]*640+(180 if dn==42 else 360)*2560
                lines += [f'{i} {gu} {dn} 0 {n} 0 0 0' for i in range(count)]
            metadata.write_text('\n'.join(lines))
            native_dir=Path(tmp)/'models'
            rows=calibration.formats(metadata,native_dir)
            self.assertEqual(len(rows),7)
            self.assertTrue(all(Path(r['shard']).parent==native_dir for r in rows))
            self.assertTrue(all(r['experts']*r['blob_bytes']>=96<<20 for r in rows))

    def row(self):
        return dict(pair=[18,20],workers=5,closed=True,exit_code=0,rows=[
            dict(kind='sample',mode='hot',nt=3,attempt=0,rep=i,iterations=10,
                 requested_bytes=1000000000,wall_s=1.2,cpu_s=6.) for i in range(5)])

    def test_individual_samples_and_rate(self):
        row=self.row();cell=calibration.summarize([row])[0]
        self.assertEqual(cell['samples'],row['rows'])
        self.assertAlmostEqual(cell['requested_GB_s_median'],1/1.2)

    def test_failure_or_missing_repetition_is_ineligible(self):
        for field,value in [('closed',False),('exit_code',1)]:
            row=self.row();row[field]=value
            with self.assertRaises(RuntimeError):calibration.summarize([row])
        row=self.row();row['rows'].pop()
        with self.assertRaises(RuntimeError):calibration.summarize([row])

    def test_under_floor_not_silently_accepted(self):
        row=self.row();row['rows'][2]['wall_s']=.999
        with self.assertRaises(RuntimeError):calibration.summarize([row])
        extended=copy.deepcopy(row['rows'])
        for r in extended:r.update(attempt=1,wall_s=2.4,iterations=20)
        row['rows']+=extended
        self.assertEqual(len(calibration.summarize([row])[0]['samples']),10)

    def test_duplicate_repetition_is_ineligible(self):
        row=self.row();row['rows'][-1]['rep']=3
        with self.assertRaises(RuntimeError):calibration.summarize([row])


if __name__=='__main__':unittest.main()
