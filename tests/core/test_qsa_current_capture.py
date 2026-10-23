"""Main-authored request/tail binding and prefix conservation; repository LICENSE.

Run: python3 -m unittest discover -s tests/core -p test_qsa_current_capture.py -v
Sparse host fixtures exercise headers only;the committed native14-case parser
separately validates selected ID/step payloads. No runtime/device is created.
"""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import capture_qsa_locality as capture


class RequestBinding(unittest.TestCase):
    def setUp(self):
        self.root=tempfile.TemporaryDirectory(prefix='qsa-binding-')
        self.addCleanup(self.root.cleanup);self.cells=117488;self.files={}
        for suffix,width in (('.ids',2051),('.steps',4)):
            p=Path(self.root.name)/('capture'+suffix);span=16+512*width*4
            with p.open('wb') as f:
                f.truncate(span*12)
                for layer in range(12):
                    f.seek(layer*span);f.write(struct.pack('<4i',layer,self.cells-512,512,width))
            self.files[suffix]=dict(path=str(p),bytes=p.stat().st_size)

    def test_actual_request_extent(self):
        self.assertEqual(capture.bind_tail(self.files,self.cells),
            dict(layers=12,first=116976,queries=512,cap=2051,steps=4,cells=117488))

    def test_other_request_tail_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'actual request tail'):
            capture.bind_tail(self.files,self.cells-1)

    def test_one_late_step_header_mismatch_is_rejected(self):
        p=Path(self.files['.steps']['path']);span=16+512*4*4
        with p.open('r+b') as f:f.seek(11*span+4);f.write(struct.pack('<i',self.cells-513))
        with self.assertRaisesRegex(RuntimeError,'actual request tail'):
            capture.bind_tail(self.files,self.cells)

    def test_short_header_is_rejected(self):
        p=Path(self.files['.ids']['path']);span=16+512*2051*4
        with p.open('r+b') as f:f.truncate(11*span+12)
        with self.assertRaisesRegex(RuntimeError,'actual request tail'):
            capture.bind_tail(self.files,self.cells)
    def test_trailing_data_is_rejected(self):
        p=Path(self.files['.steps']['path'])
        with p.open('ab') as f:f.write(b'x')
        with self.assertRaisesRegex(RuntimeError,'capture trailing bytes'):
            capture.bind_tail(self.files,self.cells)


class PrefixConservation(unittest.TestCase):
    @staticmethod
    def lines():
        cells=117488
        return [f'strata QSA locality prefix: layer={layer} first={first} queries={min(8192,cells-first)} '
            f'live={min(first+8192,cells)} page=4 packed_logical_rows={((min(first+8192,cells)+3)//4)*8} submitted=1'
            for layer in range(12) for first in range(0,cells,8192)]

    def test_full_chunk_and_tail_counts(self):
        x=capture.prefix_counts('\n'.join(self.lines()),117488)
        self.assertEqual((x['chunks_per_layer'],x['submitted_calls'],x['packed_logical_rows_per_layer'],
            x['packed_logical_rows_total']),(15,180,1955296,23463552))
        self.assertFalse(x['physical_traffic_measured'])

    def test_duplicate_or_missing_submission_is_rejected(self):
        lines=self.lines()
        for bad in (lines[:-1],lines+[lines[-1]],lines[:-1]+[lines[0]]):
            with self.assertRaisesRegex(RuntimeError,'prefix submission'):
                capture.prefix_counts('\n'.join(bad),117488)

    def test_wrong_padding_or_unparsed_record_is_rejected(self):
        lines=self.lines()
        for bad in (lines[:-1]+[lines[-1].replace('packed_logical_rows=234976','packed_logical_rows=234975')],
            lines+['strata QSA locality prefix: invalid']):
            with self.assertRaisesRegex(RuntimeError,'prefix submission'):
                capture.prefix_counts('\n'.join(bad),117488)


if __name__=='__main__':unittest.main()
