"""Regress the real RESUME/PP omission and independent request extents;no GPU."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
from native_request_protocol import RequestProtocol


def stream(n=1024,m=128):
    progress=[f'PP {i} {n} 37244 27.5' for i in range(8192,n-1,8192)]
    if n>1:progress.append(f'PP {n-1} {n} 37244 27.5')
    return ['RESUME 0',*progress,'REUSED 0',*(['T 13']*m),
            f'DONE {m} {n} 37244.0 23892.0 length 77 159 0 15260 102240 0 0 0.0 {n}']


class ProtocolTests(unittest.TestCase):
    def parse(self,lines,n=1024,m=128):
        reader=RequestProtocol(n,m)
        for line in lines:reader.feed(line)
        return reader

    def test_complete_request_shapes(self):
        for n,m in ((1,2),(1024,128),(32769,256)):
            with self.subTest(prompt=n,output=m):
                reader=self.parse(stream(n,m),n,m)
                self.assertIsNotNone(reader.done)
                self.assertEqual(reader.position,n-1)
                self.assertEqual(len(reader.tokens),m)
                self.assertEqual(reader.progress_count,0 if n==1 else (n-2)//8192+1)

    def test_original_missing_resume_and_progress_regression(self):
        # Selected generate.cpp emits RESUME before prompt work and PP through
        # n-1 before REUSED. The old controller aborted on the first notification.
        reader=RequestProtocol(1024,128)
        for line in ('RESUME 0','PP 1023 1024 37244 27.5','REUSED 0'):reader.feed(line)
        self.assertTrue(reader.resumed and reader.reused)
        self.assertEqual(reader.tokens,[])
        self.assertIsNone(reader.done)

    def test_corrupt_order_values_and_extent(self):
        good=stream();done=good[-1]
        variants=[good[1:],['RESUME 1'],good[:1]+good,good[:1]+['PP 1024 1024 1 1'],
            good[:1]+['PP 1023 1023 1 1'],good[:1]+['PP 0 1024 1 1'],
            good[:1]+['PP 1 1024 nan 1'],good[:1]+['PP 1 1024 1 -1'],
            good[:1]+['PP 1 1024 unknown 1'],good[:2]+[good[1]],
            ['RESUME 0','REUSED 0'],good[:3]+['PP 1023 1024 2 1'],
            ['T 13'],good[:3]+['REUSED 0'],good[:3]+['T 248320'],good[:3]+['T -1'],
            good[:3]+['T ٠'],good[:-1]+['T 13'],good[:3]+[done],
            good[:-1]+[done.replace('128','127',1)],good[:-1]+[done.replace('length','cancel')],
            good[:-1]+[done.replace('1024','1023')],good[:-1]+[done.replace('77 159','160 159')],
            good[:-1]+[done.replace('37244.0','nan')],good[:-1]+[done.replace('0.0','inf')],
            good+['T 13'],good+['RESUME 0'],['ERR test'],['unknown'],[''],['T '+'1'*4096]]
        for i,lines in enumerate(variants):
            with self.subTest(case=i),self.assertRaises(RuntimeError):self.parse(lines)

    def test_geometry_bounds(self):
        for n,m,v in ((0,1,1),(262145,1,1),(1,0,1),(1,262145,1),(1,1,0),
                      (True,1,1),(1,False,1),(1,1,True),(1,1,2**31)):
            with self.subTest(prompt=n,output=m,vocab=v),self.assertRaises(RuntimeError):
                RequestProtocol(n,m,v)


if __name__=='__main__':unittest.main()
