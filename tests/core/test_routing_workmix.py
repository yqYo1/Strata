"""Reject ambiguous frames and corrupt route streams before model collection."""
import copy
import io
from pathlib import Path
import sys
import struct
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import routing_workmix as route


def fixture():
    windows = [1,4]; membership = [(1<<0)|(1<<9)]*48
    formats = [(18,20,2176000)]*24+[(22,20,1971200)]*24
    data = b''.join(route.RECORD.pack(l,10,*range(10),*([0.]*10))
        for t in windows for l in range(48) for _ in range(t))
    bins = [dict(gu=g,down=d,blob_bytes=z,calls=48,CPU=960,VRAM=240,
        jobs_NT=[192,0,0,192,0,0,0,0]) for g,d,z in dict.fromkeys(formats)]
    expected = dict(windows=2,rows=5,cpu_jobs=768,cpu_entries=1920,vram_entries=480,
        jobs_NT=[384,0,0,384,0,0,0,0],logical_job_blob_bytes=384*(2176000+1971200),format_bins=bins)
    return data,windows,membership,formats,expected


class RoutingWorkmixTests(unittest.TestCase):
    def test_exact_windows_jobs_and_provider_bins(self):
        data,windows,membership,formats,expected = fixture(); stream = io.BytesIO(data)
        counts,summary = route.decode(stream,windows,membership,formats,expected)
        self.assertEqual(stream.read(),b'')
        self.assertEqual(counts[0][0],[1,0,0,1])
        self.assertEqual(counts[47][8],[1,0,0,1])
        self.assertEqual(counts[47][10],[0,0,0,0])
        self.assertEqual(summary['cpu_jobs'],768)
        encoded = route.encode_counts(counts)
        self.assertEqual(encoded[:4],b'SRWC');self.assertEqual(len(encoded),393236)
        self.assertEqual(route.read_counts(encoded),counts)

    def test_partial_reordered_ids_and_nonplaceholder_weights_rejected(self):
        data,w,m,f,e = fixture()
        record = route.RECORD
        changes = [data[:-1],record.pack(1,10,*range(10),*([0.]*10))+data[record.size:],
            record.pack(0,10,*([0]*10),*([0.]*10))+data[record.size:],
            record.pack(0,10,512,*range(1,10),*([0.]*10))+data[record.size:],
            record.pack(0,10,*range(10),*([1.]*10))+data[record.size:],
            data[:record.size]+data[2*record.size:]]
        for changed in changes:
            with self.subTest(changed=len(changed)),self.assertRaises(RuntimeError):
                route.decode(io.BytesIO(changed),w,m,f,e)

    def test_matching_global_totals_do_not_replace_format_and_membership_checks(self):
        data,w,m,f,e = fixture()
        for key in ['windows','rows','cpu_jobs','cpu_entries','vram_entries','logical_job_blob_bytes']:
            wrong=copy.deepcopy(e);wrong[key]+=1
            with self.subTest(key=key),self.assertRaises(RuntimeError):route.decode(io.BytesIO(data),w,m,f,wrong)
        wrong=copy.deepcopy(e);wrong['format_bins'][0]['CPU']+=1
        with self.assertRaises(RuntimeError):route.decode(io.BytesIO(data),w,m,f,wrong)
        with self.assertRaises(RuntimeError):route.decode(io.BytesIO(data),w,[0]*48,f,e)

    def test_only_explicit_ordered_request_phase_window_frames(self):
        text=('strata trace: request 33214 0\nstrata trace: prompt done (slots refilled) -1 -1\n'
            'strata trace: window 33213 1\nstrata trace: window 33214 4\nstrata serve: prompt summary\n')
        sample=dict(prompt_tokens=33214)
        self.assertEqual(route.request_windows(text,[sample]),[[1,4]])
        changes=[text.replace('prompt done (slots refilled)','missing'),text.replace('request 33214','request 33215'),
            text.replace('window 33214 4','window 33213 4'),text.replace('window 33214 4','window 33214 3'),
            text.replace('prompt done (slots refilled) -1 -1\n',''),text+text,
            text.replace('strata trace: window 33213 1','strata trace: read 10 tokens (windows)\nstrata trace: window 33213 1')]
        for changed in changes:
            with self.subTest(changed=changed),self.assertRaises(RuntimeError):route.request_windows(changed,[sample])

    def test_matrix_header_and_shape_refusal(self):
        data,w,m,f,e=fixture();counts,_=route.decode(io.BytesIO(data),w,m,f,e)
        raw=route.encode_counts(counts)
        for wrong in [raw[:-1],raw+b'\0',b'BAD!'+raw[4:],raw[:4]+b'\2\0\0\0'+raw[8:]]:
            with self.subTest(size=len(wrong)),self.assertRaises(RuntimeError):route.read_counts(wrong)

    def test_window_frequency_differs_from_entries_with_equal_quotas(self):
        profile=Path(__file__).resolve().parents[2]/'data/expert-profile.bin'
        raw=profile.read_bytes();order=[e for l,e in struct.iter_unpack('<HH',raw[24:24+48*512*4]) if l==0]
        counts=[[[0]*4 for _ in range(512)] for _ in range(48)]
        for e in order[:25]:counts[0][e][0]=100
        counts[0][order[25]][3]=5  # 20 entries, five unique callbacks
        counts[0][order[26]][0]=10 # ten entries, ten unique callbacks
        jobs=route.ranked_membership(counts,profile,'windows');entries=route.ranked_membership(counts,profile,'entries')
        self.assertEqual(jobs[0].bit_count(),26);self.assertEqual(entries[0].bit_count(),26)
        self.assertTrue(jobs[0]&(1<<order[26]));self.assertFalse(jobs[0]&(1<<order[25]))
        self.assertTrue(entries[0]&(1<<order[25]));self.assertFalse(entries[0]&(1<<order[26]))

    def test_counterfactual_reproduces_independently_bound_jobs_bytes_and_entries(self):
        data,w,m,f,e=fixture();counts,_=route.decode(io.BytesIO(data),w,m,f,e)
        predicted=route.counterfactual(counts,m,f)
        for k in ['cpu_jobs','cpu_entries','vram_entries','jobs_NT','logical_job_blob_bytes']:
            self.assertEqual(predicted[k],e[k])


if __name__ == '__main__':unittest.main()
