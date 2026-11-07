#!/usr/bin/env python3
"""Refuse unbound/partial native work before using it for CPU-method selection."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/sycl'))
import cpu_work_census as census


def fixture(request=1, prompt=33214):
    formats = [(18, 20, 2176000) if l % 2 == 0 else (22, 20, 1971200) for l in range(48)]
    membership = [(1 << 0) | (1 << 256)] * 48
    binding = dict(prompt_tokens=prompt, windows=2, rows=5, cpu_jobs=1008, cpu_entries=2160, pool_ms=4.8)
    lines = [f'strata cpu work: begin request={request} phase=decode prompt_tokens={prompt} '
        'layers=48 experts=512 cache=restored_decode_snapshot_immutable']
    words = ''.join(f'{(membership[0] >> (64*i)) & ((1 << 64)-1):016x}' for i in range(8))
    lines += [f'strata cpu work: cache request={request} layer={l} bin={l%2} resident_bits={words}' for l in range(48)]
    for b in range(2):
        gu, down, size = formats[b]
        lines.append(f'strata cpu work: bin request={request} bin={b} gu={gu} down={down} H=2560 FF=640 '
            f'blob_bytes={size} gu_act=15 down_act=8 gu_row=980 down_row=360 act_bytes=2920 h_bytes=680 '
            'calls=48 pool_ns=2400000 CPU=1080 VRAM=120 PCIe=0 peer=0 jobs_NT=312,0,0,192,0,0,0,0')
    lines.append(f'strata cpu work: end request={request} phase=decode valid=1 windows=2 rows=5 callbacks=96 '
        'k=10 cpu_jobs=1008 CPU=2160 VRAM=240 PCIe=0 peer=0 pcie_jobs=0 error=none')
    return '\n'.join(lines)+'\n', binding, formats, membership


class CensusReaderTests(unittest.TestCase):
    def test_exact_NT_counts_membership_word_order_and_logical_bytes(self):
        text, binding, formats, membership = fixture()
        row, = census.extract(text, [binding], formats, membership)
        self.assertEqual(row['jobs_NT'], [624, 0, 0, 384, 0, 0, 0, 0])
        self.assertEqual(row['cpu_jobs'], 1008)
        self.assertEqual(row['cpu_entries'], 2160)
        self.assertEqual(row['logical_job_blob_bytes'], 504*(2176000+1971200))
        self.assertAlmostEqual(row['NT1_fraction'], 624/1008)

    def test_requests_and_warmups_remain_separate(self):
        one, binding, formats, membership = fixture()
        two, second, _, _ = fixture(2, 33333)
        result = census.extract(one+two, [binding, second], formats, membership)
        self.assertEqual([r['request'] for r in result], [1, 2])
        self.assertEqual([r['cpu_jobs'] for r in result], [1008, 1008])
        with self.assertRaises(RuntimeError): census.extract(two+one, [binding, second], formats, membership)

    def test_refuses_missing_reordered_duplicate_unknown_or_partial_rows(self):
        text, binding, formats, membership = fixture()
        variants = [text.replace('phase=decode valid=1', 'phase=decode valid=0'),
            text.replace('error=none', 'error=request_aborted'),
            text.replace('cpu_jobs=1008', 'cpu_jobs=1009'),
            text.replace('windows=2', 'windows=3'),
            text.replace('calls=48', 'calls=47', 1),
            text.replace('jobs_NT=312,0,0,192,0,0,0,0', 'jobs_NT=312,0,0,191,0,0,0,0', 1),
            text.replace('jobs_NT=312,0,0,192,0,0,0,0', 'jobs_NT=312,0,0,192,1,0,0,0', 1),
            text.replace('layer=0 bin=0', 'layer=1 bin=0', 1),
            text.replace('pool_ns=2400000', 'pool_ns=3400000', 1),
            text.replace('gu=18', 'gu=23', 1),
            text.replace('CPU=1080', 'CPU=18446744073709551616', 1),
            text.replace('k=10', 'k=11'), text.replace('valid=1', 'valid=1 valid=1'),
            text.replace('strata cpu work: end', 'strata cpu work: unknown'),
            '\n'.join(text.splitlines()[:-1])+'\n', text+text,
            '\n'.join(text.splitlines()[:10]+text.splitlines()[11:])+'\n']
        for changed in variants:
            with self.subTest(changed=changed[-100:]), self.assertRaises(RuntimeError):
                census.extract(changed, [binding], formats, membership)

    def test_refuses_wrong_request_or_cache_or_model_binding(self):
        text, binding, formats, membership = fixture()
        changed = copy.deepcopy(binding); changed['prompt_tokens'] += 1
        with self.assertRaises(RuntimeError): census.extract(text, [changed], formats, membership)
        membership[25] ^= 1 << 300
        with self.assertRaises(RuntimeError): census.extract(text, [binding], formats, membership)

    def test_pool_rounding_bound_is_derived_from_windows_and_calls(self):
        text, binding, formats, membership = fixture()
        binding['pool_ms'] += .009
        census.extract(text, [binding], formats, membership)
        binding['pool_ms'] += .002
        with self.assertRaises(RuntimeError): census.extract(text, [binding], formats, membership)

    def test_actual_tracked_profile_admission_and_corrupt_reverse_map(self):
        root = Path(__file__).resolve().parents[2]
        raw = (root/'data/expert-profile.bin').read_bytes()
        masks = census.expected_membership(root/'data/expert-profile.bin')
        self.assertEqual([x.bit_count() for x in masks], [26]*47+[34])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'profile.bin'
            path.write_bytes(raw[:-1])
            with self.assertRaises(RuntimeError): census.expected_membership(path)
            changed = bytearray(raw); changed[-1] ^= 1; path.write_bytes(changed)
            with self.assertRaises(RuntimeError): census.expected_membership(path)


if __name__ == '__main__': unittest.main()
