"""Host corruption controls for QSA dump admission; repository LICENSE."""
from array import array
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'sycl'))
import qsa_selection_capture as capture


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'capture.bin'

    def write(self, header, rows):
        self.path.write_bytes(capture.HEADER.pack(*header) + array('i', rows).tobytes())

    def dense(self):
        self.header = (0, 0, 2, 2051)
        # Uninitialized cap suffixes deliberately have invalid values.
        self.write(self.header, [0]+[-123]*2050+[0,1]+[-456]*2049)

    def sparse(self):
        self.header = (0, 3000, 2, 2051)
        self.write(self.header, list(range(2051))*2)

    def test_dense_initialized_prefix_only(self):
        self.dense(); result = capture.inspect(self.path, [self.header])
        self.assertEqual(result['records'][0]['valid_entries'], 3)
        self.assertEqual(result['records'][0]['unused_cap_entries'], 4099)

    def test_sparse_statistics_and_exact_extraction(self):
        self.sparse(); original = self.path.read_bytes()
        result = capture.inspect(self.path, [self.header], self.header, True)
        self.assertEqual(result['locality']['adjacent_query_shared_fraction']['median'], 1)
        self.assertEqual(result['locality']['selected_entries'], 4102)
        self.assertEqual(result['locality']['unique_logical_cells'], 2051)
        self.assertEqual(sum(result['locality']['age_bin_counts']), 4102)
        kept = capture.extract(self.path, result, self.root/'fixture.bin')
        self.assertEqual((self.root/'fixture.bin').read_bytes(), original)
        self.assertEqual(kept['sha256'], result['sha256'])

    def test_exact_list_classes_ignore_uninitialized_suffix_and_keep_width(self):
        self.dense()
        result = capture.inspect(self.path, [self.header], equivalence_enabled=True)
        exact = result['records'][0]['exact_lists']
        self.assertEqual(exact['queries'], 2)
        self.assertEqual(exact['distinct_lists'], 2)
        self.assertEqual(exact['same_list_pairs'], 0)
        # Equal initialized sparse rows really share a list; invalid dense cap
        # suffixes above were not incorporated in the equivalence key.
        self.sparse()
        result = capture.inspect(self.path, [self.header], equivalence_enabled=True)
        exact = result['records'][0]['exact_lists']
        self.assertEqual(exact['distinct_lists'], 1)
        self.assertEqual(exact['queries_in_repeated_classes'], 2)
        self.assertEqual(exact['same_list_pairs'], 1)

    def test_exact_lists_do_not_confuse_overlap_or_cross_window_pairs(self):
        exact = capture.ExactLists()
        for values in ([0,1], [0,2], [0,1], [0,2], [0,1]):
            exact.add(array('i', values))
        result = exact.result()
        self.assertEqual(result['distinct_lists'], 2)
        self.assertEqual(result['same_list_pairs'], 4)
        self.assertEqual(result['maximum_class'], 3)
        self.assertEqual(result['consecutive_windows']['4']['same_list_pairs'], 2)
        self.assertEqual(result['consecutive_windows']['4']['queries_in_repeated_classes'], 4)
        self.assertEqual(result['consecutive_windows']['4']['windows'], 2)
        self.assertEqual(result['consecutive_windows']['16']['same_list_pairs'], 4)

    def test_exact_lists_are_not_merged_across_layer_records(self):
        self.sparse(); raw = self.path.read_bytes()
        other = (1, *self.header[1:])
        self.path.write_bytes(raw+capture.HEADER.pack(*other)+raw[16:])
        result = capture.inspect(self.path, [self.header, other], equivalence_enabled=True)
        self.assertEqual([r['exact_lists']['queries'] for r in result['records']], [2,2])
        self.assertEqual([r['exact_lists']['same_list_pairs'] for r in result['records']], [1,1])

    def test_truncated_and_extra_payload(self):
        self.dense(); raw = self.path.read_bytes()
        for altered in (raw[:-1], raw+b'0', raw[:15]):
            self.path.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, 'extent'):
                capture.inspect(self.path, [self.header])

    def test_untrusted_header_geometry(self):
        self.dense(); original = self.path.read_bytes()
        for header in ((0,-1,2,2051), (0,0,8193,2051), (12,0,2,2051), (0,0,2,2147483647)):
            self.path.write_bytes(capture.HEADER.pack(*header)+original[16:])
            with self.assertRaisesRegex(ValueError, 'header'):
                capture.inspect(self.path, [self.header])

    def test_bad_expected_geometry_and_duplicates(self):
        self.dense()
        for expected in ([], [self.header,self.header], [(0,0,8193,2051)], [(0,262143,2,2051)], [(False,0,2,2051)]):
            with self.assertRaises(ValueError): capture.inspect(self.path, expected)

    def test_negative_duplicate_unordered_future_ids(self):
        for cells in ([-1,1], [0,0], [1,0], [0,2]):
            self.write((0,0,2,2051), [0]+[0]*2050+cells+[0]*2049)
            with self.assertRaisesRegex(ValueError, 'logical cell'):
                capture.inspect(self.path, [(0,0,2,2051)])

    def test_record_order(self):
        self.dense(); original = self.path.read_bytes()
        self.path.write_bytes(original+capture.HEADER.pack(1,0,2,2051)+original[16:])
        with self.assertRaisesRegex(ValueError, 'header'):
            capture.inspect(self.path, [(1,0,2,2051),self.header])

    def test_uninitialized_suffix_cannot_be_retained(self):
        self.dense()
        with self.assertRaisesRegex(ValueError, 'uninitialized'):
            capture.inspect(self.path, [self.header], self.header)

    def test_changed_source_and_existing_destination_preserved(self):
        self.sparse(); result = capture.inspect(self.path, [self.header], self.header)
        dest = self.root/'fixture.bin'; dest.write_bytes(b'existing')
        with self.assertRaises(FileExistsError): capture.extract(self.path, result, dest)
        self.assertEqual(dest.read_bytes(), b'existing')
        self.path.write_bytes(self.path.read_bytes()+b'x')
        with self.assertRaises(ValueError): capture.extract(self.path, result, self.root/'new.bin')
        self.assertFalse((self.root/'new.bin').exists())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CaptureTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps(dict(cases=result.testsRun, passed=result.wasSuccessful())))
    raise SystemExit(not result.wasSuccessful())
