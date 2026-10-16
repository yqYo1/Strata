"""Main-only offline v3 contract cases, SOURCE_ONLY_UNRUN. Repository LICENSE.
Synthetic rows/statuses never become approvals, receipts or collected evidence.
No compiler, subprocess, device, runtime, process inspection or environment edits.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
V1 = D.parent / 'own-metric-query-tiny-v1'
sys.path.insert(0, str(V1))
sys.path.insert(0, str(D))
from metrics import decode
from status_reader import reconcile
from synthetic_fixtures import fixture, encode
from report_validity import assess, tiny_qualified, labels, COMPROMISED, INTERPRET_FLAGS
from catalog_identity import validate


def schema():
    return json.loads((D / 'driver-schema.json').read_text())['metrics']


def artificial_values(catalog):
    # Zero flags and positive duration/clocks/available-report count.
    return [0 if m['result_type_name'] == 'ZET_VALUE_TYPE_BOOL8' else
            0.0 if m['result_type_name'].startswith('ZET_VALUE_TYPE_FLOAT') else 1 for m in catalog]


def csv_text(catalog, samples):
    text = '=== Device #0 Metrics ===\nKernel,GlobalInstanceId,SubDeviceId,' + ','.join(labels(catalog)) + '\n'
    for instance, values in zip((11, 12), samples):
        cells = [format(v, '.6f') if type(v) is float else str(v) for v in values]
        text += '"qsa[SIMD16 {1; 1; 1} {1; 1; 1}]",' + str(instance) + ',0,' + ','.join(cells) + '\n'
    return text


class Reports(unittest.TestCase):
    def setUp(self):
        self.catalog = schema()
        self.index = {m['name']: m['metric'] for m in self.catalog}
        self.samples = [artificial_values(self.catalog) for _ in (11, 12)]
        self.codes = json.loads((D / 'result_codes.json').read_text())

    def decoded(self, text=None):
        with tempfile.TemporaryDirectory(prefix='strata-v3-offline-') as temp:
            p = Path(temp) / 'artificial.csv'
            p.write_text(csv_text(self.catalog, self.samples) if text is None else text)
            return decode(p, self.catalog, {'qsa': 2}, time.monotonic() + 10)

    def api(self, query):
        events, pid, _, group, _ = fixture()
        # Replace only synthetic property metadata with real enumerated names/types;
        # the API events and values remain artificial fixtures, never runtime proof.
        properties = [r for r in events if r.get('site') == 'schema-metric' and r['phase'] == 'result']
        for row, m in zip(properties, self.catalog):
            row['data'].update(metric_name=m['name'], units=m['units'], result_type=m['result_type'])
        return reconcile(encode(events), pid, query, group, self.catalog, time.monotonic() + 10, self.codes)

    def assess(self, query=None, catalog=None, api=None):
        query = self.decoded() if query is None else query
        api = self.api(query) if api is None else api
        return assess(query, self.catalog if catalog is None else catalog, api, schema(), time.monotonic() + 10)

    def test_valid_raw_format_api_and_semantic_layers_remain_separate(self):
        query = self.decoded(); before = copy.deepcopy(query); api = self.api(query)
        result = self.assess(query, api=api)
        self.assertTrue(result['valid']); self.assertEqual(len(result['rows']), 2)
        self.assertEqual(query, before); self.assertFalse(query['collection_qualified'])
        self.assertTrue(tiny_qualified(True, api, result)); self.assertFalse(tiny_qualified(False, api, result))

    def test_each_compromised_flag_on_second_actual_row_holds_even_when_api_passes(self):
        for name in COMPROMISED:
            with self.subTest(flag=name):
                self.samples = [artificial_values(self.catalog) for _ in (11, 12)]
                self.samples[1][self.index[name]] = 1
                query = self.decoded(); api = self.api(query); result = self.assess(query, api=api)
                self.assertTrue(api['proved']); self.assertFalse(query['collection_qualified'])
                self.assertTrue(result['rows'][0]['valid']); self.assertFalse(result['rows'][1]['valid'])
                self.assertEqual(result['rows'][1]['compromised_flags'][name]['value'], 1)
                self.assertEqual(query['rows'][1]['values'][self.index[name]], 1)
                self.assertFalse(result['valid']); self.assertFalse(tiny_qualified(True, api, result))

    def test_valid_split_frequency_and_all_midquery_metadata_are_preserved(self):
        for name in INTERPRET_FLAGS:
            self.samples[1][self.index[name]] = 1
        self.samples[1][self.index['ResultUncertainty']] = 999  # No invented percentage gate.
        self.samples[1][self.index['ReportsCount']] = 3  # Not calculated sample count.
        self.samples[1][self.index['ReportId']] = 0
        self.samples[1][self.index['QueryBeginTime']] = 0
        result = self.assess()
        self.assertTrue(result['valid'])
        for name in INTERPRET_FLAGS:
            self.assertEqual(result['rows'][1]['interpretation_metadata'][name]['value'], 1)
        self.assertEqual(result['rows'][1]['interpretation_metadata']['ReportsCount']['value'], 3)
        self.assertEqual(result['rows'][1]['interpretation_metadata']['ResultUncertainty']['value'], 999)

    def test_zero_duration_clock_or_available_count_holds(self):
        for name in ('GpuTime', 'GpuCoreClocks', 'ReportsCount'):
            with self.subTest(metadata=name):
                self.samples = [artificial_values(self.catalog) for _ in (11, 12)]
                self.samples[1][self.index[name]] = 0
                # v1 already rejects zero GpuTime; preserve that strict boundary.
                if name == 'GpuTime':
                    with self.assertRaises(ValueError): self.decoded()
                else:
                    result = self.assess()
                    self.assertFalse(result['valid']); self.assertFalse(result['rows'][1]['valid'])

    def test_unknown_missing_or_reordered_csv_column_holds_in_unchanged_decoder(self):
        text = csv_text(self.catalog, self.samples)
        header = text.splitlines()[1]
        candidates = [header.replace('ReportError', 'UnknownReportFlag'),
                      header.replace(',ReportError', ''), header + ',UnknownReportFlag',
                      header.replace('ReportLost,ReportInconsistent', 'ReportInconsistent,ReportLost')]
        for bad in candidates:
            with self.subTest(header=bad):
                with self.assertRaises(ValueError): self.decoded(text.replace(header, bad))

    def test_unknown_missing_duplicate_reordered_driver_columns_hold(self):
        query = self.decoded(); api = self.api(query)
        variants = []
        unknown = copy.deepcopy(self.catalog); unknown[76]['name'] = 'UnknownReportFlag'; variants.append(unknown)
        variants.append(self.catalog[:-1])
        duplicate = copy.deepcopy(self.catalog); duplicate[77] = copy.deepcopy(duplicate[76]); variants.append(duplicate)
        reordered = copy.deepcopy(self.catalog); reordered[76], reordered[77] = reordered[77], reordered[76]; variants.append(reordered)
        extra = copy.deepcopy(self.catalog); extra.append(dict(extra[-1], metric=83, name='Unknown')); variants.append(extra)
        for catalog in variants:
            with self.subTest(schema=catalog[-1]['name']):
                result = self.assess(query, catalog, api)
                self.assertFalse(result['schema_valid']); self.assertFalse(result['valid'])
                self.assertEqual(len(result['rows']), 2)

    def test_flag_type_description_or_numeric_type_change_holds(self):
        query = self.decoded(); api = self.api(query)
        for key, value in [('result_type_name', 'ZET_VALUE_TYPE_UINT64'), ('result_type', 1),
                           ('metric_type_name', 'ZET_METRIC_TYPE_RAW'), ('description', 'Changed description')]:
            with self.subTest(field=key):
                catalog = copy.deepcopy(self.catalog); catalog[76][key] = value
                self.assertFalse(self.assess(query, catalog, api)['valid'])

    def test_invalid_flag_encoding_still_fails_format_decoder(self):
        self.samples[1][self.index['ReportError']] = 2
        with self.assertRaises(ValueError): self.decoded()

    def test_api_false_or_missing_sample_never_qualifies(self):
        query = self.decoded(); api = self.api(query)
        for bad in ({}, {**api, 'proved': False}, {**api, 'samples': 1, 'instances': [11]}):
            with self.subTest(api=bad):
                result = self.assess(query, api=bad)
                self.assertFalse(result['valid']); self.assertFalse(tiny_qualified(True, bad, result))

    def test_last_row_and_deadline_are_checked(self):
        query = self.decoded(); api = self.api(query)
        result = assess(query, self.catalog, api, schema(), time.monotonic() - 1)
        self.assertFalse(result['valid']); self.assertTrue(all(not r['valid'] for r in result['rows']))
        query['rows'][1]['values'] = query['rows'][1]['values'][:-1]
        self.assertFalse(self.assess(query, api=api)['valid'])


class ExistingCatalog(unittest.TestCase):
    def setUp(self):
        # Read-only historical receipts; never generated or rewritten by tests.
        self.contract = json.loads((D / 'catalog-contract.json').read_text())
        def load(pin):
            p = Path(pin['path']); raw = p.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin['sha256'])
            self.assertEqual(len(raw), pin['bytes'])
            return json.loads(raw)
        self.cat = copy.deepcopy(self.contract['artifacts'])
        self.build = load(self.cat['build_receipt']); self.enum = load(self.cat['receipt'])
        self.review = load(self.contract['binary_review']); self.evidence = load(self.contract['catalog_evidence'])

    def validate(self, source_sha=None, enum_sha=None):
        return validate(self.cat, self.build, self.enum, self.review, self.evidence, self.contract,
                        self.build['boot_id'], source_sha or self.contract['compiled_catalog_source']['sha256'],
                        enum_sha or self.contract['enum_controller_package']['sha256'])

    def test_actual_enum_controller_and_compiled_query_v1_source_are_distinct_valid_identities(self):
        self.assertEqual(self.build['reviewed_package_sha256'], '1c8567870dd9990c5519a060663cd866811bfd59981b868b3f34e2c2a6c1929d')
        self.assertTrue(self.contract['compiled_catalog_source']['path'].endswith('/own-metric-query-tiny-v1/enumerate_full.cpp'))
        self.assertTrue(self.validate()['bound'])

    def test_relabeling_build_package_as_query_v1_is_rejected(self):
        self.build['reviewed_package_sha256'] = '080316bd939dd5a34934dfc3b7ece2cf3871bf84d5195379773b3c733ff6987c'
        with self.assertRaises(ValueError): self.validate()

    def test_wrong_source_enum_package_or_changed_exact_argv_is_rejected(self):
        for kw in ({'source_sha': '0' * 64}, {'enum_sha': '0' * 64}):
            with self.subTest(identity=kw):
                with self.assertRaises(ValueError): self.validate(**kw)
        command = next(c for c in self.build['commands'] if c['label'] == 'host-build')
        command['argv'] = [a.replace('own-metric-query-tiny-v1/enumerate_full.cpp', 'own-metric-query-tiny-v3/enumerate_full.cpp') for a in command['argv']]
        with self.assertRaises(ValueError): self.validate()

    def test_missing_main_review_fault_or_native_environment_change_is_rejected(self):
        self.review['approved'] = False
        with self.assertRaises(ValueError): self.validate()
        self.review['approved'] = True
        self.build['fault_window']['fresh_faults'] = ['synthetic fault']
        with self.assertRaises(ValueError): self.validate()
        self.build['fault_window']['fresh_faults'] = []
        self.enum['native_snapshot']['environment']['ZET_ENABLE_METRICS'] = '0'
        with self.assertRaises(ValueError): self.validate()


if __name__ == '__main__':
    unittest.main()
