#!/usr/bin/env python3
"""Main offline format/error checks; this does not qualify device collection.
Repository LICENSE. Temporary synthetic files only; no provider imports.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = A / 'own-metric-query-tiny-v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target = A / 'own-metric-query-reader-review-v1.json'
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not target.exists()
        source = D / 'metrics.py'
        assert sha(source) == 'ce978565e9cdd786f4269dbacb5a8e3542e17bd7dca90e85a6ce77486b73f3df'
        spec = importlib.util.spec_from_file_location('main_reviewed_metric_reader_v1', source)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        for name in ('main_controller.py', 'metrics.py', 'summary.py'):
            compile((D / name).read_text(), str(D / name), 'exec')
        catalog = json.loads((A / 'own-metric-full-catalog-evidence-v2.json').read_text())['target_compute_metrics']
        labels = []
        for m in catalog:
            units = m['units']
            if 'null' in units:
                units = ''
            elif 'percent' in units:
                units = '%'
            labels.append(m['name'] + ('[' + units + ']' if units else ''))
        name = 'test::quoted<"a,b", 3>'
        shape = '[SIMD16 {3; 1; 1} {128; 1; 1}]'
        values = ['0.000000' if m['result_type_name'].endswith(('FLOAT32', 'FLOAT64')) else '0' for m in catalog]
        values[0] = '1000'
        header = '=== Device #0 Metrics ===\nKernel,GlobalInstanceId,SubDeviceId,' + ','.join(labels) + '\n'

        def body(v=None, instance='1', subdevice='0'):
            return '"' + name + shape + '",' + instance + ',' + subdevice + ',' + ','.join(v or values) + '\n'

        valid = header + body()
        tests = []
        with tempfile.TemporaryDirectory(prefix='metric-reader-review-', dir=B) as tmp:
            p = Path(tmp) / 'synthetic.csv'

            def check(label, text, should_pass, expected=None):
                p.write_text(text)
                try:
                    result = reader.decode(p, catalog, expected or {name: 1}, time.monotonic() + 2)
                    assert result['decoded'] and not result['collection_qualified']
                    actual = True
                except ValueError:
                    actual = False
                assert actual == should_pass, label
                tests.append(dict(case=label, expected_decode=should_pass, observed_decode=actual))

            check('valid_exact83columns_with_embedded_quote_and_comma', valid, True)
            check('missing_final_newline', valid[:-1], False)
            check('unknown_header_column', valid.replace('GpuTime[ns]', 'UnknownTime[ns]', 1), False)
            check('extra_column', header + body().rstrip('\n') + ',0\n', False)
            check('missing_query_sample', header, False)
            check('duplicate_global_instance', valid + body(), False)
            check('different_subdevice_sample', header + body(subdevice='1'), False)
            check('unknown_extra_device_section', valid + '=== Device #1 Metrics ===\n', False)
            check('missing_expected_dispatch', valid, False, {name: 2})
            bad = values.copy(); bad[0] = str(2 ** 64)
            check('integer_overflow', header + body(bad), False)
            bad = values.copy(); bad[64] = '2'
            assert catalog[64]['result_type_name'] == 'ZET_VALUE_TYPE_BOOL8'
            check('invalid_boolean_encoding', header + body(bad), False)
            bad = values.copy(); bad[4] = 'nan'
            check('nonfinite_float_encoding', header + body(bad), False)
            bad = values.copy(); bad[76] = '1'
            assert catalog[76]['name'] == 'ReportError'
            # v1 is explicitly diagnostic/HOLD: valid encoding is not metric
            # validity. Preserve this missing policy rather than claiming PASS.
            check('known_limit_ReportError_true_still_format_decodes', header + body(bad), True)
        build = json.loads((B / 'main-metric-enum-build-v2/record.json').read_text())
        current_package = sha(D / 'SHA256SUMS')
        assert build['reviewed_package_sha256'] != current_package
        result = dict(schema='strata-metric-query-reader-offline-review-v1', status='OFFLINE_FORMAT_CHECKS_PASS_COLLECTION_HELD',
            GPU_executed=False, tests=tests, source_sha256=sha(source), syntax_checked_files=['main_controller.py', 'metrics.py', 'summary.py'],
            catalog_sha256=sha(A / 'own-metric-full-catalog-evidence-v2.json'),
            source_contract_findings=[
                dict(id='MQ1', finding='v1 controller binds catalog build reviewed_package_sha256 to query package SHA. Actual separately reviewed enum package1c856787 differs despite compiling exactly the pinned full source path. New explicit catalog-source/package binding is needed; do not forge the original build receipt.'),
                dict(id='MQ2', finding='v1 checks types/finite values/counts, but it decodes ReportError=true. Diagnostic-only HOLD is correct; actual report-validity policy must be reviewed before qualification.'),
                dict(id='MQ3', finding='Pinned collector internal metric API statuses are incomplete; this is source-proved, not an observed API failure.')],
            v1_native_execution='UNRUN', collection_qualified=False, originals_unchanged=True,
            scope='Synthetic offline format/error checks only, no physical counters, arithmetic/utilization/performance result.')
        target.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(path=str(target), sha256=sha(target), cases=len(tests), collection_qualified=False)))


if __name__ == '__main__':
    main()
