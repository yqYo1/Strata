"""Offline report validity, separate from v1 format and v2 API proof. Repository LICENSE.
No runtime/device I/O, environment changes, thresholds inferred from percentages,
or counter-to-hardware attribution. Input values remain untouched in query.rows.
"""
import math
import time

COMPROMISED = (
    'ReportError', 'ReportLost', 'ReportInconsistent', 'ReportContextMismatch',
    'ReportQueryModeMismatch', 'OverrunOccured', 'ReportWithoutWorkload', 'ReportCtxSwitchLost',
)
INTERPRET_FLAGS = (
    'QuerySplitOccurred', 'CoreFrequencyChanged', 'MidQueryTimer', 'MidQueryProgramming',
    'MidQueryMarker', 'MidQueryCtxSwitch', 'MidQueryC6', 'MidQueryFreqChange', 'MidQueryMmioTrigger',
)
# Available reports are OA query-report metadata, NOT the one calculated sample.
# ReportId/QueryBeginTime may be zero or wrap; do not impose positivity on IDs.
POSITIVE = ('GpuTime', 'GpuCoreClocks', 'ReportsCount')
CONTEXT = ('ResultUncertainty', 'QueryBeginTime', 'CoreFrequencyMHz',
           'AvgGpuCoreFrequencyMHz', 'ReportId', 'ReportsCount', 'GpuTime', 'GpuCoreClocks')
INTERPRETATION = {
    'QuerySplitOccurred': 'Query was split on GPU; scope/comparability requires Main interpretation, no automatic rejection.',
    'CoreFrequencyChanged': 'Core frequency changed; compare with last/average frequency, no timing-neutrality claim.',
    'MidQuery': 'Preserve every detected midquery event flag; occurrence alone is not the described loss/error condition.',
    'ResultUncertainty': 'Driver gives an uncertainty indicator in percent, without a universal acceptance threshold; retain raw value.',
    'ReportsCount': 'Positive count of available query reports, not the calculated sample count; no requirement that it equals one.',
    'limits': 'Validity is for tiny collection only. No ALU0/1/2-to-XMX mapping, utilization, bottleneck, model or clean-speed conclusion.',
}


def labels(catalog):
    out = []
    for m in catalog:
        units = m['units']
        units = '' if 'null' in units else '%' if 'percent' in units else units
        out.append(m['name'] + ('[' + units + ']' if units else ''))
    return out


def typed(value, kind):
    if kind in ('ZET_VALUE_TYPE_UINT32', 'ZET_VALUE_TYPE_UINT64', 'ZET_VALUE_TYPE_BOOL8'):
        bound = {'ZET_VALUE_TYPE_UINT32': 2**32, 'ZET_VALUE_TYPE_UINT64': 2**64,
                 'ZET_VALUE_TYPE_BOOL8': 2}[kind]
        return type(value) is int and 0 <= value < bound
    return (kind in ('ZET_VALUE_TYPE_FLOAT32', 'ZET_VALUE_TYPE_FLOAT64') and
            type(value) is float and math.isfinite(value))


def assess(query, catalog, exact_api, expected_catalog, deadline):
    """Return a durable whole-report and per-row decision, including invalid rows.

    Caller first decodes immutable CSV with unchanged v1, then reconciles exact
    v2 API evidence, then invokes this gate. Nothing mutates those inputs or
    promotes query.collection_qualified, whose format-only value stays false.
    Expected catalog comes from the package's exact pinned driver-schema.json.
    Unknown/missing/reordered columns or changed descriptions/types always hold.
    """
    result = dict(schema='strata-metric-report-validity-v3', valid=False, schema_valid=False,
                  exact_api_proved=exact_api.get('proved') is True, rows=[], reasons=[],
                  interpretation=INTERPRETATION.copy(), raw_values='Unmodified query.rows[].values; types and descriptions in catalog_selection.metrics',
                  scope='Driver report validity only; all source/ownership/numerical/map/fault/deadline gates remain required')
    def finish_reason(reason):
        result['reasons'].append(reason)
    schema_ok = (type(catalog) is list and type(expected_catalog) is list and
                 len(catalog) == len(expected_catalog) == 83 and catalog == expected_catalog and
                 [m.get('metric') for m in catalog] == list(range(83)) and
                 len({m.get('name') for m in catalog}) == 83)
    if schema_ok:
        by_name = {m['name']: m for m in catalog}
        flag_names = set(COMPROMISED + INTERPRET_FLAGS)
        schema_ok = (set(m['name'] for m in catalog if m['result_type_name'] == 'ZET_VALUE_TYPE_BOOL8' or
                         m['metric_type_name'] == 'ZET_METRIC_TYPE_FLAG') == flag_names and
                     all(n in by_name and by_name[n]['result_type_name'] == 'ZET_VALUE_TYPE_BOOL8' and
                         by_name[n]['result_type'] == 4 and by_name[n]['metric_type_name'] == 'ZET_METRIC_TYPE_FLAG' and
                         by_name[n]['metric_type'] == 5 for n in flag_names) and
                     all(n in by_name and by_name[n]['result_type_name'] == 'ZET_VALUE_TYPE_UINT64'
                         for n in POSITIVE + CONTEXT))
    result['schema_valid'] = bool(schema_ok)
    if not schema_ok:
        finish_reason('Unknown/missing/changed driver schema or BOOL8 flag definitions')
    if not result['exact_api_proved']:
        finish_reason('Exact API reconciliation is required before report validity')
    if query.get('decoded') is not True:
        finish_reason('Original strict CSV decode is required')
    if schema_ok and query.get('metric_columns') != labels(catalog):
        finish_reason('Unknown/missing/reordered decoded CSV columns')
    rows = query.get('rows')
    if type(rows) is not list or not 0 < len(rows) <= 4096:
        finish_reason('Finite nonempty actual row set required')
        return result
    ids = [r.get('global_instance_id') for r in rows]
    if (any(type(i) is not int or not 0 < i < 2**64 for i in ids) or len(set(ids)) != len(ids)):
        finish_reason('Unique finite sample identities required')
    if (exact_api.get('samples') != len(rows) or exact_api.get('instances') != sorted(i for i in ids if type(i) is int)):
        finish_reason('Exact API/actual row identity coverage required')
    index = {m['name']: m['metric'] for m in catalog} if schema_ok else {}
    global_reasons = list(result['reasons'])
    for row in rows:
        decision = dict(global_instance_id=row.get('global_instance_id'), subdevice_id=row.get('subdevice_id'),
                        valid=False, reasons=list(global_reasons), compromised_flags={}, interpretation_metadata={})
        values = row.get('values')
        if time.monotonic() >= deadline:
            decision['reasons'].append('Report-validity deadline exceeded')
        elif not schema_ok:
            pass
        elif type(values) is not list or len(values) != 83:
            decision['reasons'].append('Exact 83 typed raw values required')
        elif not all(typed(v, m['result_type_name']) for v, m in zip(values, catalog)):
            decision['reasons'].append('Invalid raw typed value')
        else:
            if row.get('subdevice_id') != 0:
                decision['reasons'].append('Exact single target subdevice required')
            for name in COMPROMISED:
                value = values[index[name]]
                decision['compromised_flags'][name] = dict(value=value, result_type='ZET_VALUE_TYPE_BOOL8',
                    description=catalog[index[name]]['description'])
                if value != 0:
                    decision['reasons'].append(name + ': ' + catalog[index[name]]['description'])
            for name in POSITIVE:
                if values[index[name]] <= 0:
                    decision['reasons'].append(name + ' must be positive: ' + catalog[index[name]]['description'])
            for name in INTERPRET_FLAGS + CONTEXT:
                m = catalog[index[name]]
                decision['interpretation_metadata'][name] = dict(value=values[index[name]],
                    result_type=m['result_type_name'], units=m['units'], description=m['description'])
        decision['valid'] = not decision['reasons']
        result['rows'].append(decision)
    if time.monotonic() >= deadline:
        finish_reason('Report-validity completion deadline exceeded')
    result['valid'] = not result['reasons'] and all(r['valid'] for r in result['rows'])
    return result


def tiny_qualified(diagnostic_complete, exact_api, report_validity):
    """Only called after original normal-close and all original safety gates."""
    return bool(diagnostic_complete and exact_api.get('proved') is True and
                report_validity.get('schema_valid') is True and report_validity.get('valid') is True)
