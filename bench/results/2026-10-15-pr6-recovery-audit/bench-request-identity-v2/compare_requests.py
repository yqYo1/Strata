#!/usr/bin/env python3
"""Pair fixed inputs and preserve natural agent feedback as a separate cohort.

All journals, keys, ordering and parameters remain mandatory. Only the body
hash of agent turn >= 1 may differ because it contains generated history.
Neither cohort qualifies model quality or inference performance.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
READER = HERE.parent / 'bench-request-identity-v1/compare_requests.py'
spec = importlib.util.spec_from_file_location('bounded_request_identity_reader_v1', READER)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
FIELDS = legacy.FIELDS
MAX_BYTES, MAX_ROWS = legacy.MAX_BYTES, legacy.MAX_ROWS
AGENT_TASK_FIELDS = ('thinking', 'reasoning_budget_tokens', 'temperature', 'model', 'quant',
                     'context', 'agent_files', 'agent_max_tokens')


def feedback(row):
    return row is not None and row['phase'] == 'agent' and type(row['turn']) is int and row['turn'] >= 1


def compare_agent_tasks(left, right):
    """Actual result metadata only; caller must also prove unchanged builders/tag."""
    identities = [suite.get('comparison_identity') for suite in (left, right)]
    report = dict(passed=False, quality_qualified=False, exact_input_paired_speed=False,
                  scope='Static task/file and generation metadata. Unchanged request builders, '
                        'tools and fixed tag require separate caller evidence.', errors=[])
    for side, identity in zip(('left', 'right'), identities):
        if not isinstance(identity, dict) or any(k not in identity for k in AGENT_TASK_FIELDS):
            report['errors'].append(dict(side=side, error='missing actual agent task metadata'))
            continue
        files = identity['agent_files']
        if not isinstance(files, list) or not files or not all(
                isinstance(item, (list, tuple)) and len(item) == 2 and
                isinstance(item[0], str) and bool(item[0]) and
                isinstance(item[1], str) and legacy.HEX.fullmatch(item[1]) for item in files):
            report['errors'].append(dict(side=side, error='invalid actual agent file identities'))
    if report['errors']:
        return report
    a, b = [{k: identity[k] for k in AGENT_TASK_FIELDS} for identity in identities]
    report.update(left=a, right=b)
    report['differences'] = {k: dict(left=a[k], right=b[k]) for k in AGENT_TASK_FIELDS if a[k] != b[k]}
    if report['differences']:
        report['errors'].append(dict(error='agent static task/generation metadata mismatch'))
    report['passed'] = not report['errors']
    return report


def compare(left, right):
    report = dict(schema='bench-request-input-identity-v2', status='FAIL', inputs_match=False,
                  fixed_inputs_match=False, agent_protocol_match=False, input_pair_qualified=False,
                  agent_exact_inputs_match=False, agent_feedback_only=True,
                  quality_qualified=False, performance_qualified=False,
                  claim='Fixed inputs (including agent turn zero) require exact actual bodies. '
                        'Later agent bodies include natural generated history; their structure and '
                        'parameters must match, but they are excluded from exact-input paired speed statistics. '
                        'Static agent task identity is a separate caller gate.',
                  limits=dict(journal_bytes_each=legacy.MAX_BYTES, rows_each=legacy.MAX_ROWS,
                              state_bytes_each=legacy.MAX_STATE_BYTES, report_bytes=legacy.MAX_REPORT_BYTES),
                  verifier_source_sha256=legacy.digest((legacy.PROJECT / 'tools/strata_bench.py').read_bytes()),
                  journal_reader_source_sha256=legacy.digest(READER.read_bytes()),
                  left={}, right={}, samples=[], errors=[])
    sides = []
    for name, path in [('left', left), ('right', right)]:
        try:
            sides.append(legacy.read_suite(path, report[name]))
        except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
            report['errors'].append(dict(side=name, error=repr(error)))
            sides.append(None)
    if any(side is None for side in sides):
        return report
    a, b = sides
    for name in ('left', 'right'):
        if report[name].get('duplicate_request_keys'):
            report['errors'].append(dict(side=name, error='duplicate request key',
                                         keys=report[name]['duplicate_request_keys']))
    for i in range(max(len(a), len(b))):
        one = a[i] if i < len(a) else None
        two = b[i] if i < len(b) else None
        differences = {k: dict(left=one.get(k) if one else None, right=two.get(k) if two else None)
                       for k in FIELDS if one is None or two is None or one[k] != two[k]}
        natural = feedback(one) and feedback(two)
        blocking = {k: v for k, v in differences.items() if not natural or k != 'request_sha256'}
        report['samples'].append(dict(position=i, left=one, right=two,
            cohort='agent-feedback' if natural else 'fixed-input', differences=differences,
            blocking_differences=blocking, match=not differences, contract_match=not blocking))
    if len(a) != len(b):
        report['errors'].append(dict(error='missing/extra request starts', left_count=len(a), right_count=len(b)))
    fixed = [r for r in report['samples'] if r['cohort'] == 'fixed-input']
    agents = [r for r in report['samples'] if r['cohort'] == 'agent-feedback']
    if not fixed:
        report['errors'].append(dict(error='no fixed-input requests'))
    if any(not row['contract_match'] for row in report['samples']):
        report['errors'].append(dict(error='ordered request structure/parameter/fixed-body mismatch'))
    report.update(fixed_input_count=len(fixed), agent_feedback_count=len(agents),
                  agent_feedback_divergent_bodies=sum(not row['match'] for row in agents))
    if not report['errors']:
        all_match = all(row['match'] for row in report['samples'])
        report.update(inputs_match=all_match, fixed_inputs_match=True, agent_protocol_match=True,
                      input_pair_qualified=True, agent_exact_inputs_match=all(row['match'] for row in agents),
                      status='REQUEST_INPUTS_MATCH' if all_match else 'FIXED_INPUTS_MATCH_AGENT_FEEDBACK')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('output')
    args = parser.parse_args(argv)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        report = compare(args.left, args.right)
        encoded = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
        legacy.require(len(encoded.encode()) <= legacy.MAX_REPORT_BYTES, 'report byte budget exceeded')
        stream.write(encoded)
    print(report['status'])
    return 0 if report['input_pair_qualified'] else 1


if __name__ == '__main__':
    sys.exit(main())
