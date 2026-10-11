# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finite CPU qualification driver. Requires unchanged qualified ledger path."""
import argparse
import importlib.util
import json
from pathlib import Path
import route_census as census


def load_ledger(path):
    spec = importlib.util.spec_from_file_location('qualified_transfer_ledger', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tokens=2048, observed=True, mmq=1, group=1, direct=0,
            accepted=1, declined=0, individual=0, dequant=0, fallback=0,
            mode='full_nonresident_stream'):
    h = (f'strata prefill expert H2D: device=0 stage=[0,1) chunk_offset=0 pos=0 '
         f'tokens={tokens} layer=0 mode={mode} issuer=inline format=native_gguf '
         'gu_type=2 down_type=2 blob_bytes=4 submitted_calls=0 submitted_bytes=0')
    values = dict(zip(census.FLAGS, (int(observed), mmq, fallback, direct, group)))
    values.update(dict(zip(census.COUNTERS, (tokens * 10 if observed else 0,
                  tokens * 10 if observed else 0, 10 if observed else 0,
                  7 if observed else 0, 502 if observed else 0,
                  accepted, declined, individual, dequant, dequant))))
    r = (f'strata prefill route census v1: device=0 stage=[0,1) chunk_offset=0 pos=0 '
         f'tokens={tokens} layer=0 mode={mode} issuer=inline '
         + ' '.join(f'{k}={values[k]}' for k in census.FLAGS + census.COUNTERS)
         + ' completion=checked')
    t = (f'strata prefill expert H2D total: scope=this_prefill_stage_run device=0 '
         f'stage=[0,1) pos=0 tokens={tokens} chunks=1 routed_calls=0 routed_bytes=0 '
         'full_stream_calls=0 full_stream_bytes=0 submitted_calls=0 submitted_bytes=0 '
         'completion=checked; request extents, not PCIe-exclusive bytes or DMA service')
    opts = dict(enabled=True, n_experts=512, route_k=10,
                moe_layers={0} if observed else set(),
                nonresident_population={0: 509} if observed else {},
                h2d_expectations=dict(expected_device=0, expected_stage=(0, 1),
                expected_pos0=0, expected_n=tokens, expected_chunk_limit=2048,
                expected_mode=mode, expected_issuer='inline', expected_format='native_gguf'))
    return [h, r, t], opts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer-ledger', required=True)
    parser.add_argument('--actual-stderr')
    parser.add_argument('--actual-expectations', help='JSON with model and h2d_expectations')
    args = parser.parse_args()
    ledger = load_ledger(args.transfer_ledger)
    positives = [fixture(), fixture(accepted=0, declined=1, individual=7),
                 fixture(mmq=0, group=0, accepted=0, dequant=7, fallback=1),
                 fixture(tokens=2047, group=0, accepted=0, individual=7, mode='routed'),
                 fixture(observed=False, group=0, accepted=0),
                 fixture(fallback=1), fixture(direct=1, group=0, accepted=0),
                 fixture(group=0, accepted=0)]
    left, mixed_opts = fixture(fallback=1)
    right, _ = fixture(mmq=0, group=0, accepted=0, dequant=7, fallback=1)
    mixed_lines = [left[0].replace('stage=[0,1)', 'stage=[0,2)'),
                   right[0].replace('stage=[0,1)', 'stage=[0,2)').replace('layer=0', 'layer=1'),
                   left[1].replace('stage=[0,1)', 'stage=[0,2)'),
                   right[1].replace('stage=[0,1)', 'stage=[0,2)').replace('layer=0', 'layer=1'),
                   left[2].replace('stage=[0,1)', 'stage=[0,2)')]
    mixed_opts['h2d_expectations']['expected_stage'] = (0, 2)
    mixed_opts['moe_layers'] = {0, 1}
    mixed_opts['nonresident_population'] = {0: 509, 1: 509}
    positives.append((mixed_lines, mixed_opts))
    passed = 0
    for lines, opts in positives:
        census.validate(lines, h2d_validate=ledger.validate, **opts)
        passed += 1
    lines, opts = fixture()
    negatives = []
    def bad_rows(new_lines):
        negatives.append((new_lines, opts))
    bad_rows([lines[0], lines[1], lines[1], lines[2]])
    bad_rows([lines[0], lines[2]])
    bad_rows([lines[1], lines[0], lines[2]])
    bad_rows([lines[0], lines[2], lines[1]])
    bad_rows([lines[0], lines[1]])
    bad_rows(lines + [lines[2]])
    for old, new in [('v1:', 'v2:'), ('device=0', 'device=1'),
                     ('stage=[0,1)', 'stage=[1,2)'), ('chunk_offset=0', 'chunk_offset=1'),
                     ('pos=0', 'pos=-0'), ('tokens=2048', 'tokens=02048'),
                     ('layer=0', 'layer=1'), ('issuer=inline', 'issuer=thread'),
                     ('mode=full_nonresident_stream', 'mode=unknown'),
                     ('routing_observed=1', 'routing_observed=0'),
                     ('mmq_enabled=1', 'mmq_enabled=2'),
                     ('expected_rows=20480', 'expected_rows=20479'),
                     ('routed_rows=20480', 'routed_rows=18446744073709551616'),
                     ('active_experts=10', 'active_experts=513'),
                     ('zero_nonresident=502', 'zero_nonresident=503'),
                     ('mmq_direct=0', 'mmq_direct=1'),
                     ('mmq_enabled=1', 'mmq_enabled=0'),
                     ('native_dequant_down_calls=0', 'native_dequant_down_calls=1'),
                     ('gather_group_declined_calls=0', 'gather_group_declined_calls=1')]:
        bad_rows([lines[0], lines[1].replace(old, new), lines[2]])
    bad_rows([lines[0], lines[1] + ' extra=1', lines[2]])
    bad_rows([lines[0], lines[1].replace('completion=checked', 'completion=pending'), lines[2]])
    bad_rows([lines[0], lines[1], lines[2].replace('submitted_calls=0', 'submitted_calls=1')])
    for override in [dict(n_experts=0), dict(route_k=513), dict(moe_layers=set()),
                     dict(nonresident_population={}), dict(nonresident_population={0: 508}),
                     dict(enabled=False), dict(enabled=0)]:
        negatives.append((lines, dict(opts, **override)))
    changed_plan = list(mixed_lines)
    changed_plan[3] = changed_plan[3].replace('plan_fallback=1', 'plan_fallback=0')
    negatives.append((changed_plan, mixed_opts))
    swapped = list(mixed_lines)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    negatives.append((swapped, mixed_opts))
    overflow_lines, overflow_opts = fixture(tokens=(1 << 63) - 2)
    overflow_opts['h2d_expectations']['expected_chunk_limit'] = (1 << 63) - 2
    negatives.append((overflow_lines, overflow_opts))
    aggregate_lines = [line.replace('tokens=2048', 'tokens=1000000000000000000')
                       .replace('routed_rows=20480', 'routed_rows=10000000000000000000')
                       .replace('expected_rows=20480', 'expected_rows=10000000000000000000')
                       for line in mixed_lines]
    aggregate_opts = dict(mixed_opts, h2d_expectations=dict(mixed_opts['h2d_expectations'],
                         expected_n=1000000000000000000,
                         expected_chunk_limit=1000000000000000000))
    negatives.append((aggregate_lines, aggregate_opts))
    rejected = 0
    for capture, config in negatives:
        try:
            census.validate(capture, h2d_validate=ledger.validate, **config)
        except (census.ValidationError, ledger.ValidationError):
            rejected += 1
        else:
            raise AssertionError(f'negative case {rejected + 1} accepted')
    disabled = dict(opts, enabled=False)
    result = census.validate('', h2d_validate=ledger.validate, **disabled)
    assert result['absent'] and 'aggregates' not in result
    actual = None
    if bool(args.actual_stderr) != bool(args.actual_expectations):
        parser.error('actual stderr and expectations must be supplied together')
    if args.actual_stderr:
        config = json.loads(Path(args.actual_expectations).read_text())
        config['moe_layers'] = set(config['moe_layers'])
        if config.get('nonresident_population') is not None:
            config['nonresident_population'] = {int(k): v for k, v in config['nonresident_population'].items()}
        actual = census.validate(Path(args.actual_stderr).read_text(),
                                 h2d_validate=ledger.validate, **config)
    print(json.dumps({'positive_cases': passed + 1, 'negative_cases': rejected,
                      'actual': actual}, sort_keys=True))


if __name__ == '__main__':
    main()
