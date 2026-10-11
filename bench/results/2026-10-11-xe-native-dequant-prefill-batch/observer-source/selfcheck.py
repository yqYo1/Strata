# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""SOURCE ONLY until Main executes with --ledger /absolute/transfer_ledger.py."""
import argparse
import importlib.util
from pathlib import Path
import unittest
import batch_census as census


def load_ledger(path):
    spec = importlib.util.spec_from_file_location('qualified_transfer_ledger', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate


def fixture(empty=False, full=False):
    groups, transfers, routes = [], [], []
    active, selected, logical = (1, 0, 0) if empty else (3, 2, 2)
    flush = int(not empty)
    mode, issuer = ('full_nonresident_stream', 'thread') if full else ('routed', 'inline')
    calls = 4 if full else selected
    for offset in (0, 4):
        groups.append('[prefill-native-dequant-group-v2] device=0 lb=0 le=2 '
            f'chunk_offset={offset} pos={8+offset} tokens=4 layer=0 fused_prefix=1 '
            f'cpu_experts=0 gpu_order={active} gu_admitted={flush} gu_declined=0 '
            f'down_admitted=0 down_declined={flush} gu_covered={logical} down_covered=0 '
            f'gu_fallback_experts=0 down_fallback_experts={logical} flushes={flush} '
            f'held_reuse_flushes=0 released_slots={selected}')
        for layer in (0, 1):
            scope = (f'device=0 stage=[0,2) chunk_offset={offset} pos={8+offset} '
                     f'tokens=4 layer={layer} mode={mode} issuer={issuer}')
            transfers.append(f'strata prefill expert H2D: {scope} format=native_gguf '
                f'gu_type=1 down_type=1 blob_bytes=10 submitted_calls={calls} submitted_bytes={calls*10}')
            routes.append(f'strata prefill route census v1: {scope} routing_observed=1 '
                f'mmq_enabled={layer} plan_fallback=1 mmq_direct=0 group_gather=0 '
                f'routed_rows=4 expected_rows=4 active_experts={active} active_nonresident={selected} '
                f'zero_nonresident={4-selected} gather_group_accepted_calls=0 '
                'gather_group_declined_calls=0 gather_individual_calls=0 native_dequant_gu_calls=0 '
                f'native_dequant_down_calls={0 if layer else logical} completion=checked')
    rc, fc = (0, calls*4) if full else (calls*4, 0)
    total = ('strata prefill expert H2D total: scope=this_prefill_stage_run device=0 '
        'stage=[0,2) pos=8 tokens=8 chunks=2 '
        f'routed_calls={rc} routed_bytes={rc*10} full_stream_calls={fc} full_stream_bytes={fc*10} '
        f'submitted_calls={rc+fc} submitted_bytes={(rc+fc)*10} '
        'completion=checked; request extents, not PCIe-exclusive bytes or DMA service')
    return groups + transfers + routes + [total]


class Selfcheck(unittest.TestCase):
    ledger = None

    def check(self, text, empty=False, full=False, pins=None):
        return census.validate(text, enabled=True, h2d_validate=type(self).ledger,
            h2d_expectations=dict(expected_device=0, expected_stage=(0, 2), expected_pos0=8,
                expected_n=8, expected_chunk_limit=4,
                expected_mode='full_nonresident_stream' if full else 'routed',
                expected_issuer='thread' if full else 'inline', expected_format='native_gguf',
                expected_layers={layer: dict(blob_bytes=10, gu_type=1, down_type=1) for layer in (0, 1)}),
            n_experts=5, route_k=1, moe_layers={0, 1}, nonresident_population={0: 4, 1: 4},
            row_expectations=({(offset, 0): dict(fused_prefix=1, cpu_experts=0,
                             selected_streamed_experts=0 if empty else 2) for offset in (0, 4)}
                              if pins is None else pins))

    def test_two_chunks_mixed_mmq_independent_refusal(self):
        result = self.check(fixture())
        self.assertEqual([row['offset'] for row in result['group_rows']], [0, 4])
        self.assertEqual(len(result['rows']), 4)

    def test_zero_post_prefix(self):
        self.assertEqual(self.check(fixture(True), empty=True)['group_rows'][0]['flushes'], 0)

    def test_full_stream(self):
        self.assertEqual(len(self.check(fixture(full=True), full=True)['group_rows']), 2)

    def test_all_context_corruptions(self):
        good = fixture()
        for field in census.GROUP_CONTEXT:
            original = dict(token.split('=') for token in good[0].split()[1:])[field]
            # Unique exact field replacement avoids replacing an offset substring.
            replacement = str(int(original) + 1)
            changed = good[0].replace(' ' + field + '=' + original + ' ',
                                     ' ' + field + '=' + replacement + ' ')
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check([changed] + good[1:])
        for field in census.GROUP_CONTEXT:
            original = dict(token.split('=') for token in good[0].split()[1:])[field]
            changed = good[0].replace(' ' + field + '=' + original + ' ',
                                     ' ' + field + '=18446744073709551616 ')
            with self.subTest(overflow=field), self.assertRaises(ValueError):
                self.check([changed] + good[1:])

    def test_rejected_structures(self):
        good = fixture()
        cases = [good[1:], [good[1], good[0]] + good[2:],
                 [good[0], good[0]] + good[1:], good + [good[0]], good[:-1],
                 good + good, [good[0][:-1]] + good[1:],
                 [good[0].replace('group-v2', 'group-v1')] + good[1:],
                 [good[0] + ' gu_declined=0'] + good[1:],
                 [good[2], good[0]] + good[1:2] + good[3:],
                 good[:2] + good[3:],
                 good[:6] + [good[6].replace('down_calls=2', 'down_calls=1')] + good[7:]]
        replacements = [('gu_admitted=1', 'gu_admitted=2'), ('gu_covered=2', 'gu_covered=17'),
                        ('released_slots=2', 'released_slots=3'), ('flushes=1', 'flushes=0'),
                        ('held_reuse_flushes=0', 'held_reuse_flushes=2'),
                        ('gu_declined=0', 'gu_declined=-1'), ('gu_declined=0', 'gu_declined=00'),
                        ('gu_declined=0', 'gu_declined=18446744073709551616')]
        cases.extend([[good[0].replace(before, after)] + good[1:] for before, after in replacements])
        for index, text in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.check(text)

    def test_rejected_independent_pins(self):
        good = {(offset, 0): dict(fused_prefix=1, cpu_experts=0, selected_streamed_experts=2)
                for offset in (0, 4)}
        cases = [{}, {(4, 0): good[(4, 0)]}, dict(good, **{'bad': good[(0, 0)]})]
        for field, value in [('fused_prefix', 0), ('cpu_experts', 1),
                             ('selected_streamed_experts', 1), ('fused_prefix', True)]:
            changed = {key: dict(pin) for key, pin in good.items()}
            changed[(0, 0)][field] = value
            cases.append(changed)
        for index, pins in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.check(fixture(), pins=pins)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    Selfcheck.ledger = staticmethod(load_ledger(args.ledger))
    unittest.main(argv=['selfcheck.py'])
