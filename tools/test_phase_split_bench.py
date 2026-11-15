"""Source/phase binding and independent decision gates for B1 comparisons."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / 'sycl'))
import phase_split_bench


class SplitUnitComparisonTests(unittest.TestCase):
    def setUp(self):
        self.bench = phase_split_bench.controller('prefill')
        self.row = dict(native_builds={arm: dict(binary=dict(path='/owned/' + arm, sha256=v[5]),
            native_source_commit=v[3], native_validation_probe={}, boot='boot')
            for arm, v in self.bench.BUILDS.items()})

    def plan(self, index):
        read, arm = self.bench.cases()[index]
        p = dict(index=index, kernel=arm, heads=1, adapt_swaps=0, root='/owned/run', port=1234)
        if read is not None: p['read'] = read
        return p

    def prefill(self, slowdown=False):
        rows = []
        for i, (read, arm) in enumerate(self.bench.cases()):
            n = self.row['native_builds'][arm]
            value = 110 if slowdown and arm == 'candidate' and read == 65537 else 100
            rows.append(dict(index=i, read=read, kernel=arm, binary=n['binary'],
                native_source_commit=n['native_source_commit'], passed=True, complete=True,
                ownership_closed=True, native_exit_code=0, fresh_faults=[],
                samples=[dict(prompt_ms=value, ttft_s=value / 1000)] * 3))
        return rows

    def decode(self, slowdown=False):
        self.bench = phase_split_bench.controller('decode'); rows = []
        for i, (_, arm) in enumerate(self.bench.cases()):
            n = self.row['native_builds'][arm]
            rows.append(dict(index=i, kernel=arm, binary=n['binary'],
                native_source_commit=n['native_source_commit'], passed=True, complete=True,
                ownership_closed=True, native_exit_code=0, fresh_faults=[], samples=[
                    dict(exact_native_decode_tps=9 if slowdown and arm == 'candidate' and j == 1 else 10,
                        prompt_ms=100, ttft_s=.1, acceptance_rate=.8 if j == 0 else .2)
                    for j in range(2)]))
        return rows

    def test_each_phase_selects_exact_original_A_or_B1(self):
        for phase in ('prefill', 'decode'):
            self.bench = phase_split_bench.controller(phase)
            for i, (_, arm) in enumerate(self.bench.cases()):
                self.assertEqual(self.bench.native_context(self.row, self.plan(i)), self.row['native_builds'][arm])

    def test_wrong_phase_source_image_and_case_are_refused(self):
        for p in [dict(self.plan(0), index=True), dict(self.plan(0), read=82),
                  dict(self.plan(0), kernel='candidate'), dict(self.plan(0), adapt_swaps=1)]:
            with self.assertRaises(RuntimeError): self.bench.native_context(self.row, p)
        for field, value in [('native_source_commit', '0' * 40), ('binary', dict(sha256='0' * 64))]:
            row = copy.deepcopy(self.row); row['native_builds']['reference'][field] = value
            with self.assertRaises(RuntimeError): self.bench.native_context(row, self.plan(0))

    def test_both_arms_keep_same_immutable_KV_and_safety(self):
        with patch.object(self.bench.kv.base, 'environment', return_value=dict(
                EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1')):
            a = self.bench.environment(self.row, self.plan(0))
            self.assertEqual(a, self.bench.environment(self.row, self.plan(1)))
            self.assertEqual(a['STRATA_SYCL_DECODE_KV_COALESCED'], '1')
            self.assertEqual(a['STRATA_SYCL_KV_COPY_GROUPS'], '0')

    def test_decode_driver_uses_selected_image(self):
        self.bench = phase_split_bench.controller('decode'); ctx = self.bench.context()
        for i, (_, arm) in enumerate(self.bench.cases()):
            self.assertEqual(ctx.native_context(self.row, self.plan(i)), self.row['native_builds'][arm])

    def test_official_commands_keep_thinking_and_workload_scope(self):
        argv = self.bench.command(self.plan(0))
        self.assertIn('--think', argv); self.assertIn('--prefill-exact-tokens', argv)
        self.assertEqual(argv[argv.index('--runs') + 1], '3')
        self.bench = phase_split_bench.controller('decode'); argv = self.bench.command(self.plan(0))
        self.assertIn('--think', argv); self.assertIn('--long-generation-only', argv)

    def test_prefill_loss_is_not_offset_or_given_naming_exception(self):
        rows = self.prefill(slowdown=True); result = self.bench.decision(rows)
        self.assertFalse(result['phase_preservation_eligible'])
        self.assertFalse(result['naming_identity_exception_applied'])
        self.assertNotIn('naming_control_eligible', result)
        with self.assertRaises(RuntimeError): self.bench.decision(rows[:-1])

    def test_decode_loss_is_independently_refused(self):
        rows = self.decode(slowdown=True)
        result = self.bench.decision(rows)
        self.assertFalse(result['phase_preservation_eligible'])
        self.assertFalse(result['naming_identity_exception_applied'])
        rows = self.decode(); rows[1]['samples'][1]['exact_native_decode_tps'] = float('nan')
        with self.assertRaises(RuntimeError): self.bench.decision(rows)

    def test_success_still_needs_observed_MTP_classes_and_is_not_adoption(self):
        rows = self.decode(); result = self.bench.decision(rows)
        self.assertTrue(result['phase_preservation_eligible']); self.assertFalse(result['inference_adopted'])
        rows[1]['samples'][1]['acceptance_rate'] = .8
        self.assertFalse(self.bench.decision(rows)['phase_preservation_eligible'])


if __name__ == '__main__': unittest.main()
