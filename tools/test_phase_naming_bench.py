"""Admission/arm/decision controls for the owned naming comparison."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / 'sycl'))
import phase_naming_bench as bench


class NamingComparisonTests(unittest.TestCase):
    def setUp(self):
        bench.configure('prefill')
        self.row = dict(native_builds={arm: dict(binary=dict(path='/owned/' + arm, sha256=v[5]),
            native_source_commit=v[3], native_validation_probe={}, boot='boot')
            for arm, v in bench.BUILDS.items()})

    def plan(self, index):
        read, arm = bench.cases()[index]
        plan = dict(index=index, kernel=arm, heads=1, adapt_swaps=0, root='/owned/run', port=1234)
        if read is not None: plan['read'] = read
        return plan

    def test_every_planned_arm_selects_its_own_source_and_image(self):
        for phase in ('prefill', 'decode'):
            bench.configure(phase)
            for i, (_, arm) in enumerate(bench.cases()):
                self.assertEqual(bench.native_context(self.row, self.plan(i)), self.row['native_builds'][arm])

    def test_misbound_phase_source_and_binary_are_refused(self):
        plan = self.plan(0)
        bad = [dict(plan, index=-1), dict(plan, index=True), dict(plan, read=8193),
               dict(plan, kernel='candidate'), dict(plan, heads=0), dict(plan, adapt_swaps=1)]
        for value in bad:
            with self.assertRaises(RuntimeError): bench.native_context(self.row, value)
        for field, value in [('native_source_commit', '0'*40), ('binary', dict(path='/foreign', sha256='0'*64))]:
            row = copy.deepcopy(self.row); row['native_builds']['reference'][field] = value
            with self.assertRaises(RuntimeError): bench.native_context(row, plan)

    def test_both_phase_environments_keep_identical_kv_and_safety(self):
        base = dict(EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1')
        with patch.object(bench.kv.base, 'environment', return_value=base):
            for phase in ('prefill', 'decode'):
                bench.configure(phase)
                a = bench.environment(self.row, self.plan(0)); b = bench.environment(self.row, self.plan(1))
                self.assertEqual(a, b)
                self.assertEqual(a['STRATA_SYCL_DECODE_KV_COALESCED'], '1')
                self.assertEqual(a['STRATA_SYCL_KV_COPY_GROUPS'], '0')
                self.assertEqual(a['EnableDirectSubmission'], '0')
                self.assertEqual(a['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'], '1')

    def test_decode_driver_uses_the_selected_native_context(self):
        bench.configure('decode'); ctx = bench.context()
        self.assertIs(ctx.native_context, bench.native_context)
        for i in range(6):
            self.assertEqual(ctx.native_context(self.row, self.plan(i)),
                self.row['native_builds'][bench.ARMS[i]])
        with patch.object(bench.kv, 'api_window', return_value={'checked': True}) as observed:
            self.assertEqual(ctx.api_window('native', {}, self.row, 'reference'), {'checked': True})
            self.assertEqual(observed.call_args.args[-1], 'candidate')

    def test_legacy_single_image_adapter_stays_explicit(self):
        row = dict(self.row['native_builds']['reference'], unused='excluded')
        self.assertEqual(bench.kv.native_context(row, {}), self.row['native_builds']['reference'])

    def test_commands_keep_thinking_and_separate_workloads(self):
        plan = self.plan(0); argv = bench.command(plan)
        self.assertIn('--think', argv); self.assertIn('--prefill-exact-tokens', argv)
        self.assertEqual(argv[argv.index('--runs')+1], '3')
        bench.configure('decode'); argv = bench.command(self.plan(0))
        self.assertIn('--think', argv); self.assertIn('--long-generation-only', argv)
        self.assertEqual(argv[argv.index('--workload-id')+1], bench.kv.WORKLOAD_ID)

    def test_prefill_failure_is_preserved_without_decode_offset(self):
        processes = []
        for i, (read, arm) in enumerate(bench.cases()):
            native = self.row['native_builds'][arm]
            value = 110.0 if arm == 'candidate' and read == 65537 else 100.0
            processes.append(dict(index=i, read=read, kernel=arm, binary=native['binary'],
                native_source_commit=native['native_source_commit'], passed=True, complete=True,
                ownership_closed=True, native_exit_code=0, fresh_faults=[],
                samples=[dict(prompt_ms=value, ttft_s=value/1000)]*3))
        result = bench.decision(processes)
        self.assertFalse(result['median_preservation_screen_passed'])
        self.assertFalse(result['naming_control_eligible'])
        with self.assertRaises(RuntimeError): bench.decision(processes[:-1])
        processes[0]['binary']['sha256'] = '0'*64
        with self.assertRaises(RuntimeError): bench.decision(processes)

    def test_naming_control_does_not_hide_a_clear_slowdown(self):
        self.assertTrue(bench.paired_control([1, 1, 1])['no_clear_slowdown'])
        self.assertTrue(bench.paired_control([.99, 1.01, 1.002])['no_clear_slowdown'])
        self.assertFalse(bench.paired_control([1.01, 1.01, 1.01])['no_clear_slowdown'])
        for bad in ([1, 1], [1, 1, float('nan')], [1, 1, 0], [1, 1, True]):
            with self.assertRaises(RuntimeError): bench.paired_control(bad)


if __name__ == '__main__': unittest.main()
