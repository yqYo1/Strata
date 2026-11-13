"""Fail-closed controls for current fixture provenance and launch selection."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import current_profile_control as control


class CurrentControlTest(unittest.TestCase):
    def proof(self):
        inputs = {'tools/sycl/owned.py': 'a', **{p: 'b' for p in control.EXTRA}}
        proof = dict(observer_source_files=len(inputs), observer_source_sha256=control.signature(inputs),
            declared_observers=dict(inputs), additional_observers=sorted(inputs), native_source_commit='original')
        return proof, inputs

    def test_remove_only_explicit_uncompiled_observers(self):
        proof, inputs = self.proof()
        before = copy.deepcopy(proof)
        projected = control.project_proof(proof, inputs)
        self.assertEqual(projected['declared_observers'], {'tools/sycl/owned.py': 'a'})
        self.assertEqual(projected['observer_source_files'], 1)
        self.assertEqual(projected['native_source_commit'], 'original')
        self.assertEqual(proof, before)

    def test_refuse_inconsistent_census_and_missing_observer(self):
        proof, inputs = self.proof()
        for change in (lambda p: p.update(observer_source_files=9),
                       lambda p: p.update(observer_source_sha256='wrong'),
                       lambda p: p['declared_observers'].pop(control.THIS),
                       lambda p: p['additional_observers'].remove(control.TEST)):
            bad = copy.deepcopy(proof); change(bad)
            with self.assertRaises(RuntimeError): control.project_proof(bad, inputs)

    def test_active_environment_and_command_never_start_paused(self):
        env = {'UNITRACE_HostTiming': '1'}
        profile = dict(control.adapter.profile.profiling(Path('/private'), {
            'launcher': {'path': '/launcher'}, 'collector': {'path': '/collector'}}), mode='aggregate')
        row = {'native_validation_probe': {'library': {'path': '/probe'}}}
        with patch.object(control.adapter.engine_once, 'native_environment', return_value=dict(env, LD_PRELOAD='/probe')):
            direct = control.expected_environment(row, {'environment': env, 'profile': None})
            active = control.expected_environment(row, {'environment': env, 'profile': profile})
        self.assertEqual(direct['LD_PRELOAD'], '/probe')
        self.assertEqual(active['LD_PRELOAD'], '/collector:/probe')
        self.assertNotIn('UNITRACE_StartPaused', active)
        self.assertEqual(active['UNITRACE_Verbose'], '1')
        self.assertEqual(control.adapter.engine_once.profile_command(profile, ['/fixture']),
            ['/launcher', '--session', profile['session'], '--device-timing', '--verbose', '/fixture'])

    def test_order_rejects_skips_retries_and_failed_advance(self):
        row = dict(stage='ready', host={'passed': True}, cases=[])
        self.assertEqual(control.next_case(row, 'direct'), 0)
        with self.assertRaises(RuntimeError): control.next_case(row, 'active')
        row['cases'].append({'passed': False})
        with self.assertRaises(RuntimeError): control.next_case(row, 'active')
        row['cases'][0]['passed'] = True
        self.assertEqual(control.next_case(row, 'active'), 1)
        with self.assertRaises(RuntimeError): control.next_case(row, 'direct')
        row['cases'].append({'passed': True})
        with self.assertRaises(RuntimeError): control.next_case(row, 'active')


if __name__ == '__main__': unittest.main()
