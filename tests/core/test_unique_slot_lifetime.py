"""Host-only negative admission checks; no native/device launches.

Main-authored; repository LICENSE.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
import unique_slot_lifetime as t


def cases():
    env = t.previous.SAFETY | t.previous.BACKEND | dict(CUDA_MODULE_LOADING='EAGER',
        ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
        ZE_ENABLE_THREADING_VALIDATION='1', ZE_ENABLE_HANDLE_LIFETIME='0',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION')
    values = []
    for mode, placement, width in t.descriptors():
        label = mode + '-' + placement + '-w' + str(width)
        kernels, copies, ready = t.counts(mode)
        values.append(dict(mode=mode, placement=placement, width=width, label=label,
            root=str(t.ROOT / label), environment=dict(env), expected_kernels=kernels,
            expected_payload_copies=copies, expected_ready_barriers=(ready + width - 1) // width,
            argv=[str(t.ROOT / 'fixture'), 'host'] if mode == 'host' else
                 [str(t.ROOT / 'fixture'), mode, placement, str(width)]))
    return values


class Admission(unittest.TestCase):
    def test_complete_order_not_an_arbitrary_subset(self):
        row = dict(cases=cases(), model_trial_allowed=False, performance_qualified=False)
        t.plan_gate(row)
        self.assertEqual(len(row['cases']), 15)
        for changed in [row['cases'][:-1], list(reversed(row['cases'])), row['cases'] + row['cases'][:1]]:
            with self.assertRaises(RuntimeError): t.plan_gate(dict(row, cases=changed))

    def test_all_safety_and_validation_settings_are_required(self):
        original = cases()[1]
        for key in original['environment']:
            wrong = copy.deepcopy(original); wrong['environment'][key] = 'different'
            with self.subTest(key=key), self.assertRaises(RuntimeError): t.case_gate(wrong)

    def test_case_identity_and_typed_counts_cannot_drift(self):
        for key, value in [('root', '/tmp/other'), ('argv', ['other']), ('width', True),
                           ('expected_kernels', False), ('expected_payload_copies', 9),
                           ('expected_ready_barriers', 1), ('label', 'other')]:
            wrong = copy.deepcopy(cases()[1]); wrong[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): t.case_gate(wrong)

    def test_tail_and_alien_copy_are_counted(self):
        self.assertEqual(t.counts('partial129'), (129, 130, 129))
        partial = next(c for c in cases() if c['mode'] == 'partial129')
        self.assertEqual(partial['expected_ready_barriers'], 3)
        self.assertEqual(t.counts('full512'), (512, 513, 512))
        self.assertEqual(t.counts('throw'), (0, 2, 0))
        with self.assertRaises(RuntimeError): t.counts('unplanned')

    def test_component_pass_cannot_admit_model_or_speed(self):
        row = dict(cases=cases(), model_trial_allowed=False, performance_qualified=False)
        for key in ['model_trial_allowed', 'performance_qualified']:
            with self.subTest(key=key), self.assertRaises(RuntimeError): t.plan_gate(dict(row, **{key: True}))

    def test_changed_source_and_actual_dependency_refused(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); manifest = root / 'runtime.json'; manifest.write_text('{}')
            used = root / 'used'; used.write_text('original')
            binary = root / 'fixture'; binary.write_text('executable')
            unused = root / 'unused'; unused.write_text('original')
            row = dict(cases=cases(), model_trial_allowed=False, performance_qualified=False,
                project={'source_sha256': {'source': 'pinned'}}, driver={'same': True},
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), recipe_tools={},
                runtime_manifest={'path': str(manifest), 'sha256': t.sha(manifest)},
                phase_references={}, binary={'path': str(binary), 'sha256': t.sha(binary)},
                actual_inputs={'compiled': {str(used): t.sha(used)}}, allowed={str(unused): t.sha(unused)})
            pin = t.base.original.coherent.build.common.source
            with patch.object(pin, 'pin', return_value=row['project']) as source, \
                 patch.object(t.base.guards, 'driver_guard', return_value=row['driver']), \
                 patch.object(t, 'PHASE_REFERENCES', {}):
                unused.write_text('not consumed by this build')
                t.check(row, False)  # Unused SDK members are not actual inputs.
                used.write_text('changed actual input')
                with self.assertRaises(RuntimeError): t.check(row, False)
                used.write_text('original')
                source.return_value = {'source_sha256': {'source': 'different'}}
                with self.assertRaises(RuntimeError): t.check(row, False)

    def test_prelaunch_failure_is_terminal_and_does_not_start_owner(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); record = root / 'record.json'
            row = dict(stage='prepared', passed=False, complete=False, active=False,
                ownership_closed=True, commands=[], results=[], driver={'same': True},
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            record.write_text(json.dumps(row))
            with patch.object(t, 'ROOT', root), patch.object(t, 'RECORD', record), \
                 patch.object(t, 'check', side_effect=RuntimeError('source mismatch')), \
                 patch.object(t, 'save', side_effect=lambda r: record.write_text(json.dumps(r))), \
                 patch.object(t.base.guards, 'driver_guard', return_value=row['driver']), \
                 patch.object(t.base.owner_api, 'Owner') as owner:
                with self.assertRaisesRegex(RuntimeError, 'source mismatch'): t.execute('build')
                failed = json.loads(record.read_text())
                self.assertEqual(failed['stage'], 'failed'); self.assertFalse(failed['active'])
                self.assertTrue(failed['ownership_closed']); owner.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'wrong/closed'): t.execute('build')

    def test_owner_setup_failure_leaves_one_shot_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); record = root / 'record.json'
            row = dict(stage='prepared', commands=[], results=[], driver={},
                native_source_commit='pinned', boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            record.write_text(json.dumps(row))
            with patch.object(t, 'ROOT', root), patch.object(t, 'RECORD', record), \
                 patch.object(t, 'check'), patch.object(t, 'save', side_effect=lambda r: record.write_text(json.dumps(r))), \
                 patch.object(t.base.guards, 'driver_guard', return_value={}), \
                 patch.object(t.base.owner_api, 'Owner', side_effect=RuntimeError('setup failed')):
                with self.assertRaisesRegex(RuntimeError, 'setup failed'): t.execute('build')
                self.assertTrue((root / 'build.started').is_file())
                self.assertEqual(json.loads(record.read_text())['stage'], 'failed')
                self.assertTrue(json.loads(record.read_text())['ownership_closed'])


if __name__ == '__main__': unittest.main()
