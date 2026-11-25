"""Host checks for the build-stage pin and the matched upload comparison."""

import copy
import unittest
from unittest.mock import patch

import dflash2_mode16_batch_model as model


class Mode16BatchModelTest(unittest.TestCase):
    def test_result_commit_does_not_relabel_candidate(self):
        source = 'fd72dce1fea69788e66070d5783093c30d8957af'
        frozen = model.candidate.plan(source)
        actual = model.candidate.recorded_plan(dict(native_source_commit=source, mode16_batch_plan=frozen))
        self.assertEqual(actual['source_commit'], source)
        self.assertNotEqual(model.candidate.plan()['source_commit'], source)
        with self.assertRaisesRegex(RuntimeError, 'frozen build sources'):
            model.candidate.plan('de4494888')

    def test_model_inputs_change_only_upload_policy(self):
        old_build, _ = model.candidate.references()
        with patch.object(model.candidate, 'admitted_build', return_value=old_build):
            row = model.baseline()
        self.assertEqual(len(row['request_hashes'][str(model.READ)]), 3)
        self.assertEqual(len(row['prefill_work']), 3)
        ctx = model.experiment()
        plans = [dict(index=i, arm=arm, root=str(model.ROOT / str(i)), port=36000 + i,
                      read=model.READ, heads=1, kv_arm='retain') for i, arm in enumerate(model.ORDER)]
        environments = []
        for plan in plans:
            command = ctx.command(plan)
            self.assertEqual(command[command.index('--prefill-exact-tokens') + 1], str(model.READ))
            self.assertEqual(command[command.index('--runs') + 1], '3')
            self.assertIn('--think', command)
            cfg = ctx.configuration(row, plan)
            self.assertEqual(cfg['exe'], old_build['binaries']['strata']['path'])
            self.assertEqual(cfg['args'][cfg['args'].index('--draft') + 1], 'dflash2')
            env = ctx.environment(row, plan)
            self.assertEqual(env.pop(model.UPLOAD), plan['arm'])
            environments.append({k: v.replace(plan['root'], '<process>') for k, v in env.items()})
        self.assertTrue(all(env == environments[0] for env in environments))

    def test_repeatable_gain_uses_the_warm_pairs(self):
        control = model.bank.previous.historic(model.upload.CONTROL, model.curve.KEY)['processes'][3]
        processes = []
        for i, arm in enumerate(model.ORDER):
            sample = copy.deepcopy(control)
            sample.update(arm=arm, native_identity=dict(pid=i + 1, start_ticks=i + 1))
            if arm == 'batched':
                sample['samples'][0]['prompt_ms'] *= 1.01
                scale = 1.01 if i == 5 else 0.99
                for warm in sample['samples'][1:]:
                    warm['prompt_ms'] *= scale
            processes.append(sample)
        decision = model.decision(processes)
        self.assertTrue(decision['repeatable_warm_gain'])
        self.assertTrue(all(pair['first_ms'] > 1 for pair in decision['batched_over_demand']))
        for warm in processes[2]['samples'][1:]:
            warm['prompt_ms'] *= 1.02
        self.assertFalse(model.decision(processes)['repeatable_warm_gain'])


if __name__ == '__main__':
    unittest.main()
