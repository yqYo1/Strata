"""Saved-role replacement through the public launch configuration helpers.

Main-authored regression tests; repository LICENSE applies. No GPU execution.
"""
import copy
import unittest

from strata_cli.launch import settings as S
from strata_cli.launch.runner import apply_cards, derive_config
from strata_control import gpu_roles


class LaunchGpuOrder(unittest.TestCase):
    def saved(self):
        return dict(exe='strata', args=[], gpu=[0, 1], gpu_order='as_given',
            gpu_roles={'0': 'helper', '1': 'main'}, gpu_cards={'0': 'A', '1': 'B'},
            gpu_none=False, gpus_asked=True, layer_split='auto', hip_ordinal=0)

    def test_plain_list_replaces_saved_roles_and_order(self):
        for selected in ([1], [1, 0]):
            with self.subTest(selected=selected):
                cfg = self.saved()
                apply_cards(cfg, selected)
                self.assertEqual(cfg['gpu'], selected[0] if len(selected) == 1 else selected)
                self.assertTrue(cfg['gpus_asked'])
                for stale in ('gpu_roles', 'gpu_cards', 'gpu_none', 'gpu_order', 'hip_ordinal'):
                    self.assertNotIn(stale, cfg)
                self.assertEqual(cfg.get('layer_split'), 'auto' if len(selected) > 1 else None)

    def test_explicit_roles_keep_main_last_and_explicit_order(self):
        cards = [dict(index=0, name='A', vram_gb=12, speed=1),
                 dict(index=1, name='B', vram_gb=12, speed=2)]
        roles = gpu_roles.mirror({0: 'helper', 1: 'main'}, cards)
        cfg = self.saved()
        apply_cards(cfg, roles)
        self.assertEqual((cfg['gpu'], cfg['gpu_order']), ([0, 1], 'as_given'))
        self.assertEqual(cfg['gpu_roles'], {'0': 'helper', '1': 'main'})

    def test_no_selection_keeps_installed_choice(self):
        for selected in (None, []):
            with self.subTest(selected=selected):
                cfg = self.saved(); original = copy.deepcopy(cfg)
                apply_cards(cfg, selected)
                self.assertEqual(cfg, original)

    def test_launch_derivation_keeps_saved_config_independent(self):
        base = self.saved(); original = copy.deepcopy(base)
        cfg = derive_config(base, S.LaunchSettings(family='qwen', model='IQ3_XXS'),
            bool_flags=frozenset(), gpus=[1, 0])
        self.assertEqual(base, original)
        self.assertEqual(cfg['gpu'], [1, 0])
        self.assertNotIn('gpu_order', cfg)
        self.assertNotIn('gpu_roles', cfg)


if __name__ == '__main__':
    unittest.main()
