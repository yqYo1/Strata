# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""GPU-free setup replica config validation; no model downloads or subprocesses."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class SetupReplicas(unittest.TestCase):
    def config(self):
        return {"gpu_pci": "05:00.0", "split_pci": ["06:00.0"], "args": []}

    def test_defaults_leave_single_engine_config_unchanged(self):
        cfg = self.config()
        before = dict(cfg)
        self.assertEqual(setup.configure_replicas(cfg), [])
        self.assertEqual(cfg, before)

    def test_opt_in_uses_explicit_pci_groups_and_reservations(self):
        cfg = self.config()
        self.assertEqual(setup.configure_replicas(cfg, 2, 120000, 8000),
                         [["0000:05:00.0"], ["0000:06:00.0"]])
        self.assertEqual((cfg["replicas"], cfg["replica_ram_mib"], cfg["replica_vram_mib"]), (2, 120000, 8000))
        self.assertEqual(setup.configure_replicas(cfg, 1), [])

    def test_incomplete_or_overlapping_configs_are_refused(self):
        for count, ram, vram in ((2, None, None), (2, True, 8000), (2, 120000, 0), (3, 120000, 8000),
                                 (-1, 120000, 8000)):
            with self.subTest(count=count, ram=ram, vram=vram), self.assertRaises(ValueError):
                setup.configure_replicas(self.config(), count, ram, vram)
        cfg = self.config()
        cfg["split_pci"] = ["0000:05:00.0"]
        with self.assertRaises(ValueError):
            setup.configure_replicas(cfg, 2, 120000, 8000)


if __name__ == "__main__":
    unittest.main()
