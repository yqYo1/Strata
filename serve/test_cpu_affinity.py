# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU-only topology and child-only exec-affinity fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from serve.cpu_affinity import launch_args, partition_cpu_sets


class CpuAffinity(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for cpu in range(8):
            path = self.root / "devices/system/cpu" / f"cpu{cpu}" / "topology"
            path.mkdir(parents=True)
            (path / "physical_package_id").write_text("0")
            (path / "core_id").write_text(str(cpu % 4))

    def test_siblings_stay_together_and_reserved_core_is_excluded(self):
        self.assertEqual(partition_cpu_sets(2, set(range(8)), sys_root=self.root), [[0, 1, 4, 5], [2, 3, 6, 7]])
        self.assertEqual(partition_cpu_sets(2, set(range(8)), {4}, self.root), [[1, 2, 5, 6], [3, 7]])
        self.assertEqual(partition_cpu_sets(2, {0, 1, 4}, sys_root=self.root), [[0, 4], [1]])

    def test_missing_topology_and_insufficient_cores_fail_closed(self):
        for count, allowed in ((3, {0, 4, 1}), (1, set()), (0, {0}), (2, {0, 9})):
            with self.subTest(count=count, allowed=allowed), self.assertRaises(ValueError):
                partition_cpu_sets(count, allowed, sys_root=self.root)

    def test_no_wrapper_for_default_and_no_preexec_for_opt_in(self):
        self.assertEqual(launch_args("strata", ["--serve"], None), ["strata", "--serve"])
        with mock.patch("serve.cpu_affinity.os.sched_getaffinity", return_value={2, 3}), \
                mock.patch("serve.cpu_affinity.os.access", return_value=True):
            command = launch_args("/engine/strata", ["--serve"], [2])
            self.assertEqual(command[-3:], ["--command", "/engine/strata", "--serve"])
            self.assertIn("-I", command)
            for cpus in ([], [2, 2], [4], [True]):
                with self.assertRaises(ValueError):
                    launch_args("/engine/strata", [], cpus)
            with self.assertRaises(ValueError):
                launch_args("relative/strata", [], [2])

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux affinity required")
    def test_actual_owned_child_mask_and_parent_unchanged(self):
        before = os.sched_getaffinity(0)
        selected = min(before)
        program = "import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))"
        command = launch_args(str(Path(sys.executable).resolve()), ["-c", program], [selected])
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout), [selected])
        self.assertEqual(os.sched_getaffinity(0), before)


if __name__ == "__main__":
    unittest.main()
