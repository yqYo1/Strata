# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""GPU-free resource/partial-start admission and owned replica generator regressions."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.replica_admission import memory_available_mib, model_file_floor_mib, validate_resources
from serve.replicas import BLOCK, ReplicaConfigError, ReplicaEngine
from serve.server import start_replicas
from serve.test_replicas import FakeEngine


class ResourceAdmission(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.proc = self.root / "proc"
        self.cg = self.root / "cgroup"
        (self.proc / "self").mkdir(parents=True)
        (self.cg / "job").mkdir(parents=True)
        (self.proc / "meminfo").write_text("MemAvailable: 65536 kB\n")
        (self.proc / "self/cgroup").write_text("0::/job\n")
        (self.proc / "self/mountinfo").write_text(f"42 1 0:1 / {self.cg} rw - cgroup2 cgroup rw\n")
        (self.cg / "memory.max").write_text("max")
        (self.cg / "job/memory.max").write_text(str(32 << 20))
        (self.cg / "job/memory.current").write_text(str(8 << 20))

    def test_v2_membership_and_ancestor_intersection(self):
        self.assertEqual(memory_available_mib(self.proc, self.cg), 24)
        (self.cg / "memory.max").write_text(str(16 << 20))
        (self.cg / "memory.current").write_text(str(10 << 20))
        self.assertEqual(memory_available_mib(self.proc, self.cg), 6)

    def test_real_unlimited_v2_root_needs_mount_proof(self):
        (self.cg / "memory.max").unlink()
        (self.proc / "self/mountinfo").write_text(f"42 1 0:1 / {self.cg} rw - cgroup2 cgroup rw\n")
        self.assertEqual(memory_available_mib(self.proc, self.cg), 24)
        (self.proc / "self/mountinfo").write_text(f"42 1 0:1 /hidden {self.cg} rw - cgroup2 cgroup rw\n")
        with self.assertRaises(ReplicaConfigError):
            memory_available_mib(self.proc, self.cg)

    def test_v1_and_unknown_layout(self):
        (self.proc / "self/cgroup").write_text("7:memory:/job\n")
        (self.proc / "self/mountinfo").write_text(f"43 1 0:2 / {self.cg / 'memory'} rw - cgroup memory rw\n")
        for path in (self.cg / "memory", self.cg / "memory/job"):
            path.mkdir(parents=True, exist_ok=True)
            (path / "memory.limit_in_bytes").write_text(str(16 << 20))
            (path / "memory.usage_in_bytes").write_text(str(4 << 20))
        self.assertEqual(memory_available_mib(self.proc, self.cg), 12)
        (self.proc / "self/cgroup").write_text("0::/missing\n")
        with self.assertRaises(ReplicaConfigError):
            memory_available_mib(self.proc, self.cg)

    def test_file_floor_and_missing_reservations_fail_before_hardware(self):
        model = self.root / "native.gguf"
        model.write_bytes(b"tiny valid path fixture")
        cfg = {"args": ["--native", str(model)]}
        self.assertEqual(model_file_floor_mib(cfg), 1)
        for fields in ({}, {"replica_ram_mib": True}, {"replica_ram_mib": 1, "replica_vram_mib": 0}):
            with mock.patch("serve.replica_admission.memory_available_mib") as memory, \
                    self.assertRaises(ReplicaConfigError):
                validate_resources({**cfg, **fields}, [["0000:05:00.0"]])
            memory.assert_not_called()

    def test_native_floor_counts_all_shards_and_refuses_a_missing_one(self):
        first = self.root / "model-00001-of-00002.gguf"
        second = self.root / "model-00002-of-00002.gguf"
        first.write_bytes(b"x")
        cfg = {"args": ["--native", str(first)]}
        with self.assertRaises(ReplicaConfigError):
            model_file_floor_mib(cfg)
        second.write_bytes(b"x" * (1 << 20))
        self.assertEqual(model_file_floor_mib(cfg), 2)
        cfg.update(replica_ram_mib=1, replica_vram_mib=1)
        with mock.patch("serve.replica_admission.memory_available_mib") as memory, \
                self.assertRaises(ReplicaConfigError):
            validate_resources(cfg, [["0000:05:00.0"]])
        memory.assert_not_called()

    def test_total_ram_and_vram_unknown_are_refused(self):
        cfg = {"replica_ram_mib": 100, "replica_vram_mib": 50}
        groups = [["0000:05:00.0"], ["0000:06:00.0"]]
        with mock.patch("serve.replica_admission.model_file_floor_mib", return_value=1), \
                mock.patch("serve.replica_admission.memory_available_mib", return_value=6300), \
                mock.patch("serve.replica_admission.Path.is_dir", return_value=True):
            with self.assertRaises(ReplicaConfigError):
                validate_resources(cfg, groups, free_vram=lambda _: 100)
            validate_resources(cfg, groups[:1], pending=1, free_vram=lambda _: 100)
            with self.assertRaises(ReplicaConfigError):
                validate_resources(cfg, groups[:1], pending=1, free_vram=lambda _: None)

    def test_partial_start_closes_every_created_owner(self):
        owners = []

        class Engine(FakeEngine):
            def __init__(self, *args, **kwargs):
                super().__init__(0)
                self.info = {"arena_mib": 1}
                self.closed = False
                owners.append(self)

            def restart(self, tries=1):
                if len(owners) == 2:
                    raise RuntimeError("second READY failed")

            def close(self):
                self.closed = True

        cfg = {"args": [], "replica_ram_mib": 10, "replica_vram_mib": 10}
        with mock.patch("serve.replica_admission.validate_resources"), \
                mock.patch("serve.replicas.cpu_sets", return_value=[[0], [1]]), \
                mock.patch("serve.server.StrataEngine", Engine), \
                self.assertRaises(RuntimeError):
            start_replicas(cfg, [["0000:05:00.0"], ["0000:06:00.0"]], "/engine/strata", [], False)
        self.assertEqual(len(owners), 2)
        self.assertTrue(all(owner.closed for owner in owners))


class GeneratorOwnership(unittest.TestCase):
    def test_cancelled_lock_wait_never_reaches_or_recaches_a_child(self):
        engines = [FakeEngine(0), FakeEngine(0)]
        for engine in engines:
            engine.generate = mock.Mock()
        front = ReplicaEngine(engines)
        cancel = threading.Event()
        gate = mock.Mock()

        def wait(**_):
            cancel.set()
            return False

        gate.acquire.side_effect = wait
        front.run_locks[0] = gate
        self.assertEqual(list(front.generate(list(range(BLOCK * 2)), 4, {}, cancel)), [])
        self.assertEqual(front.inflight, [0, 0])
        self.assertEqual(front.router.matched(list(range(BLOCK * 2))), [0, 0])
        gate.release.assert_not_called()
        for engine in engines:
            engine.generate.assert_not_called()

    def test_cancel_done_cannot_recache_a_fully_exhausted_generator(self):
        class Engine(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                yield 7
                self.last = {"generated": 1, "prompt_tokens": len(ids), "finish": "cancel"}

        front = ReplicaEngine([Engine(0), Engine(0)])
        ids = list(range(BLOCK * 2))
        list(front.generate(ids, 4, {}, threading.Event()))
        self.assertEqual(front.router.matched(ids), [0, 0])

    def test_consumer_eos_close_reuses_only_the_completed_input_prefix(self):
        for finish, expected in (("stop", BLOCK * 2), ("cancel", 0)):
            class Engine(FakeEngine):
                def generate(self, ids, max_new, sampling, cancel):
                    try:
                        yield 7
                    finally:
                        self.last = {"generated": 1, "prompt_tokens": len(ids), "finish": finish}

            front = ReplicaEngine([Engine(0), Engine(0)])
            ids = list(range(BLOCK * 2))
            gen = front.generate(ids, 4, {}, threading.Event())
            self.assertEqual(next(gen), 7)
            gen.close()
            self.assertEqual(front.router.matched(ids), [expected, 0])

    def test_close_drains_before_unlock_and_does_not_recache_partial_output(self):
        events = []

        class Engine(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                try:
                    yield 7
                    yield 8
                finally:
                    events.append(front.run_locks[0].locked())

        engines = [Engine(0), Engine(0)]
        front = ReplicaEngine(engines)
        ids = list(range(BLOCK * 2))
        gen = front.generate(ids, 2, {}, threading.Event())
        self.assertEqual(next(gen), 7)
        gen.close()
        self.assertEqual(events, [True])
        self.assertEqual(front.inflight, [0, 0])
        self.assertEqual(front.router.matched(ids), [0, 0])

    def test_done_statistics_snapshot_survives_next_request(self):
        class Engine(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                yield 7
                self.last = {"generated": 1, "prompt_tokens": len(ids)}

        engines = [Engine(0), Engine(0)]
        front = ReplicaEngine(engines)
        list(front.generate([1], 1, {}, threading.Event()))
        snapshot = front.last
        front.current.last = {"generated": 99}
        self.assertIs(front.last, snapshot)
        self.assertEqual(front.last["generated"], 1)

    def test_shutdown_disallows_new_route_and_close_visits_every_child(self):
        engines = [FakeEngine(0), FakeEngine(0)]
        for engine in engines:
            engine.close = mock.Mock()
        front = ReplicaEngine(engines)
        front.close()
        self.assertEqual(front.route([1]), -1)
        for engine in engines:
            engine.close.assert_called_once_with()

    def test_lazy_first_load_rolls_back_if_one_replica_fails_ready(self):
        engines = [FakeEngine(0), FakeEngine(0)]
        for engine in engines:
            engine.up = False
            engine.close = mock.Mock()
        engines[0].restart = mock.Mock(side_effect=lambda _: setattr(engines[0], "up", True))
        engines[1].restart = mock.Mock(side_effect=RuntimeError("second READY failed"))
        front = ReplicaEngine(engines)
        with self.assertRaisesRegex(RuntimeError, "second READY failed"):
            front.restart(1)
        for engine in engines:
            engine.close.assert_called_once_with()

    def test_unload_visits_other_owners_after_first_failure(self):
        engines = [FakeEngine(0), FakeEngine(0)]
        engines[0].unload = mock.Mock(side_effect=RuntimeError("first unload failed"))
        engines[1].unload = mock.Mock()
        front = ReplicaEngine(engines)
        with self.assertRaisesRegex(RuntimeError, "first unload failed"):
            front.unload()
        for engine in engines:
            engine.unload.assert_called_once_with()

    def test_assigned_last_is_request_local(self):
        engines = [FakeEngine(0), FakeEngine(0)]
        front = ReplicaEngine(engines)
        original = engines[0].last
        requested = {"generated": 1}
        front.last = requested
        self.assertIs(front.last, requested)
        self.assertIs(engines[0].last, original)


if __name__ == "__main__":
    unittest.main()
