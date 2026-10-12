# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Data-parallel replicas (serve/replicas.py): the config, the router, and the real Service over HTTP with two fake
engines (the fake of test_parallel.py) behind a ReplicaEngine."""
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.replicas import BLOCK, ReplicaConfigError, ReplicaEngine, Router, cpu_sets, gpu_groups, replica_cfg
from serve.server import ByteTokenizer, Service, StrataEngine, serve
from serve.test_parallel import FAKE_BATCH


class Config(unittest.TestCase):
    def test_number_and_explicit_pci_groups(self):
        cfg = {"gpu_pci": "05:00.0", "split_pci": ["06:00.0"], "replicas": 2}
        self.assertEqual(gpu_groups(cfg), [["0000:05:00.0"], ["0000:06:00.0"]])
        self.assertEqual(gpu_groups({"replicas": [{"devices": ["05:00.0"]}, {"devices": ["06:00.0"]}]}),
                         gpu_groups(cfg))
        self.assertEqual(gpu_groups({"replicas": 1}), [])
        self.assertEqual(gpu_groups({}), [])

    def test_rejects_overlap_indices_and_bad_geometry(self):
        for cfg in ({"replicas": 2}, {"replicas": -1}, {"replicas": True},
                    {"gpu_pci": "05:00.0", "split_pci": ["0000:05:00.0"], "replicas": 2},
                    {"replicas": [{"devices": [0]}, {"devices": [1]}]},
                    {"replicas": [{"devices": ["05:20.0"]}, {"devices": ["06:00.0"]}]},
                    {"replicas": [{"devices": []}, {"devices": ["06:00.0"]}]}):
            with self.subTest(cfg=cfg), self.assertRaises(ReplicaConfigError):
                gpu_groups(cfg)

    def test_replica_config_is_owned_and_does_not_mutate_base(self):
        cfg = {"args": ["--conversation-save", "/cache/conversations"], "gpu_pci": "05:00.0",
               "split_pci": ["06:00.0"], "log": "/log/engine.log", "expert_profile_save": "/profiles/new.bin",
               "env": {"A": "1"}, "replicas": 2}
        child = replica_cfg(cfg, 1, ["0000:06:00.0"])
        self.assertEqual(child["env"]["STRATA_GPU_PCI"], "0000:06:00.0")
        self.assertEqual(child["split_pci"], [])
        self.assertEqual(child["log"], "/log/engine.r1.log")
        self.assertEqual(child["expert_profile_save"], "/profiles/new.r1.bin")
        self.assertEqual(child["args"], ["--conversation-save", "/cache/conversations.r1"])
        self.assertEqual(cfg["args"], ["--conversation-save", "/cache/conversations"])
        self.assertNotIn("replicas", child)

    def test_device_override_arguments_and_env_are_rejected(self):
        for change in ({"args": ["--gpu-pci", "07:00.0"]}, {"env": {"STRATA_LAYER_SPLIT": "off"}}):
            with self.assertRaises(ReplicaConfigError):
                replica_cfg({"args": [], **change}, 0, ["0000:05:00.0"])

    def test_entry_arguments_keep_model_and_device_ownership(self):
        cfg = {"args": [], "replicas": [{"devices": ["05:00.0"], "args": ["--temperature", "0"]},
                                        {"devices": ["06:00.0"]}]}
        self.assertEqual(replica_cfg(cfg, 0, ["0000:05:00.0"])["args"], ["--temperature", "0"])
        cfg["replicas"][0]["args"] = ["--native", "/other/model.gguf"]
        with self.assertRaises(ReplicaConfigError):
            replica_cfg(cfg, 0, ["0000:05:00.0"])

    def test_cpu_sets_off(self):
        self.assertIsNone(cpu_sets(2, enabled=False))
        self.assertIsNone(cpu_sets(1))


class Routing(unittest.TestCase):
    def test_matched_prefix_per_replica(self):
        r = Router(3)
        a = list(range(BLOCK * 4))
        r.note(0, a)
        r.note(1, a[:BLOCK * 2])
        self.assertEqual(r.matched(a + [9] * 70), [BLOCK * 4, BLOCK * 2, 0])
        self.assertEqual(r.matched([7] * 200), [0, 0, 0])
        r.forget(0)
        self.assertEqual(r.matched(a), [0, BLOCK * 2, 0])

    def test_the_index_is_bounded(self):
        r = Router(2, cap=10)
        for i in range(50):
            r.note(i % 2, [i] * (BLOCK * 2))
        self.assertLessEqual(len(r.blocks), 10)


class FakeEngine:
    def __init__(self, n=4):
        self.batch, self.up = n, True
        self.max_context = 4096
        self.info = {}
        self.last = {}

    def alive(self):
        return self.up


class RouteChoice(unittest.TestCase):
    def test_least_loaded_then_affinity(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        picks = [e.route([i] * 100) for i in range(4)]             # nothing cached: spread
        self.assertEqual(sorted(picks), [0, 0, 1, 1])
        ids = list(range(BLOCK * 8))
        e.inflight = [0, 0]
        k = e.route(ids)
        e.router.note(k, ids)
        e.inflight = [0, 0]
        self.assertEqual(e.route(ids + [5] * 10), k)                # the conversation goes back
        e.inflight = [0, 0]
        e.inflight[k] = 1                                           # 512 tokens held < 3000 per running request
        self.assertNotEqual(e.route(ids + [5] * 10), k)

    def test_a_long_prefix_outweighs_load(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        ids = list(range(BLOCK * 200))                              # 12,800 tokens held on replica 0
        e.router.note(0, ids)
        e.inflight = [2, 0]
        self.assertEqual(e.route(ids + [1]), 0)
        e.inflight = [5, 0]
        self.assertEqual(e.route(ids + [1]), 1)

    def test_a_dead_replica_is_skipped(self):
        a, b = FakeEngine(), FakeEngine()
        e = ReplicaEngine([a, b])
        e.heal = lambda: None
        ids = list(range(BLOCK * 3))
        e.router.note(0, ids)
        a.up = False
        self.assertEqual(e.route(ids), 1)
        b.up = False
        self.assertEqual(e.route(ids), -1)

    def test_nonfinite_forgets_the_replica(self):
        """#879: after DONE ... nonfinite the engine holds no conversation: the router forgets that replica."""
        from serve.server import EngineNonFinite

        class NonFinite(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                yield 7
                raise EngineNonFinite("the engine's logits became non-finite (#879)")

        e = ReplicaEngine([NonFinite(), NonFinite()])
        other, ids = [9] * (BLOCK * 4), list(range(BLOCK * 3))
        e.router.note(0, other)
        e.router.note(1, other)
        k = 0
        e.inflight[k] += 1                                          # as route() counts it
        with mock.patch.object(e, "route", return_value=k):
            with self.assertRaises(EngineNonFinite):
                list(e.generate(ids, 10, {}, threading.Event()))
        self.assertEqual(e.router.matched(ids)[k], 0)
        self.assertEqual(e.router.matched(other)[k], 0)             # its older conversations too
        self.assertEqual(e.router.matched(other)[1 - k], len(other))
        self.assertEqual(e.inflight[k], 0)

    def test_batch_is_the_sum(self):
        self.assertEqual(ReplicaEngine([FakeEngine(4), FakeEngine(4)]).batch, 8)
        self.assertEqual(ReplicaEngine([FakeEngine(0), FakeEngine(0)]).batch, 2)


class ReplicaService(unittest.TestCase):
    def start(self, slots=4, n=2):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        real = server.subprocess.Popen
        self.logs = [Path(self.tmp.name) / f"r{i}.log" for i in range(n)]
        engines = []
        for i in range(n):
            extra = (["--batch", str(slots)] if slots else []) + ["--log", str(self.logs[i])]
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engines.append(StrataEngine("strata", extra))
        self.engine = ReplicaEngine(engines)
        self.svc = Service(self.engine, ByteTokenizer(),
                           ChatTemplate(Path(__file__).parent / "test_chat_template.jinja"))
        self.svc.telemetry = mock.Mock()
        self.svc.telemetry.snapshot.return_value = {"now": {}, "history": {}, "static": {}}
        self.svc.telemetry.gpu_activity.return_value = None
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.close()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def post(self, messages, max_tokens=64):
        body = {"messages": messages, "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def chat(self, text, max_tokens=64):
        return self.post([{"role": "user", "content": text}], max_tokens)

    def served(self):
        counts = []
        for path in self.logs:
            lines = path.read_text().splitlines() if path.exists() else []
            counts.append(sum(1 for line in lines if line.split()[0] in ("GEN", "BGEN")))
        return counts

    def burst(self, prompts):
        out = [None] * len(prompts)

        def one(i):
            out[i] = self.chat(prompts[i])
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return out

    def test_requests_spread_over_both_and_answers_are_the_solo_answer(self):
        self.start()
        solo = self.chat("hello there")
        before = self.served()
        out = self.burst([f"request number {i} " + "x" * 300 for i in range(6)])
        got = [b - a for a, b in zip(before, self.served())]
        self.assertTrue(all(g >= 2 for g in got), got)
        self.assertEqual(set(out), {solo})

    def test_metrics_and_status_see_all_slots(self):
        self.start()
        self.assertEqual(self.svc.v1_status()["concurrency"]["serving"], 8)
        m = self.svc.metrics()
        self.assertEqual(len(m["live"]["slots"]), 8)
        self.assertEqual([r["replica"] for r in m["live"]["replicas"]], [0, 1])
        self.assertEqual(m["engine"]["replicas"], 2)

    def test_health_reports_each_replica_without_hardware_sampling(self):
        self.start(slots=0)
        with urllib.request.urlopen(self.base + "/health", timeout=10) as response:
            health = json.loads(response.read())
        self.assertTrue(health["loaded"])
        self.assertEqual([replica["alive"] for replica in health["replicas"]], [True, True])

    def test_slot_ids_have_their_replica_owner(self):
        self.start()
        with urllib.request.urlopen(self.base + "/slots", timeout=10) as response:
            slots = json.loads(response.read())
        self.assertEqual([slot["id"] for slot in slots], list(range(8)))
        self.assertEqual([slot["replica"] for slot in slots], [0] * 4 + [1] * 4)

    def test_single_file_sessions_cannot_select_an_arbitrary_replica(self):
        self.start(slots=0)
        self.svc.slot_save_path = self.tmp.name
        for child in self.engine.engines:
            child.session_file = mock.Mock()
        for action in ("save", "restore"):
            code, _ = self.svc.slot_action("0", action, "session.bin")
            self.assertEqual(code, 501)
        for child in self.engine.engines:
            child.session_file.assert_not_called()

    def test_engines_without_slots_run_one_each_at_once(self):
        self.start(slots=0)
        out = self.burst(["a" * 40, "b" * 40])
        self.assertEqual(self.served(), [1, 1])
        self.assertEqual(len(set(out)), 1)

    def test_a_conversation_goes_back_to_its_replica(self):
        self.start()
        first = "begin " + "lorem ipsum " * 40
        reply = self.chat(first)
        k = self.served().index(1)
        self.post([{"role": "user", "content": first}, {"role": "assistant", "content": reply},
                   {"role": "user", "content": "and then?"}], 16)
        self.assertEqual(self.served()[k], 2)

    def test_a_replica_that_died_is_skipped(self):
        self.start()
        self.engine.heal = lambda: None
        self.engine.engines[0].proc.kill()
        self.engine.engines[0].proc.wait()
        self.assertTrue(self.engine.alive())
        for i in range(3):
            self.assertIn("ok, done.", self.chat(f"after the loss {i}"))
        self.assertEqual(self.served()[1], 3)


if __name__ == "__main__":
    unittest.main()
