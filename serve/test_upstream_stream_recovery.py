# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU fake regressions for answer reserve, held streams and owned-engine inactivity."""
import collections
import io
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serve.server import EngineSilent, Service, StrataEngine, reasoning_repeat_coverage
from serve.telemetry import Telemetry
from serve.test_reasoning_loop_recovery import RecordingEngine, Tokenizer


class StreamRecovery(unittest.TestCase):
    def test_periodic_tail_and_nonperiodic_text(self):
        for period in ("7", "12345678901", "Ab" * 32):
            self.assertEqual(reasoning_repeat_coverage(period * 2048), 1)
        self.assertEqual(reasoning_repeat_coverage("1" * 2047), 0)
        self.assertEqual(reasoning_repeat_coverage("".join(str(i) for i in range(2000))), 0)

    def test_answer_reserve_boundaries(self):
        for maximum, expected in ((1, 1), (2, 1), (100, 50), (1024, 512), (4096, 3072)):
            with self.subTest(maximum=maximum):
                tok = Tokenizer()
                engine = RecordingEngine(tok, ["thinking", "answer"])
                svc = Service(engine, tok, None)
                gen = svc.run([1], True, [], maximum, {"reasoning_budget_tokens": 10000}, threading.Event())
                try:
                    next(gen)
                    self.assertEqual(svc.status["thinking_budget"], expected)
                finally:
                    gen.close()

    def stream(self, text, tools):
        tok = Tokenizer()
        svc = Service(RecordingEngine(tok, text), tok, None)
        clock = iter(range(100000))
        with patch("serve.server.time.perf_counter", side_effect=lambda: next(clock)), \
                patch("serve.server.KEEPALIVE_S", 10):
            return list(svc.run([1], False, tools, 4000, {}, threading.Event()))

    def test_held_argument_pings_but_normal_content_does_not(self):
        tools = [{"name": "read", "parameters": {"properties": {"data": {"type": "object"}}}}]
        text = ('<tool_call>\n<function=read>\n<parameter=data>\n{"x":"' + "a" * 200 +
                '"}\n</parameter>\n</function>\n</tool_call>')
        held = self.stream(text, tools)
        self.assertTrue(any(kind == "ping" for kind, _ in held))
        self.assertTrue(any(kind == "event" and event.kind == "tool_call" for kind, event in held))
        normal = self.stream("ordinary answer " * 20, [])
        self.assertFalse(any(kind == "ping" for kind, _ in normal))


class Watchdog(unittest.TestCase):
    def engine(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = Mock(pid=123, stdin=io.StringIO())
        engine.proc.poll.return_value = None
        engine.gpu_busy = lambda: False
        engine._silent = Mock(return_value=EngineSilent("frozen"))
        return engine

    def test_frozen_only_with_known_idle_gpu_and_comparable_activity(self):
        cases = [(True, (0, 0, 1), False), (None, (0, 0, 1), False),
                 (False, (1, 0, 1), False), (False, (0, 2 << 20, 1), False),
                 (False, None, False), (False, (0, 0, 2), False), (False, (0, 0, 1), True)]
        for busy, current, fails in cases:
            with self.subTest(busy=busy, current=current):
                engine = self.engine()
                engine.gpu_busy = lambda: busy
                state = {"proc": engine.proc, "base": (0, 0, 1), "next": 90}
                with patch.object(engine, "_activity", return_value=current), \
                        patch("serve.server.time.monotonic", return_value=100):
                    if fails:
                        with self.assertRaises(EngineSilent):
                            engine._stall_check(0, state, 90)
                    else:
                        engine._stall_check(0, state, 90)
                self.assertEqual(engine._silent.called, fails)

    def test_missing_callback_error_and_replaced_owner_fail_open(self):
        for callback in (None, Mock(side_effect=RuntimeError("no sample"))):
            engine = self.engine()
            engine.gpu_busy = callback
            state = {"proc": engine.proc, "base": (0, 0, 1), "next": 0}
            with patch.object(engine, "_activity", return_value=(0, 0, 1)):
                engine._stall_check(0, state, 90)
            engine._silent.assert_not_called()
        engine = self.engine()
        state = {"proc": object(), "base": (0, 0, 1), "next": 0}
        with patch.object(engine, "_activity", return_value=(0, 0, 1)):
            engine._stall_check(0, state, 90)
        engine._silent.assert_not_called()
        self.assertIs(state["proc"], engine.proc)

    def test_gpu_work_or_unknown_resets_the_entire_idle_window(self):
        engine = self.engine()
        state = {"proc": engine.proc, "base": (0, 0, 1), "next": 90}
        with patch.object(engine, "_activity", return_value=(0, 0, 1)), \
                patch("serve.server.time.monotonic", return_value=100):
            engine.gpu_busy = lambda: True
            engine._stall_check(0, state, 90)
            self.assertIsNone(state["base"])
            engine.gpu_busy = lambda: False
            engine._stall_check(0, state, 90)
            self.assertEqual(state["next"], 190)
        engine._silent.assert_not_called()

    def test_serial_poll_cancel_stops_and_drains(self):
        engine = self.engine()
        engine.alive = lambda: True
        engine.sampling_keys = lambda _: ""
        engine.can_stop = True
        engine.lines = Mock()
        engine.lines.get.side_effect = [queue.Empty(), "DONE 0\n"]
        engine._parse_done = Mock()
        engine._stall_check = Mock()
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(engine.generate([1], 128, {}, cancel)), [])
        self.assertEqual(engine.proc.stdin.getvalue().splitlines(), ["GEN 128 1", "STOP"])
        engine._stall_check.assert_not_called()
        engine._parse_done.assert_called_once_with("DONE 0\n")

    def test_psutil_unavailable_and_dead_child(self):
        engine = self.engine()
        with patch.dict("sys.modules", {"psutil": None}):
            self.assertIsNone(engine._activity())
        engine.proc.poll.return_value = 0
        process = Mock()
        with patch.dict("sys.modules", {"psutil": SimpleNamespace(Process=process)}):
            self.assertIsNone(engine._activity())
        process.assert_not_called()

    def test_activity_uses_only_owned_pid(self):
        engine = self.engine()
        process = Mock()
        process.create_time.return_value = 1
        process.cpu_times.return_value = SimpleNamespace(user=2, system=3)
        process.io_counters.return_value = SimpleNamespace(read_bytes=4, write_bytes=5)
        factory = Mock(return_value=process)
        with patch.dict("sys.modules", {"psutil": SimpleNamespace(Process=factory)}):
            self.assertEqual(engine._activity(), (5, 9, 1))
        factory.assert_called_once_with(123)

    def test_invalid_activity_does_not_establish_idle(self):
        engine = self.engine()
        for user, reads in ((-1, 0), (0, -1), (float("nan"), 0), (float("inf"), 0)):
            process = Mock()
            process.create_time.return_value = 1
            process.cpu_times.return_value = SimpleNamespace(user=user, system=0)
            process.io_counters.return_value = SimpleNamespace(read_bytes=reads, write_bytes=0)
            factory = Mock(return_value=process)
            with patch.dict("sys.modules", {"psutil": SimpleNamespace(Process=factory)}):
                self.assertIsNone(engine._activity())

    def test_control_cancel_sends_stop_once_and_drains_without_watchdog(self):
        engine = self.engine()
        cancel = threading.Event()
        cancel.set()
        engine.lines = Mock()
        engine.lines.get.side_effect = [queue.Empty(), "DONE 0\n", "BADM\n"]
        engine._send = Mock()
        engine._parse_done = Mock()
        engine._ctl_mode = "solo"
        engine._stall_check = Mock()
        list(engine._control(cancel, lambda _: None))
        engine._send.assert_called_once_with("STOP")
        engine._stall_check.assert_not_called()

    def test_serial_generator_close_stops_and_drains(self):
        engine = self.engine()
        engine.alive = lambda: True
        engine.sampling_keys = lambda _: ""
        engine.can_stop = True
        engine.lines = queue.Queue()
        engine.lines.put("T 42\n")
        engine.lines.put("DONE 1\n")
        engine._parse_done = Mock()
        gen = engine.generate([1], 128, {}, threading.Event())
        self.assertEqual(next(gen), 42)
        gen.close()
        self.assertEqual(engine.proc.stdin.getvalue().splitlines(), ["GEN 128 1", "STOP"])
        engine._parse_done.assert_called_once_with("DONE 1\n")
        engine._silent.assert_not_called()

    def test_telemetry_freshness_and_invalid_values(self):
        telemetry = Telemetry.__new__(Telemetry)
        telemetry.lock = threading.Lock()
        for value, sampled, expected in ((0, 99, None), (.01, 99, True), (None, 99, None),
                                         (float("nan"), 99, None), (-1, 99, None), (101, 99, None),
                                         (False, 99, None), (0, 90, None), (0, None, None)):
            telemetry.now = {"gpu_util": value}
            telemetry.sampled_at = sampled
            with patch("serve.telemetry.time.monotonic", return_value=100):
                self.assertIs(telemetry.gpu_activity(), expected)

    def test_zero_monitor_gauge_never_ends_a_silent_engine(self):
        telemetry = Telemetry.__new__(Telemetry)
        telemetry.lock = threading.Lock()
        telemetry.now = {"gpu_util": 0}
        telemetry.sampled_at = 100
        engine = self.engine()
        engine.gpu_busy = telemetry.gpu_activity
        state = {"proc": engine.proc, "base": (0, 0, 1), "next": 90}
        with patch.object(engine, "_activity", return_value=(0, 0, 1)), \
                patch("serve.server.time.monotonic", return_value=100), \
                patch("builtins.print") as note:
            engine._stall_check(0, state, 90)
            engine._stall_check(0, state, 90)
        engine._silent.assert_not_called()
        self.assertEqual(note.call_count, 1)

    def test_slow_sample_does_not_refresh_old_idle_reading(self):
        telemetry = Telemetry.__new__(Telemetry)
        telemetry.lock = threading.Lock()
        telemetry.hist = collections.defaultdict(collections.deque)
        clock = [100.0]

        def sample():
            clock[0] = 120.0
            return {"gpu_util": 0}

        telemetry.sample = sample
        with patch("serve.telemetry.time.monotonic", side_effect=lambda: clock[0]), \
                patch("serve.telemetry.time.sleep", side_effect=RuntimeError("stop sampling")):
            with self.assertRaisesRegex(RuntimeError, "stop sampling"):
                telemetry._loop()
            self.assertIsNone(telemetry.gpu_activity())


if __name__ == "__main__":
    unittest.main()
