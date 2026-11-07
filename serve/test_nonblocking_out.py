"""Bounded console regressions for the Niko b776e1cd functional port.

Main-authored; repository LICENSE. These CPU-only checks must run at a
closed GPU boundary. Child pipes are deliberately unread until child exit.
"""
import io
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from serve.nonblocking_out import NonBlockingOut, decoupled_console


class HeldSink:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.text = []

    def write(self, text):
        self.entered.set()
        if not self.release.wait(10):
            raise OSError("test sink deadline")
        self.text.append(text)

    def flush(self):
        pass


class ConsoleQueue(unittest.TestCase):
    def test_posix_descriptor_preserves_unicode_and_closes_its_copy(self):
        if os.name != "posix":
            self.skipTest("POSIX descriptor ownership")
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as sink, \
                patch.object(sys, "__stdout__", sink):
            out = NonBlockingOut(sink)
            descriptor = out._fd
            self.assertIsNotNone(descriptor)
            self.assertNotEqual(descriptor, sink.fileno())
            try:
                out.write("日本語\n")
                self.assertTrue(out.drain(2))
            finally:
                out.close(.1)
                out._thread.join(1)
            self.assertFalse(out._thread.is_alive())
            with self.assertRaises(OSError):
                os.fstat(descriptor)
            sink.seek(0)
            self.assertEqual(sink.read(), "日本語\n")

    def held(self, **limits):
        sink = HeldSink()
        out = NonBlockingOut(sink, **limits)

        def finish():
            sink.release.set()
            out.close(.5)
            out._thread.join(1)
            self.assertFalse(out._thread.is_alive(), "owned console thread survived")

        self.addCleanup(finish)
        self.assertEqual(out.write("held\n"), 5)
        self.assertTrue(sink.entered.wait(2), "writer did not reach held sink")
        return sink, out

    def test_blocked_sink_keeps_latest_chunks_in_order(self):
        sink, out = self.held(max_bytes=16, max_chunks=3)
        for i in range(10):
            self.assertEqual(out.write(f"{i}\n"), 2)
        self.assertEqual((out.dropped_chunks, out.dropped_bytes), (7, 14))
        self.assertFalse(out.drain(.02))
        sink.release.set()
        self.assertTrue(out.drain(2))
        self.assertEqual("".join(sink.text), "held\n7\n8\n9\n")

    def test_close_does_not_wait_forever_for_console(self):
        _, out = self.held()
        started = time.monotonic()
        out.close(.02)
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(out.write("late\n"), 5)

    def test_broken_sink_does_not_stop_later_diagnostics(self):
        class Sink(io.StringIO):
            first = True

            def write(self, text):
                if self.first:
                    self.first = False
                    raise OSError("closed console")
                return super().write(text)

        sink = Sink()
        out = NonBlockingOut(sink)
        try:
            out.write("lost\n")
            out.write("retained\n")
            self.assertTrue(out.drain(2))
            self.assertEqual(sink.getvalue(), "retained\n")
        finally:
            out.close(.1)
            out._thread.join(1)
        self.assertFalse(out._thread.is_alive())

    def test_context_wraps_owned_streams_and_restores_them(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.multiple(sys, stdout=stdout, __stdout__=stdout,
                            stderr=stderr, __stderr__=stderr):
            with decoupled_console() as installed:
                self.assertEqual(len(installed), 2)
                print("out")
                print("err", file=sys.stderr)
            self.assertIs(sys.stdout, stdout)
            self.assertIs(sys.stderr, stderr)
        self.assertEqual(stdout.getvalue(), "out\n")
        self.assertEqual(stderr.getvalue(), "err\n")

    def test_context_preserves_replaced_capture_streams(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.multiple(sys, stdout=stdout, stderr=stderr):
            with decoupled_console() as installed:
                self.assertEqual(installed, [])
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)


class ConsoleChildExit(unittest.TestCase):
    def child(self, wrapped, unbuffered=False, server_entry=False):
        script = (
            "import os\n"
            "os.write(2,b'READY\\n')\n"
            "from serve.nonblocking_out import decoupled_console\n"
            "import contextlib\n"
            f"with decoupled_console() if {wrapped!r} else contextlib.nullcontext():\n"
            "    for i in range(256):\n"
            "        print('x'*8192,flush=True)\n"
            "os.write(2,b'DONE\\n')\n"
        )
        if server_entry:
            script = (
                "import os\n"
                "os.write(2,b'READY\\n')\n"
                "from serve import server\n"
                "def body():\n"
                "    for i in range(256):\n"
                "        print('x'*8192,flush=True)\n"
                "    os.write(2,b'DONE\\n')\n"
                "    return 0\n"
                "server._main=body\n"
                "raise SystemExit(server.main())\n"
            )
        env = dict(os.environ)
        env.pop("PYTHONUNBUFFERED", None)
        if server_entry:
            env["STRATA_BLOCKING_CONSOLE"] = "0" if wrapped else "1"
        argv = [sys.executable, "-B", "-X", "faulthandler"]
        if unbuffered:
            argv.append("-u")
        proc = subprocess.Popen([*argv, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertTrue(select.select([proc.stderr], [], [], 3)[0], "child startup deadline")
            self.assertEqual(proc.stderr.readline(), b"READY\n")
            if not wrapped:
                with self.assertRaises(subprocess.TimeoutExpired):
                    proc.wait(timeout=.2)
            else:
                code = proc.wait(timeout=8)
                diagnostics = proc.stderr.read(8192)
                self.assertEqual(code, 0, "console child must exit normally: "
                                 + diagnostics.decode(errors="replace"))
                self.assertIn(b"DONE\n", diagnostics)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=2)
            proc.stdout.close()
            proc.stderr.close()

    def test_unwrapped_unread_pipe_blocks_control(self):
        self.child(False)

    def test_wrapped_unread_pipe_exits_buffered(self):
        self.child(True)

    def test_wrapped_unread_pipe_exits_unbuffered(self):
        self.child(True, unbuffered=True)

    def test_server_entry_unread_pipe_exits_buffered(self):
        self.child(True, server_entry=True)

    def test_server_entry_unread_pipe_exits_unbuffered(self):
        self.child(True, unbuffered=True, server_entry=True)

    def test_server_entry_blocking_bypass_blocks_control(self):
        self.child(False, server_entry=True)

    def test_server_cli_help_preserves_default_and_bypass(self):
        for value in ("0", "1"):
            with self.subTest(blocking=value):
                env = dict(os.environ, STRATA_BLOCKING_CONSOLE=value)
                proc = subprocess.run([sys.executable, "-X", "faulthandler", "-B", "-m",
                    "serve.server", "--help"], env=env, cwd=Path(__file__).resolve().parents[1],
                    capture_output=True, timeout=10)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
                self.assertIn(b"--engine", proc.stdout)
                self.assertIn(b"--api-key", proc.stdout)


if __name__ == "__main__":
    unittest.main()
