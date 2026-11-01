#!/usr/bin/env python3
"""Actual Linux child/pidfd checks for bounded /proc exit observation; MIT."""
import contextlib
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/sycl'))
from proc_observer import ProcObserver

CHILD = '''
import ctypes,signal,sys
signal.alarm(5)
print('READY',flush=True)
assert sys.stdin.buffer.read(1)==b'X'
libc=ctypes.CDLL(None,use_errno=True)
libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
assert libc.prctl(4,0,0,0,0)==0
print('UNREADABLE',flush=True)
assert sys.stdin.buffer.read(1)==b'Y'
'''


@contextlib.contextmanager
def child():
    proc = subprocess.Popen([sys.executable, '-I', '-c', CHILD], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        env=dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1'))
    fd = os.pidfd_open(proc.pid)
    watch = select.poll()
    watch.register(fd, select.POLLIN)
    try:
        assert select.select([proc.stdout], [], [], 3)[0], 'child readiness deadline'
        assert proc.stdout.readline() == b'READY\n'
        yield proc, watch
    finally:
        # This child runs the fixed source above and never creates descendants.
        if not watch.poll(0):
            proc.kill()
        proc.wait(timeout=3)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()
        os.close(fd)


def unreadable(proc):
    proc.stdin.write(b'X')
    proc.stdin.flush()
    assert select.select([proc.stdout], [], [], 3)[0], 'child /proc transition deadline'
    assert proc.stdout.readline() == b'UNREADABLE\n'


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'), 'Linux pidfd required')
class ExitObservation(unittest.TestCase):
    def test_permission_denial_is_not_exit(self):
        with child() as (proc, watch):
            observer = ProcObserver(.05)
            observe = lambda: Path('/proc', str(proc.pid), 'environ').read_bytes()
            self.assertTrue(observer.sample(observe, lambda: bool(watch.poll(0))))
            unreadable(proc)
            self.assertFalse(watch.poll(0))
            self.assertFalse(observer.sample(observe, lambda: bool(watch.poll(0))))
            self.assertFalse(observer.exit_confirmed_after_unavailable)
            deadline = time.monotonic() + 1
            with self.assertRaisesRegex(RuntimeError, 'remains live'):
                while time.monotonic() < deadline:
                    observer.sample(observe, lambda: bool(watch.poll(0)))
                    time.sleep(.002)
            self.assertFalse(watch.poll(0))
            self.assertGreater(observer.permission_denials, 0)

    def test_exact_child_exit_ends_unreadable_gap(self):
        with child() as (proc, watch):
            observer = ProcObserver()
            observe = lambda: Path('/proc', str(proc.pid), 'maps').read_text()
            self.assertTrue(observer.sample(observe, lambda: bool(watch.poll(0))))
            unreadable(proc)
            self.assertFalse(observer.sample(observe, lambda: bool(watch.poll(0))))
            proc.stdin.write(b'Y')
            proc.stdin.flush()
            deadline = time.monotonic() + 3
            while not watch.poll(0) and time.monotonic() < deadline:
                observer.sample(observe, lambda: bool(watch.poll(0)))
                time.sleep(.002)
            self.assertTrue(watch.poll(0))
            self.assertFalse(observer.sample(observe, lambda: bool(watch.poll(0))))
            self.assertTrue(observer.exit_confirmed_after_unavailable)
            self.assertEqual(proc.wait(timeout=1), 0)

    def test_observation_failure_is_not_masked(self):
        observer = ProcObserver()
        def bad_provider():
            raise RuntimeError('unplanned provider')
        with self.assertRaisesRegex(RuntimeError, 'unplanned provider'):
            observer.sample(bad_provider, lambda: True)
        self.assertEqual(observer.successful_samples, 0)

    def test_missing_file_requires_exit_or_finite_grace(self):
        clock = [0.]
        observer = ProcObserver(.5, clock=lambda: clock[0])
        def absent():
            raise FileNotFoundError('maps')
        self.assertFalse(observer.sample(absent, lambda: False))
        self.assertFalse(observer.exit_confirmed_after_unavailable)
        clock[0] = .5
        with self.assertRaisesRegex(RuntimeError, 'remains live'):
            observer.sample(absent, lambda: False)
        self.assertFalse(observer.sample(absent, lambda: True))
        self.assertTrue(observer.exit_confirmed_after_unavailable)

    def test_grace_cannot_be_unbounded(self):
        for value in (0, -1, 5.01, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ProcObserver(value)


if __name__ == '__main__':
    unittest.main()
