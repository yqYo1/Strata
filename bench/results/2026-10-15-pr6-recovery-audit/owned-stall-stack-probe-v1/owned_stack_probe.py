"""Source proposal: reuse the pinned Main Owner; never own/adopt the inferior.

All process creation, process limits, signals and reaping remain in direct_owner.
This adapter owns only an additional pidfd, never signals it and never calls
Popen.poll/wait/communicate. It runs one GDB sibling via a separate Owner.
"""
from pathlib import Path
import hashlib
import importlib.util
import os
import re
import selectors
import time

OWNER_SHA256 = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
GDB_TEXT_CAP = 256 << 10
GDB_WALL_SECONDS = 30
JOB_CAP = 256 << 20


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_owner(path):
    require(sha(path) == OWNER_SHA256, 'Main Owner pin mismatch')
    spec = importlib.util.spec_from_file_location('main_pinned_direct_owner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def all_tracers(pid):
    """Bounded, conservative snapshot; disappearing task causes fail closed."""
    paths = list(Path('/proc', str(pid), 'task').iterdir())
    require(len(paths) <= 2048, 'native task count cap')
    result = {}
    for task in paths:
        text = (task / 'status').read_text()
        match = re.search(r'^TracerPid:\s*(\d+)\s*$', text, re.M)
        require(match, 'missing task TracerPid')
        result[task.name] = int(match[1])
    require(str(pid) in result, 'native leader task missing')
    return result


class StallCaptured(RuntimeError):
    """Main must stop submitting requests and select its owned shutdown path."""


class StackProbe:
    def __init__(self, owner_module, debugger_owner, io, entry, record,
                 persist, cwd, *, silence_seconds=240, shutdown_payload=None):
        require(silence_seconds > 0, 'positive silence threshold required')
        self.module, self.debugger_owner = owner_module, debugger_owner
        self.io, self.entry, self.record, self.persist = io, entry, record, persist
        self.cwd = Path(cwd)
        self.original_poll, self.original_read, self.original_send = io.poll, io.read, io.send
        self.anchor = dict(entry['direct_identity'])
        require(self.anchor['pid'] == io.proc.pid, 'Owner direct identity mismatch')
        require(self.anchor['ppid'] == os.getpid(), 'native must be direct controller child')
        require(self.anchor['sid'] == self.anchor['pid'], 'native must be owned session leader')
        self.boot = boot_id()
        self.fd = os.pidfd_open(self.anchor['pid'])
        self.last_progress = time.monotonic()
        self.silence_seconds = silence_seconds
        self.shutdown_payload = shutdown_payload
        self.attempted = False
        self.requests_stopped = False
        self.capture_active = False
        self.record.update(owner_sha256=OWNER_SHA256, boot_id=self.boot,
                           direct_identity=self.anchor, silence_seconds=silence_seconds,
                           target_pidfd_held=True, state='armed', captures=[])
        try:
            self.snapshot(alive=True)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise
        io.poll, io.read, io.send = self.poll, self.read, self.send
        self.persist()

    def snapshot(self, *, alive=False):
        require(self.fd is not None, 'target pin closed')
        require(self.io.proc.returncode is None, 'native was reaped/polled outside Owner')
        require(boot_id() == self.boot, 'boot changed')
        now = self.module.identity(self.anchor['pid'])
        require(now is not None, 'unreaped native identity disappeared')
        for key in ('pid', 'start_ticks', 'ppid', 'pgid', 'sid'):
            require(now[key] == self.anchor[key], 'native identity changed: ' + key)
        if alive:
            require(not self.io.exit_watch.poll(0), 'native exited before capture')
            require(now['state'] not in ('Z', 'X', 'T', 't'), 'native already dead/stopped')
        return dict(boot_id=self.boot, identity=now, tracers=all_tracers(now['pid']))

    def read(self):
        before = self.io.stdout.tell()
        self.original_read()
        if self.io.stdout.tell() != before:
            self.last_progress = time.monotonic()

    def require_requests_allowed(self):
        require(not self.requests_stopped, 'new requests forbidden after first stall')

    def send(self, data, seconds=20):
        if self.requests_stopped:
            require(self.shutdown_payload is not None and data == self.shutdown_payload,
                    'only the predeclared native shutdown payload allowed after stall')
        return self.original_send(data, seconds=seconds)

    def poll(self):
        self.original_poll()
        if self.capture_active or self.attempted:
            return
        elapsed = time.monotonic() - self.last_progress
        if elapsed >= self.silence_seconds:
            self.requests_stopped = True
            self.record['requests_stopped'] = True
            self.capture_now('first_stdout_silence', silence_seconds=elapsed)
            raise StallCaptured('one stack snapshot completed; no further requests')

    def capture_now(self, reason, *, silence_seconds=None):
        require(not self.attempted, 'only one stack snapshot permitted')
        self.attempted, self.capture_active = True, True
        capture = dict(reason=reason, silence_seconds=silence_seconds, started_utc=self.module.utc(),
                       state='capturing', complete=False, tracer_clear=False,
                       resumed_proven=False,
                       limits=dict(wall_seconds=GDB_WALL_SECONDS, cpu_seconds=30,
                                   rss_bytes=2 << 30, text_bytes=GDB_TEXT_CAP,
                                   file_bytes=GDB_TEXT_CAP, job_bytes=JOB_CAP))
        self.record['captures'].append(capture)
        # GDB receives a minimal independent environment, never native LD_PRELOAD.
        debugger_env = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C',
                            HOME=str(self.debugger_owner.output.parent), DEBUGINFOD_URLS='')
        argv = ['/usr/bin/gdb', '-nx', '-nh', '-batch',
                '-iex', 'set auto-load off', '-iex', 'set debuginfod enabled off',
                '-iex', 'set may-call-functions off',
                '-iex', 'set startup-with-shell off',
                '-ex', 'set pagination off', '-ex', 'set confirm off',
                '-ex', 'set print frame-arguments none', '-ex', 'set print elements 16',
                '-ex', 'attach ' + str(self.anchor['pid']),
                '-ex', 'echo OWNED_PROBE_ATTACHED\\n', '-ex', 'info inferiors',
                '-ex', 'info threads', '-ex', 'thread apply all bt 12',
                '-ex', 'detach', '-ex', 'echo OWNED_PROBE_DETACHED\\n', '-ex', 'quit']
        error = None
        try:
            capture['before'] = self.snapshot(alive=True)
            require(not any(capture['before']['tracers'].values()), 'native already traced')
            self.persist()

            def collect(debug_io, debug_entry):
                # The original native Owner continues enforcing its wall/RSS/log/job
                # budgets even while this synchronous nested debugger Owner is running.
                debug_io.proc.stdin.close()
                with selectors.DefaultSelector() as ready:
                    ready.register(debug_io.proc.stdout, selectors.EVENT_READ)
                    while not debug_io.exit_watch.poll(0) or not debug_io.eof:
                        self.original_poll()
                        debug_io.poll()
                        if ready.select(.02):
                            debug_io.read()
                            debug_io.pending.clear()

            command, so, se = self.debugger_owner.run(
                'stack-once', argv, debugger_env, self.cwd, interaction=collect,
                wall=GDB_WALL_SECONDS, cpu=30, text_cap=GDB_TEXT_CAP,
                rss_cap=2 << 30, total_cap=JOB_CAP, file_cap=GDB_TEXT_CAP)
            capture['debugger'] = command
            require(command.get('direct_identity', {}).get('ppid') == os.getpid(),
                    'GDB is not direct controller child')
            require(command.get('direct_identity', {}).get('sid') != self.anchor['sid'],
                    'GDB must have separate owned session')
            require(self.module.closed(command) and command['exit_code'] == 0,
                    'GDB failed, timed out, exceeded budget, or needed cleanup')
            output = so.read_text(errors='replace')
            require('OWNED_PROBE_ATTACHED\n' in output and
                    'OWNED_PROBE_DETACHED\n' in output, 'missing GDB capture markers')
            require(re.search(r'\bprocess\s+' + str(self.anchor['pid']) + r'\b', output),
                    'GDB did not identify the pinned native PID')
            capture['backtrace_thread_ids'] = re.findall(r'^Thread (\d+)\b', output, re.M)
            require(capture['backtrace_thread_ids'], 'missing all-thread backtrace')
            require(so.stat().st_size + se.stat().st_size <= GDB_TEXT_CAP,
                    'GDB output exceeded finite cap')
            capture['logs'] = {str(p): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in (so, se)}
        except BaseException as exc:
            error = exc
            capture['error'] = repr(exc)
        finally:
            # Owner.run has already closed/reaped GDB. Never try another attach,
            # numeric signal, ptrace detach, PR_SET_PTRACER_ANY or target adoption.
            # Check all native task TracerPid fields, not just the leader.
            try:
                after = self.snapshot()
                deadline = time.monotonic() + 3
                while any(after['tracers'].values()) and time.monotonic() < deadline:
                    self.original_poll()
                    time.sleep(.02)
                    after = self.snapshot()
                capture['after'] = after
                capture['tracer_clear'] = not any(after['tracers'].values())
                require(capture['tracer_clear'], 'native still has a tracer after GDB closure')
                require(after['identity']['state'] not in ('T', 't'),
                        'native remains stopped after GDB closure')
                capture['native_exited_during_capture'] = bool(self.io.exit_watch.poll(0))
            except BaseException as exc:
                capture['closure_error'] = repr(exc)
                if error is None:
                    error = exc
            self.capture_active = False
            capture['finished_utc'] = self.module.utc()
            capture['complete'] = error is None
            capture['state'] = 'captured' if error is None else 'incomplete'
            self.record['state'] = capture['state']
            self.persist()
        if error is not None:
            # Propagate to Main's interaction. Its existing native Owner remains
            # responsible for safely terminating/closing the original native.
            raise RuntimeError('stack diagnostic incomplete; no new requests') from error
        return capture

    def close(self):
        # Invoke inside interaction finally, before Owner reaps the native.
        self.io.poll, self.io.read, self.io.send = self.original_poll, self.original_read, self.original_send
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.record['target_pidfd_held'] = False
            self.persist()
