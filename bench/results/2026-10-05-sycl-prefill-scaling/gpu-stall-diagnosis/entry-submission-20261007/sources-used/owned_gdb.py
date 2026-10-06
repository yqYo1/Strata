"""Own a diagnostic child through GDB/MI; inspect and resume without root.

This is for initial correctness/debug runs, not performance measurements.
The caller supplies diagnostic_environment(), a private output directory and
the finite job's deadline. Never attach to or signal an unrelated process.
"""
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time


def process_identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


class OwnedGdb:
    def __init__(self, argv, output, env, *, inferior_tty_fd=None):
        # The caller owns a newly allocated PTY and its protocol I/O. GDB's MI
        # channel must remain separate from a serving child's stdin/stdout.
        self.terminal = None
        if inferior_tty_fd is not None:
            if not os.isatty(inferior_tty_fd):
                raise ValueError('inferior_tty_fd is not a terminal')
            self.terminal = os.ttyname(inferior_tty_fd)
        self.output = Path(output)
        self.output.mkdir(exist_ok=True)
        if any((self.output / name).exists() for name in
               ['gdb-mi.stdout', 'engine-and-gdb.stderr', 'inferior.stderr']):
            raise FileExistsError('Refusing to overwrite a previous debugger log')
        self.responses, self.stops, self.inferior = {}, [], None
        self.exit_code, self.exit_signal, self.token = None, None, 0
        self.pending, self.snapshots = bytearray(), []
        self.raw = (self.output / 'gdb-mi.stdout').open('wb')
        self.stderr = (self.output / 'engine-and-gdb.stderr').open('wb')
        # GDB's no-shell startup does not interpret argument quoting. A tiny
        # exec helper preserves the real argv through JSON, including spaces,
        # quotes and literal dollar signs, without involving a shell. Exec
        # keeps the inferior PID and GDB follows the real executable's symbols.
        self.launch = tempfile.TemporaryDirectory(prefix='strata-gdb-args-', dir='/tmp')
        launch = Path(self.launch.name)
        (launch / 'argv.json').write_text(json.dumps({
            'argv': [str(x) for x in argv],
            'stderr': str(self.output.resolve() / 'inferior.stderr') if self.terminal else None}))
        (launch / 'exec.py').write_text(
            'import json,os,sys\n'
            'with open(sys.argv[1]) as f: config=json.load(f)\n'
            'if config["stderr"] is not None:\n'
            ' fd=os.open(config["stderr"],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\n'
            ' os.dup2(fd,2);os.close(fd)\n'
            'argv=config["argv"]\n'
            'os.execvpe(argv[0],argv,os.environ)\n')
        (self.output / 'inferior-argv.json').write_text(json.dumps([str(x) for x in argv], indent=2) + '\n')
        self.child = subprocess.Popen(
            ['/usr/bin/gdb', '-nx', '-q', '--interpreter=mi2'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            env=env, start_new_session=True, bufsize=0)
        self.debugger_identity = process_identity(self.child.pid)
        os.set_blocking(self.child.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.child.stdout, selectors.EVENT_READ)
        try:
            for command in ['-gdb-set pagination off', '-gdb-set confirm off',
                            '-gdb-set startup-with-shell off',
                            '-gdb-set debuginfod enabled off', '-gdb-set mi-async on']:
                self.command(command)
            self.command('-file-exec-and-symbols /usr/bin/python3')
            if self.terminal:
                self.command('-inferior-tty-set ' + self.terminal)
            self.command('-exec-arguments ' + str(launch / 'exec.py') + ' ' + str(launch / 'argv.json'))
        except BaseException:
            self.close()
            raise

    def poll(self, seconds=0):
        for key, _ in self.selector.select(seconds):
            data = os.read(key.fd, 65536)
            if not data:
                self.selector.unregister(key.fileobj)
                continue
            self.raw.write(data)
            self.raw.flush()
            self.pending.extend(data)
            while b'\n' in self.pending:
                line, _, tail = self.pending.partition(b'\n')
                self.pending[:] = tail
                line = line.decode(errors='replace').strip()
                match = re.match(r'(\d+)\^(.*)', line)
                if match:
                    self.responses[int(match[1])] = match[2]
                if line.startswith('=thread-group-started,'):
                    pid = int(re.search(r'\bpid="(\d+)"', line)[1])
                    self.inferior = process_identity(pid)
                if line.startswith('*stopped,'):
                    self.stops.append(line)
                    if 'reason="exited-normally"' in line:
                        self.exit_code = 0
                    elif 'reason="exited"' in line:
                        # GDB/MI reports the inferior's nonzero exit in octal.
                        self.exit_code = int(re.search(r'exit-code="([0-7]+)"', line)[1], 8)
                    elif 'reason="exited-signalled"' in line:
                        self.exit_signal = re.search(r'signal-name="([^"]+)"', line)[1]

    def command(self, command, seconds=10):
        self.token += 1
        token = self.token
        self.child.stdin.write(f'{token}{command}\n'.encode())
        end = time.monotonic() + seconds
        while token not in self.responses and time.monotonic() < end:
            self.poll(min(.1, max(0, end - time.monotonic())))
            if self.child.poll() is not None:
                self.poll(0)
                break
        result = self.responses.pop(token, None)
        if result is None or result.startswith('error'):
            raise RuntimeError(f'GDB command failed: {command}: {result}')
        return result

    def run(self):
        self.command('-exec-run')

    def snapshot(self, label, resume=True):
        if self.exit_code is not None or self.exit_signal is not None:
            return None
        start = self.raw.tell()
        if not self.stops or self.stops[-1] == 'resumed':
            count = len(self.stops)
            self.command('-exec-interrupt --all')
            end = time.monotonic() + 10
            while len(self.stops) == count and time.monotonic() < end:
                self.poll(.1)
            if len(self.stops) == count:
                raise TimeoutError('GDB did not report the inferior stopped')
        if self.exit_code is not None or self.exit_signal is not None:
            return None
        stop = self.stops[-1]
        self.command('-thread-info')
        for text in ['thread apply all bt 48', 'info registers', 'info sharedlibrary']:
            self.command('-interpreter-exec console ' + json.dumps(text))
        self.raw.flush()
        path = self.output / f'{label}.mi.txt'
        with (self.output / 'gdb-mi.stdout').open('rb') as source, path.open('wb') as dest:
            source.seek(start)
            shutil.copyfileobj(source, dest)
        record = {'label': label, 'stop': stop, 'path': str(path), 'resumed': False}
        self.snapshots.append(record)
        # Resume only our requested SIGINT. Preserve real crashes at their first
        # stop instead of letting a later cleanup failure conceal the fault.
        if resume and 'signal-name="SIGINT"' in stop:
            self.command('-exec-continue --all')
            record['resumed'] = True
            self.stops.append('resumed')
        return record

    def close(self):
        cleanup = {'forced': False, 'inferior_survived': False, 'gdb_survived': False}
        try:
            if self.inferior and self.exit_code is None and self.exit_signal is None:
                identity = process_identity(self.inferior['pid'])
                cleanup['forced'] = bool(identity and identity['state'] != 'Z')
            if self.child.poll() is None:
                try:
                    self.command('-gdb-exit', seconds=2)
                except (RuntimeError, OSError):
                    cleanup['forced'] = True
            end = time.monotonic() + 2
            while self.child.poll() is None and time.monotonic() < end:
                self.poll(.05)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if self.child.poll() is not None:
                    break
                cleanup['forced'] = True
                try:
                    os.killpg(self.child.pid, sig)
                except ProcessLookupError:
                    break
                end = time.monotonic() + 1
                while self.child.poll() is None and time.monotonic() < end:
                    self.poll(.05)
            # An inferior can have its own process group. Pin its identity with
            # pidfd before cleanup, rather than relying on the debugger's group.
            if self.inferior:
                identity = process_identity(self.inferior['pid'])
                if identity and identity['start_ticks'] == self.inferior['start_ticks'] and identity['state'] != 'Z':
                    cleanup['forced'] = True
                    try:
                        fd = os.pidfd_open(identity['pid'])
                        again = process_identity(identity['pid'])
                        try:
                            if again and again['start_ticks'] == identity['start_ticks']:
                                signal.pidfd_send_signal(fd, signal.SIGKILL)
                        finally:
                            os.close(fd)
                    except ProcessLookupError:
                        pass
                    end = time.monotonic() + 2
                    while time.monotonic() < end:
                        identity = process_identity(self.inferior['pid'])
                        if not identity or identity['state'] == 'Z':
                            break
                        time.sleep(.05)
                    cleanup['inferior_survived'] = bool(identity and identity['state'] != 'Z')
                    if cleanup['inferior_survived']:
                        (self.output.parent / 'stalled-writer.json').write_text(
                            json.dumps(self.inferior, indent=2) + '\n')
            cleanup['gdb_survived'] = self.child.poll() is None
            if cleanup['gdb_survived'] and not cleanup['inferior_survived']:
                (self.output.parent / 'stalled-writer.json').write_text(
                    json.dumps(self.debugger_identity, indent=2) + '\n')
            self.poll(0)
            return cleanup
        finally:
            self.selector.close()
            self.child.stdin.close()
            self.child.stdout.close()
            self.raw.close()
            self.stderr.close()
            self.launch.cleanup()
