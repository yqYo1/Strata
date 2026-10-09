"""SOURCE ONLY / UNTESTED. Root executes this serially; no model or GPU.

Compile, collect and render are separate supervised children under one lock.
No application SIGPROF handler, timer, ptrace, perf, preload or host changes.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

DEADLINE = 60
WORK_DEADLINE = 57  # reserve three seconds for exact-owned termination/reaping
BUDGET = 64 * 1024**2
SOURCE = Path(__file__).resolve().with_name('native_target.c')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(fields[19]), 'ppid': int(fields[1]),
                'state': fields[0]}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def same(row):
    now = identity(row['pid'])
    return now if now and now['start_ticks'] == row['start_ticks'] else None


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Supervisor:
    def __init__(self, out, compiler, gprofng):
        self.out, self.compiler, self.gprofng = out, compiler, gprofng
        self.start = time.monotonic()
        self.owners, self.pidfds, self.children = {}, {}, []
        self.pending = bytearray()
        self.record = {'version': 2, 'active': True, 'passed': False,
                       'source_status': 'SOURCE ONLY / UNTESTED until this receipt closes',
                       'runtime_gate': 'open', 'started_utc': utc(),
                       'deadline_seconds': DEADLINE, 'combined_budget_bytes': BUDGET,
                       'gpu_executed': False, 'model_opened': False,
                       'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                       'controller_sha256': sha(__file__), 'source_sha256': sha(SOURCE),
                       'compiler': str(compiler), 'compiler_sha256': sha(compiler),
                       'gprofng': str(gprofng), 'gprofng_sha256': sha(gprofng),
                       'environment': {k: v for k, v in os.environ.items() if k in
                                       {'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LANG', 'LC_ALL',
                                        'LC_NUMERIC', 'LC_CTYPE', 'TZ', 'GLIBC_TUNABLES',
                                        'SP_COLLECTOR_PARAMS', 'SP_COLLECTOR_FOLLOW'}},
                       'environment_scope': 'Loader, locale and collector controls; unrelated environment values omitted.',
                       'uname': list(os.uname()),
                       'cpu_description': Path('/proc/cpuinfo').read_text().split('\n\n', 1)[0],
                       'owners': self.owners, 'steps': [], 'protocol': [], 'cleanup': []}
        self.record['collector_library'] = {}
        lib = Path('/usr/lib/x86_64-linux-gnu/gprofng/libgp-collector.so')
        if lib.is_file():
            self.record['collector_library'] = {'path': str(lib), 'sha256': sha(lib)}

    def save(self):
        self.record['elapsed_seconds'] = time.monotonic() - self.start
        tmp = self.out / 'record.json.tmp'
        tmp.write_text(json.dumps(self.record, indent=2) + '\n')
        tmp.replace(self.out / 'record.json')

    def own(self, row):
        key = str(row['pid'])
        if key in self.owners:
            assert self.owners[key]['start_ticks'] == row['start_ticks'], 'PID reuse'
            return
        # pidfd prevents check/kill PID reuse races. Require Linux/Python support.
        fd = os.pidfd_open(row['pid'])
        assert same(row), 'identity changed while opening pidfd'
        self.owners[key] = row
        self.pidfds[key] = fd

    def discover(self):
        rows = [row for p in Path('/proc').iterdir() if p.name.isdigit()
                and (row := identity(int(p.name)))]
        changed = True
        while changed:
            changed = False
            for row in rows:
                parent = self.owners.get(str(row['ppid']))
                if str(row['pid']) not in self.owners and parent and same(parent):
                    try:
                        self.own(row)
                        changed = True
                    except ProcessLookupError:
                        pass

    def poll(self):
        self.discover()
        self.save()
        assert time.monotonic() - self.start < WORK_DEADLINE, 'work deadline; cleanup reserve reached'
        size = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        self.record['observed_combined_bytes'] = size
        assert size < BUDGET, '64-MiB combined text/binary/experiment budget exceeded'

    def launch(self, name, argv):
        self.poll()
        step = {'name': name, 'argv': argv, 'started_utc': utc()}
        self.record['steps'].append(step)
        with (self.out / (name + '.stdout')).open('wb') as stdout, \
                (self.out / (name + '.stderr')).open('wb') as stderr:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr,
                                     start_new_session=True)
        self.children.append((child, step))
        row = identity(child.pid)
        assert row, 'child disappeared before ownership was recorded'
        self.own(row)
        self.save()
        return child

    def wait(self, child):
        while child.poll() is None:
            self.poll()
            time.sleep(.02)
        self.poll()
        for proc, step in self.children:
            if proc is child:
                step['exit_code'] = child.returncode
                step['finished_utc'] = utc()
        self.save()
        assert child.returncode == 0, 'child failed: ' + str(child.returncode)

    def read_line(self, child, reply):
        while True:
            # Drain first even if collector/target has already exited. v1 checked
            # exit first and could reject a BYE already buffered in the FIFO.
            while True:
                try:
                    data = os.read(reply, 4096)
                except BlockingIOError:
                    break
                if not data:
                    break
                self.pending.extend(data)
                assert len(self.pending) < 4096, 'unexpected oversized FIFO protocol'
            if b'\n' in self.pending:
                value, _, tail = self.pending.partition(b'\n')
                self.pending[:] = tail
                line = value.decode('ascii')
                self.record['protocol'].append(line)
                self.save()
                return line
            self.poll()
            if child.poll() is not None:
                # Last drain in the next iteration after observing exit closes
                # the writer-versus-exit race; the current drain may predate exit.
                try:
                    data = os.read(reply, 4096)
                except BlockingIOError:
                    data = b''
                if data:
                    self.pending.extend(data)
                    continue
                raise RuntimeError('collector exited with no complete FIFO reply: '
                                   + str(child.returncode))
            time.sleep(.02)

    def toggle(self, action, row, binary):
        now = same(row)
        assert now and now['state'] != 'Z', 'target identity no longer live'
        assert Path('/proc', str(row['pid']), 'exe').resolve() == binary.resolve(), \
            'READY PID is not the compiled native target'
        status = Path('/proc', str(row['pid']), 'status').read_text()
        caught = int(re.search(r'^SigCgt:\s*([0-9a-fA-F]+)$', status, re.M)[1], 16)
        assert caught & (1 << (signal.SIGUSR2 - 1)), 'collector SIGUSR2 handler absent'
        signal.pidfd_send_signal(self.pidfds[str(row['pid'])], signal.SIGUSR2)
        self.record['steps'].append({'action': action, 'target': now,
                                     'signal': 'SIGUSR2', 'sigcgt': hex(caught)})
        self.save()

    def evidence(self):
        experiment = self.out / 'result.er'
        xml = experiment / 'log.xml'
        if not xml.is_file():
            self.record['profile_validation'] = {'error': 'missing log.xml'}
            return False
        raw = xml.read_bytes()
        # Installed gprofng can pad its XML stream with trailing NUL bytes.
        # Strip only that tail; an embedded NUL remains a framing failure.
        unpadded = raw.rstrip(b'\0')
        self.record['collector_xml'] = {'sha256': sha(xml), 'raw_bytes': len(raw),
                                        'trailing_nul_padding_bytes': len(raw) - len(unpadded)}
        assert b'\0' not in unpadded, 'embedded NUL in collector XML'
        text = unpadded.decode('utf-8')
        # gprofng log.xml contains successive XML fragments rather than one root.
        text = re.sub(r'<\?xml[^>]*\?>', '', text)
        tree = ET.fromstring('<root>' + text + '</root>')
        errors = [{'attributes': e.attrib, 'text': ''.join(e.itertext())}
                  for e in tree.iter('event') if e.get('kind') == 'cerror']
        profiles = [ET.tostring(p, encoding='unicode') for p in tree.iter('profile')
                    if 'ptimer' in p.attrib.values()]
        dataptrs = [p.attrib for p in tree.iter('dataptr')]
        sizes = {p.name: p.stat().st_size for p in experiment.iterdir()
                 if p.is_file() and p.name.startswith('data.')}
        frame_ok = sizes.get('data.frameinfo', 0) > 0
        # ptimer writes data.profile; require the declared profile data stream,
        # not incidental sample/metadata files.
        event_ok = sizes.get('data.profile', 0) > 0
        self.record['profile_validation'] = {
            'collector_errors': errors, 'ptimer_profiles': profiles,
            'dataptrs': dataptrs, 'data_file_sizes': sizes,
            'frame_nonempty': frame_ok, 'profile_events_nonempty': event_ok,
            'xml_sha256': sha(xml)}
        return not errors and bool(profiles) and frame_ok and event_ok

    def attribution(self):
        text = (self.out / 'functions.stdout').read_text(errors='replace')
        values = []
        for line in text.splitlines():
            if re.search(r'\bstrata_native_busy\b', line):
                # Explicit e.user metric: first field must be finite numeric
                # CPU seconds. Never infer samples from Total or report length.
                match = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s+', line)
                if match:
                    values.append({'line': line, 'exclusive_user_cpu_seconds':
                                   float(match[1])})
        self.record['native_attribution'] = values
        return any(v['exclusive_user_cpu_seconds'] > 0 for v in values)

    def cleanup(self):
        self.discover()
        for sig, grace in [(signal.SIGTERM, 2), (signal.SIGKILL, 1)]:
            for key, row in reversed(list(self.owners.items())):
                now = same(row)
                if not now or now['state'] == 'Z':
                    continue
                try:
                    signal.pidfd_send_signal(self.pidfds[key], sig)
                    self.record['cleanup'].append({'identity': row, 'signal': sig.name})
                except ProcessLookupError:
                    pass
            until = time.monotonic() + grace
            while time.monotonic() < until:
                for child, _ in self.children:
                    child.poll()
                if not any((now := same(r)) and now['state'] != 'Z'
                           for r in self.owners.values()):
                    break
                time.sleep(.02)
        for child, step in self.children:
            step['exit_code'] = child.poll()
        self.record['survivors'] = [r for r in self.owners.values()
                                   if (now := same(r)) and now['state'] != 'Z']
        for fd in self.pidfds.values():
            os.close(fd)
        self.record['passed'] &= not self.record['survivors'] and not self.record['cleanup']


def main():
    if not __debug__:
        raise RuntimeError('run without Python -O; acceptance assertions are required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True,
                        help='root-owned B containing owned-v0141-measurement.lock')
    parser.add_argument('--run-name', default='gprofng-native-protocol-smoke-v2')
    parser.add_argument('--compiler', type=Path, default=Path('/usr/bin/cc'))
    parser.add_argument('--gprofng', type=Path, default=Path('/usr/bin/gprofng'))
    args = parser.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', args.run_name), 'unsafe run name'
    base = args.base.resolve(strict=True)
    assert base.is_dir() and base.stat().st_uid == os.getuid(), 'base must be owned'
    # Lock is held through compilation, collection, display and owned cleanup.
    lockfd = os.open(base / 'owned-v0141-measurement.lock',
                     os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lockfd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = base / args.run_name
        out.mkdir(mode=0o700)  # fresh-only, never overwrite historical evidence
        supervisor = Supervisor(out, args.compiler.resolve(strict=True),
                                args.gprofng.resolve(strict=True))
        supervisor.record['measurement_lock'] = str(base / 'owned-v0141-measurement.lock')
        supervisor.record['controller_identity'] = identity(os.getpid())
        request = reply = None
        stopping = False
        def interrupted(signum, frame):
            supervisor.record.setdefault('controller_signals', []).append(signum)
            if not stopping:
                raise RuntimeError('controller interrupted: ' + str(signum))
        old_handlers = {sig: signal.signal(sig, interrupted)
                        for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            # Root's lock is the coordination authority. Refuse visible known
            # competing workloads too; never signal or change those processes.
            prohibited = {'strata', 'strata-xe-health', 'gdb', 'ninja', 'vtune',
                          'gprofng', 'gp-collect-app'}
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    name = (proc / 'comm').read_text().strip()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                assert name not in prohibited, 'competing workload: ' + proc.name + ' ' + name
            for name in ('request', 'reply'):
                os.mkfifo(out / name, 0o600)
            request = os.open(out / 'request', os.O_RDWR | os.O_NONBLOCK)
            reply = os.open(out / 'reply', os.O_RDWR | os.O_NONBLOCK)
            binary = out / 'native_target'
            compile_argv = [str(supervisor.compiler), '-g', '-O2', '-fno-omit-frame-pointer',
                            '-std=c11', '-Wall', '-Wextra', str(SOURCE), '-o', str(binary)]
            supervisor.wait(supervisor.launch('compile', compile_argv))
            supervisor.record['binary_sha256'] = sha(binary)
            argv = [str(supervisor.gprofng), 'collect', 'app', '-p', '10m', '-S', 'off',
                    '-F', 'off', '-a', 'off', '-y', 'USR2', '-o', str(out / 'result.er'),
                    str(binary), str(out / 'request'), str(out / 'reply')]
            child = supervisor.launch('collector', argv)
            ready = supervisor.read_line(child, reply)
            assert re.fullmatch(r'READY [0-9]+', ready), 'invalid READY'
            supervisor.discover()
            row = identity(int(ready.split()[1]))
            assert row and str(row['pid']) in supervisor.owners, 'READY PID not owned child'
            assert same(supervisor.owners[str(row['pid'])]), 'READY identity mismatch'
            supervisor.record['target_identity'] = row
            supervisor.toggle('resume', row, binary)
            assert os.write(request, b'RUN\n') == 4
            assert re.fullmatch(r'DONE [0-9]+', supervisor.read_line(child, reply)), 'invalid DONE'
            supervisor.toggle('pause', row, binary)
            assert os.write(request, b'QUIT\n') == 5
            assert supervisor.read_line(child, reply) == 'BYE', 'missing BYE'
            supervisor.wait(child)
            assert not supervisor.pending, 'extra protocol data'
            valid = supervisor.evidence()
            # Render even on invalid collection so failure evidence remains useful.
            display = [str(supervisor.gprofng), 'display', 'text', '-limit', '0',
                       '-metrics', 'e.user', '-functions', str(out / 'result.er')]
            supervisor.wait(supervisor.launch('functions', display))
            attributed = supervisor.attribution()
            assert valid, 'collector errors, missing ptimer schema, or empty frame/event data'
            assert attributed, 'no positive native busy-function CPU-time attribution'
            supervisor.record['passed'] = True
        except BaseException as error:
            supervisor.record['error'] = repr(error)
        finally:
            stopping = True
            try:
                supervisor.cleanup()
            except BaseException as error:
                supervisor.record['cleanup_error'] = repr(error)
                supervisor.record['passed'] = False
            # Retain available XML/debug/error evidence even when an earlier stage
            # failed. No privileged probes and no deletion/compression here.
            if 'profile_validation' not in supervisor.record:
                try:
                    supervisor.evidence()
                except BaseException as error:
                    supervisor.record['evidence_error'] = repr(error)
            for fd in (request, reply):
                if fd is not None:
                    os.close(fd)
            supervisor.record['active'] = False
            if supervisor.record.get('controller_signals'):
                supervisor.record['passed'] = False
            supervisor.record['finished_utc'] = utc()
            supervisor.record['runtime_gate'] = 'passed' if supervisor.record['passed'] else 'rejected'
            supervisor.record['retention'] = {
                'owner': 'root gprofng native smoke v2', 'byte_budget': BUDGET,
                'consumer': 'ptimer usability and unresolved collector errno diagnosis',
                'next_review': 'root reviews terminal receipt and extracts compact evidence',
                'raw_deleted': False}
            supervisor.save()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        print(json.dumps({k: supervisor.record.get(k) for k in
                          ('passed', 'error', 'cleanup_error', 'survivors', 'elapsed_seconds')}))
        return 0 if supervisor.record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
