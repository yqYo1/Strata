#!/bin/bash
# Read-only by default. Linux DRM recovery requires all device clients to exit.
# See docs/INTEL_GPU_RECOVERY.md. No sudo inside this script, no firmware flash.
set -eu
exec /usr/bin/python3 - "$@" <<'PY'
import argparse
import datetime
import errno
import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time

BDF_PATTERN = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
FAULT = re.compile(r"page fault|fault response|timestamp stuck|wedg|engine reset|gt reset|"
                   r"guc.*(?:timeout|timed out|failed)|memory.*cat|device.*lost|"
                   r"Timed out wait for G2H|Check job timeout|Timedout job|"
                   r"Schedule disable failed to respond|trying reset|"
                   r"\breset (?:queued|started|done)\b", re.I)


class RecoveryError(RuntimeError):
    pass


class RecoveryRefused(RecoveryError):
    """No reset may be attempted until the blocking condition is resolved."""
    pass


class Runner:
    """Never wait indefinitely for a process trapped in uninterruptible sleep."""
    def __init__(self, output):
        self.output = output
        self.calls = []
        self.stranded = False

    def run(self, label, argv, seconds=20, env=None):
        out = self.output / (label + '.stdout')
        err = self.output / (label + '.stderr')
        started = time.monotonic()
        with out.open('wb') as stdout, err.open('wb') as stderr:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout,
                                     stderr=stderr, start_new_session=True, env=env)
            interrupted = None
            try:
                deadline = started + seconds
                while child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
            except BaseException as e:
                interrupted = e
            timed_out = child.poll() is None
            if timed_out:
                for sig, grace in [(signal.SIGTERM, 1), (signal.SIGKILL, 1)]:
                    try:
                        os.killpg(child.pid, sig)
                    except ProcessLookupError:
                        break
                    until = time.monotonic() + grace
                    while child.poll() is None and time.monotonic() < until:
                        time.sleep(0.05)
            alive = child.poll() is None
            self.stranded |= alive
        receipt = dict(label=label, argv=argv, pid=child.pid, exit_code=child.returncode,
                       timed_out=timed_out, still_alive=alive,
                       elapsed_seconds=time.monotonic() - started)
        self.calls.append(receipt)
        if alive:
            raw = Path(f'/proc/{child.pid}/stat').read_text()
            ticks = int(raw[raw.rfind(')') + 2:].split()[19])
            (self.output.parent / 'stalled-writer.json').write_text(json.dumps(dict(pid=child.pid, start_ticks=ticks)))
            raise RecoveryRefused(f'{label}: PID {child.pid} did not exit after SIGKILL; '
                                'no further reset will be attempted')
        if interrupted:
            raise interrupted
        if timed_out or child.returncode != 0:
            raise RecoveryError(f'{label}: failed; see {out} and {err}')
        return out.read_text(errors='replace')

    def write(self, label, path, value):
        # Opening/writing sysfs can itself block in the driver. Isolate that task.
        code = 'import sys; f=open(sys.argv[1],"w"); f.write(sys.argv[2]); f.close()'
        return self.run(label, ['/usr/bin/python3', '-c', code, str(path), value])


class Device:
    def __init__(self, bdf, sysroot=Path('/sys'), procroot=Path('/proc')):
        self.bdf, self.sysroot, self.procroot = bdf, sysroot, procroot
        self.path = sysroot / 'bus/pci/devices' / bdf
        self.driver = sysroot / 'bus/pci/drivers/xe'

    def read(self, name, fallback='unavailable'):
        try:
            return (self.path / name).read_text().strip()
        except FileNotFoundError:
            return fallback

    def describe(self):
        if not self.path.is_dir() or self.read('vendor') != '0x8086':
            raise RecoveryError('Selected PCI function is not an Intel GPU')
        if self.read('device') not in ('0xe20c', '0xe209', '0xe20b'):
            raise RecoveryError('This helper is restricted to Arc B570/B580 PCI IDs')
        if (self.path / 'driver').exists() and (self.path / 'driver').resolve() != self.driver.resolve():
            raise RecoveryError('Selected function is bound to a driver other than xe')
        statuses = {p.parent.name: p.read_text().strip()
                    for p in (self.path / 'drm').glob('card*/*/status')}
        return dict(bdf=self.bdf, vendor=self.read('vendor'), device=self.read('device'),
                    subsystem_vendor=self.read('subsystem_vendor'),
                    subsystem_device=self.read('subsystem_device'),
                    driver='xe' if (self.path / 'driver').exists() else 'unbound',
                    boot_vga=self.read('boot_vga'), connectors=statuses,
                    reset_methods=self.read('reset_method'),
                    survivability_mode=self.read('survivability_mode'),
                    runtime_status=self.read('power/runtime_status'))

    def clients_unbounded(self):
        # xpu-smi ps omits clients with no active engines. Inspect descriptors AND
        # surviving mappings; a closing D-state process can have neither left.
        node_names = {p.name for p in (self.path / 'drm').glob('*')
                      if re.fullmatch(r'(card\d+|renderD\d+)', p.name)}
        device_numbers = set()
        for n in node_names:
            value = (self.path / 'drm' / n / 'dev').read_text().strip().split(':')
            device_numbers.add(os.makedev(*map(int, value)))
        found, unreadable = [], []
        for p in self.procroot.iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                raw = (p / 'stat').read_text()
                tail = raw[raw.rfind(')') + 2:].split()
                comm = (p / 'comm').read_text().strip()
                reasons = set()
                for f in (p / 'fd').iterdir():
                    try:
                        info = f.stat()
                        if stat.S_ISCHR(info.st_mode) and info.st_rdev in device_numbers:
                            reasons.add('open GPU descriptor')
                        fdinfo = (p / 'fdinfo' / f.name).read_text()
                        if re.search(r'^drm-pdev:\s*' + re.escape(self.bdf) + r'\s*$', fdinfo, re.M):
                            reasons.add('DRM PCI descriptor')
                    except FileNotFoundError:
                        pass
                mappings = (p / 'maps').read_text()
                if any('/dev/dri/' + n in mappings for n in node_names):
                    reasons.add('GPU memory mapping')
                if tail[0] == 'D' and os.geteuid() == 0:
                    stack = (p / 'stack').read_text()
                    if re.search(r'\bxe_[a-z_]+|\bdrm_release', stack):
                        reasons.add('blocked GPU close/driver stack')
                if reasons:
                    found.append(dict(pid=int(p.name), comm=comm, state=tail[0],
                                      start_ticks=int(tail[19]), reasons=sorted(reasons)))
            except (FileNotFoundError, ProcessLookupError):
                pass
            except PermissionError:
                if p.exists():
                    unreadable.append(int(p.name))
        return found, unreadable

    def clients(self):
        # /proc/maps or stack inspection can block on a kernel lock too. A
        # separate child bounds inspection, without leaving a waiting thread.
        fd, name = tempfile.mkstemp(prefix='strata-xe-clients-')
        os.close(fd)
        result = Path(name)
        pid = os.fork()
        if pid == 0:
            try:
                result.write_text(json.dumps(self.clients_unbounded()))
                os._exit(0)
            except BaseException:
                os._exit(1)
        deadline = time.monotonic() + 10
        try:
            while time.monotonic() < deadline:
                ended, status = os.waitpid(pid, os.WNOHANG)
                if ended:
                    if os.waitstatus_to_exitcode(status):
                        raise RecoveryRefused('GPU client inspection failed')
                    return json.loads(result.read_text())
                time.sleep(0.05)
            os.kill(pid, signal.SIGKILL)
            # SIGKILL cannot immediately end a D-state task. Do not wait for it.
            raise RecoveryRefused(f'GPU client inspection timed out (PID {pid}); reset refused')
        finally:
            result.unlink(missing_ok=True)


def stop_client(device, pid):
    current, _ = device.clients()
    record = next((r for r in current if r['pid'] == pid), None)
    if not record:
        raise RecoveryError(f'PID {pid} is not a verified client of this GPU')
    # pidfd pins the process identity: never signal a recycled numeric PID.
    fd = os.pidfd_open(pid)
    try:
        again, _ = device.clients()
        if not any(r['pid'] == pid and r['start_ticks'] == record['start_ticks'] for r in again):
            raise RecoveryError(f'PID {pid} changed while inspecting it')
        for sig, grace in [(signal.SIGTERM, 3), (signal.SIGKILL, 2)]:
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if not (device.procroot / str(pid)).exists():
                    return
                time.sleep(0.05)
    finally:
        os.close(fd)
    raise RecoveryError(f'PID {pid} is still present; reset refused')


def require_idle(device):
    d = device.describe()
    if d['boot_vga'] != '0' or any(x == 'connected' for x in d['connectors'].values()):
        raise RecoveryRefused('GPU is a boot/display device; recovery requires closing its display session first')
    if d['survivability_mode'] not in ('unavailable', '0'):
        raise RecoveryRefused('Firmware survivability mode is active; rebind/FLR is not the firmware recovery procedure')
    clients, unreadable = device.clients()
    if clients or unreadable:
        raise RecoveryRefused('GPU clients or uninspectable processes remain; see report.json')


def reset(device, runner, method):
    require_idle(device)
    methods = device.read('reset_method').split()
    if method != 'rebind' and (method not in methods or not (device.path / 'reset').exists()):
        raise RecoveryError(f'{method} reset is not supported')
    if method == 'bus':
        # A secondary-bus reset must not touch another device. The actual B570
        # is alone behind 04:01.0; its audio endpoint is behind 04:02.0 instead.
        endpoints = {p.name for p in device.path.resolve().parent.iterdir()
                     if BDF_PATTERN.fullmatch(p.name)}
        if endpoints != {device.bdf}:
            raise RecoveryRefused('Bus reset refused: GPU is not the sole function on its secondary bus')
    unbound = not (device.path / 'driver').exists()
    selected = False
    try:
        if not unbound:
            runner.write('unbind', device.driver / 'unbind', device.bdf)
            unbound = True
        if (device.path / 'driver').exists():
            raise RecoveryError('Unbind returned but driver is still bound')
        if method != 'rebind':
            # Restrict PCI reset to FLR: do not silently fall back to bus reset,
            # affect audio/other devices, remove/rescan, or unload all xe cards.
            selected = True
            runner.write('select-' + method, device.path / 'reset_method', method)
            runner.write('function-reset', device.path / 'reset', '1')
            runner.write('restore-reset-methods', device.path / 'reset_method', ' '.join(methods))
            selected = False
        runner.write('bind', device.driver / 'bind', device.bdf)
        unbound = False
        if not (device.path / 'driver').exists():
            raise RecoveryError('Bind returned but the driver is not attached')
    except BaseException:
        # No competing sysfs write if a previous writer is still stuck.
        if not runner.stranded:
            if selected:
                runner.write('restore-after-failure', device.path / 'reset_method', ' '.join(methods))
            if not (device.path / 'driver').exists():
                runner.write('bind-after-failure', device.driver / 'bind', device.bdf)
        raise


def journal_cursor(runner, label):
    text = runner.run(label, ['/usr/bin/journalctl', '-k', '-n', '0', '--show-cursor', '--no-pager'])
    match = re.search(r'^-- cursor: (.+)$', text, re.M)
    if not match:
        raise RecoveryError('Cannot obtain the kernel journal cursor; health cannot be certified')
    return match[1]


def health_environment():
    # Probe as the invoking user even when resets run under sudo. Clear unrelated
    # tuning flags; use the installed runtime, V2, cache off, no copy offload.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_'))}
    adapter = '/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0'
    if not Path(adapter).is_file():
        raise RecoveryError('Expected installed Level Zero V2 adapter is unavailable')
    env.update(SYCL_CACHE_PERSISTENT='0', UR_ADAPTERS_FORCE_LOAD=adapter,
               UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu')
    env['LD_LIBRARY_PATH'] = ':'.join(['/opt/intel/oneapi/compiler/2026.1/lib',
                                     '/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib',
                                     '/opt/intel/oneapi/umf/1.1/lib',
                                     '/usr/lib/x86_64-linux-gnu'])
    return env


def diagnostic_environment(env=None):
    """Trace API entry/return to stderr; use only for correctness/debug runs."""
    env = dict(health_environment() if env is None else env)
    # Level Zero 1.32 API logging requires the validation layer. Successful
    # returns matter too: an entry without a return can locate a blocked API.
    env.update(ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOG_CONSOLE='1',
               ZEL_LOADER_LOGGING_LEVEL='trace',
               ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1',
               ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
               UR_LOG_LOADER='level:debug;flush:debug;output:stderr',
               UR_LOG_LEVEL_ZERO='level:debug;flush:debug;output:stderr',
               UR_LOG_TRACING='level:info;flush:info;output:stderr',
               STRATA_TRACE='1')
    # Preserve any deliberately selected validation layers in a debug run.
    layers = [x for x in env.get('UR_ENABLE_LAYERS', '').split(',') if x]
    if 'UR_LAYER_TRACING' not in layers:
        layers.append('UR_LAYER_TRACING')
    env['UR_ENABLE_LAYERS'] = ','.join(layers)
    return env


def validate_probe_runtime(runner, binary):
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RecoveryError(f'Health executable is unavailable: {binary}')
    env = health_environment()
    # Loading the adapter resolves its shared-library dependencies without
    # calling a GPU API. A missing runtime must be reported before any reset.
    runner.run('health-runtime', ['/usr/bin/python3', '-c',
               'import ctypes,sys; ctypes.CDLL(sys.argv[1])', env['UR_ADAPTERS_FORCE_LOAD']],
               seconds=5, env=env)
    return env


def check_health(device, runner, binary, cursor=None):
    env = diagnostic_environment(validate_probe_runtime(runner, binary))
    settings = [k for k in env
                if k.startswith(('SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_', 'STRATA_'))
                or k == 'LD_LIBRARY_PATH']
    (runner.output / 'health-environment.json').write_text(
        json.dumps({k: env[k] for k in settings}, indent=2) + '\n')
    if cursor is None:
        cursor = journal_cursor(runner, 'before-health-cursor')
    argv = [str(binary.resolve()), device.bdf]
    if os.geteuid() == 0:
        user = os.environ.get('SUDO_USER')
        if not user or user == 'root':
            raise RecoveryError('Run with sudo from your ordinary account to execute the probe unprivileged')
        argv = ['/usr/sbin/runuser', '-u', user, '--', '/usr/bin/env'] + [k + '=' + env[k] for k in settings] + argv
    probe_error = None
    try:
        text = runner.run('health', argv, seconds=30, env=env)
        if f'PASS {device.bdf}: 3 rounds, 16384 exact words each' not in text:
            raise RecoveryError('Health probe did not record the complete BDF-matched integer check')
    except RecoveryError as e:
        probe_error = e
    # Even exit 0 and exact data are insufficient if the driver faults/resets.
    log = runner.run('health-kernel', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor,
                                    '--no-pager', '-o', 'short-iso-precise'])
    relevant = [line for line in log.splitlines() if device.bdf in line or 'xe ' in line]
    faults = [line for line in relevant if FAULT.search(line)]
    (runner.output / 'health-xe.txt').write_text('\n'.join(relevant) + '\n')
    if probe_error:
        raise probe_error
    if faults:
        raise RecoveryError(f'Probe data matched, but {len(faults)} new xe fault/reset messages occurred')


def acquire_recovery_lock(base, held_lock=None):
    path = base / 'recovery.lock'
    if held_lock is None:
        lock = path.open('a')
    else:
        owned, expected = os.fstat(held_lock.fileno()), path.stat()
        if (owned.st_dev, owned.st_ino) != (expected.st_dev, expected.st_ino):
            raise RecoveryRefused('Borrowed recovery lock belongs to another file')
        lock = os.fdopen(os.dup(held_lock.fileno()), 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock


def publish_summary(report, output, runner):
    user = os.environ.get('SUDO_USER')
    if os.geteuid() == 0 and user and user != 'root':
        # Publish a small summary as the caller, avoiding privileged writes
        # through home-directory symlinks. Raw root logs remain private.
        public = {k: v for k, v in report.items() if k != 'steps'}
        public['steps'] = [{k: v for k, v in s.items() if k != 'argv'} for s in runner.calls]
        public['root_report'] = str(output / 'report.json')
        writer = ('from pathlib import Path; import sys; '
                  'p=Path.home()/".local/state/strata-sycl/gpu-recovery"; '
                  'p.mkdir(parents=True,exist_ok=True); '
                  '(p/"latest.json").write_text(sys.argv[1])')
        try:
            runner.run('publish-summary', ['/usr/sbin/runuser', '-u', user, '--',
                       '/usr/bin/python3', '-c', writer, json.dumps(public, indent=2) + '\n'], seconds=5)
        except (RecoveryError, OSError) as e:
            print(f'Could not publish unprivileged summary: {e}', file=sys.stderr)


def main(argv=None, held_lock=None, publish=True):
    parser = argparse.ArgumentParser(description='Arc xe recovery without host reboot; default: inspect only')
    parser.add_argument('--bdf', default='0000:05:00.0')
    parser.add_argument('--apply', action='store_true', help='perform one recovery attempt (requires sudo)')
    parser.add_argument('--method', choices=['rebind', 'flr', 'bus'], default='rebind')
    parser.add_argument('--stop-pid', type=int, action='append', default=[], help='terminate this verified GPU client before reset')
    parser.add_argument('--check', type=Path, help='compiled sycl/tools/xe-health.cpp; run after recovery')
    args = parser.parse_args(argv)
    if not BDF_PATTERN.fullmatch(args.bdf):
        parser.error('Invalid PCI BDF')
    if args.stop_pid and not args.apply:
        parser.error('--stop-pid requires --apply')
    if args.apply and os.geteuid() != 0:
        parser.error('--apply requires sudo; inspection remains available without it')
    os.umask(0o077)
    if os.geteuid() == 0:
        base = Path('/var/log/strata-gpu-recovery')
    else:
        base = Path.home() / '.local/state/strata-sycl/gpu-recovery'
    base.mkdir(parents=True, exist_ok=True)
    if base.is_symlink() or base.stat().st_uid != os.geteuid() or base.stat().st_mode & 0o022:
        raise RecoveryError('Recovery log directory must be owned by the caller and not writable by others')
    output = Path(tempfile.mkdtemp(prefix=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-'), dir=base))
    runner, device = Runner(output), Device(args.bdf)
    report = dict(action='recover' if args.apply else 'inspect', method=args.method,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  healthy=False, recovery_completed=False, dump_saved=False, steps=runner.calls)
    code = 1
    lock = None
    try:
        # Root-owned log directory also holds the lock; no /tmp symlink targets.
        lock = acquire_recovery_lock(base, held_lock)
        pending = base / 'stalled-writer.json'
        if args.apply and pending.exists():
            old = json.loads(pending.read_text())
            try:
                raw = Path(f'/proc/{old["pid"]}/stat').read_text()
                ticks = int(raw[raw.rfind(')') + 2:].split()[19])
                if ticks == old['start_ticks']:
                    raise RecoveryRefused(f'Previous recovery writer PID {old["pid"]} remains; no competing reset')
            except FileNotFoundError:
                pass
            pending.unlink()
        report['device_before'] = device.describe()
        report['clients'], report['uninspectable_pids'] = device.clients()
        print(json.dumps(dict(device=report['device_before'], clients=report['clients'],
                              uninspectable_process_count=len(report['uninspectable_pids'])), indent=2))
        if not args.apply:
            print(f'Inspection only. Apply with sudo and --apply --method {args.method}.')
            code = 0
        else:
            # Validate a requested probe before any destructive action.
            if args.check and (not args.check.is_file() or not os.access(args.check, os.X_OK)):
                raise RecoveryError('Requested health executable is unavailable')
            if args.check:
                try:
                    validate_probe_runtime(runner, args.check)
                except RecoveryError as error:
                    raise RecoveryRefused('Health runtime is unavailable; no reset: ' + str(error)) from error
            for pid in args.stop_pid:
                stop_client(device, pid)
            report['clients'], report['uninspectable_pids'] = device.clients()
            reset(device, runner, args.method)
            report['device_after'] = device.describe()
            report['recovery_completed'] = True
            if args.check:
                check_health(device, runner, args.check)
                report['healthy'] = True
                print('Recovery verified: exact H2D/kernel/D2H, no new xe fault/reset messages.')
                code = 0
            else:
                print('Driver reattached; GPU execution has not been verified. No workloads were restarted.')
                code = 3
    except (RecoveryError, OSError) as e:
        report['error'] = str(e)
        if isinstance(e, RecoveryRefused) or (isinstance(e, OSError) and e.errno in (errno.EAGAIN, errno.EWOULDBLOCK)):
            report['refused'] = True
            code = 4
        print(str(e), file=sys.stderr)
    finally:
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Report: {output / "report.json"}')
        if publish:
            publish_summary(report, output, runner)
        if lock:
            lock.close()
    return code


if __name__ == '__main__':
    sys.exit(main())
PY
