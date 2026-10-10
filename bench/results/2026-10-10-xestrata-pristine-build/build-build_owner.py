"""Root-only finite owner for the copied-expression GDN numerical discriminator.

host-checks runs CPU-only supervision checks. gpu is a single diagnostic run,
never model performance, driver recovery, service management or a retry loop.
"""
from pathlib import Path
import argparse, csv, datetime, fcntl, hashlib, json, os, re, resource
import select, signal, subprocess, sys, time, types

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-gate-factor-probe-20261010')
HEAD = '609a270daac23c9b1a26d26ec092b2f60d2da335'
BUILD_RECEIPT = B / 'gdn-gate-factor-probe-cpu-build-v4/record.json'
BUILD_HASH = '91730e83d082d55ab9d9b751ac2627a61c6747d1fea961e6f845d96ca693f9fc'
RECOVER_HASH = '2f90005f79d2309747d917cff51dc53d8ee6192d68cb1feca074f84af8b0796c'
LENGTHS = [0, 1, 3, 5, 15, 16, 17, 127, 128, 129, 8191, 8192, 3, 8192, 8192]
STAGES = ['baseline_producer', 'candidate_producer', 'scalar_reader', 'SG16_reader', 'SG32_reader']


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_identity(pid):
    try:
        v = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=v[0], ppid=int(v[1]), pgid=int(v[2]), sid=int(v[3]),
                    start_ticks=int(v[19]), rss_bytes=int(v[21]) * os.sysconf('SC_PAGE_SIZE'))
    except (FileNotFoundError, ProcessLookupError):
        return None


class Owner:
    def __init__(self, output):
        self.output = output
        self.commands = []
        self.active = None
        self.persist = lambda: None

    def run(self, label, argv, env, wall=120, text_cap=8 << 20, file_cap=4 << 20):
        entry = dict(label=label, argv=argv, environment=env, started_utc=utc(),
                     wall_seconds=wall, normal_exit=False, session_empty=False,
                     observation_complete=False, cleanup=[], errors=[], owners=[])
        self.commands.append(entry)
        self.persist()
        start = time.monotonic()
        so, se = self.output / (label + '.stdout'), self.output / (label + '.stderr')
        proc = None
        owned = {}
        exit_watch = select.poll()

        def register(v):
            key = (v['pid'], v['start_ticks'])
            if key in owned:
                return
            fd = os.pidfd_open(v['pid'])
            try:
                after = process_identity(v['pid'])
                if after is None or after['start_ticks'] != v['start_ticks']:
                    os.close(fd)
                    return
                owned[key] = (fd, dict(v))
                entry['owners'].append({k: v[k] for k in ['pid', 'start_ticks', 'sid', 'pgid']})
            except BaseException:
                os.close(fd)
                raise

        def scan():
            members = []
            if proc is None:
                return members
            for p in Path('/proc').iterdir():
                if not p.name.isdecimal():
                    continue
                try:
                    v = process_identity(int(p.name))
                except PermissionError:
                    # This unprivileged app cannot adopt another user's process.
                    # Inaccessible same-user entries leave ownership unobserved.
                    try:
                        if p.stat().st_uid != os.getuid():
                            continue
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    raise
                if v and v['sid'] == proc.pid and v['state'] != 'Z':
                    register(v)
                    members.append(v)
            return members

        def signal_owned(sig):
            try:
                scan()
            except BaseException as e:
                entry['errors'].append('cleanup scan: ' + repr(e))
            for (pid, ticks), (fd, original) in list(owned.items()):
                attempt = dict(pid=pid, start_ticks=ticks, signal=sig.name, time_utc=utc())
                try:
                    current = process_identity(pid)
                    if current is None or current['start_ticks'] != ticks or current['state'] == 'Z':
                        continue
                    signal.pidfd_send_signal(fd, sig)
                    attempt['result'] = 'sent'
                except ProcessLookupError:
                    attempt['result'] = 'exited'
                except BaseException as e:
                    attempt['result'] = 'error: ' + repr(e)
                    entry['errors'].append(attempt['result'])
                entry['cleanup'].append(attempt)
                try:
                    self.persist()
                except BaseException as save_error:
                    entry['errors'].append('signal persistence: ' + repr(save_error))

        def child_limits():
            for key, values in [(resource.RLIMIT_AS, (16 << 30, 16 << 30)),
                                (resource.RLIMIT_CPU, (600, 601)),
                                (resource.RLIMIT_FSIZE, (file_cap, file_cap)),
                                (resource.RLIMIT_NOFILE, (256, 256)),
                                (resource.RLIMIT_CORE, (0, 0))]:
                resource.setrlimit(key, values)

        try:
            with so.open('wb') as out, se.open('wb') as err:
                proc = subprocess.Popen(argv, cwd=W, env=env, stdout=out, stderr=err,
                                        start_new_session=True, preexec_fn=child_limits)
                self.active = proc
                entry['pid'] = proc.pid
                first = process_identity(proc.pid)
                assert first is not None and first['sid'] == proc.pid, 'missing direct process/session identity'
                register(first)
                direct_fd = owned[(first['pid'], first['start_ticks'])][0]
                exit_watch.register(direct_fd, select.POLLIN)
                entry['direct_identity'] = first
                self.persist()
                # Keep the leader unreaped until the last session scan. Its PID
                # anchors the SID and cannot be recycled into an unrelated run.
                while not exit_watch.poll(0):
                    members = scan()
                    rss = sum(v['rss_bytes'] for v in members)
                    entry['peak_session_rss_bytes'] = max(entry.get('peak_session_rss_bytes', 0), rss)
                    assert rss <= 24 << 30, 'owned RSS limit'
                    assert so.stat().st_size + se.stat().st_size <= text_cap, 'combined text limit'
                    assert time.monotonic() - start < wall, 'wall deadline'
                    self.persist()
                    time.sleep(.05)
                members = scan()
                entry['session_empty'] = not members
                assert not members, 'owned session survives direct process exit'
                assert so.stat().st_size + se.stat().st_size <= text_cap, 'combined text limit at exit'
                entry['observation_complete'] = True
        except BaseException as e:
            entry['error'] = type(e).__name__ + ': ' + str(e)
            entry['failure_observed_utc'] = utc()
            try:
                self.persist()
            except BaseException as save_error:
                entry['errors'].append('failure persistence: ' + repr(save_error))
        finally:
            if proc is not None:
                try:
                    needs_stop = not exit_watch.poll(0) or bool(scan())
                except BaseException as e:
                    needs_stop = True
                    entry['errors'].append('final scan: ' + repr(e))
                if needs_stop:
                    for sig in [signal.SIGTERM, signal.SIGKILL]:
                        signal_owned(sig)
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline:
                            try:
                                if exit_watch.poll(0) and not scan():
                                    break
                            except BaseException as e:
                                entry['errors'].append('stop observation: ' + repr(e))
                                break
                            time.sleep(.05)
                try:
                    entry['survivors'] = scan()
                    entry['session_empty'] = not entry['survivors']
                    entry['ownership_status'] = 'observed'
                except BaseException as e:
                    entry['session_empty'] = False
                    entry['ownership_status'] = 'unknown'
                    entry['errors'].append('closure: ' + repr(e))
                # Do not scan the numeric SID again after this reap: once its
                # last member exits that identifier may belong to a new session.
                try:
                    entry['exit_code'] = proc.wait(timeout=.1)
                    entry['normal_exit'] = proc.returncode >= 0
                    entry['direct_child_reaped'] = True
                except BaseException as e:
                    entry['direct_child_reaped'] = False
                    entry['errors'].append('wait: ' + repr(e))
            try:
                entry.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - start,
                             logs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in [so, se] if p.exists()})
            except BaseException as e:
                entry['errors'].append('log identities: ' + repr(e))
            finally:
                for fd, _ in owned.values():
                    try:
                        os.close(fd)
                    except OSError as e:
                        entry['errors'].append('pidfd close: ' + repr(e))
                self.active = proc if proc is not None and (not entry.get('direct_child_reaped') or not entry.get('session_empty')) else None
        return entry, so, se


def completed(entry):
    return (entry.get('observation_complete') and entry.get('direct_child_reaped')
            and entry.get('session_empty') and not entry.get('error')
            and not entry.get('errors') and not entry.get('cleanup'))

