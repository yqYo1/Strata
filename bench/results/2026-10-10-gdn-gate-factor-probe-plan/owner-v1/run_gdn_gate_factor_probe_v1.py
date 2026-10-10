"""Root-only finite owner for the copied-expression GDN numerical discriminator.

host-checks runs CPU-only supervision checks. gpu is a single diagnostic run,
never model performance, driver recovery, service management or a retry loop.
"""
from pathlib import Path
import argparse, csv, datetime, fcntl, hashlib, json, os, re, resource
import select, signal, subprocess, sys, time, types

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-gate-factor-probe-20261010')
HEAD = '00ccf742bb0a39a6c6d5d7aba9b43d7a1a9f4518'
BUILD_RECEIPT = B / 'gdn-gate-factor-probe-cpu-build-v3/record.json'
BUILD_HASH = '6fdb4234d361f310d0b9e9894561da484906ac2b8492ed180d57bf2d5a0f1880'
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

    def run(self, label, argv, env, wall=120, text_cap=8 << 20, file_cap=4 << 20):
        entry = dict(label=label, argv=argv, environment=env, started_utc=utc(),
                     wall_seconds=wall, normal_exit=False, session_empty=False,
                     observation_complete=False, cleanup=[], errors=[], owners=[])
        self.commands.append(entry)
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

        def child_limits():
            for key, values in [(resource.RLIMIT_AS, (16 << 30, 16 << 30)),
                                (resource.RLIMIT_CPU, (120, 121)),
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
                # Keep the leader unreaped until the last session scan. Its PID
                # anchors the SID and cannot be recycled into an unrelated run.
                while not exit_watch.poll(0):
                    members = scan()
                    rss = sum(v['rss_bytes'] for v in members)
                    entry['peak_session_rss_bytes'] = max(entry.get('peak_session_rss_bytes', 0), rss)
                    assert rss <= 2 << 30, 'owned RSS limit'
                    assert so.stat().st_size + se.stat().st_size <= text_cap, 'combined text limit'
                    assert time.monotonic() - start < wall, 'wall deadline'
                    time.sleep(.05)
                members = scan()
                entry['session_empty'] = not members
                assert not members, 'owned session survives direct process exit'
                assert so.stat().st_size + se.stat().st_size <= text_cap, 'combined text limit at exit'
                entry['observation_complete'] = True
        except BaseException as e:
            entry['error'] = type(e).__name__ + ': ' + str(e)
            entry['failure_observed_utc'] = utc()
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


def parse_probe(text, env):
    rows = list(csv.reader(text.splitlines()))
    it = iter(rows)
    identity = next(it)
    assert identity[0] == 'IDENTITY' and len(identity) == 27, identity
    fields = dict(zip(identity[1::2], identity[2::2]))
    assert len(fields) == 13 and fields['backend'] == 'LevelZero'
    for key, value in [('vendor', 0x8086), ('device', 0xe20c), ('domain', 0), ('bus', 5),
                       ('pci_device', 0), ('function', 0), ('root', 1), ('passed', 1)]:
        assert int(fields[key]) == value, (key, fields[key])
    assert int(fields['driver_version']) > 0 and int(fields['pci_extension_version']) >= 0x10000
    assert 'B570' in fields['name']
    for key in ['ZE_FLAT_DEVICE_HIERARCHY', 'ZE_AFFINITY_MASK', 'ONEAPI_DEVICE_SELECTOR']:
        raw = env.get(key)
        displayed = 'unset' if raw is None else ''.join(c if 32 <= ord(c) <= 126 and c != ',' else '_' for c in raw)
        assert next(it) == ['IDENTITY_ENV', key, displayed]
    assert next(it) == ['SCOPE', 'necessary_bitwise_gate_screen_not_recurrence_model_or_performance']
    assert next(it) == ['GEOMETRY', 'HV48', 'HK16', 'S128', 'MaxT8192', 'device_allocations_below16MiB']
    device = next(it)
    assert len(device) == 6 and device[0] == 'DEVICE' and 'B570' in device[1]
    assert device[2:4] == ['vendor', str(0x8086)] and device[4] == 'driver' and device[5]
    hashes = []
    for case, length in enumerate(LENGTHS):
        assert next(it) == ['CASE', str(case), 'T', str(length), 'specials', '0']
        for name in STAGES:
            assert next(it) == ['STAGE', name]
        if length == 8192:
            row = next(it)
            assert len(row) == 2 and row[0] == 'FULL_REPEAT_HASH'
            hashes.append(int(row[1]))
        assert next(it) == ['CASE_PASS', str(case + 1), str(length * 48)]
    assert next(it) == ['TERMINAL', 'pass', 'cases', '15', 'comparisons', '7970640',
                        'necessary_screen_only', 'no_state_no_norm_no_model_no_timing']
    assert list(it) == [] and len(hashes) == 3 and len(set(hashes)) == 1
    return dict(identity=fields, cases=15, stage_entries=75, comparisons=7970640,
                full8K_repeat_hashes=hashes, necessary_screen_only=True)


def main():
    mode = argparse.ArgumentParser()
    mode.add_argument('mode', choices=['host-checks', 'gpu'])
    selected = mode.parse_args().mode
    assert __debug__
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = B / ('gdn-owner-controller-host-checks-v1' if selected == 'host-checks' else 'gdn-gate-factor-probe-runtime-v1')
        output.mkdir(mode=0o700)
        owner = Owner(output)
        record = dict(active=True, complete=False, passed=False, mode=selected,
                      started_utc=utc(), controller_sha256=sha(__file__), commands=owner.commands,
                      GPU_runtime_launched=False, model_opened=False, adopted=False,
                      performance_eligible=False, full_lifecycle_passed=False,
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

        def save():
            body = json.dumps(record, indent=2) + '\n'
            assert len(body.encode()) < 256 << 10, 'record byte cap'
            temp = output / 'record.json.tmp'
            temp.write_text(body)
            temp.replace(output / 'record.json')

        save()
        clean = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        try:
            if selected == 'host-checks':
                for label, code, wall, cap in [
                    ('normal', 'print("closed")', 3, 8 << 20),
                    ('nonzero', 'raise SystemExit(2)', 3, 8 << 20),
                    ('timeout', 'import time;time.sleep(20)', .15, 8 << 20),
                    ('output-limit', 'import os,time;os.write(1,b"x"*262144);time.sleep(20)', 3, 65536),
                    ('reparented-group', 'import os,time\npid=os.fork()\nif pid==0:\n os.setpgid(0,0);time.sleep(20)\nelse:\n time.sleep(.05);os._exit(0)', 3, 8 << 20),
                ]:
                    e, _, _ = owner.run(label, ['/usr/bin/python3', '-c', code], clean, wall=wall, text_cap=cap)
                    assert e.get('direct_child_reaped') and e.get('session_empty') and not e['errors'], e
                    if label == 'normal':
                        assert completed(e) and e['exit_code'] == 0
                    elif label == 'nonzero':
                        assert completed(e) and e['exit_code'] == 2
                    else:
                        assert e.get('error') and e['cleanup'] and not completed(e)
                    save()
                record.update(complete=True, passed=True, no_GPU_API_or_model=True,
                              checked=['normal0', 'nonzero2 preserved', 'deadline owned cleanup',
                                       'text-limit owned cleanup', 'same-SID reparented child/new-PGID cleanup'])
            else:
                assert sha(BUILD_RECEIPT) == BUILD_HASH
                build = json.loads(BUILD_RECEIPT.read_text())
                assert build['passed'] and build['complete'] and not build['active'] and not build['cleanup'] and not build['survivors']
                assert subprocess.check_output(['/usr/bin/git', 'rev-parse', 'HEAD'], cwd=W, text=True, timeout=5).strip() == HEAD
                assert not subprocess.check_output(['/usr/bin/git', 'status', '--porcelain'], cwd=W, text=True, timeout=5)

                def check_pins():
                    assert sha(BUILD_RECEIPT) == BUILD_HASH and sha(build['binary']['path']) == build['binary']['sha256']
                    for rel, x in build['source_pins'].items():
                        assert sha(W / rel) == x['sha256'], rel
                    for path, x in build['identity_dependency_pins'].items():
                        assert sha(path) == x['sha256'] and str(Path(path).resolve()) == x['realpath'], path
                    for path, x in record.get('runtime_pins', {}).items():
                        assert sha(path) == x['sha256'] and str(Path(path).resolve()) == x['realpath'], path
                    assert sha(__file__) == record['controller_sha256']

                check_pins()
                helper = W / 'sycl/tools/recover-xe.sh'
                assert sha(helper) == RECOVER_HASH
                source = helper.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
                environment_module = types.ModuleType('pure_diagnostic_environment')
                exec(compile(source, str(helper), 'exec'), environment_module.__dict__)
                env = dict(clean, LD_LIBRARY_PATH=':'.join(['/opt/intel/oneapi/compiler/2026.1/lib',
                           '/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib', '/opt/intel/oneapi/umf/1.1/lib',
                           '/usr/lib/x86_64-linux-gnu']), SYCL_CACHE_PERSISTENT='0',
                           UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',
                           UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                           EnableDirectSubmission='0')
                env = environment_module.diagnostic_environment(env)
                paths = [Path('/opt/intel/oneapi/compiler/2026.1/lib') / p for p in
                         ['libsycl.so.9', 'libur_loader.so.0', 'libur_adapter_level_zero_v2.so.0']]
                paths += [Path('/usr/lib/x86_64-linux-gnu') / p for p in ['libze_loader.so.1', 'libze_intel_gpu.so.1']]
                record.update(environment=env, build_receipt_sha256=BUILD_HASH, source_head=HEAD,
                              binary=build['binary'], runtime_pins={str(p): dict(realpath=str(p.resolve()), sha256=sha(p)) for p in paths},
                              recover_environment_helper_sha256=RECOVER_HASH,
                              exact_environment_set_unset={key: dict(present=key in env, value=env.get(key)) for key in
                                                          ['ZE_FLAT_DEVICE_HIERARCHY', 'ZE_AFFINITY_MASK', 'ONEAPI_DEVICE_SELECTOR']},
                              limits=dict(AS_each_bytes=16 << 30, RSS_session_bytes=2 << 30, CPU_each_seconds=[120, 121],
                                          stdout_stderr_bytes=8 << 20, FSIZE_each_bytes=4 << 20, NOFILE=256, CORE=0, wall_seconds=120))
                save()
                journal, so, se = owner.run('journal-cursor', ['/usr/bin/journalctl', '-k', '-n', '0', '--show-cursor', '--no-pager'], clean, wall=5, text_cap=256 << 10, file_cap=256 << 10)
                assert completed(journal) and journal['exit_code'] == 0 and not se.read_text().strip(), 'journal cursor unavailable'
                cursor = re.search(r'^-- cursor: (.+)$', so.read_text(), re.M)
                assert cursor, 'kernel cursor absent'
                record['kernel_cursor'] = cursor.group(1)
                fault_path = Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
                record['devcoredump_before'] = fault_path.exists()
                assert not record['devcoredump_before'], 'existing device fault requires review'
                host_checks = json.loads((B / 'gdn-owner-controller-host-checks-v1/record.json').read_text())
                assert host_checks['passed'] and host_checks['complete'] and not host_checks['active']
                assert host_checks['controller_sha256'] == record['controller_sha256'], 'host checks are for different controller bytes'
                record['host_checks_sha256'] = sha(B / 'gdn-owner-controller-host-checks-v1/record.json')
                check_pins()
                record['GPU_runtime_launch_attempted'] = True
                save()
                e, so, se = owner.run('probe', [build['binary']['path']], env)
                record['GPU_runtime_launched'] = 'pid' in e
                record['probe_process_result'] = dict(exit_code=e.get('exit_code'), completed=completed(e), normal_exit=e.get('normal_exit'))
                stdout, stderr = so.read_text(), se.read_text()
                record['last_progress_lines'] = stdout.splitlines()[-8:]
                record['submitted_stage_seen'] = '\nSTAGE,' in stdout
                assert owner.active is None, 'unresolved owned probe process/session'
                interval, jo, je = owner.run('kernel-interval', ['/usr/bin/journalctl', '-k', '--after-cursor', record['kernel_cursor'], '--no-pager'], clean, wall=5, text_cap=1 << 20, file_cap=1 << 20)
                record['journal_status'] = 'complete' if completed(interval) and interval.get('exit_code') == 0 and not je.read_text().strip() else 'unavailable_or_partial'
                record['visible_kernel_GPU_entries'] = [line for line in jo.read_text().splitlines() if re.search(r'\b(?:xe|i915|drm)\b|gpu.*(?:hang|fault|reset)|devcoredump', line, re.I)]
                record['devcoredump_after'] = fault_path.exists()
                assert record['journal_status'] == 'complete', 'kernel fault interval unknown'
                assert not record['visible_kernel_GPU_entries'] and not record['devcoredump_after'], 'new driver entry/fault requires review'
                assert completed(e) and e['exit_code'] == 0, 'probe did not close normally0'
                assert not any(x in stderr for x in ['SUBMITTED_STAGE_FAIL', 'ASYNC,', 'FIRST_DIFFERENCE', 'TERMINAL,fail']), 'probe error marker'
                record['numerical_result'] = parse_probe(stdout, env)
                check_pins()
                assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                record.update(complete=True, passed=True, kernel_fault_observation='complete_interval_no_new_entries',
                              scope='Necessary copied-expression primitive only; no recurrence/state/norm/model/performance/full-context qualification')
        except BaseException as e:
            record['error'] = type(e).__name__ + ': ' + str(e)
        finally:
            unresolved = any('pid' in e and (not e.get('direct_child_reaped') or not e.get('session_empty')) for e in owner.commands)
            record.update(active=unresolved, finished_utc=utc(), GPU_completion='source_drains_and_normal0' if record['passed'] and selected == 'gpu' else 'unproven_if_GPU_launched')
            try:
                save()
            except BaseException as e:
                print('CONTROLLER_RECORD_FAILURE,' + repr(e), file=sys.stderr, flush=True)
                record['passed'] = False
        print(json.dumps(dict(record=str(output / 'record.json'), sha256=sha(output / 'record.json'),
                              passed=record['passed'], error=record.get('error'), GPU_runtime_launched=record['GPU_runtime_launched'])), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
