#!/usr/bin/env python3
"""Main-owned named-unitrace request window. Diagnostic, not bench.sh timing.

No native rebuild or runtime environment mutation. Startup is paused; the
window includes long prefill, state/weight restoration and ONE generated token.
Run component first, inspect its result, then explicitly run long.
"""
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import time

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
PTI = Path('/home/yayoi/ghq/github.com/intel/pti-gpu')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
OWNER = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
WRAPPER = A/'external-session-decode-profile-v2/main_controller.py'
sys.path.insert(0, str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id

FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
NATIVE_ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|stopping with owners live')


def read_ready(io, record, seconds):
    """Native protocol emits its effective INFO before READY, exactly once."""
    line = io.line(seconds)
    require(line.startswith('INFO '), 'native configuration INFO required')
    fields = {}
    for item in line.split()[1:]:
        require('=' in item, 'malformed native INFO')
        key, value = item.split('=', 1)
        require(key and value and key not in fields, 'invalid or duplicate native INFO field')
        fields[key] = value
    expected = dict(context='262144', kv='int8', kv_resident='32768',
                    expert_slots='128', spec='4', pool_workers='5', pcie_frac='0.00')
    require(all(fields.get(k) == v for k, v in expected.items()), 'effective native INFO differs')
    require(io.line(seconds) == 'READY 262144 stop', 'native ready protocol after INFO')
    record['native_INFO'] = fields


def main():
    require(sys.argv[1:] in (['component'], ['long']), 'explicit component or long diagnostic')
    mode = sys.argv[1]
    module = load_owner(OWNER)
    require(sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner pin')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs = {k: p for k, p in dict(
            baseline=A/'prefill-only-clean-iq1-c4096-v1.json',
            build=A/'selective-iq-native-build-v1.json',
            tool=B/'main-primary-unitrace-build-root-v1/record.json',
            natural=A/'natural-layout-long-pair-v2.json',
            retired=A/'prefill-window-component-v1-retirement.json',
            ready_host=A/'prefill-window-profile-ready-host-v2.json').items()}
        data = {k: json.loads(p.read_text()) for k, p in refs.items()}
        baseline, build, tool, natural = [data[k] for k in ('baseline', 'build', 'tool', 'natural')]
        require(all(d['passed'] and d['complete'] and not d['active'] for d in data.values()), 'closed reference gates')
        require(data['retired']['session_released'] and data['ready_host']['recipe_sha256'] == sha(Path(__file__)), 'prior failure retirement and current host parser qualification')
        require(not baseline['fresh_faults'] and baseline['boot_unchanged'] and not baseline['driver_dump_exists'] and baseline['boot_id'] == boot_id(), 'latest GPU closure')
        for command in baseline['commands']:
            require(module.closed(command), 'baseline ownership')
            for old in command['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'old identity still present')
        if mode == 'long':
            refs['component'] = A/'prefill-window-profile-component-v2.json'
            component = json.loads(refs['component'].read_text())
            require(component['passed'] and component['complete'] and not component['active'] and
                    component['provider_window_valid'] and not component['fresh_faults'] and
                    component['boot_id'] == boot_id(), 'component collection gate')
        require(all(sha(P/n) == h for n, h in build['source_pins'].items()), 'native source pin')
        require(all(sha(Path(n)) == h for n, h in build['linked_inputs'].items()), 'native linked inputs')
        exe = Path(build['binary'])
        require(sha(exe) == build['binary_sha256'] == baseline['binary_sha256'], 'native binary')
        provider = Path(tool['build_path'])
        require(tool['source_revision'] == '6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033' and
                all(sha(provider/n) == h for n, h in tool['binaries'].items()) and
                all(sha(Path(n)) == h for n, h in tool['XPTI_provider_pins'].items()), 'actual collector pins')
        for field in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
            require(all(sha(Path(n)) == h for n, h in baseline[field].items()), 'model metadata')
        for n, v in baseline['model_identity'].items():
            s = Path(n).stat()
            require(s.st_size == v['bytes'] and s.st_mtime_ns == v['mtime_ns'], 'model file identity')
        tokens = Path(natural['engine_args'][natural['engine_args'].index('--tokens-file')+1])
        require(sha(tokens) == natural['input']['tokens_sha256'], 'fixed natural input')
        all_ids = [int(x) for x in tokens.read_text().strip().split(',')]
        require(len(all_ids) == 34603, 'input length')
        ids = all_ids[:513] if mode == 'component' else all_ids
        require((DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists(), 'GPU preflight')
        root = B/('main-prefill-window-profile-'+mode+'-root-v2')
        receipt = A/('prefill-window-profile-'+mode+'-v2.json')
        require(not root.exists() and not receipt.exists(), 'fresh diagnostic root')
        root.mkdir(mode=0o700)
        reports = root/'provider'
        reports.mkdir(mode=0o700)
        session = 'StrataPrefill'+secrets.token_hex(16)
        shm = Path('/dev/shm', 'uctrl'+session)
        require(not shm.exists(), 'unique unused named session')
        env = dict(baseline['environment'], XDG_CACHE_HOME=str(root/'cache'))
        for key in ('STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
            env.pop(key, None)
        require(not any(k.startswith(('UNITRACE_', 'PTI_', 'XPTI_')) or k in ('LD_PRELOAD', 'UR_ENABLE_LAYERS') or 'TOKEN' in k or 'API_KEY' in k for k in env), 'uninstrumented input environment')
        env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                   UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                   UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
                   STRATA_TRACE='1', STRATA_WATCHDOG_S='1800')
        # Raw Chrome history is needed for overlapping transfers/device intervals
        # and host waits. A finite 1GiB exception is recorded BEFORE collection.
        raw_cap, total_cap = 1 << 30, 1152 << 20
        wall, request_wall = (600, 300) if mode == 'component' else (2400, 1800)
        control_env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), LANG='C.UTF-8',
                           LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        args = list(baseline['engine_args']) + ['--serve']
        argv = [str(provider/'unitrace'), '--device-timing', '--host-timing',
                '--chrome-call-logging', '--chrome-kernel-logging',
                '--chrome-event-buffer-size', '8192', '--session', session, '--start-paused',
                '--result-dir', str(reports), '/usr/bin/python3', '-B', str(WRAPPER),
                '--runtime-child', str(root), str(exe), *args]
        record = dict(schema='owned-prefill-window-profile-v2', mode=mode, active=True,
            passed=False, complete=False, GPU_executed=False, provider_window_valid=False,
            performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
            restore_qualified=False, commands=[], control_commands=[], environment=env, argv=argv,
            boot_id=boot_id(), recipe_sha256=sha(Path(__file__)), wrapper_sha256=sha(WRAPPER),
            owner_sha256=sha(OWNER), references={k: sha(p) for k, p in refs.items()},
            source_pins=build['source_pins'], binary_sha256=sha(exe),
            profiler_binaries=tool['binaries'], profiler_source_revision=tool['source_revision'],
            actual_input_tokens=len(ids), canonical_tokens_sha256=sha(tokens),
            limits=dict(wall_seconds=wall, request_seconds=request_wall, RSS_bytes=88 << 30,
                        raw_text_and_provider_bytes=raw_cap, job_bytes=total_cap, file_bytes=raw_cap),
            retention_exception=dict(owner='prefill-window-profile-'+mode+'-v2',
                question='Which transfer, GPU execution and host-wait intervals occupy the actual prefill request dependency path?',
                reason='Complete bounded Chrome history is required to compute interval unions and join host/device events; aggregate duration sums cannot answer this.',
                consumer='Main trace interval/correlation analyzer', path=str(reports), byte_budget=raw_cap,
                next_review='After compact verified interval summary; retire raw successful traces.'),
            scope='Paused startup, external resume before one GEN, pause after DONE; includes prefill, weight/state restore and one output token. Not a pure-prefill-only window, not clean throughput or an official bench.sh performance sample.')

        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')

        owner = module.Owner(root/'owned', persist)
        controls = module.Owner(root/'controls', persist)
        record['commands'], record['control_commands'] = owner.commands, controls.commands
        persist()
        last_budget = 0.

        def budget(force=False):
            nonlocal last_budget
            now = time.monotonic()
            if not force and now-last_budget < .25:
                return
            last_budget = now
            files = [p for p in root.rglob('*') if p.is_file()]
            sizes = {p: p.stat().st_size for p in files}
            raw = sum(n for p, n in sizes.items() if reports in p.parents or p.suffix in ('.stdout', '.stderr'))
            require(raw <= raw_cap and sum(sizes.values()) <= total_cap, 'raw/provider or job budget: incomplete')
            require(sum(1 for p in files if reports in p.parents) <= 64, 'provider file count cap')

        def utility(label, cmd):
            c, out, err = owner.run(label, cmd, control_env, P, wall=60, cpu=60,
                rss_cap=2 << 30, text_cap=8 << 20, total_cap=total_cap, file_cap=8 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label+': '+err.read_text()[-2000:])
            return out.read_text()

        def session_identity():
            s = shm.lstat()
            require(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and
                    stat.S_IMODE(s.st_mode) == 0o600 and s.st_size == 1024, 'private collector session')
            identity = dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, path=str(shm))
            require('session_object' not in record or record['session_object'] == identity, 'session object changed')
            record['session_object'] = identity
            return identity

        def control(action, native_poll=None):
            session_identity()

            def interaction(ci, entry):
                ci.proc.stdin.close()
                end = time.monotonic()+5
                while not ci.exit_watch.poll(0) or not ci.eof:
                    if native_poll:
                        native_poll()
                    budget()
                    ci.poll()
                    require(time.monotonic() < end, 'finite collector CLI control')
                    ci.read()
                    ci.pending.clear()
                    ci.exit_watch.poll(5)

            c, out, err = controls.run('session-'+action, [str(provider/'unitrace'), '--'+action, session],
                control_env, P, interaction=interaction, wall=6, cpu=6, rss_cap=1 << 30,
                text_cap=1 << 20, total_cap=total_cap, file_cap=1 << 20)
            state = dict(resume='resumed', pause='paused', stop='stopped')[action]
            require(module.closed(c) and c['exit_code'] == 0 and
                    f'[INFO] Session {session} is {state}' in out.read_text()+err.read_text(), 'collector control acknowledgement')
            record.setdefault('controls', []).append(dict(action=action, completed_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
            persist()

        cursor = None
        profile_entry = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
            record['journal_before_cursor'] = cursor

            def interaction(io, entry):
                original = io.poll

                def checked_poll():
                    original()
                    budget()

                io.poll = checked_poll
                read_ready(io, record, min(wall, 300))
                start = json.loads((root/'runtime-start.json').read_text())
                native = start['native']
                require(start['binary_sha256'] == sha(exe) and native['sid'] == entry['pid'], 'actual native identity')
                until = time.monotonic()+1
                while not any(v['pid'] == native['pid'] and v['start_ticks'] == native['start_ticks'] for v in entry['owners']):
                    checked_poll()
                    require(time.monotonic() < until, 'native not pinned in original owner')
                    select_wait = io.exit_watch.poll(5)
                    require(not select_wait, 'leader died before request')
                now = module.identity(native['pid'])
                require(now and now['start_ticks'] == native['start_ticks'] and
                        sha(Path('/proc', str(native['pid']), 'exe')) == sha(exe), 'native changed before resume')
                record['actual_native'] = native
                session_identity()
                control('resume', checked_poll)
                wire = ('GEN 1 temperature=0 seed=42 '+','.join(map(str, ids))+'\n').encode()
                record['request_wire_sha256'] = hashlib.sha256(wire).hexdigest()
                record['request_begin_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                io.send(wire)
                end = time.monotonic()+request_wall
                generated, progress = [], []
                while True:
                    remaining = end-time.monotonic()
                    require(remaining > 0, 'request deadline')
                    line = io.line(remaining)
                    if re.fullmatch(r'T \d+', line):
                        generated.append(int(line[2:]))
                        require(len(generated) <= 1, 'one output maximum')
                    elif line.startswith('DONE '):
                        f = line.split()
                        require(len(f) == 15, 'DONE protocol fields')
                        done = dict(generated=int(f[1]), prompt=int(f[2]), prompt_ms=float(f[3]),
                            decode_ms=float(f[4]), finish=f[5], accepted=int(f[6]), offered=int(f[7]),
                            reused=int(f[8]), read=int(f[14]))
                        require(done['generated'] == len(generated) == 1 and done['finish'] == 'length' and
                                done['prompt'] == done['read'] == len(ids) and done['reused'] == 0 and
                                all(math.isfinite(done[k]) and done[k] >= 0 for k in ('prompt_ms', 'decode_ms')), 'actual request incomplete')
                        record.update(done=done, generated_ids=generated, progress=progress,
                            request_done_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
                        persist()
                        break
                    else:
                        require(re.fullmatch(r'(?:RESUME \d+|REUSED \d+|PP \d+ \d+ [\d.]+ [\d.]+)', line), 'unexpected native protocol')
                        progress.append(line)
                        require(len(progress) <= 1024, 'progress count cap')
                control('pause', checked_poll)
                io.send(b'QUIT\n')
                io.exit(120)

            record['GPU_executed'] = True
            persist()
            print('Named unitrace '+mode+' diagnostic: paused startup; one '+str(len(ids))+'-token request / one output', flush=True)
            profile_entry, out, err = owner.run('request-profile', argv, env, P, interaction=interaction,
                wall=wall, cpu=wall*12, rss_cap=88 << 30, text_cap=64 << 20,
                total_cap=total_cap, file_cap=raw_cap)
            require(module.closed(profile_entry) and profile_entry['exit_code'] == 0, 'profile process closure')
            exit_record = json.loads((root/'runtime-exit.json').read_text())
            require(exit_record['exit_code'] == 0 and exit_record['native_child_reaped'] and
                    exit_record['native'] == record['actual_native'] and
                    exit_record['binary_sha256'] == sha(exe), 'actual native normal closure')
            require(not NATIVE_ERROR.search(err.read_text()), 'native/collector error')
            record['native_exit'] = exit_record
            # Unit trace normally unlinks its own named session at exit. If it
            # remains, use only the same pinned private session's public stop CLI.
            if shm.exists():
                control('stop')
            record['session_released'] = not shm.exists()
            require(record['session_released'], 'collector session release')
            budget(force=True)
            files = [p for p in reports.rglob('*') if p.is_file()]
            record['provider_files'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(files)]
            native_dirs = [p for p in reports.glob('*.{}'.format(record['actual_native']['pid'])) if p.is_dir()]
            require(len(native_dirs) == 1, 'one native collector directory')
            traces = list(native_dirs[0].glob('*.json'))
            traces = [p for p in traces if p.name != 'run_config.json']
            require(len(traces) == 1, 'one native Chrome trace')
            trace = json.loads(traces[0].read_text())
            events = trace['traceEvents']
            counts = collections.Counter((e.get('ph'), e.get('cat')) for e in events)
            require(counts[('X', 'gpu_op')] > 0 and counts[('X', 'cpu_op')] > 0, 'host/device window events missing')
            require(all(math.isfinite(float(e['ts'])) and math.isfinite(float(e['dur'])) and float(e['dur']) >= 0
                        for e in events if e.get('ph') == 'X'), 'invalid trace intervals')
            record.update(provider_window_valid=True,
                native_trace=dict(path=str(traces[0]), sha256=sha(traces[0]), events=len(events),
                    event_counts=[dict(phase=k[0], category=k[1], count=n) for k, n in counts.items()]))
            require(all(sha(P/n) == h for n, h in build['source_pins'].items()) and sha(exe) == build['binary_sha256'], 'source/binary changed during capture')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                # Never call the model's old session scan after its anchor is reaped.
                # A leftover private control object can be released only if all
                # original owned commands are conclusively closed.
                if owner.active is None and controls.active is None and all(module.closed(c) for c in owner.commands+controls.commands):
                    if shm.exists() and 'session_object' in record and not any(c['label'] == 'session-stop' for c in controls.commands):
                        control('stop')
                    record['session_released'] = not shm.exists()
                if cursor and owner.active is None:
                    entries = [json.loads(line) for line in utility('journal-final',
                        ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if line.strip()]
                    record.update(fresh_faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE', '')))],
                        driver_dump_exists=(DEVICE/'devcoredump').exists(), boot_unchanged=boot_id() == record['boot_id'])
            finally:
                record.update(active=owner.active is not None or controls.active is not None, finished_utc=module.utc())
                if (record['active'] or not all(module.closed(c) for c in owner.commands+controls.commands) or
                        'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or
                        record.get('driver_dump_exists') or not record.get('session_released')):
                    record.update(passed=False, complete=False)
                persist()
                with receipt.open('x') as stream:
                    json.dump(record, stream, indent=2)
                    stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], mode=mode,
                             actual_input_tokens=len(ids), native_trace=record.get('native_trace'))), flush=True)


if __name__ == '__main__':
    # The launcher respects this umask when it creates its public session shm.
    os.umask(0o077)
    main()
