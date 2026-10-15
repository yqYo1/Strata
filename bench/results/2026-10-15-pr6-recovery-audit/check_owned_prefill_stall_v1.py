#!/usr/bin/env python3
"""One owned diagnostic; host gate first, no HTTP/retry/native rebuild."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HOST = BASE / 'main-owned-stack-host-preflight-root-v2'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import StackProbe, StallCaptured, load_owner, require, sha, boot_id

FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


class ProgressProbe(StackProbe):
    """Observe real layer/chunk progress, without treating warnings as progress."""
    def __init__(self, *args, stderr, **kwargs):
        self.stderr = stderr
        self.trace_offset = stderr.stat().st_size
        self.trace_pending = b''
        self.next_trace_check = 0
        super().__init__(*args, **kwargs)
        self.record['progress_policy'] = 'consumed stdout bytes or new stderr layer/chunk/issuer trace'

    def poll(self):
        now = time.monotonic()
        if now >= self.next_trace_check:
            self.next_trace_check = now + .05
            with self.stderr.open('rb') as stream:
                stream.seek(self.trace_offset)
                data = stream.read((256 << 10) + 1)
            require(len(data) <= 256 << 10, 'stderr progress read cap')
            self.trace_offset += len(data)
            lines = (self.trace_pending + data).split(b'\n')
            self.trace_pending = lines.pop()
            require(len(self.trace_pending) <= 65536, 'stderr progress line cap')
            for line in lines:
                if re.match(rb'^strata trace: (?:layer-major layer |prompt chunk |prefill issuer )', line):
                    self.last_progress = now
                    self.record['semantic_progress_count'] = self.record.get('semantic_progress_count', 0) + 1
                    self.record['last_semantic_progress'] = line.decode(errors='replace')[:400]
        super().poll()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['host-auto', 'model'])
    args = parser.parse_args()
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    root = BASE / ('main-owned-prefill-stall-' + args.mode + '-root-v1')
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host_gate = json.loads((HOST / 'preflight-v1/record.json').read_text())
        require(host_gate['passed'] and host_gate['complete'] and not host_gate['active'], 'host gate not closed/pass')
        require(not root.exists(), 'fresh independent diagnostic root required')
        root.mkdir(mode=0o700)
        shim, target = HOST / 'libowned_ptracer_parent.so', HOST / 'host_target'
        require(sha(shim) == host_gate['inputs'][str(shim)], 'host-tested shim changed')
        record = dict(schema='main-owned-prefill-stall-v1', mode=args.mode, active=True,
                      complete=False, passed=False, GPU_executed=args.mode == 'model',
                      boot_id=boot_id(), commands=[], debugger_commands=[], requests=[],
                      recipe_sha256=sha(Path(__file__)), host_gate_sha256=sha(HOST / 'preflight-v1/record.json'),
                      shim_sha256=sha(shim), probe={}, performance_qualified=False, adopted=False,
                      quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
                      scope='Independent diagnostic. Does not repair historical failed Direct0/1 performance pair.')

        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        native_owner = module.Owner(root / 'native-owned', persist)
        debugger_owner = module.Owner(root / 'gdb-owned', persist)
        record['commands'], record['debugger_commands'] = native_owner.commands, debugger_owner.commands
        host_env = dict(PATH='/usr/bin:/bin', HOME=str(root), LANG='C', LC_ALL='C')

        def utility(label, argv):
            entry, out, err = native_owner.run(label, argv, host_env, PROJECT, wall=30, cpu=30,
                                               rss_cap=2 << 30, text_cap=1 << 20,
                                               total_cap=256 << 20, file_cap=1 << 20)
            require(module.closed(entry) and entry['exit_code'] == 0, label + ' failed: ' + err.read_text()[-1000:])
            return out.read_text()

        cursor = None
        try:
            if args.mode == 'host-auto':
                require(sha(target) == host_gate['inputs'][str(target)], 'host target changed')
                env = dict(host_env, LD_PRELOAD=str(shim))

                def interaction(io, entry):
                    require(io.line(10) == f'READY {io.proc.pid} {os.getpid()}', 'host READY')
                    io.send(b'PING\n')
                    first = int(io.line(5).split()[1])
                    probe = ProgressProbe(module, debugger_owner, io, entry, record['probe'], persist,
                                          PROJECT, stderr=root / 'native-owned/host-auto.stderr',
                                          silence_seconds=.25, shutdown_payload=b'PING\nQUIT\n')
                    try:
                        try:
                            io.line(10)
                            raise RuntimeError('automatic silence capture did not occur')
                        except StallCaptured:
                            require(probe.requests_stopped, 'request gate did not close')
                            try:
                                io.send(b'GEN 1 1\n')
                            except RuntimeError:
                                record['forbidden_send_blocked'] = True
                            else:
                                raise RuntimeError('new request allowed after capture')
                            io.send(b'PING\nQUIT\n')
                            second = int(io.line(5).split()[1])
                            require(second > first, 'post-capture worker did not advance')
                            record['heartbeats'] = [first, second]
                            record['probe']['captures'][0]['resumed_proven'] = True
                            io.exit(5)
                    finally:
                        probe.close()

                entry, _, _ = native_owner.run('host-auto', [str(target)], env, PROJECT, interaction=interaction,
                                                wall=65, cpu=60, rss_cap=256 << 20,
                                                text_cap=1 << 20, total_cap=256 << 20, file_cap=1 << 20)
                require(module.closed(entry) and entry['exit_code'] == 7, 'automatic host ownership/exit')
                require(len(debugger_owner.commands) == 1 and module.closed(debugger_owner.commands[0]), 'one closed GDB required')
                record.update(passed=True, complete=True, result='automatic_capture_gate_PASS')
            else:
                automatic = json.loads((BASE / 'main-owned-prefill-stall-host-auto-root-v1/record.json').read_text())
                require(automatic['passed'] and automatic['complete'] and not automatic['active'], 'automatic capture gate not pass')
                prior = json.loads((HERE / 'prompt-direct-bench-full-0-v1.json').read_text())
                build = json.loads((HERE / 'cpu-gu-reuse-build-v3.json').read_text())
                inputs = json.loads((HERE / 'prompt-direct-bench-failure-evidence-v1.json').read_text())['reconstructed_public_requests']
                exe = Path(build['build_path']) / 'strata'
                require(sha(exe) == prior['binary_sha256'] == build['binaries']['strata'], 'native binary changed')
                require(all(sha(PROJECT / n) == h for n, h in build['source_pins'].items()), 'native sources changed')
                require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'device driver/dump gate')
                for name, pin in prior['model_identity'].items():
                    st = Path(name).stat()
                    require(st.st_size == pin['bytes'] and st.st_mtime_ns == pin['mtime_ns'], 'model identity changed')
                for kind in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
                    require(all(sha(Path(n)) == h for n, h in prior[kind].items()), 'model metadata changed')
                token_sets = []
                for item in inputs:
                    p = Path(item['tokens_path'])
                    require(sha(p) == item['tokens_sha256'], 'diagnostic token input changed')
                    ids = list(map(int, p.read_text().strip().split(',')))
                    require(len(ids) == item['actual_prompt_tokens'] == 32089, 'input token count')
                    token_sets.append(ids)
                env = dict(prior['environment'])
                for key in ('STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
                    env.pop(key, None)
                env.update(LD_PRELOAD=str(shim), XDG_CACHE_HOME=str(root / 'cache'), STRATA_TRACE='1',
                           UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                           UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                           UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                           UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
                require(not any('TOKEN' in k or 'API_KEY' in k or k.startswith(('UNITRACE_', 'PTI_')) for k in env), 'private/profiler env forbidden')
                for k, v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1', EnableDirectSubmission='0',
                                  STRATA_SYCL_PREFILL_SAME_QUEUE='1', SYCL_CACHE_PERSISTENT='0', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items():
                    require(env[k] == v, 'safety environment mismatch ' + k)
                argv = [str(exe), *prior['engine_args'], '--serve']
                record.update(argv=argv, environment=env, input=inputs, binary_sha256=sha(exe),
                              source_pins=build['source_pins'], historical_receipt_sha256=sha(HERE / 'prompt-direct-bench-full-0-v1.json'),
                              sequence_limit='Only the three exact32K requests; earlier chat/agent requests omitted. A non-reproduction does not exonerate the original sequence.',
                              limits=dict(wall_seconds=2400, silence_seconds=150, RSS_bytes=88 << 30, text_bytes=64 << 20, job_bytes=256 << 20))
                cursor = re.search(r'-- cursor: (\S+)', utility('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
                record['journal_before_cursor'] = cursor

                def interaction(io, entry):
                    while True:
                        line = io.line(600)
                        if line.startswith('READY '):
                            require(line == 'READY 262144 stop', 'native READY mismatch')
                            break
                    stderr = root / 'native-owned/model.stderr'
                    grant = f'owned-ptracer grant pid={io.proc.pid} parent={os.getpid()}\n'
                    require(grant in stderr.read_text(), 'tested per-process ptrace grant missing')
                    probe = ProgressProbe(module, debugger_owner, io, entry, record['probe'], persist, PROJECT,
                                          stderr=stderr, silence_seconds=150, shutdown_payload=b'QUIT\n')

                    def consume(line, request):
                        if re.fullmatch(r'T \d+', line):
                            request['generated_ids'].append(int(line[2:]))
                            require(len(request['generated_ids']) <= 24, 'generation count cap')
                        elif re.fullmatch(r'(?:RESUME \d+|REUSED \d+|PP \d+ \d+ [\d.]+ [\d.]+)', line):
                            request['progress'].append(line)
                        elif line.startswith('DONE '):
                            f = line.split()
                            require(len(f) == 15, 'DONE shape')
                            request['done'] = dict(generated=int(f[1]), prompt=int(f[2]), prompt_ms=float(f[3]),
                                                   decode_ms=float(f[4]), finish=f[5], accepted=int(f[6]), offered=int(f[7]),
                                                   reused=int(f[8]), read=int(f[14]))
                            require(request['done']['generated'] == len(request['generated_ids']) and
                                    request['done']['prompt'] == request['done']['read'] == 32089 and
                                    request['done']['reused'] == 0, 'DONE work mismatch')
                            return True
                        else:
                            raise RuntimeError('unexpected native protocol ' + line[:200])
                        return False

                    try:
                        try:
                            for index, ids in enumerate(token_sets):
                                probe.require_requests_allowed()
                                request = dict(index=index, started_utc=module.utc(), generated_ids=[], progress=[], complete=False)
                                record['requests'].append(request)
                                wire = ('GEN 24 ' + ','.join(map(str, ids)) + '\n').encode()
                                import hashlib
                                request['wire_sha256'] = hashlib.sha256(wire).hexdigest()
                                persist()
                                print('Diagnostic request ' + str(index) + ' input32089/gen24', flush=True)
                                io.send(wire)
                                while not consume(io.line(1000), request):
                                    pass
                                request.update(complete=True, finished_utc=module.utc())
                                persist()
                                print('Diagnostic DONE ' + json.dumps(request['done']), flush=True)
                            io.send(b'QUIT\n')
                            io.exit(60)
                        except StallCaptured:
                            record['stall_captured'] = True
                            persist()
                            print('Diagnostic captured one owned thread stack; no further requests', flush=True)
                            # Observe original watchdog exit, or finish this one request and QUIT.
                            deadline = time.monotonic() + 630
                            with selectors.DefaultSelector() as ready:
                                ready.register(io.proc.stdout, selectors.EVENT_READ)
                                while not io.exit_watch.poll(0) or not io.eof:
                                    io.poll()
                                    require(time.monotonic() < deadline, 'post-capture exit deadline')
                                    if ready.select(.02):
                                        io.read()
                                        while b'\n' in io.pending:
                                            raw, _, tail = io.pending.partition(b'\n')
                                            io.pending[:] = tail
                                            if consume(raw.decode().strip(), record['requests'][-1]):
                                                io.send(b'QUIT\n')
                            require(not io.pending.strip(), 'partial final protocol')
                    finally:
                        probe.close()

                entry, _, err = native_owner.run('model', argv, env, PROJECT, interaction=interaction,
                                                 wall=2400, cpu=2400 * len(os.sched_getaffinity(0)),
                                                 rss_cap=86 << 30, text_cap=(64 << 20) - (256 << 10),
                                                 total_cap=256 << 20, file_cap=64 << 20)
                record['native_original_exit_code'] = entry.get('exit_code')
                require(module.closed(entry), 'native diagnostic ownership not normally closed')
                record['complete'] = True
                record['passed'] = entry['exit_code'] == 0 and not record.get('stall_captured') and all(x['complete'] for x in record['requests'])
                record['result'] = 'three_requests_no_stall_observed' if record['passed'] else 'diagnostic_failure_or_stall_preserved'
                require(sha(exe) == record['binary_sha256'], 'native binary changed during diagnostic')
                require(all(sha(PROJECT / n) == h for n, h in build['source_pins'].items()), 'native source changed during diagnostic')
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    entries = [json.loads(s) for s in utility('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if s.strip()]
                    record['fresh_faults'] = [e for e in entries if FAULT.search(str(e.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = boot_id() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record['passed'] = False
            finally:
                record['active'] = native_owner.active is not None or debugger_owner.active is not None
                record['finished_utc'] = module.utc()
                persist()
                (HERE / ('owned-prefill-stall-' + args.mode + '-v1.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(result=record['result'], passed=record['passed'], GPU_executed=record['GPU_executed'])), flush=True)


if __name__ == '__main__':
    main()
