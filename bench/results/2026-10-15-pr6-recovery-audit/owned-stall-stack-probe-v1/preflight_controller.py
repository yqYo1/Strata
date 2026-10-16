#!/usr/bin/env python3
"""Main may run AFTER separately building the two pure host fixture artifacts.

This is proposal source, NOT_RUN. No compiler/GPU/server/retry inside controller.
Only pinned direct_owner may spawn, signal or reap target and GDB.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
from owned_stack_probe import JOB_CAP, StackProbe, boot_id, load_owner, require, sha

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_OWNER = PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEFAULT_LOCK = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/owned-v0141-measurement.lock')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--shim', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--owner', type=Path, default=DEFAULT_OWNER)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    require(os.getuid() != 0, 'preflight must run as ordinary Main user')
    require(Path('/proc/sys/kernel/yama/ptrace_scope').read_text().strip() == '1',
            'preflight scoped to Yama ptrace_scope=1; do not change sysctl')
    require(Path('/usr/bin/gdb').is_file(), 'installed /usr/bin/gdb required')
    module = load_owner(args.owner.resolve())
    target, shim, output = args.target.resolve(), args.shim.resolve(), args.output.resolve()
    require(target.is_file() and shim.is_file(), 'Main must build host target and preload first')
    # The caller must explicitly supply artifacts built from host_target.c and
    # ptracer_parent.c. This controller never accepts arbitrary process PIDs.
    with args.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not output.exists(), 'fresh private output required')
        output.mkdir(mode=0o700)
        record = dict(schema='owned-stack-host-preflight-v1', active=True, passed=False,
                      complete=False, GPU_executed=False, boot_id=boot_id(),
                      controller_pid=os.getpid(), commands=[], debugger_commands=[], probe={},
                      inputs={str(p): sha(p) for p in (target, shim, args.owner.resolve(), Path('/usr/bin/gdb'))},
                      source={p.name: sha(p) for p in Path(__file__).parent.iterdir()
                              if p.suffix in ('.py', '.c')}, lock=str(args.lock.resolve()),
                      original_target_exit_expected=7, performance_qualified=False)

        def persist():
            (output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        native_owner = module.Owner(output / 'native-owned', persist)
        debug_owner = module.Owner(output / 'gdb-owned', persist)
        record['commands'], record['debugger_commands'] = native_owner.commands, debug_owner.commands
        persist()
        env = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', HOME=str(output), LD_PRELOAD=str(shim))

        def interaction(io, entry):
            require(io.line(10) == f'READY {io.proc.pid} {os.getpid()}', 'host target/parent handshake')
            stderr = native_owner.output / 'host-target.stderr'
            require(stderr.read_text() == f'owned-ptracer grant pid={io.proc.pid} parent={os.getpid()}\n',
                    'preload grant must precede target READY')
            io.send(b'PING\n')
            first = io.line(5).split()
            require(len(first) == 2 and first[0] == 'PONG', 'host initial heartbeat')
            probe = StackProbe(module, debug_owner, io, entry, record['probe'], persist,
                               PROJECT, silence_seconds=150, shutdown_payload=b'QUIT\n')
            try:
                capture = probe.capture_now('host_preflight_explicit')
                require(len(set(capture['backtrace_thread_ids'])) >= 2,
                        'host main and worker stacks must both be present')
                io.send(b'PING\n')
                second = io.line(5).split()
                require(len(second) == 2 and second[0] == 'PONG', 'post-detach target response')
                require(int(second[1]) > int(first[1]), 'CPU worker did not advance')
                capture['resumed_proven'] = True
                record['heartbeats'] = [int(first[1]), int(second[1])]
                io.send(b'QUIT\n')
                io.exit(5)
                persist()
            finally:
                probe.close()

        try:
            command, _, _ = native_owner.run(
                'host-target', [str(target)], env, PROJECT, interaction=interaction,
                wall=65, cpu=60, rss_cap=256 << 20, text_cap=1 << 20,
                total_cap=JOB_CAP, file_cap=1 << 20)
            require(module.closed(command), 'host native ownership not closed')
            require(command['exit_code'] == 7, 'original target exit not preserved')
            require(len(debug_owner.commands) == 1 and module.closed(debug_owner.commands[0]),
                    'exactly one normally closed debugger required')
            require(record['probe']['captures'][0]['resumed_proven'], 'resume not proven')
            require(boot_id() == record['boot_id'], 'boot changed')
            require(sha(target) == record['inputs'][str(target)] and sha(shim) == record['inputs'][str(shim)],
                    'host artifacts changed')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = native_owner.active is not None or debug_owner.active is not None
            record['finished_utc'] = module.utc()
            persist()


if __name__ == '__main__':
    main()
