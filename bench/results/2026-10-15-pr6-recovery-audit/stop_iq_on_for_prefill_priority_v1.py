#!/usr/bin/env python3
"""One-shot user-priority cancellation of our exact, already-owned bench client.

Only SIGINT the pinned HTTP benchmark child. Its existing parent finally block
owns graceful HTTP-server/native shutdown. No GPU/native/reset/foreign signals.
The original owner's final receipt must remain incomplete, not be rewritten PASS.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import signal

A = Path(__file__).resolve().parent
P = A.parents[2]
ROOT = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-selective-iq-clean-bench-full-1-root-v2')


def identity(pid):
    f = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, state=f[0], ppid=int(f[1]), sid=int(f[3]), start_ticks=int(f[19]))


def main():
    leader = dict(pid=323620, sid=323620, start_ticks=39561427)
    bench = dict(pid=326347, ppid=323620, sid=323620, start_ticks=39568041)
    state = json.loads((ROOT / 'record.json').read_text())
    command = next(c for c in state['commands'] if c['label'] == 'bench-full-1')
    assert state['active'] and command['pid'] == leader['pid']
    expected_argv = ['python3', 'tools/strata_bench.py', '--url', 'http://127.0.0.1:18097',
                     '--config', str(ROOT / 'config.json'), '--runs', '3', '--out', str(ROOT / 'stock-results'),
                     '--ready-timeout', '480', '--long', '--think', '--fail-fast-engine-error',
                     '--long-generation', '--workload-id', '1792036964']
    for expected in (leader, bench):
        now = identity(expected['pid'])
        assert all(now[k] == v for k, v in expected.items()) and now['state'] != 'Z'
    assert Path('/proc', str(bench['pid']), 'cwd').resolve() == P
    argv = [s.decode() for s in Path('/proc', str(bench['pid']), 'cmdline').read_bytes().split(b'\0') if s]
    assert argv == expected_argv
    text = (ROOT / 'owned/bench.stdout').read_text()
    cohort = text.split('32K prompt:', 1)[1].split('long generation:', 1)[0]
    assert all(f'run {i}: 32089 tokens, prompt ' in cohort for i in (1, 2, 3))
    fd = os.pidfd_open(bench['pid'])
    try:
        watcher = select.poll(); watcher.register(fd, select.POLLIN)
        assert not watcher.poll(0)
        now = identity(bench['pid'])
        assert all(now[k] == v for k, v in bench.items())
        record = dict(schema='owned-bench-priority-stop-v1',
                      reason='User requests exclusive long-prefill optimization until PP exceeds600; retire remaining decode workload',
                      utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      leader=leader, benchmark=bench, benchmark_argv=argv,
                      prefill_cohort_already_reported=True, signal='SIGINT', target='benchmark-client-only',
                      sent=False, original_suite_complete=False, performance_adopted=False)
        destination = A / 'selective-iq-on-priority-stop-v1.json'
        with destination.open('x') as stream:
            json.dump(record, stream, indent=2); stream.write('\n')
        signal.pidfd_send_signal(fd, signal.SIGINT)
        record['sent'] = True
        destination.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(sent=True, pid=bench['pid'], signal='SIGINT', native_signaled=False)))
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
