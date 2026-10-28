#!/usr/bin/env python3
"""Finish the existing frozen cohort; no retry, new plan or adoption.

Main-only one-use orchestration. Wait for the exact running qualifier3 parent,
commit its closed result, then execute4 and5 serially with the existing owner,
budgets and source/runtime/native gates. Retire this file after cohort closure.
"""
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import qualify_qsa_k_model as q

SELF = str(Path(__file__).resolve().relative_to(q.P))
REL = str(q.RESULT.relative_to(q.P))
require = q.require


def git(*args):
    return subprocess.check_output(['git', *args], cwd=q.P, text=True)


def save(index):
    require(git('branch', '--show-current').strip() == 'feature/sycl', 'wrong branch')
    require(not git('diff', '--cached', '--name-only').strip(), 'unrelated staged edits')
    old = json.loads(git('show', 'HEAD:' + REL))
    new = json.loads(q.RESULT.read_text())
    a, b = old[q.m.KEY][q.KEY], new[q.m.KEY][q.KEY]
    require(b == q.compact(json.loads((q.ROOT / 'record.json').read_text())) and
            len(a['processes']) == index and len(b['processes']) == index + 1 and
            b['processes'][:-1] == a['processes'] and
            b['processes'][-1]['index'] == index and not b['active'],
            'exact new closed process required')
    # The existing controller alone may update its cohort; preserve everything
    # else, including failed experiments, source identities and frozen plans.
    allowed = {'stage', 'complete', 'passed', 'ownership_closed', 'active', 'processes', 'decision'}
    require({k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} <= allowed,
            'frozen cohort definition changed')
    old[q.m.KEY][q.KEY] = b
    require(old == new, 'unrelated result edits')
    subprocess.run(['git', 'diff', '--check', '--', REL], cwd=q.P, check=True)
    subprocess.run(['git', 'add', '--', REL], cwd=q.P, check=True)
    subprocess.run(['git', 'commit', '-m', f'perf(sycl): retain official cohort process{index}'],
                   cwd=q.P, check=True)
    subprocess.run(['git', 'push', 'origin', 'feature/sycl'], cwd=q.P, check=True)
    result = b['processes'][-1]
    require(result['passed'] and q.accepted(result) and b['ownership_closed'],
            'failed result preserved;stop without retry or next process')
    print(f'Process{index} closed and committed;no adoption implied', flush=True)


def main():
    require(len(sys.argv) == 3, 'PID START_TICKS of existing qualifier3 required')
    require(git('show', 'HEAD:' + SELF) == (q.P / SELF).read_text(), 'commit exact coordinator first')
    pid, start = map(int, sys.argv[1:])
    identity = q.m.runtime.supervision.identity(pid)
    require(identity and identity['start_ticks'] == start, 'exact live qualifier3 required')
    argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    require(len(argv) >= 4 and argv[1].endswith(b'/qualify_qsa_k_model.py') and
            argv[2:4] == [b'run', b'3'], 'wrong predecessor command')
    fd = os.pidfd_open(pid)
    try:
        poll = select.poll(); poll.register(fd, select.POLLIN)
        deadline = time.monotonic() + 5400
        while not poll.poll(50000):
            require(time.monotonic() < deadline, 'predecessor deadline;no duplicate launch')
        print('Exact qualifier3 parent exited;checking its closed result', flush=True)
    finally:
        os.close(fd)
    with (q.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(3)
        for index in (4, 5):
            try:
                q.run(index)
            finally:
                # Failed closed evidence is committed too. A pre-admission
                # refusal has no new row and stops here without rewriting it.
                save(index)


if __name__ == '__main__':
    main()
