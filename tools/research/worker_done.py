#!/usr/bin/env python3
"""Send one successful Orca research completion with hashes from its files.

Run from the worker report directory after the final inbox check. Supply the
same Orca executable and task/dispatch IDs as in the dispatched preamble.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('orca', 'task_id', 'dispatch_id', 'subject', 'summary'):
        parser.add_argument(name)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    report = root / 'report.md'
    filesystem = root / 'worker-filesystem-proof.json'
    ipc = root / 'worker-orca-status.json'
    mount = json.loads(filesystem.read_bytes())
    ipc_bytes = ipc.read_bytes()
    receipt = json.loads(ipc_bytes)
    if not (mount['passed'] and mount['report_directory'] == str(root)
            and receipt['ok'] and receipt['reachable']
            and receipt['task_id'] == args.task_id and receipt['dispatch_id'] == args.dispatch_id):
        parser.error('the recorded worker proofs must match this directory and dispatched task')
    report_bytes = report.read_bytes()
    if not report_bytes.strip():
        parser.error('write a nonempty report.md before reporting success')

    body = (f'{args.summary.strip()}\n\n'
            f'task_id={args.task_id} dispatch_id={args.dispatch_id}\n'
            f'report.md={hashlib.sha256(report_bytes).hexdigest()}\n'
            f'worker-orca-status.json={hashlib.sha256(ipc_bytes).hexdigest()}')
    subprocess.run([args.orca, 'orchestration', 'send', '--from', receipt['terminal'],
                    '--type', 'worker_done', '--subject', args.subject, '--body', body,
                    '--task-id', args.task_id, '--dispatch-id', args.dispatch_id, '--outcome', 'succeeded',
                    '--files-modified', ','.join(p.name for p in (report, filesystem, ipc)),
                    '--report-path', str(report), '--json'], check=True, timeout=30)


if __name__ == '__main__':
    main()
