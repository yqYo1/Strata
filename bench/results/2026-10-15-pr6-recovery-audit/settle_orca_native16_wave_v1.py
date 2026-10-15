#!/usr/bin/env python3
"""Settle this fixed wave without printing research or delivery bodies."""
import hashlib
import json
from pathlib import Path
import subprocess

A = Path(__file__).resolve().parent
RUN = 'run_c8f478972ec8'
HANDLE = 'term_6fbfd718-e3a2-46d8-ab62-73c36d2851a6'


def call(args):
    p = subprocess.run(['orca', 'orchestration', *args, '--json'],
                       capture_output=True, text=True, timeout=30)
    r = json.loads(p.stdout)
    assert p.returncode == 0 and r['ok'], (p.returncode, p.stderr)
    return r


def main():
    names = {781: 'orca-implement-query-registers-round781-launch-v1.json',
             782: 'orca-research-round782-launch-v1.json',
             783: 'orca-research-round783-launch-v1.json',
             784: 'orca-implement-native16-gemm-round784-launch-v1.json',
             786: 'orca-research-round786-launch-v1.json',
             787: 'orca-implement-native16-attention-round787-launch-v1.json',
             789: 'orca-research-round789-launch-v1.json',
             791: 'orca-research-round791-launch-v1.json',
             793: 'orca-research-round793-launch-v1.json',
             794: 'orca-research-round794-launch-v1.json',
             797: 'orca-research-round797-launch-v1.json',
             798: 'orca-research-round798-launch-v1.json',
             799: 'orca-implement-native16-onednn-round799-launch-v1.json',
             802: 'orca-implement-native16-quality-round802-launch-v1.json',
             803: 'orca-implement-paired-gemm-bench-round803-launch-v1.json',
             804: 'orca-implement-native16-onednn-functional-round804-launch-v1.json',
             805: 'orca-research-round805-launch-v1.json',
             807: 'orca-research-round807-launch-v1.json',
             809: 'orca-implement-onednn-qualification-round809-launch-v1.json',
             810: 'orca-implement-vllm-xe2-native16-round810-launch-v1.json',
             811: 'orca-research-round811-launch-v1.json',
             813: 'orca-research-round813-launch-v1.json'}
    known = {}
    for n, name in names.items():
        v = json.loads((A / name).read_text())['receipt']['result']
        known[v['dispatchId']] = (n, v['taskId'])
    statefile = A / 'orca-native16-wave-settlement-state-v1.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else dict(
        processed=[], completed={}, check_count=0, pending_receipt=None)
    pending = A / state['pending_receipt'] if state['pending_receipt'] else None
    while True:
        if pending is None or not json.loads(pending.read_text())['result'].get('deliveryId'):
            r = call(['check', '--terminal', HANDLE, '--run', RUN])
            state['check_count'] += 1
            pending = A / f"orca-native16-wave-delivery-{state['check_count']}-v1.json"
            assert not pending.exists()
            pending.write_text(json.dumps(r, indent=2) + '\n')
            state['pending_receipt'] = pending.name
            statefile.write_text(json.dumps(state, indent=2) + '\n')
        r = json.loads(pending.read_text())['result']
        for m in r['messages']:
            if m['id'] in state['processed']:
                continue
            p = json.loads(m['payload']) if isinstance(m['payload'], str) else m['payload']
            n, task = known[p['dispatchId']]
            assert p['taskId'] == task
            if m['type'] == 'heartbeat':
                state['processed'].append(m['id'])
                continue
            assert m['type'] == 'worker_done' and p['outcome'] in ('succeeded', 'failed')
            path = Path(p['reportPath'])
            if not path.is_absolute():
                path = A.parents[2] / path
            assert path.exists() and path.is_relative_to(A)
            settlement = A / f'orca-round{n}-settlement-v1.json'
            if not settlement.exists():
                settlement.write_text(json.dumps(call(['worker-show', '--dispatch', p['dispatchId']]), indent=2) + '\n')
            check = json.loads(settlement.read_text())['result']
            d = check['dispatch']
            assert d['taskId'] == task and d['id'] == p['dispatchId']
            assert d['status'] == ('completed' if p['outcome'] == 'succeeded' else 'failed')
            assert check['projection']['outcome'] == p['outcome']
            release = A / f'orca-round{n}-release-v1.json'
            if not release.exists():
                release.write_text(json.dumps(call(['worker-release', '--dispatch', p['dispatchId']]), indent=2) + '\n')
            state['completed'][str(n)] = dict(path=str(path.relative_to(A)), outcome=p['outcome'],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
            state['processed'].append(m['id'])
            statefile.write_text(json.dumps(state, indent=2) + '\n')
            print(json.dumps(dict(round=n, outcome=p['outcome'], body_printed=False)), flush=True)
        statefile.write_text(json.dumps(state, indent=2) + '\n')
        if not r.get('deliveryId'):
            break
        following = call(['check', '--terminal', HANDLE, '--run', RUN, '--ack', r['deliveryId']])
        state['check_count'] += 1
        pending = A / f"orca-native16-wave-delivery-{state['check_count']}-v1.json"
        assert not pending.exists()
        pending.write_text(json.dumps(following, indent=2) + '\n')
        state['pending_receipt'] = pending.name
        statefile.write_text(json.dumps(state, indent=2) + '\n')
        if not following['result'].get('deliveryId'):
            break
    print(json.dumps(dict(completed=sorted(state['completed']), bodies_printed=False)))


if __name__ == '__main__':
    main()
