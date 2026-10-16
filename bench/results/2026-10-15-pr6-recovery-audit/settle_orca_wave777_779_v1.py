#!/usr/bin/env python3
"""Settle known research deliveries; print identifiers, never report bodies."""
import hashlib
import json
from pathlib import Path
import subprocess
A = Path(__file__).resolve().parent
RUN = 'run_c8f478972ec8'
HANDLE = 'term_6fbfd718-e3a2-46d8-ab62-73c36d2851a6'


def call(args):
    p = subprocess.run(['orca', 'orchestration', *args, '--json'], capture_output=True, text=True, timeout=30)
    r = json.loads(p.stdout)
    assert p.returncode == 0 and r['ok'], (p.returncode, p.stderr)
    return r


def main():
    known = {}
    for n in range(777, 780):
        r = json.loads((A / f'orca-research-round{n}-launch-v1.json').read_text())['receipt']['result']
        known[r['dispatchId']] = dict(round=n, task=r['taskId'])
    statefile = A / 'orca-wave777-779-settlement-state-v1.json'
    if statefile.exists():
        state = json.loads(statefile.read_text())
        pending = A / state['pending_receipt']
    else:
        state = dict(processed=[], completed={}, pending_receipt='orca-wave777-779-delivery-initial-v1.json', check_count=0)
        pending = A / state['pending_receipt']
    if not json.loads(pending.read_text())['result'].get('deliveryId'):
        following = call(['check', '--terminal', HANDLE, '--run', RUN])
        state['check_count'] += 1
        pending = A / f"orca-wave777-779-delivery-ack-{state['check_count']}-v1.json"
        assert not pending.exists()
        pending.write_text(json.dumps(following, indent=2) + '\n')
        state['pending_receipt'] = pending.name
        statefile.write_text(json.dumps(state, indent=2) + '\n')
    while True:
        r = json.loads(pending.read_text())['result']
        for message in r['messages']:
            if message['id'] in state['processed']:
                continue
            payload = message['payload']
            if isinstance(payload, str):
                payload = json.loads(payload)
            k = known[payload['dispatchId']]
            assert payload['taskId'] == k['task']
            if message['type'] == 'heartbeat':
                state['processed'].append(message['id'])
                statefile.write_text(json.dumps(state, indent=2) + '\n')
                continue
            assert message['type'] == 'worker_done', 'Unexpected message: retain delivery, inspect operationally'
            assert payload['outcome'] in ('succeeded', 'failed')
            n = k['round']
            report = A / f'research-round{n}-v1.txt'
            assert payload['reportPath'].endswith(str(report.relative_to(A.parents[2]))) and report.is_file()
            settlement = A / f'orca-round{n}-settlement-v1.json'
            if settlement.exists():
                check = json.loads(settlement.read_text())
            else:
                check = call(['worker-show', '--dispatch', payload['dispatchId']])
                settlement.write_text(json.dumps(check, indent=2) + '\n')
            d = check['result']['dispatch']
            expected_status = 'completed' if payload['outcome'] == 'succeeded' else 'failed'
            assert d['id'] == payload['dispatchId'] and d['taskId'] == k['task'] and d['status'] == expected_status
            assert check['result']['projection']['outcome'] == payload['outcome']
            release = A / f'orca-round{n}-release-v1.json'
            if not release.exists():
                release.write_text(json.dumps(call(['worker-release', '--dispatch', payload['dispatchId']]), indent=2) + '\n')
            state['completed'][str(n)] = dict(report=str(report.name), sha256=hashlib.sha256(report.read_bytes()).hexdigest(), outcome=payload['outcome'], release_receipt=release.name)
            state['processed'].append(message['id'])
            statefile.write_text(json.dumps(state, indent=2) + '\n')
            print(json.dumps(dict(round=n, outcome=payload['outcome'], report=report.name, body_read_by_Main=False)), flush=True)
        args = ['check', '--terminal', HANDLE, '--run', RUN]
        if r.get('deliveryId'):
            args += ['--ack', r['deliveryId']]
        else:
            statefile.write_text(json.dumps(state, indent=2) + '\n')
            break
        following = call(args)
        state['check_count'] += 1
        pending = A / f"orca-wave777-779-delivery-ack-{state['check_count']}-v1.json"
        assert not pending.exists()
        pending.write_text(json.dumps(following, indent=2) + '\n')
        state['pending_receipt'] = pending.name
        statefile.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(dict(completed=sorted(state['completed']), body_read_by_Main=False)))


if __name__ == '__main__':
    main()
