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
    p = subprocess.run(['/home/yayoi/.nix-profile/bin/orca', 'orchestration', *args, '--json'],
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
             813: 'orca-research-round813-launch-v1.json',
             816: 'orca-implement-vllm-qualification-round816-launch-v1.json',
             817: 'orca-research-round817-launch-v1.json',
             818: 'orca-implement-native16-qsa-round818-launch-v1.json',
             820: 'orca-research-round820-launch-v1.json',
             821: 'orca-implement-vllm-engine-hook-round821-launch-v1.json',
             822: 'orca-research-round822-launch-v1.json',
             825: 'orca-research-round825-launch-v1.json',
             826: 'orca-research-round826-launch-v1.json',
             828: 'orca-research-round828-launch-v1.json',
             829: 'orca-research-round829-launch-v1.json',
             831: 'orca-research-round831-launch-v1.json',
             832: 'orca-research-round832-launch-v1.json',
             833: 'orca-implement-vllm-public-hook-round833-launch-v1.json',
             835: 'orca-research-round835-launch-v1.json',
             836: 'orca-research-round836-launch-v1.json',
             837: 'orca-implement-vllm-quality-round837-launch-v1.json',
             839: 'orca-research-round839-launch-v1.json',
             840: 'orca-research-round840-launch-v1.json',
             842: 'orca-research-round842-launch-v1.json',
             843: 'orca-research-round843-launch-v1.json',
             844: 'orca-implement-qsa16-scale-round844-launch-v1.json',
             846: 'orca-implement-vllm-public-qualification-round846-launch-v1.json',
             847: 'orca-research-round847-launch-v1.json',
             849: 'orca-research-round849-launch-v1.json',
             850: 'orca-research-round850-launch-v1.json',
             852: 'orca-implement-vllm-selected-quality-round852-launch-v1.json',
             854: 'orca-research-round854-launch-v1.json',
             855: 'orca-research-round855-launch-v1.json',
             856: 'orca-implement-qsa-device-events-round856-launch-v1.json',
             860: 'orca-implement-current-unitrace-round860-launch-v1.json',
             861: 'orca-implement-prompt-softmax-subgroups-round861-launch-v1.json',
             863: 'orca-research-round863-launch-v1.json',
             864: 'orca-implement-unitrace-direct-collector-round864-launch-v1.json',
             867: 'orca-research-round867-launch-v1.json',
             868: 'orca-research-round868-launch-v1.json',
             869: 'orca-research-round869-launch-v1.json',
             871: 'orca-research-round871-launch-v1.json',
             872: 'orca-research-round872-launch-v1.json',
             873: 'orca-research-round873-launch-v1.json',
             875: 'orca-research-round875-launch-v2.json',
             876: 'orca-research-round876-launch-v1.json',
             877: 'orca-research-round877-launch-v1.json',
             878: 'orca-resource-timeline-sol-launch-v1.json',
             879: 'orca-pr6-faithful-restore-sol-launch-v1.json',
             883: 'orca-pr6-independent-safety-review-sol-launch-v1.json',
             885: 'orca-research-round885-launch-v1.json',
             886: 'orca-research-round886-launch-v1.json',
             887: 'orca-metric-enumeration-sol-launch-v1.json',
             888: 'orca-pr6-safety-corrections-sol-launch-v2.json',
             890: 'orca-pr6-independent-safety-review-sol-launch-v2.json',
             891: 'orca-metric-query-tiny-sol-launch-v1.json',
             893: 'orca-research-round893-launch-v1.json',
             895: 'orca-metric-status-sol-launch-v2.json',
             896: 'orca-research-round896-launch-v1.json',
             897: 'orca-pr6-residual-fixes-sol-launch-v3.json',
             899: 'orca-metric-contract-sol-launch-v3.json',
             900: 'orca-pr6-independent-safety-review-sol-launch-v3.json',
             902: 'orca-research-round902-launch-v1.json',
             904: 'orca-pr6-prefill-api-harness-sol-launch-v1.json',
             905: 'orca-research-round905-launch-v1.json'}
    known = {}
    for n, name in names.items():
        receipt = json.loads((A / name).read_text())
        v = receipt.get('receipt', receipt)['result']
        known[v['dispatchId']] = (n, v['taskId'])
    statefile = A / 'orca-native16-wave-settlement-state-v1.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else dict(
        processed=[], completed={}, check_count=0, pending_receipt=None)
    pending = A / state['pending_receipt'] if state['pending_receipt'] else None
    while True:
        if pending is None or not json.loads(pending.read_text())['result'].get('deliveryId'):
            r = call(['check', '--terminal', HANDLE, '--run', RUN])
            if not r['result'].get('deliveryId'):
                assert not r['result']['messages']
                state['pending_receipt'] = None
                statefile.write_text(json.dumps(state, indent=2) + '\n')
                break
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
            if m['type'] == 'status':
                # A payload-free liveness notification is not completion.
                # Keep the durable delivery, without releasing a live worker.
                assert set(p) == {'taskId', 'dispatchId'}
                state['processed'].append(m['id'])
                continue
            if m['type'] == 'escalation' and set(p) == {'taskId', 'dispatchId'}:
                # Inspect the exact dispatch; a payload-free attention notice
                # never itself authorizes release, retry or a live-worker nudge.
                inspected = call(['worker-show', '--dispatch', p['dispatchId']])
                shown = inspected['result']
                assert shown['dispatch']['taskId'] == task and shown['dispatch']['id'] == p['dispatchId']
                assert shown['projection']['attention']['requiresAction'] is False
                archive = A / ('orca-wave-escalation-' + m['id'] + '-v1.json')
                if not archive.exists():
                    archive.write_text(json.dumps(inspected, indent=2) + '\n')
                state.setdefault('escalation_inspections', {})[m['id']] = archive.name
                state['processed'].append(m['id'])
                continue
            assert m['type'] == 'worker_done' and p['outcome'] in ('succeeded', 'failed')
            path = Path(p['reportPath'])
            workspace = (Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-pr6-prefill-restore')
                         if n in (879, 883, 888, 890, 897, 900, 904) else A.parents[2])
            if not path.is_absolute():
                path = workspace / path
            if n == 879:
                assert path == workspace / 'bench/results/2026-10-16-pr6-faithful-restore/report.ja.md'
            elif n == 883:
                assert path == workspace / 'bench/results/2026-10-16-pr6-faithful-restore/independent-safety-review-v1.md'
            elif n == 888:
                assert path == workspace / 'bench/results/2026-10-16-pr6-faithful-restore/safety-corrections-v2.md'
            elif n == 890:
                assert path == workspace / 'bench/results/2026-10-16-pr6-faithful-restore/independent-safety-review-v2.md'
            elif n == 897:
                assert path == workspace / 'bench/results/2026-10-16-pr6-faithful-restore/safety-corrections-v3.md'
            elif n == 900:
                assert path == workspace / 'bench/results/2026-10-16-pr6-faithful-restore/independent-safety-review-v3.md'
            elif n == 904:
                assert path == workspace / 'bench/results/2026-10-16-pr6-faithful-restore/prefill-api-harness-v1.md'
            else:
                assert path.is_relative_to(A)
            assert path.exists()
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
            state['completed'][str(n)] = dict(path=str(path if n in (879, 883, 888, 890, 897, 900, 904) else path.relative_to(A)), outcome=p['outcome'],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
            state['processed'].append(m['id'])
            statefile.write_text(json.dumps(state, indent=2) + '\n')
            print(json.dumps(dict(round=n, outcome=p['outcome'], body_printed=False)), flush=True)
        statefile.write_text(json.dumps(state, indent=2) + '\n')
        if not r.get('deliveryId'):
            break
        following = call(['check', '--terminal', HANDLE, '--run', RUN, '--ack', r['deliveryId']])
        if not following['result'].get('deliveryId'):
            assert not following['result']['messages']
            state['pending_receipt'] = None
            statefile.write_text(json.dumps(state, indent=2) + '\n')
            break
        state['check_count'] += 1
        pending = A / f"orca-native16-wave-delivery-{state['check_count']}-v1.json"
        assert not pending.exists()
        pending.write_text(json.dumps(following, indent=2) + '\n')
        state['pending_receipt'] = pending.name
        statefile.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(dict(completed=sorted(state['completed']), bodies_printed=False)))


if __name__ == '__main__':
    main()
