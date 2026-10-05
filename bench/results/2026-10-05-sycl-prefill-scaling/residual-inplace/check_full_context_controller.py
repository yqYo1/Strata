"""Check the full-context controller with a CPU protocol stub, never a GPU.

This checks request counts and rejects deliberately invalid replies. It does
not validate Strata's arithmetic, allocated context, or GPU execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


STUB = r'''#!/usr/bin/python3
import os, sys
from pathlib import Path
capacity = int(sys.argv[sys.argv.index('--max-context') + 1])
mode = os.environ.get('STRATA_CONTROLLER_STUB_MODE', 'valid')
print(f'READY {capacity} stop', flush=True)
for request in sys.stdin:
    if request.strip() == 'QUIT':
        break
    _, count, _, encoded = request.split()
    count = int(count)
    n = len(encoded.split(','))
    tail = n == capacity - 2 and count == 2
    if n + count > capacity:
        if mode == 'refusal-executes':
            print('strata trace: window 0 1', file=sys.stderr, flush=True)
        print(f'ERR prompt ({n} tokens) + max_new ({count}) exceeds the context ({capacity})', flush=True)
        continue
    if not (tail and mode == 'stale-head'):
        Path(os.environ['STRATA_DUMP_FIRST_LOGITS']).write_bytes(bytes(248320 * 4))
    windows = [(n - 1, 1)]
    if count > 1:
        windows.append((n, min(4, capacity - n)))
    if tail:
        if mode == 'overrun':
            windows[-1] = (n, 4)
        elif mode == 'last-cell-missing':
            windows[-1] = (n, 1)
        elif mode == 'wrong-tail':
            windows[-1] = (n + 1, 1)
    for pos, length in windows:
        print(f'strata trace: window {pos} {length}', file=sys.stderr, flush=True)
    print('REUSED 0')
    emitted = 1 if tail and mode == 'early-eos' else count
    for i in range(emitted):
        print(f'T {100 + i}')
        print('LP nan' if tail and mode == 'nan-logprob' else f'LP -0.5 {100 + i}:-0.5')
    prompt = n - 1 if tail and mode == 'wrong-count' else n
    finish = 'stop' if tail and mode in ('early-eos', 'wrong-finish') else 'length'
    print(f'DONE {emitted} {prompt} 1.0 1.0 {finish} 0 0 0 0 0 0 0 0.0 {n} 0', flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    controller = Path(__file__).with_name('check_full_context.py').resolve()
    # The controller removes inherited STRATA_* tuning variables. The stub
    # therefore receives its test mode through an ordinary environment key.
    stub = STUB.replace('STRATA_CONTROLLER_STUB_MODE', 'CONTROLLER_STUB_MODE')
    checks = [('valid', 64, True), ('valid', 262144, True)]
    checks += [(mode, 64, False) for mode in (
        'overrun', 'last-cell-missing', 'wrong-tail', 'early-eos',
        'wrong-count', 'wrong-finish', 'nan-logprob', 'stale-head',
        'refusal-executes')]
    result = {
        'scope': 'CPU protocol stub only; no real model or GPU execution',
        'controller_sha256': hashlib.sha256(controller.read_bytes()).hexdigest(),
        'stub_sha256': hashlib.sha256(stub.encode()).hexdigest(),
        'checks': [],
    }
    with tempfile.TemporaryDirectory(prefix='strata-context-controller-') as scratch:
        recovery = Path(scratch)
        exe = recovery / 'protocol-stub'
        exe.write_text(stub)
        exe.chmod(0o755)
        (recovery / 'coding-context-256k-tokens.txt').write_text('123 ' * 262144 + '\n')
        (recovery / 'reference.json').write_text(json.dumps({
            'runs': [{'args': ['--pack', 'unused', '--native', 'unused', '--expert-profile', 'unused']}],
            'env': {},
        }))
        for mode, capacity, expected in checks:
            name = f'{mode}-{capacity}'
            env = dict(os.environ, CONTROLLER_STUB_MODE=mode)
            command = ['/usr/bin/python3', str(controller), '--stage', 'serve',
                       '--context', str(capacity), '--recovery', str(recovery),
                       '--executable', str(exe)]
            # Only a CPU stub is timed out here; this never kills a GPU process.
            run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
            summary = json.loads((recovery / f'serve-{capacity}-{exe.stem}' / 'summary.json').read_text())
            (args.out / f'{name}.json').write_text(json.dumps(summary, indent=2) + '\n')
            (args.out / f'{name}.log').write_text(run.stdout + run.stderr)
            observed = run.returncode == 0 and summary['completed']
            assert observed == expected, (name, run.returncode, run.stderr)
            if not expected:
                error = summary.get('terminal_error', '')
                assert error.startswith(('AssertionError(', 'FileNotFoundError(')), (name, error)
            else:
                rows = summary['runs']
                assert len(rows) == 5
                assert [r['input_tokens'] for r in rows] == [capacity-4, capacity-2, capacity, capacity-2, 37]
                assert rows[1]['clipped_verify_tail'] == 2
                assert rows[1]['last_executed_kv_cell'] == capacity-1
            result['checks'].append({'name': name, 'exit_code': run.returncode,
                                     'expected_completion': expected, 'completed': summary['completed'],
                                     'terminal_error': summary.get('terminal_error')})
    result['completed'] = True
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
