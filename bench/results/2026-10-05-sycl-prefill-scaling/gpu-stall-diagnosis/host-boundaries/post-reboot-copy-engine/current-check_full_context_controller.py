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
import os, signal, sys, time
from pathlib import Path
capacity = int(sys.argv[sys.argv.index('--max-context') + 1])
mode = os.environ.get('STRATA_CONTROLLER_STUB_MODE', 'valid')
ple_io = sys.argv[sys.argv.index('--ple-io') + 1]
if mode == 'environment-file':
    assert os.environ['CAPACITY_ENV_SENTINEL'] == 'current'
if '--serve' not in sys.argv:
    n = len(Path(sys.argv[sys.argv.index('--tokens-file') + 1]).read_text().split())
    count = int(sys.argv[sys.argv.index('--max-new') + 1])
    if n + count > capacity:
        print('must fit the prompt and generation', flush=True)
        sys.exit(2)
    if mode == 'cli-hangs':
        time.sleep(60)
    Path(os.environ['STRATA_DUMP_FIRST_LOGITS']).write_bytes(bytes(248320 * 4))
    print(f'prefill {n - 1} tokens', flush=True)
    print('output : ' + ' '.join(str(100 + i) for i in range(count)), flush=True)
    sys.exit(0)
if ple_io == 'ram' and mode != 'missing-ram-startup':
    print('strata generate: PLE table loaded (not locked) (--ple-io ram) in 1.5 s', file=sys.stderr, flush=True)
if mode == 'no-ready':
    time.sleep(60)
if mode == 'ignores-quit':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(f'READY {capacity} stop', flush=True)
if mode == 'does-not-read':
    time.sleep(60)
for request in sys.stdin:
    if request.strip() == 'QUIT':
        if mode == 'quit-writes':
            for _ in range(8):
                print('x' * 262144, flush=True)
        if mode == 'ignores-quit':
            time.sleep(60)
        if mode == 'exit-fails':
            sys.exit(7)
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
    checks = [('valid', 64, True, 'direct'), ('valid', 262144, True, 'direct')]
    checks += [(mode, 64, False, 'direct') for mode in (
        'overrun', 'last-cell-missing', 'wrong-tail', 'early-eos',
        'wrong-count', 'wrong-finish', 'nan-logprob', 'stale-head',
        'refusal-executes')]
    checks += [('valid', 64, True, 'ram'), ('valid', 262144, True, 'ram'),
               ('missing-ram-startup', 64, False, 'ram')]
    checks += [('no-ready',64,False,'direct'),('does-not-read',262144,False,'direct'),
               ('ignores-quit',64,False,'direct'),('exit-fails',64,False,'direct'),
               ('quit-writes',64,True,'direct'),('environment-file',64,True,'direct')]
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
        for mode, capacity, expected, ple_io in checks:
            name = f'{mode}-{capacity}' + ('-ple-ram' if ple_io == 'ram' else '')
            env = dict(os.environ, CONTROLLER_STUB_MODE=mode)
            command = ['/usr/bin/python3', str(controller), '--stage', 'serve',
                       '--context', str(capacity), '--recovery', str(recovery),
                       '--executable', str(exe), '--ple-io', ple_io]
            if mode=='environment-file':
                current=recovery/'environment.json'
                current.write_text(json.dumps({'PATH':'/usr/bin:/bin','LD_LIBRARY_PATH':'',
                    'CONTROLLER_STUB_MODE':mode,'CAPACITY_ENV_SENTINEL':'current'}))
                env['CONTROLLER_STUB_MODE']='exit-fails'
                env['CAPACITY_ENV_SENTINEL']='inherited'
                command+=['--environment-file',str(current)]
            bounded = mode in ('no-ready','does-not-read','ignores-quit','exit-fails')
            if bounded:
                command += ['--protocol-timeout','0.5','--shutdown-timeout','0.1','--job-timeout','5']
            # Only a CPU stub is timed out here; this never kills a GPU process.
            run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
            variant = '-ple-ram' if ple_io == 'ram' else ''
            summary = json.loads((recovery / (f'serve-{capacity}-{exe.stem}' + variant) / 'summary.json').read_text())
            (args.out / f'{name}.json').write_text(json.dumps(summary, indent=2) + '\n')
            (args.out / f'{name}.log').write_text(run.stdout + run.stderr)
            observed = run.returncode == 0 and summary['completed']
            assert observed == expected, (name, run.returncode, run.stderr)
            if not expected:
                error = summary.get('terminal_error', '')
                assert error.startswith(('AssertionError(', 'FileNotFoundError(', 'RuntimeError(', 'TimeoutError(')), (name, error)
                if bounded:
                    proc=summary['processes'][0]
                    assert not proc['still_alive'] and proc['exit_code'] is not None
                    if mode=='ignores-quit':assert proc['cleanup_actions']==['QUIT','SIGTERM','SIGKILL'],proc
                    if mode=='exit-fails':assert proc['exit_code']==7,proc
                    if mode in ('no-ready','does-not-read'):assert summary['processing_error'].startswith('TimeoutError(')
            else:
                assert summary['ple_io'] == ple_io
                assert summary['args'][summary['args'].index('--ple-io') + 1] == ple_io
                if ple_io == 'ram':
                    assert summary['ple_table_locked'] is False
                    assert summary['ple_table_startup_seconds'] == 1.5
                rows = summary['runs']
                assert len(rows) == 5
                assert [r['input_tokens'] for r in rows] == [capacity-4, capacity-2, capacity, capacity-2, 37]
                assert rows[1]['clipped_verify_tail'] == 2
                assert rows[1]['last_executed_kv_cell'] == capacity-1
                if mode=='environment-file':assert summary['env']['LD_LIBRARY_PATH']==''
                if mode=='quit-writes':
                    raw=Path(summary['memory']['path']).with_name('protocol.stdout.raw')
                    assert raw.stat().st_size>=8*262144
            result['checks'].append({'name': name, 'exit_code': run.returncode,
                                     'ple_io': ple_io,
                                     'expected_completion': expected, 'completed': summary['completed'],
                                     'terminal_error': summary.get('terminal_error'),
                                     'processes':summary.get('processes')})
        for stage,capacity,mode,expected in [('boundary',64,'valid',True),('boundary',262144,'valid',True),
                ('cli',64,'valid',True),('cli',262144,'valid',True),('cli',64,'cli-hangs',False)]:
            name=f'{stage}-{mode}-{capacity}'
            command=['/usr/bin/python3',str(controller),'--stage',stage,'--context',str(capacity),
                     '--recovery',str(recovery),'--executable',str(exe),'--job-timeout','0.5',
                     '--shutdown-timeout','0.1']
            run=subprocess.run(command,env=dict(os.environ,CONTROLLER_STUB_MODE=mode),text=True,capture_output=True,timeout=60)
            summary=json.loads((recovery/f'{stage}-{capacity}-{exe.stem}'/'summary.json').read_text())
            (args.out/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n')
            (args.out/f'{name}.log').write_text(run.stdout+run.stderr)
            assert (run.returncode==0 and summary['completed'])==expected,(name,run.stderr)
            assert all(not p['still_alive'] for p in summary['processes'])
            if stage=='cli' and expected:assert summary['runs'][0]['logical_length']==capacity
            if stage=='boundary':assert all(r['exit_code']==2 for r in summary['runs'])
            if mode=='cli-hangs':assert summary['processing_error'].startswith('TimeoutExpired(')
            result['checks'].append(dict(name=name,exit_code=run.returncode,expected_completion=expected,
                                        completed=summary['completed'],terminal_error=summary.get('terminal_error'),
                                        processes=summary['processes']))
    result['completed'] = True
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
