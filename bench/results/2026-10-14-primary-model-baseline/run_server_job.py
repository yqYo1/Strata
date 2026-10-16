#!/usr/bin/env python3
"""One owned server/client session; graceful shutdown and real engine status."""
import importlib.util
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
U = Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0143-pristine-20261012')
PY = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'


def main():
    root = Path(sys.argv[1])
    mode = sys.argv[2]
    plan = json.loads((root / 'plan.json').read_text())
    base = plan['url']
    with socket.socket() as check:
        check.bind(('127.0.0.1', plan['port']))
    server_log = (root / 'owned/server.stdout').open('wb')
    server_err = (root / 'owned/server.stderr').open('wb')
    result = dict(passed=False, mode=mode, requests=[], commands=[])
    server = None
    def persist():
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    try:
        argv = [PY, str(P / 'serve/server.py'), '--engine', 'strata', '--config',
                str(root / 'config.json'), '--host', '127.0.0.1', '--port', str(plan['port'])]
        result['commands'].append(argv)
        print('primary model: starting owned HTTP server', flush=True)
        server = subprocess.Popen(argv, cwd=P, stdout=server_log, stderr=server_err)
        result['server_pid'] = server.pid
        persist()
        deadline = time.monotonic() + 480
        while True:
            if server.poll() is not None:
                raise RuntimeError('server exited before readiness: ' + str(server.returncode))
            try:
                with urllib.request.urlopen(base + '/v1/models', timeout=1) as response:
                    result['models'] = json.load(response)
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('primary server readiness')
                time.sleep(.5)
        result['ready'] = True
        print('primary model: HTTP ready', flush=True)
        if mode == 'diagnostic':
            sys.path.insert(0, str(U / 'tools'))
            import strata_bench as client
            for label, messages, limit in [
                ('initial-short-correctness-only', [{'role': 'user', 'content':
                 'Explain in one sentence why reusing expert weights helps processing a long prompt.'}], 8),
                ('long64K-diagnostic-not-standard-speed-suite',
                 [{'role': 'user', 'content': '[primary diagnostic 64K]\n' + client.long_prompt(58000)}], 1),
            ]:
                print('primary model: begin ' + label, flush=True)
                raw = client.chat(base, '', messages, limit, timeout=480)
                sample = client.digest(raw)
                result['requests'].append(dict(label=label, sample=sample, text=raw['text']))
                persist()
                assert not raw['err'], raw['err']
                assert sample['generated_tokens'] and sample['generated_tokens'] <= limit, sample
                assert raw['text'], 'empty diagnostic generation'
                for key in ['prompt_tps', 'decode_tps', 'total_s']:
                    assert sample.get(key) is None or math.isfinite(float(sample[key])), (key, sample)
                if label.startswith('long'):
                    assert sample['prompt_tokens'] >= 32768 and (sample['reused_tokens'] or 0) == 0, sample
                print(json.dumps(dict(completed=label, sample=sample)), flush=True)
        elif mode == 'stock-bench':
            argv = [str(U / 'bench.sh'), '--url', base, '--long', '--runs', '3',
                    '--out', str(root / 'stock-results'), '--ready-timeout', '480']
            result['commands'].append(argv)
            print('primary model: unchanged Niko bench.sh --long --runs 3', flush=True)
            with (root / 'owned/bench.stdout').open('wb') as out, (root / 'owned/bench.stderr').open('wb') as err:
                completed = subprocess.run(argv, cwd=U, stdout=out, stderr=err, timeout=2100)
            result['bench_exit_code'] = completed.returncode
            assert completed.returncode == 0, completed.returncode
        else:
            raise ValueError(mode)
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        if server is not None:
            print('primary model: graceful server shutdown', flush=True)
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
            try:
                result['server_exit_code'] = server.wait(timeout=120)
            finally:
                server_log.close()
                server_err.close()
        persist()
    engine = json.loads((root / 'engine-exit.json').read_text())
    result['real_engine_exit'] = engine
    assert result['server_exit_code'] == 0 and engine['exit_code'] == 0, result
    persist()
    print('primary model: server and real engine exited normally', flush=True)


if __name__ == '__main__':
    main()
