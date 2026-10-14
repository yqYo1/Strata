#!/usr/bin/env python3
"""Single-owner model session with a guarded, early STOP for the long diagnostic."""
import datetime
import hashlib
import json
import math
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
U = Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0143-pristine-20261012')
PY = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'


def identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]), sid=int(fields[3]), start_ticks=int(fields[19]))


def stop_single_long(root, action):
    """Only this runner's already-parsed, sole active long request may be stopped."""
    entry = json.loads((root / 'engine-start.json').read_text())
    native, wrapper = entry['native'], entry['wrapper']
    action['engine_identity'] = entry
    for expected in [native, wrapper]:
        current = identity(expected['pid'])
        assert all(current[k] == v for k, v in expected.items()) and current['state'] != 'Z'
        assert current['sid'] == os.getsid(0), 'engine outside this owned session'
    assert native['ppid'] == wrapper['pid']
    assert Path('/proc', str(native['pid']), 'exe').resolve() == Path(entry['engine']).resolve()
    pidfd = os.pidfd_open(native['pid'])
    try:
        watcher = select.poll()
        watcher.register(pidfd, select.POLLIN)
        assert not watcher.poll(0), 'native has exited'
        log = (root / 'owned/engine.stderr').read_text()
        expected_prompt = int(json.loads((root / 'plan.json').read_text())['expected_long_prompt_tokens'])
        assert f'strata trace: prompt start {expected_prompt - 1} -1' in log, 'long payload not parsed'
        assert f'of {expected_prompt - 1}' in log, 'no matching long chunk observation'
        fd = os.open('/proc/' + str(native['pid']) + '/fd/0', os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            pipe = os.fstat(fd)
            assert stat.S_ISFIFO(pipe.st_mode)
            assert (pipe.st_dev, pipe.st_ino) == (entry['stdin']['device'], entry['stdin']['inode'])
            current = identity(native['pid'])
            assert current['start_ticks'] == native['start_ticks'] and not watcher.poll(0)
            assert os.write(fd, b'STOP\n') == 5
            action['sent'] = True
        finally:
            os.close(fd)
    finally:
        os.close(pidfd)


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
        if mode in ['diagnostic', 'short-diagnostic']:
            sys.path.insert(0, str(U / 'tools'))
            import strata_bench as client
            requests = [
                ('initial-short-correctness-only', [{'role': 'user', 'content':
                 'Explain in one sentence why reusing expert weights helps processing a long prompt.'}], 8),
                ('long64K-diagnostic-not-standard-speed-suite',
                 [{'role': 'user', 'content': '[primary diagnostic 64K]\n' + client.long_prompt(58000)}], 1),
            ]
            if mode == 'short-diagnostic':
                requests = requests[:1]
            for label, messages, limit in requests:
                print('primary model: begin ' + label, flush=True)
                cancel = threading.Event()
                action = dict(sent=False, budget_stop=False, label=label)
                guard = None
                if label.startswith('long'):
                    def watchdog():
                        if cancel.wait(plan['long_soft_deadline_seconds']):
                            return
                        action.update(budget_stop=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                        try:
                            stop_single_long(root, action)
                        except BaseException as error:
                            action['error'] = repr(error)
                        (root / 'control-action.json').write_text(json.dumps(action, indent=2) + '\n')
                        print('primary model: long diagnostic soft deadline: ' + json.dumps(action), flush=True)
                    guard = threading.Thread(target=watchdog, daemon=True)
                    guard.start()
                try:
                    raw = client.chat(base, '', messages, limit, timeout=480)
                finally:
                    cancel.set()
                    if guard:
                        guard.join(timeout=10)
                        assert not guard.is_alive(), 'STOP controller did not close'
                sample = client.digest(raw)
                result['requests'].append(dict(label=label, sample=sample, text=raw['text'],
                    messages_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                    budget_stop=action['budget_stop']))
                persist()
                assert not action['budget_stop'], 'long diagnostic stopped for budget; incomplete'
                assert not raw['err'], raw['err']
                assert sample['generated_tokens'] and sample['generated_tokens'] <= limit, sample
                assert raw['text'], 'empty diagnostic generation'
                for key in ['prompt_tps', 'decode_tps', 'total_s']:
                    assert sample.get(key) is None or math.isfinite(float(sample[key])), (key, sample)
                if label.startswith('long'):
                    assert sample['prompt_tokens'] == plan['expected_long_prompt_tokens'] and (sample['reused_tokens'] or 0) == 0, sample
                print(json.dumps(dict(completed=label, sample=sample)), flush=True)
        elif mode == 'stock-bench':
            argv = [str(U / 'bench.sh'), '--url', base, '--config', str(root / 'config.json'), '--long', '--runs', '3',
                    '--out', str(root / 'stock-results'), '--ready-timeout', '480']
            result['commands'].append(argv)
            print('primary model: unchanged Niko bench.sh --long --runs 3', flush=True)
            with (root / 'owned/bench.stdout').open('wb') as out, (root / 'owned/bench.stderr').open('wb') as err:
                completed = subprocess.run(argv, cwd=U, stdout=out, stderr=err, timeout=plan['bench_timeout_seconds'])
            result['bench_exit_code'] = completed.returncode
            assert completed.returncode == 0, completed.returncode
        else:
            raise ValueError(mode)
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        try:
            if server is not None:
                print('primary model: graceful server shutdown', flush=True)
                if server.poll() is None:
                    server.send_signal(signal.SIGINT)
                result['server_exit_code'] = server.wait(timeout=120)
        except BaseException as error:
            result.update(passed=False, shutdown_error=repr(error))
            raise
        finally:
            server_log.close()
            server_err.close()
            if (root / 'engine-exit.json').exists():
                result['real_engine_exit'] = json.loads((root / 'engine-exit.json').read_text())
            normal_native = result.get('real_engine_exit', {}).get('exit_code') == 0
            normal_server = result.get('server_exit_code') == 0
            result['passed'] = result['passed'] and normal_native and normal_server
            if not (normal_native and normal_server):
                result['shutdown_status'] = 'normal native/server exit not proved'
            persist()
    assert result['server_exit_code'] == 0 and result['real_engine_exit']['exit_code'] == 0, result
    print('primary model: server and real engine exited normally', flush=True)


if __name__ == '__main__':
    main()
