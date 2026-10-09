"""Root-owned bounded, serialized fake fixtures for prompt preparer v1."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-static-quota-oracle-v0141-20261010')
T = W / 'sycl/tools'
PINS = {
    T / 'prepare_independent_prompt_tokens.py': '365d3a32b55252cafcb1947e7743dfc9d0758589d91f9242ca730fe71a9925a3',
    T / 'test_prepare_independent_prompt_tokens.py': '016bb3a42b15f2fd21211cc30f6c687c7e87c9a96de4785f56a4de41a1a4b2f7',
    B / 'research-20261009/implementation-independent-prompt-preparer-v1.txt': 'd9ba46149b38e31547094603f1a6e6194ce6a8731d8b98d28974d186191cbada',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert __debug__
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, expected in PINS.items():
            assert digest(path) == expected, str(path)
        out = B / 'independent-prompt-fixtures-cpu-validation-v1'
        out.mkdir(mode=0o700)
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        argv = ['/usr/bin/python3', str(T / 'test_prepare_independent_prompt_tokens.py')]
        record = dict(active=True, complete=False, passed=False,
                      scope='Eight independent fake tokenizer/renderer host fixtures; no production tokenizer imported or pack opened',
                      controller_sha256=digest(Path(__file__)), pins={str(p): h for p, h in PINS.items()},
                      argv=argv, cwd=str(T), environment=env, deadline_seconds=30, text_budget_bytes=1048576,
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      gpu_work_submitted=False, model_opened=False, actual_pack_tokenized=False,
                      inference_run=False, adopted=False, performance_eligible=False, full_lifecycle_passed=False,
                      cleanup=[], survivors=[])
        rp = out / 'record.json'
        def save():
            rp.write_text(json.dumps(record, indent=2) + '\n')
        save()
        begin = time.monotonic()
        proc = None
        stdout, stderr = out / 'fixtures.stdout', out / 'fixtures.stderr'
        try:
            with stdout.open('wb') as so, stderr.open('wb') as se:
                proc = subprocess.Popen(argv, cwd=T, env=env, stdout=so, stderr=se)
                ticks = int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                record['owner'] = dict(pid=proc.pid, start_ticks=ticks)
                save()
                while proc.poll() is None:
                    assert time.monotonic() - begin < 30, 'CPU deadline'
                    assert stdout.stat().st_size + stderr.stat().st_size <= 1048576, 'CPU log budget'
                    time.sleep(.05)
            record['exit_code'] = proc.returncode
            assert proc.returncode == 0, 'fixture child failed'
            log = stderr.read_text()
            assert 'Ran 8 tests' in log and log.rstrip().endswith('OK'), 'fixture count/status'
            for path, expected in PINS.items():
                assert digest(path) == expected, 'input changed'
            record.update(complete=True, passed=True, tests=8)
        except BaseException as error:
            record['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                record['cleanup'].append('TERM owned CPU fixture')
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    record['cleanup'].append('KILL owned CPU fixture')
                    proc.wait(timeout=5)
            if proc is not None and proc.poll() is None:
                record['survivors'].append(record['owner'])
            record['logs'] = {str(p): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in (stdout, stderr) if p.exists()}
            record.update(active=False, elapsed_seconds=time.monotonic() - begin,
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save()
        print(json.dumps(dict(record=str(rp), sha256=digest(rp), passed=record['passed'], elapsed_seconds=record['elapsed_seconds'])))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
