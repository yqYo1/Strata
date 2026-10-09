"""Root-owned bounded arithmetic-only CPU qualification. No model/GPU work."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-static-quota-oracle-v0141-20261010')
PINS = {
    W/'sycl/tools/static_cache_quota.py': '6025cbef7718d3f26f086f7de03cd65370b6edf4dce511c434357bc6c79f0220',
    W/'sycl/tools/test_static_cache_quota.py': '0554c59c6f69940db5316103786213c3e9ce635105108b92b98a3c3933bcd8cb',
    B/'research-20261009/implementation-static-cache-quota-oracle-v1.txt': '06cfa60bcae2851e6bc4f021f5aece63960095018a87bc87ab944f6d89a87326',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert __debug__, 'assertions must remain enabled'
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, sha in PINS.items():
            assert digest(path) == sha, str(path)
        out = B/'static-cache-quota-cpu-validation-v1'
        out.mkdir(mode=0o700)
        record = dict(active=True, complete=False, passed=False,
                      scope='Pure exact prefix-count DP, independent exhaustive tiny oracle and production geometry; no actual trace scoring or engine change',
                      gpu_work_submitted=False, model_opened=False, adopted=False,
                      performance_eligible=False, full_lifecycle_passed=False,
                      pins={str(p): h for p,h in PINS.items()},
                      wrapper_sha256=digest(Path(__file__)),
                      deadline_seconds=30, text_budget_bytes=2*1024**2,
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      argv=['/usr/bin/python3', str(W/'sycl/tools/test_static_cache_quota.py')],
                      cwd=str(W), cleanup=[], survivors=[])
        rp = out/'record.json'
        def save():
            rp.write_text(json.dumps(record, indent=2)+'\n')
        save()
        start = time.monotonic()
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        record['environment'] = env
        proc = None
        try:
            with (out/'stdout').open('wb') as stdout, (out/'stderr').open('wb') as stderr:
                proc = subprocess.Popen(record['argv'], cwd=W, env=env, stdout=stdout, stderr=stderr)
                status = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
                record['owner'] = dict(pid=proc.pid, start_ticks=int(status[19]))
                save()
                while proc.poll() is None:
                    assert time.monotonic()-start < 30, 'CPU qualification deadline'
                    assert (out/'stdout').stat().st_size+(out/'stderr').stat().st_size <= 2*1024**2, 'CPU qualification text budget'
                    time.sleep(.05)
                record['exit_code'] = proc.returncode
            for path, sha in PINS.items():
                assert digest(path) == sha, 'source changed during qualification'
            for name in ('stdout','stderr'):
                record[name] = dict(bytes=(out/name).stat().st_size, sha256=digest(out/name))
            assert record['exit_code'] == 0, 'fixture exit'
            assert 'Ran 8 tests' in (out/'stderr').read_text() and (out/'stderr').read_text().rstrip().endswith('OK'), 'fixture count/result'
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = type(error).__name__+': '+str(error)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                record['cleanup'].append('TERM owned fixture')
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    record['cleanup'].append('KILL owned fixture')
                    proc.wait(timeout=5)
            if proc is not None:
                record['exit_code'] = proc.poll()
                if proc.poll() is None:
                    record['survivors'].append(record['owner'])
            record.update(active=False, elapsed_seconds=time.monotonic()-start,
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save()
        print(json.dumps(dict(record=str(rp), sha256=digest(rp), passed=record['passed'], elapsed_seconds=record['elapsed_seconds'])))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
