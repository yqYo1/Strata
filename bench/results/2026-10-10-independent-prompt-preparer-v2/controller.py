"""Root-owned bounded, serialized fake fixtures for prompt preparer v2."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-static-quota-oracle-v0141-20261010')
T = W / 'sycl/tools'
PINS = {
    T / 'prepare_independent_prompt_tokens_v2.py': '829f6335ff7b47d587832c40b9f30b9aafddce112e7b980095aa17225b0940cd',
    T / 'test_prepare_independent_prompt_tokens_v2.py': 'a6e36b7da9bc8d7928bc7417a6eb4d4277f87f72ae435dda95afe3191c68a729',
    B / 'research-20261009/implementation-independent-prompt-preparer-v2.txt': '5fd23cbb593fde95d25a6e9141575b45432ee4317efaed67295d5f8a12f8c8ef',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert __debug__
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, expected in PINS.items():
            assert digest(path) == expected, str(path)
        out = B / 'independent-prompt-fixtures-cpu-validation-v2'
        out.mkdir(mode=0o700)
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        argv = ['/usr/bin/python3', str(T / 'test_prepare_independent_prompt_tokens_v2.py')]
        record = dict(active=True, complete=False, passed=False,
                      scope='Seventeen independent fake tokenizer/renderer host fixtures; no production tokenizer imported or pack opened',
                      controller_sha256=digest(Path(__file__)), pins={str(p): h for p, h in PINS.items()},
                      argv=argv, cwd=str(T), environment=env, deadline_seconds=60, text_budget_bytes=1048576,
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      gpu_work_submitted=False, model_opened=False, actual_pack_tokenized=False,
                      inference_run=False, adopted=False, performance_eligible=False, full_lifecycle_passed=False,
                      cleanup=[], survivors=[], resource_limits=dict(AS_bytes=1073741824,RSS_poll_bytes=805306368,CPU_soft_seconds=30,CPU_hard_seconds=31,FSIZE_bytes=33554432,NOFILE=64,CORE_bytes=0),peak_rss_bytes=0)
        rp = out / 'record.json'
        def save():
            rp.write_text(json.dumps(record, indent=2) + '\n')
        save()
        begin = time.monotonic()
        proc = None
        stdout, stderr = out / 'fixtures.stdout', out / 'fixtures.stderr'
        try:
            with stdout.open('wb') as so, stderr.open('wb') as se:
                def child_limits():
                    for kind,limits in ((resource.RLIMIT_AS,(1073741824,1073741824)),(resource.RLIMIT_CPU,(30,31)),(resource.RLIMIT_FSIZE,(33554432,33554432)),(resource.RLIMIT_NOFILE,(64,64)),(resource.RLIMIT_CORE,(0,0))):
                        resource.setrlimit(kind,limits)
                proc = subprocess.Popen(argv, cwd=T, env=env, stdout=so, stderr=se,preexec_fn=child_limits,start_new_session=True)
                ticks = int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                record['owner'] = dict(pid=proc.pid, start_ticks=ticks)
                save()
                while proc.poll() is None:
                    assert time.monotonic() - begin < 60, 'CPU deadline'
                    assert stdout.stat().st_size + stderr.stat().st_size <= 1048576, 'CPU log budget'
                    try:
                        status=Path(f'/proc/{proc.pid}/status').read_text()
                    except FileNotFoundError:
                        status=''
                    for line in status.splitlines():
                        if line.startswith('VmRSS:'):
                            rss=int(line.split()[1])*1024
                            record['peak_rss_bytes']=max(record['peak_rss_bytes'],rss)
                            assert rss<=805306368,'CPU RSS budget'
                    time.sleep(.05)
            record['exit_code'] = proc.returncode
            assert proc.returncode == 0, 'fixture child failed'
            log = stderr.read_text()
            assert 'Ran 17 tests' in log and log.rstrip().endswith('OK'), 'fixture count/status'
            for path, expected in PINS.items():
                assert digest(path) == expected, 'input changed'
            record.update(complete=True, passed=True, tests=17)
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
