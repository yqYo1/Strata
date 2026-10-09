"""Root owns arithmetic fixtures plus actual closed-trace replay after cleanup."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-static-quota-oracle-v0141-20261010')
T = W/'sycl/tools'
PINS = {
    T/'static_cache_quota.py': '6025cbef7718d3f26f086f7de03cd65370b6edf4dce511c434357bc6c79f0220',
    T/'static_cache_quota_trace.py': '1ab9bf4aea79eb121507d7a59df58546f79674ebcb2475ee4379a7413f43f981',
    T/'test_static_cache_quota_trace.py': '11e7f61a7d9f34647f994b927a1078cc3695ae664daed4b6637efba0953b9033',
    B/'research-20261009/implementation-static-cache-quota-trace-v1.txt': 'd71ac932d8a876e446a2824b7a8adb5a0123a6cbbe85d5f23b697b49fff4b090',
    B/'owned-cache-route-pairs-v0141-code32k-tasks6-diagnostic-r6/record.json': 'f606f5a1f2d00dba240213a7feeb3cc84c9da4f1487075d1e97c336a558e4c89',
    B/'cache-route-r6-artifact-retirement-v1.json': '20e9767e5e954710e464af41b563bf1fcc57b214b8f257c2bcadd3e49377ac43',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert __debug__
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, sha in PINS.items(): assert digest(path) == sha, str(path)
        retired = json.loads((B/'cache-route-r6-artifact-retirement-v1.json').read_text())
        assert retired['active'] is False and retired['complete'] is True
        assert all(x['removed'] is True and not Path(x['path']).exists() for x in retired['entries'])
        out = B/'static-cache-quota-trace-cpu-validation-v1'
        out.mkdir(mode=0o700)
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        record = dict(active=True, complete=False, passed=False,
                      scope='Seven independent trace-curve fixtures and actual frozen r6 entry-count selection/diagnostic scoring after candidate tensor retirement; no new model inference',
                      gpu_work_submitted=False, model_opened=False, adopted=False, performance_eligible=False, full_lifecycle_passed=False,
                      controller_sha256=digest(Path(__file__)), pins={str(p): h for p,h in PINS.items()},
                      environment=env, deadline_seconds=30, text_budget_bytes=2*1024**2,
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), steps=[], cleanup=[], survivors=[])
        rp=out/'record.json'
        def save(): rp.write_text(json.dumps(record,indent=2)+'\n')
        save();start=time.monotonic();proc=None
        commands=[('trace-fixture',['/usr/bin/python3',str(T/'test_static_cache_quota_trace.py')]),
                  ('actual-trace',['/usr/bin/python3',str(T/'static_cache_quota_trace.py'),
                                   '--self-sha256',PINS[T/'static_cache_quota_trace.py'],'--helper-root',str(B),
                                   '--receipt',str(B/'owned-cache-route-pairs-v0141-code32k-tasks6-diagnostic-r6/record.json'),
                                   '--receipt-sha256',PINS[B/'owned-cache-route-pairs-v0141-code32k-tasks6-diagnostic-r6/record.json'],
                                   '--controller',str(B/'run_owned_cache_route_pairs_v0141_code32k_v6.py')])]
        try:
            for label,argv in commands:
                step=dict(label=label,argv=argv,cwd=str(T));record['steps'].append(step)
                stdout=out/(label+'.stdout');stderr=out/(label+'.stderr')
                begin=time.monotonic()
                with stdout.open('wb') as so, stderr.open('wb') as se:
                    proc=subprocess.Popen(argv,cwd=T,env=env,stdout=so,stderr=se)
                    ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
                    step['owner']=dict(pid=proc.pid,start_ticks=ticks);save()
                    while proc.poll() is None:
                        assert time.monotonic()-begin<30,'CPU step deadline'
                        assert sum(p.stat().st_size for p in out.glob('*.stdout'))+sum(p.stat().st_size for p in out.glob('*.stderr')) <= 2*1024**2,'CPU log budget'
                        time.sleep(.05)
                step.update(exit_code=proc.returncode,elapsed_seconds=time.monotonic()-begin,
                            stdout_bytes=stdout.stat().st_size,stdout_sha256=digest(stdout),stderr_bytes=stderr.stat().st_size,stderr_sha256=digest(stderr));save()
                assert proc.returncode==0,'CPU step failed: '+label
            log=(out/'trace-fixture.stderr').read_text()
            assert 'Ran 7 tests' in log and log.rstrip().endswith('OK')
            result=json.loads((out/'actual-trace.stdout').read_text())
            assert result['fit_request']==dict(ordinal=1,name='A-read0')
            assert result['objective']=='routed entries N' and result['uniform_MAXBLOB_slots']==128
            assert sum(result['fit']['quotas'])==128 and result['fit']['score']>=result['fit']['baseline_score']
            assert [q['default_entries'] for q in result['requests']]==[2116,1874,2116,1874]
            assert len(result['requests'])==4
            assert all(result[k] is False for k in ('adopted','performance_eligible','full_lifecycle_passed'))
            for path,sha in PINS.items(): assert digest(path)==sha,'input changed'
            record.update(complete=True,passed=True,actual_closed_trace_scoring_executed=True,
                          original_run_receipt_unchanged=True,post_retirement_consumer_passed=True)
        except BaseException as error:
            record['error']=type(error).__name__+': '+str(error)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate();record['cleanup'].append('TERM owned CPU step')
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill();record['cleanup'].append('KILL owned CPU step');proc.wait(timeout=5)
            if proc is not None and proc.poll() is None: record['survivors'].append(record['steps'][-1]['owner'])
            record.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        print(json.dumps(dict(record=str(rp),sha256=digest(rp),passed=record['passed'],elapsed_seconds=record['elapsed_seconds'])))
        return 0 if record['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
