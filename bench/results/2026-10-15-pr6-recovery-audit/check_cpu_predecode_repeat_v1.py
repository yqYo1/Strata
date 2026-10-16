#!/usr/bin/env python3
"""Main-only independent processes for the already-correct CPU streaming fixture."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha


def main():
    baseline_path = D / 'cpu-predecode-probe-v1.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['passed'] and baseline['complete'] and not baseline['active']
    assert all(c['exit_code'] == 0 and c['session_empty'] and c['direct_child_reaped'] and not c.get('cleanup') for c in baseline['commands'])
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    exe = B / 'main-primary-cpu-predecode-root-v1/clean/fixture'
    assert sha(exe) == baseline['correctness']['clean']['binary_sha256']
    assert 0 in os.sched_getaffinity(0)
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert all(sha(Path(n)) == h for n, h in baseline['source_pins'].items())
        assert all(sha(Path(n)) == h for n, h in baseline['archive_sha256'].items())
        output = B / 'main-primary-cpu-predecode-repeat-root-v1'
        assert not output.exists(); output.mkdir(mode=0o700)
        env = dict(baseline['environment'])
        record = dict(active=True, passed=False, complete=False, commands=[], repetitions=[],
            environment=env, core=0, source_pins=baseline['source_pins'], archive_sha256=baseline['archive_sha256'],
            binary_sha256=sha(exe), baseline_sha256=sha(baseline_path), recipe_sha256=sha(Path(__file__)),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), GPU_executed=False,
            adopted=False, model_performance_qualified=False, quality_qualified=False,
            limits=dict(wall_seconds_per_process=300, RSS_bytes=2 << 30, text_bytes=64 << 20, job_bytes=256 << 20),
            scope='Three new independent processes,64 distinct synthetic experts,nt1. Individual paired CPU/wall samples retained. No model cache hit-rate, DRAM bandwidth, or full-engine speed claim.')
        def persist(): (output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(output / 'owned', persist); record['commands'] = owner.commands; persist()
        try:
            for rep in range(3):
                print('CPU predecode streaming repetition ' + str(rep), flush=True)
                argv = ['/usr/bin/taskset','-c','0',str(exe),'--timing','64','1','12','1','--correctness-approved-by-main']
                e, out, err = owner.run('stream-' + str(rep), argv, env, P, wall=300,
                    cpu=300, rss_cap=2 << 30, text_cap=64 << 20, total_cap=256 << 20, file_cap=64 << 20)
                assert module.closed(e) and e['exit_code'] == 0, (e.get('error'), err.read_text()[-4000:])
                rows = [json.loads(s) for s in out.read_text().splitlines() if s.strip()]
                samples = [s for s in rows if s['kind'] == 'timing_sample']
                assert len(samples) == 432 and rows[-1]['kind'] == 'timing_close' and rows[-1]['complete']
                assert all(not s['migration_observed'] and s['cpu_start'] == s['cpu_end'] == 0 for s in samples)
                record['repetitions'].append(dict(repetition=rep, rows=rows, stdout_sha256=sha(out), stderr=err.read_text(), stderr_sha256=sha(err)))
                persist()
            assert sha(exe) == record['binary_sha256']
            assert all(sha(Path(n)) == h for n, h in baseline['source_pins'].items())
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error); raise
        finally:
            record['active'] = False; persist()
            (D / 'cpu-predecode-repeat-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], repetitions=len(record['repetitions']), samples=1296)), flush=True)


if __name__ == '__main__':
    main()
