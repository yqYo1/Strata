#!/usr/bin/env python3
"""Apply Main-reviewed bench change and serialize three host-only suites."""
import fcntl
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = BASE / 'main-bench-failfast-host-root-v1'
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not ROOT.exists(), 'fresh host test root required')
        ROOT.mkdir(mode=0o700)
        record = dict(schema='main-bench-failfast-host-v1', active=True, passed=False,
                      GPU_executed=False, commands=[], recipe_sha256=sha(Path(__file__)))

        def persist():
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(ROOT / 'owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                   HOME=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        proposal = HERE / 'bench-failfast-engine-error-v1'
        commands = [('apply-reviewed-proposal', ['/usr/bin/python3', '-B', str(proposal / 'apply_proposal.py'), str(PROJECT)])]
        for name in ('test_strata_bench_failfast.py', 'test_strata_bench.py', 'test_benchmark_evidence.py'):
            commands.append((name[:-3], ['/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-s', 'tools', '-p', name]))
        try:
            for label, argv in commands:
                entry, out, err = owner.run(label, argv, env, PROJECT, wall=60, cpu=60,
                                           rss_cap=2 << 30, text_cap=4 << 20,
                                           total_cap=32 << 20, file_cap=4 << 20)
                require(module.closed(entry) and entry['exit_code'] == 0,
                        label + ' failed: ' + err.read_text()[-2000:])
            record.update(passed=True, source={str(p.relative_to(PROJECT)): sha(p) for p in
                          (PROJECT / 'tools/strata_bench.py', PROJECT / 'tools/test_strata_bench_failfast.py')})
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            (HERE / 'bench-failfast-host-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False, commands=len(owner.commands))))


if __name__ == '__main__':
    main()
