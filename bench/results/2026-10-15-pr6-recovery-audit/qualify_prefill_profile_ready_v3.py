#!/usr/bin/env python3
"""Owned CPU receiver qualification after v1's actual normal INFO mismatch."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        retired = A/'prefill-window-component-v2-retirement.json'
        d = json.loads(retired.read_text())
        require(d['passed'] and d['complete'] and not d['active'] and d['session_released'], 'v2 retirement')
        root = B/'main-prefill-window-ready-host-root-v3'
        receipt = A/'prefill-window-profile-ready-host-v3.json'
        require(not root.exists() and not receipt.exists(), 'fresh host root')
        root.mkdir(mode=0o700)
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
            recipe_sha256=sha(A/'run_prefill_window_profile_v3.py'),
            fixture_sha256=sha(A/'test_prefill_profile_ready_v3.py'),
            retirement_sha256=sha(retired), source_sha256=sha(Path(__file__)))
        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), LANG='C.UTF-8',
                   LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        try:
            c, out, err = owner.run('ready-parser-host', ['/usr/bin/python3', '-B',
                str(A/'test_prefill_profile_ready_v3.py')], env, P, wall=30, cpu=30,
                rss_cap=1 << 30, text_cap=1 << 20, total_cap=8 << 20, file_cap=1 << 20)
            require(module.closed(c) and c['exit_code'] == 0, err.read_text())
            result = json.loads(out.read_text())
            require(result['passed'] and result['cases'] == 27 and not result['GPU_executed'] and
                    not result['provider_executed'], 'host cases')
            record.update(result=result, passed=True, complete=True)
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with receipt.open('x') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], cases=record.get('result', {}).get('cases'))), flush=True)


if __name__ == '__main__':
    main()
