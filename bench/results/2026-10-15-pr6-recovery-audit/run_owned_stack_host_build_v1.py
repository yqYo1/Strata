#!/usr/bin/env python3
"""Main-reviewed CPU-only build and real owned attach/detach gate. No GPU."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PROPOSAL = HERE / 'owned-stall-stack-probe-v1'
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
LOCK = BASE / 'owned-v0141-measurement.lock'
sys.path.insert(0, str(PROPOSAL))
from owned_stack_probe import load_owner, sha, require, boot_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.output.resolve()
    require(ROOT.parent == BASE, 'CPU artifacts must stay under the named experiment base')
    owner_path = PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    owner_module = load_owner(owner_path)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not ROOT.exists(), 'fresh CPU build root required')
        ROOT.mkdir(mode=0o700)
        record = dict(schema='main-owned-stack-host-build-v1', active=True,
                      passed=False, GPU_executed=False, boot_id=boot_id(), commands=[],
                      source={str(p.relative_to(PROJECT)): sha(p)
                              for p in sorted(PROPOSAL.iterdir()) if p.is_file()},
                      controller_sha256=sha(Path(__file__)), owner_sha256=sha(owner_path),
                      lock=str(LOCK), purpose='CPU-only debugger attach/detach/resume gate')

        def persist():
            (ROOT / 'build-record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = owner_module.Owner(ROOT / 'build-owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME=str(ROOT), LANG='C', LC_ALL='C')
        compiler = '/usr/bin/cc'
        shim, target = ROOT / 'libowned_ptracer_parent.so', ROOT / 'host_target'
        commands = [
            ('compiler-version', [compiler, '--version']),
            ('build-shim', [compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fPIC', '-shared', str(PROPOSAL / 'ptracer_parent.c'), '-o', str(shim)]),
            ('build-target', [compiler, '-std=c11', '-O0', '-g', '-Wall', '-Wextra', '-Werror',
                              '-pthread', str(PROPOSAL / 'host_target.c'), '-o', str(target)]),
        ]
        try:
            for path in sorted(PROPOSAL.glob('*.py')):
                compile(path.read_text(), str(path), 'exec')
            record['python_syntax'] = 'PASS'
            for label, argv in commands:
                entry, _, _ = owner.run(label, argv, env, PROJECT, wall=60, cpu=60,
                                        text_cap=1 << 20, rss_cap=1 << 30,
                                        total_cap=32 << 20, file_cap=8 << 20)
                require(owner_module.closed(entry) and entry['exit_code'] == 0,
                        label + ' failed or ownership incomplete')
            record['artifacts'] = {str(p): dict(bytes=p.stat().st_size, sha256=sha(p))
                                   for p in (shim, target)}
            require(boot_id() == record['boot_id'], 'boot changed')
            record['passed'] = True
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = owner.active is not None
            record['finished_utc'] = owner_module.utc()
            persist()
    # Build jobs have closed and lock is released before preflight takes it.
    spec = importlib.util.spec_from_file_location('main_cpu_stack_preflight', PROPOSAL / 'preflight_controller.py')
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    sys.argv = [str(PROPOSAL / 'preflight_controller.py'), '--target', str(target),
                '--shim', str(shim), '--output', str(ROOT / 'preflight-v1'),
                '--owner', str(owner_path), '--lock', str(LOCK)]
    preflight.main()
    result = json.loads((ROOT / 'preflight-v1/record.json').read_text())
    print(json.dumps(dict(build=record['passed'], preflight=result['passed'],
                          original_exit=result['commands'][0]['exit_code'],
                          GPU_executed=False, root=str(ROOT))))


if __name__ == '__main__':
    main()
