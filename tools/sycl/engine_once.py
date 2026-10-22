#!/usr/bin/env python3
"""One native launch per owned benchmark root; no retries or environment mutation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]))


def native_environment(plan, parent):
    environment=dict(parent)
    probe=plan.get('native_validation_probe')
    if probe is not None:
        if environment.get('LD_PRELOAD') or environment.get('CUDA_MODULE_LOADING') != 'EAGER':
            raise RuntimeError('probe requires private native preload and immutable startup environment')
        for artifact in (probe['library'],probe['loader']):
            with Path(artifact['path']).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != artifact['sha256']:
                    raise RuntimeError('validation probe/provider changed')
        environment.update(LD_PRELOAD=probe['library']['path'],
            STRATA_UR_PROBE_EXPECTED_LOADER=str(Path(probe['loader']['path']).resolve()))
    return environment


def profile_command(profiling, command):
    mode = profiling.get('mode', 'trace')
    if mode == 'trace':
        flags = ['--start-paused', '--host-timing', '--device-timing',
                 '--chrome-kernel-logging', '--chrome-call-logging',
                 '--chrome-event-buffer-size', '128']
    elif mode == 'aggregate':
        # Record from exec: paused graph construction can lose later replays.
        flags = ['--device-timing', '--verbose']
    else:
        raise RuntimeError('unsupported profile mode: ' + str(mode))
    return [profiling['launcher']['path'], '--session', profiling['session'], *flags, *command]


def main():
    root = Path(os.environ['STRATA_MEASUREMENT_ROOT'])
    plan = json.loads((root / 'plan.json').read_text())
    binary = plan['binary']
    with Path(binary['path']).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != binary['sha256']:
            raise RuntimeError('engine binary changed')
    claim = dict(wrapper=identity(os.getpid()), binary=binary, args=sys.argv[1:],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    # A server restart must fail before another native process/GPU launch.
    with (root / 'launch.json').open('x') as stream:
        json.dump(claim, stream)
    environment=native_environment(plan,os.environ)
    if 'native_validation_probe' in plan:
        claim['native_validation_probe']=plan['native_validation_probe']
    command=[binary['path'],*sys.argv[1:]]
    profiling=plan.get('native_profile')
    if profiling is not None:
        command=profile_command(profiling,command)
        if not plan.get('native_validation_probe') or environment.get('ZES_ENABLE_SYSMAN')!='0':
            raise RuntimeError('profile requires qualified target validation and Sysman-off')
        for artifact in (profiling['launcher'],profiling['collector']):
            with Path(artifact['path']).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=artifact['sha256']:
                    raise RuntimeError('profile launcher/collector changed')
        environment['STRATA_UNITRACE_TARGET_PRELOAD']=environment.pop('LD_PRELOAD')
        allowed={'UNITRACE_SystemTime','UNITRACE_TraceOutputDir','UNITRACE_FollowChildProcess','LD_DEBUG'}
        if set(profiling['environment'])-allowed or profiling['environment'].get('UNITRACE_SystemTime')!='1':
            raise RuntimeError('unsupported profile environment override')
        environment.update(profiling['environment'])
        if 'LD_PRELOAD' in environment:
            raise RuntimeError('profile launcher must not preload target libraries')
        claim['native_profile']=profiling
    child = subprocess.Popen(command, env=environment)
    claim['native'] = identity(child.pid)
    (root / 'engine-start.json').write_text(json.dumps(claim) + '\n')
    collector = None
    if 'resource_collection' in plan:
        diagnostic = plan['resource_collection']
        with Path(diagnostic['source']['path']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != diagnostic['source']['sha256']:
                raise RuntimeError('resource collector source changed')
        collector = subprocess.Popen([sys.executable, '-B', diagnostic['source']['path'],
            str(root / 'engine-start.json'), '--out', str(root / 'client-resources.jsonl'),
            '--seconds', str(diagnostic['seconds']), '--interval', str(diagnostic['interval'])],
            env=dict(os.environ), stdout=subprocess.DEVNULL)
        claim['resource_collector'] = identity(collector.pid)
    code = child.wait()
    if collector is not None:
        # The reader sees confirmed native exit on its next interval. Owner's
        # finite process budget handles an abnormal blocked reader; no retry.
        claim['resource_collector_exit_code'] = collector.wait(timeout=12)
    (root / 'engine-exit.json').write_text(json.dumps(dict(claim, exit_code=code)) + '\n')
    return code or claim.get('resource_collector_exit_code', 0)


if __name__ == '__main__':
    sys.exit(main())
