#!/usr/bin/env python3
"""Serial immutable OFF/ON device arrays. No product apply/speed adoption."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['quick', 'full'])
    args = parser.parse_args()
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build_path = HERE / 'parallel-iq-component-build-v1.json'
        build = json.loads(build_path.read_text())
        require(build['passed'] and build['complete'] and not build['active'], 'standalone build gate')
        exe = Path(build['binary'])
        require(sha(exe) == build['binary_sha256'], 'standalone executable changed')
        require(all(sha(PROJECT / n) == h for n, h in build['source_pins'].items()), 'candidate sources changed')
        require(all(sha(Path(n)) == h for n, h in build['linked_archives'].items()), 'linked archives changed')
        if args.mode == 'full':
            quick = json.loads((HERE / 'parallel-iq-component-quick-v1.json').read_text())
            require(quick['passed'] and quick['complete'] and not quick['active'] and
                    not quick['fresh_faults'] and quick['binary_sha256'] == build['binary_sha256'], 'quick array/fault gate')
        require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'device driver/dump gate')
        root = BASE / ('main-parallel-iq-component-' + args.mode + '-root-v1')
        require(not root.exists(), 'fresh independent capture root required')
        root.mkdir(mode=0o700)
        record = dict(schema='main-parallel-iq-component-v1', mode=args.mode, active=True,
                      passed=False, complete=False, GPU_executed=False, commands=[], arms=[],
                      boot_id=boot_id(), binary_sha256=sha(exe), build_receipt_sha256=sha(build_path),
                      recipe_sha256=sha(Path(__file__)), product_applied=False, performance_qualified=False,
                      source_pins=build['source_pins'], adopted=False,
                      retention_consumer='Complete OFF/ON comparison; retain first useful divergence if any. Review raw arrays after comparator and compact hashes are verified.',
                      limits=dict(text_bytes=64 << 20, job_bytes=512 << 20, arm_capture_bytes=128 << 20))

        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        host_env = dict(PATH='/usr/bin:/bin', HOME=str(root), LANG='C', LC_ALL='C')

        def utility(label, argv):
            entry, out, err = owner.run(label, argv, host_env, PROJECT, wall=60, cpu=60,
                                        rss_cap=2 << 30, text_cap=4 << 20, total_cap=512 << 20, file_cap=4 << 20)
            require(module.closed(entry) and entry['exit_code'] == 0, label + ' failed: ' + err.read_text()[-2000:])
            return out.read_text()

        before = None
        try:
            before = re.search(r'-- cursor: (\S+)', utility('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
            record['journal_before_cursor'] = before
            previous = json.loads((HERE / 'prompt-direct-bench-full-0-v1.json').read_text())
            env = dict(previous['environment'])
            for key in ('STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
                env.pop(key, None)
            env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                       UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                       UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                       UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
            if args.mode == 'quick':
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                           UR_LOG_TRACING='level:info;flush:info;output:stderr')
            require(not any('TOKEN' in k or 'API_KEY' in k or k == 'LD_PRELOAD' or k.startswith(('UNITRACE_', 'PTI_')) for k in env), 'unexpected private/profiler environment')
            for arm in ('off', 'on'):
                cap = root / arm
                arm_env = dict(env, STRATA_SYCL_PARALLEL_IQ_DEQUANT='0' if arm == 'off' else '1',
                               XDG_CACHE_HOME=str(root / ('cache-' + arm)))
                print('IQ device ' + args.mode + ' ' + arm, flush=True)
                entry, out, err = owner.run('device-' + arm, [str(exe), arm, args.mode, str(cap)], arm_env, PROJECT,
                                             wall=180 if args.mode == 'quick' else 600,
                                             cpu=600 * len(os.sched_getaffinity(0)), rss_cap=8 << 30,
                                             text_cap=16 << 20, total_cap=512 << 20, file_cap=16 << 20)
                record['GPU_executed'] = True
                journal = utility('journal-after-' + arm, ['journalctl', '-k', '-b', '--after-cursor', before, '--no-pager', '-o', 'json'])
                entries = [json.loads(s) for s in journal.splitlines() if s.strip()]
                faults = [e for e in entries if FAULT.search(str(e.get('MESSAGE', '')))]
                item = dict(arm=arm, exit_code=entry.get('exit_code'), fresh_faults=faults,
                            driver_dump_exists=(DEVICE / 'devcoredump').exists(), boot_unchanged=boot_id() == record['boot_id'])
                record['arms'].append(item)
                persist()
                require(module.closed(entry) and entry['exit_code'] == 0, arm + ' failed: ' + err.read_text()[-2000:])
                require(not faults and not item['driver_dump_exists'] and item['boot_unchanged'], arm + ' fresh fault')
                require(sum(p.stat().st_size for p in cap.iterdir() if p.is_file()) <= 128 << 20, 'arm capture cap')
                require('B570' in (cap / 'device.txt').read_text(), 'wrong GPU')
                item['completion'] = json.loads((cap / 'complete.json').read_text())
                require(item['completion']['complete'], 'missing capture completion')
            comparison = root / 'comparison.json'
            utility('compare-full-arrays', ['/usr/bin/python3', '-B', str(HERE / 'parallel-iq-dequant-v1/compare_captures.py'),
                                            str(root / 'off'), str(root / 'on'), str(comparison)])
            result = json.loads(comparison.read_text())
            require(result['array_comparison'] == 'PASS', 'complete array mismatch')
            record['comparison_sha256'] = sha(comparison)
            record['cases'] = len(result['samples'])
            record['comparison_report'] = 'parallel-iq-component-' + args.mode + '-arrays-v1.json'
            (HERE / record['comparison_report']).write_bytes(comparison.read_bytes())
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if before:
                    entries = [json.loads(s) for s in utility('journal-final', ['journalctl', '-k', '-b', '--after-cursor', before, '--no-pager', '-o', 'json']).splitlines() if s.strip()]
                    record['fresh_faults'] = [e for e in entries if FAULT.search(str(e.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = boot_id() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record['passed'] = False
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                persist()
                (HERE / ('parallel-iq-component-' + args.mode + '-v1.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], cases=record['cases'], fresh_faults=len(record['fresh_faults']), performance_qualified=False)), flush=True)


if __name__ == '__main__':
    main()
