#!/usr/bin/env python3
"""Main-owned smoke and alternating independent-process resident timing."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import statistics
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
    parser.add_argument('mode', choices=['smoke', 'screen', 'wide'])
    args = parser.parse_args()
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build = json.loads((HERE / 'parallel-iq-timing-build-v2.json').read_text())
        require(build['passed'] and build['complete'] and not build['active'], 'timing build gate')
        exe = Path(build['binary'])
        require(sha(exe) == build['binary_sha256'] and all(sha(PROJECT / n) == h for n, h in build['source_pins'].items()), 'timing source/binary changed')
        require(all(sha(Path(p)) == h for p, h in {**build['linked_archives'], **build['linked_mkl']}.items()), 'linked library changed')
        if args.mode != 'smoke':
            previous = json.loads((HERE / ('parallel-iq-timing-' + ('screen' if args.mode == 'wide' else 'smoke') + '-v2.json')).read_text())
            require(previous['passed'] and previous['complete'] and not previous['active'] and not previous['fresh_faults'] and previous['binary_sha256'] == build['binary_sha256'], 'prior smoke/screen gate')
        require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'driver/dump gate')
        root = BASE / ('main-parallel-iq-timing-' + args.mode + '-root-v2')
        require(not root.exists(), 'fresh timing root required')
        root.mkdir(mode=0o700)
        types = [(18, 'gu'), (22, 'gu'), (21, 'gu'), (23, 'gu'), (20, 'down'), (42, 'down')]
        if args.mode == 'smoke':
            cases = [(21, 'gu', 64, 'gemm')]
        elif args.mode == 'screen':
            cases = [(t, role, 64, 'gemm') for t, role in types]
        else:
            cases = [(t, role, 4, 'dequant') for t, role in types]
            cases += [(t, role, ne, 'gemm') for ne in (4,16,128,512) for t, role in types]
        jobs = []
        for round_index in range(1 if args.mode == 'smoke' else 3):
            for case_index, case in enumerate(cases):
                arms = ('0', '1') if (round_index + case_index) % 2 == 0 else ('1', '0')
                jobs.extend((round_index, case, arm) for arm in arms)
        record = dict(schema='main-parallel-iq-timing-v2', mode=args.mode, active=True, passed=False,
                      complete=False, GPU_executed=False, commands=[], samples=[], boot_id=boot_id(),
                      binary_sha256=sha(exe), build_receipt_sha256=sha(HERE / 'parallel-iq-timing-build-v2.json'),
                      recipe_sha256=sha(Path(__file__)), repeats=20, trials=7, warmup=10,
                      expected_processes=len(jobs), product_applied=False, performance_adopted=False,
                      scope='Warm device-resident completed dequant(+oneMKL GEMM) batch wall, no exclusive kernel/model/transfer/quality claim.',
                      limits=dict(each_process_wall_seconds=180, whole_seconds=7200, text_bytes=64 << 20, job_bytes=256 << 20))

        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        host_env = dict(PATH='/usr/bin:/bin', HOME=str(root), LANG='C', LC_ALL='C')

        def utility(label, argv):
            c, out, err = owner.run(label, argv, host_env, PROJECT, wall=60, cpu=60,
                                     rss_cap=2 << 30, text_cap=4 << 20, total_cap=256 << 20, file_cap=4 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ' failed: ' + err.read_text()[-1000:])
            return out.read_text()

        old = json.loads((HERE / 'parallel-iq-component-build-v1.json').read_text())
        env = dict(old['environment'], ONEAPI_DEVICE_SELECTOR='level_zero:gpu', SYCL_CACHE_PERSISTENT='0',
                   UR_ADAPTERS_FORCE_LOAD=str(Path(old['environment']['LD_LIBRARY_PATH'].split(':')[0]) / 'libur_adapter_level_zero_v2.so.0'),
                   UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1', EnableDirectSubmission='0',
                   STRATA_SYCL_PREFILL_SAME_QUEUE='1', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0')
        if args.mode == 'smoke':
            env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                       UR_LOG_LOADER='level:warning;flush:warning;output:stderr', UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        require(not any('TOKEN' in k or 'API_KEY' in k or k == 'LD_PRELOAD' or k.startswith(('UNITRACE_', 'PTI_')) for k in env), 'unexpected environment')
        before = None
        import time
        started = time.monotonic()
        try:
            before = re.search(r'-- cursor: (\S+)', utility('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
            record['journal_before_cursor'] = before
            for index, (round_index, (type_id, role, ne, mode), arm) in enumerate(jobs):
                require(time.monotonic() - started < 7200, 'whole timing job deadline')
                label = f'case-{index:03d}'
                arm_env = dict(env, STRATA_SYCL_PARALLEL_IQ_DEQUANT=arm, XDG_CACHE_HOME=str(root / ('cache-' + label)))
                argv = [str(exe), arm, str(type_id), role, str(ne), mode, '20', '7', '10']
                if args.mode == 'smoke':
                    argv.append(str(root / ('capture-' + arm + '.bin')))
                print(f'{args.mode} {index+1}/{len(jobs)} round{round_index} type{type_id} ne{ne} {mode} arm{arm}', flush=True)
                entry, out, err = owner.run(label, argv, arm_env, PROJECT, wall=180,
                                             cpu=180 * len(os.sched_getaffinity(0)), rss_cap=8 << 30,
                                             text_cap=4 << 20, total_cap=256 << 20, file_cap=16 << 20)
                record['GPU_executed'] = True
                require(module.closed(entry) and entry['exit_code'] == 0, label + ' failed: ' + err.read_text()[-2000:])
                sample = json.loads(out.read_text())
                require(sample['schema'] == 2 and sample['complete'] and sample['queue_retired'] and sample['full_half_oracle']
                        and sample['guards'] and sample['inputs_unchanged'] and sample['full_checks'] == 2 and sample['allocation'] == 'device_usm'
                        and sample['transfers_between_trials'] == 0 and sample['arm'] == arm and len(sample['trials']) == 7
                        and sample['repeats_per_batch'] == 20 and 'B570' in sample['device'], 'incomplete/wrong component result')
                require(mode != 'gemm' or (sample['full_gemm_reference'] and sample['max_output_error'] == 0), 'GEMM reference failure')
                sample.update(process_index=index, round=round_index,
                              median_operation_ns=statistics.median(t['completed_batch_wall_ns'] / 20 for t in sample['trials']))
                if args.mode == 'smoke':
                    cap = root / ('capture-' + arm + '.bin')
                    sample['capture'] = dict(path=str(cap), bytes=cap.stat().st_size, sha256=sha(cap))
                record['samples'].append(sample)
                journal = utility('journal-' + label, ['journalctl', '-k', '-b', '--after-cursor', before, '--no-pager', '-o', 'json'])
                faults = [json.loads(s) for s in journal.splitlines() if s.strip() and FAULT.search(str(json.loads(s).get('MESSAGE', '')))]
                require(not faults and not (DEVICE / 'devcoredump').exists() and boot_id() == record['boot_id'], 'fresh GPU fault')
                persist()
            groups = {}
            for sample in record['samples']:
                key = f"{sample['type']}/{sample['role']}/{sample['ne']}/{sample['mode']}"
                groups.setdefault(key, {'0': [], '1': []})[sample['arm']].append(sample)
            summary = {}
            for key, arms in groups.items():
                require(len(arms['0']) == len(arms['1']) == (1 if args.mode == 'smoke' else 3), 'independent process count mismatch')
                identity = ('w_fnv1a64','u_fnv1a64','x_fnv1a64','dense_fnv1a64','output_fnv1a64','N','K','ne','device','driver')
                require(all(tuple(s[k] for k in identity) == tuple(arms['0'][0][k] for k in identity) for arm in arms.values() for s in arm), 'cross-arm input/output/identity mismatch')
                off = [s['median_operation_ns'] for s in arms['0']]
                on = [s['median_operation_ns'] for s in arms['1']]
                summary[key] = dict(off_process_medians_ns=off, on_process_medians_ns=on,
                                    median_reduction_fraction=1 - statistics.median(on) / statistics.median(off),
                                    ranges_separated=max(on) < min(off))
            if args.mode == 'smoke':
                require((root / 'capture-0.bin').read_bytes() == (root / 'capture-1.bin').read_bytes(), 'smoke full output bytes differ')
                record['full_external_output_equality'] = True
            record.update(passed=True, complete=True, summary=summary)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if before:
                    entries = [json.loads(s) for s in utility('journal-final', ['journalctl', '-k', '-b', '--after-cursor', before, '--no-pager', '-o', 'json']).splitlines() if s.strip()]
                    record['fresh_faults'] = [e for e in entries if FAULT.search(str(e.get('MESSAGE', '')))]
                    record['boot_unchanged'] = boot_id() == record['boot_id']
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    if record['fresh_faults'] or not record['boot_unchanged'] or record['driver_dump_exists']:
                        record['passed'] = False
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                persist()
                (HERE / ('parallel-iq-timing-' + args.mode + '-v2.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=record['passed'], samples=len(record['samples']), summary=record['summary'])), flush=True)


if __name__ == '__main__':
    main()
