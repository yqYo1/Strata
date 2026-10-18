#!/usr/bin/env python3
"""Main-owned production-shaped resident GEMM reference; not model throughput."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
ROOT = B / 'resident-gemm-v2'
ENGINE = B / 'main-native-gdn-engine'
RESULT = A / 'resident-gemm-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
TARGET = 'sycl_native16_paired_gemm_bench'
HIST_COMMIT = '66d5c545939ade5c6f1732d8a7b4d3a7fe5d1676'
HIST_SHA = '4aa422bcaddb00220f44ca5ad931328df36425f226cc5d49f21aca1e0f39d0b3'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('smoke', 'diagnostic', 'perf'),
                   'use measure_gemm.py smoke|diagnostic|perf')
    mode = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        build = json.loads((ENGINE / 'record.json').read_text())
        guards.require(build['complete'] and build['passed'] and build['ownership_closed'] and
                       not build['active'], 'closed build required')
        for source in [build['project'], *build['dependencies']]:
            for name, expected in source['source_sha256'].items():
                guards.require(digest(Path(source['root']) / name) == expected, 'source changed: ' + name)
        fixture = build['fixtures'][TARGET]
        guards.require(digest(fixture['path']) == fixture['sha256'], 'fixture changed')
        if mode == 'smoke':
            guards.require(not ROOT.exists(), 'inspect/close/retire the previous job before starting')
            ROOT.mkdir(mode=0o700)
            historical = json.loads(subprocess.check_output(['git', 'show', HIST_COMMIT + ':' +
                'bench/results/2026-10-15-pr6-recovery-audit/moe-window-census-current.json'], cwd=P))
            hist = historical['window_census']['aggregate_job_rows_histogram']
            guards.require(hashlib.sha256(json.dumps(hist, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                           == HIST_SHA, 'historical row histogram changed')
            record = dict(source_commit=build['source_commit'], fixture=fixture, checks=[],
                          complete=False, passed=False, active=False, model_performance_measured=False,
                          model_quality_qualified=False, physical262144_RESTORE_qualified=False,
                          histogram=dict(commit=HIST_COMMIT, sha256=HIST_SHA, bins=len(hist),
                              total_rows=sum(m*n for m,n in hist),
                              rows_at8192=sum(m*n for m,n in hist if m == 8192)),
                          consumer='Current MoE resource reference; retain compact samples, retire raw logs/JIT after interpretation.',
                          limits=['Resident reused FP16 operands/weights, BF16 output, FP32 accumulation; no IQ3 dequant, SwiGLU or routing/scatter.',
                                  'Service reference for this implementation and shapes, not a hardware peak or a prediction of whole-model speed.',
                                  'Initial uploads use pageable fixture memory, not the production pinned-weight pipeline or a PCIe ceiling.',
                                  'Event timestamps share a device timebase; never subtract them from host clocks. Profiling does not guarantee exact timing accuracy.'])
        else:
            record = json.loads(RESULT.read_text())
            guards.require(record['mode'] in ('smoke', 'diagnostic') and record['passed'] and record['ownership_closed'] and
                record['source_commit'] == build['source_commit'] and record['fixture'] == fixture,
                'matching successful smoke required')
            guards.require(subprocess.check_output(['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P)
                           == RESULT.read_bytes(), 'commit the preceding smoke first')
        guards.require(not (ROOT / ('owned-' + mode)).exists(), 'this phase already started; inspect its owner')
        settings = json.loads(SETTINGS.read_text())
        for path, item in settings['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime changed: ' + path)
        env = dict(settings['environment'])
        if mode == 'perf':
            for key in list(env):
                if key.startswith(('UR_LOG_', 'IGC_')) or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE'):
                    del env[key]
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir(exist_ok=True)
        env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        record.update(mode=mode, passed=False, complete=False, environment=env)
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(dict(record,
                owned_commands=owner.commands if owner else []), indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + mode), persist)
        jobs = ([('--smoke', None, 0), ('--resident', 4, 0), ('--resident', 5, 0)] if mode == 'smoke' else
                [('--resident', 8, process) for process in range(3)] if mode == 'diagnostic' else
                [('--resident', shape, process) for shape in (4,5,0,1,6,7,8,9,2,3) for process in range(3)])
        try:
            for flag, shape, process in jobs:
                command = [fixture['path'], flag] + ([] if shape is None else [str(shape)]) + ['fp16', '--process', str(process)]
                label = f'{flag[2:]}-{shape}-{process}'
                entry, out, err = owner.run(label, command, env, P, wall=240, cpu=480,
                    rss_cap=4 << 30, text_cap=64 << 20, file_cap=64 << 20, total_cap=64 << 20)
                stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
                events = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
                result = [e for e in events if e.get('kind') == 'result']
                plans = [e for e in events if e.get('kind') == 'plan']
                samples = [e for e in events if e.get('kind') == 'sample']
                validations = [e for e in events if e.get('kind') == 'validation']
                checked = dict(phase=mode, command=command, plan=plans[0] if plans else None, process=process,
                    exit_code=entry.get('exit_code'), ownership_closed=supervision.closed(entry), passed=False,
                    validation_count=len(validations), native_result=result[-1] if result else None,
                    samples=[s for s in samples if not s['warmup']])
                record['checks'].append(checked)
                persist()
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0 and len(result) == len(plans) == 1
                    and result[0]['status'] == 'PASS' and not re.search(
                        r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|CHECKED_DRAIN_FAILED|cleanup failed', stderr),
                    label + ': ' + json.dumps(result[-1] if result else {'output_tail': stdout[-4096:]}) + stderr[-1024:])
                guards.require(len(samples) == len(validations) == result[0]['validated_invocations'], 'validation count')
                for sample, validation in zip(samples, validations):
                    guards.require(sample['ordinal'] == validation['ordinal'] and sample['route'] == validation['route']
                        and validation['status'] == 'PASS' and validation['all_finite'] and validation['closed_form_oracle']
                        and validation['guards'] and validation['inputs_immutable'], 'unvalidated sample')
                    if plans[0]['resident_only']:
                        guards.require(sample['submits'] == 8 and len(sample['device_timestamps_ns']) == 8 and
                            sample['kernel_total_seconds'] > 0, 'missing burst events')
                        sample['logical_kernel_TFLOPs'] = sample['logical_flop_per_submit'] * 8 / sample['kernel_total_seconds'] / 1e12
                        sample['logical_wall_TFLOPs'] = sample['logical_flop_per_submit'] * 8 / sample['submit_to_completion_wall_seconds'] / 1e12
                    sample['status'] = 'PASS'
                checked['passed'] = True
                persist()
            if mode == 'perf':
                record['summary'] = []
                for shape in range(10):
                    checks = [c for c in record['checks'] if c['phase'] == 'perf' and c['plan']['shape'] == shape]
                    guards.require(len(checks) == 3, 'three independent processes per shape required')
                    plan = checks[0]['plan']
                    for route in ('vllm_compact16x64_bf16_output', 'vllm_wide128x128_bf16_output'):
                        medians = [statistics.median(s['logical_kernel_TFLOPs'] for s in c['samples'] if s['route'] == route)
                                   for c in checks]
                        record['summary'].append(dict(shape=shape, M=plan['T'], N=plan['N'], K=plan['K'], route=route,
                            production_selected=(('compact' in route) == (plan['T'] < 128)),
                            per_process_median_kernel_TFLOPs=medians, median_kernel_TFLOPs=statistics.median(medians),
                            sample_stdev_process_medians=statistics.stdev(medians)))
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
            record['passed'] = False
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [x for x in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try:
                record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record['driver_guard_error'] = str(error)
                record['passed'] = False
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'complete', 'passed', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
