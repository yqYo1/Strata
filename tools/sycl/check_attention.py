#!/usr/bin/env python3
"""Main-owned QSA checks and same-binary four-value K versus K/V timing."""
import fcntl
import hashlib
import json
import math
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
ROOT = B / 'qsa-attention-current'
ENGINE = B / 'main-native-gdn-engine'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-attention-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
TARGETS = ('sycl_qsa_bf16_late_flags_test', 'sycl_qsa_bf16_prefix_optimized_test',
           'sycl_qsa_bf16_prefix_qcache_test', 'sycl_qsa_bf16_prefix_kvec4_test',
           'sycl_qsa_bf16_prefix_kvvec4_test')
BENCH = 'sycl_qsa_bf16_prefix_qcache_bench'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(len(sys.argv) in (2, 3) and sys.argv[1] in ('smoke', 'component', 'perf') and sys.argv[2:] in ([], ['ordered-exp']),
                   'use check_attention.py smoke|component|perf [ordered-exp]')
    mode = sys.argv[1]
    ordered = sys.argv[2:] == ['ordered-exp']
    root = B / 'qsa-ordered-exp-component-v1' if ordered else ROOT
    result_path = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-ordered-exp-current.json' if ordered else RESULT
    targets = ('sycl_qsa_bf16_prefix_kvvec4_test', 'sycl_qsa_bf16_prefix_ordered_exp_test') if ordered else TARGETS
    bench = 'sycl_qsa_bf16_prefix_ordered_exp_bench' if ordered else BENCH
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        build = json.loads((ENGINE / 'record.json').read_text())
        guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and
                       not build['active'], 'closed qualified build required')
        for group in [build['project'], *build['dependencies']]:
            for name, expected in group['source_sha256'].items():
                guards.require(digest(Path(group['root']) / name) == expected, 'build source changed: ' + name)
        fixtures = {name: build['fixtures'][name] for name in (*targets, bench)}
        for item in fixtures.values():
            guards.require(digest(item['path']) == item['sha256'], 'fixture binary changed')
        if mode == 'smoke':
            guards.require(not root.exists(), 'close/retire the previous check before replacing it')
            root.mkdir(mode=0o700)
            record = dict(source_commit=build['source_commit'], fixtures=fixtures, checks=[],
                          passed=False, complete=False, active=False, model_performance_measured=False,
                          model_quality_qualified=False, physical262144_model_RESTORE_qualified=False,
                          baseline='qsa_bf16_prefix_kvvec4_batch' if ordered else 'qsa_bf16_prefix_kvec4_batch',
                          candidate='qsa_bf16_prefix_ordered_exp_batch' if ordered else 'qsa_bf16_prefix_kvvec4_batch',
                          consumer='Immediate same-binary component comparison; preserve individual samples and retire raw checks after closure.',
                          prospective_gate='For ordered-exp:3 independent processes,6 rotated samples per arm/shape. Geomean candidate/base<=0.95 across3 matched shapes; no shape>1.05; each process improves for both cap2051 patterns. Component advance only, no model or decode adoption.',
                          limits='Complete wrapper wall includes admission, pack, attention and checked drain. No hardware saturation, physical traffic, occupancy or model speed inference.')
        else:
            record = json.loads(result_path.read_text())
            if ordered:
                committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(result_path.relative_to(P))], cwd=P)
                guards.require(committed == result_path.read_bytes(), 'commit preceding result first')
            guards.require(record['passed'] and record['ownership_closed'] and record['source_commit'] ==
                           build['source_commit'] and record['fixtures'] == fixtures, 'matching smoke required')
            guards.require(record['mode'] == ('component' if mode == 'perf' else 'smoke'),
                           'require smoke then component then perf on this exact build')
        guards.require(not (root / ('owned-' + mode)).exists(), 'this check already started; inspect its receipt')
        settings = json.loads(SETTINGS.read_text())
        for path, item in settings['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime asset changed: ' + path)
        env = dict(settings['environment'])
        if mode == 'component':
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
            env.pop('UR_LOG_TRACING', None)
        if mode == 'perf':
            for key in list(env):
                if key.startswith('UR_LOG_') or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE'):
                    del env[key]
        for name in ('tmp', 'cache'):
            (root / name).mkdir(exist_ok=True)
        env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        record.update(passed=False, complete=False, ownership_closed=False, mode=mode, environment=env, GPU_executed=True)
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(root / ('owned-' + mode), persist)
        try:
            jobs = [(bench, i) for i in range(3 if ordered else 1)] if mode == 'perf' else [
                (name, 0) for name in ((*targets, bench) if mode == 'smoke' else targets)]
            for name, process in jobs:
                command = [fixtures[name]['path'], '--component' if mode == 'perf' else '--' + mode]
                if ordered and mode == 'perf': command.append(str(process))
                entry, out, err = owner.run(name + ('-' + str(process) if mode == 'perf' else ''), command, env, P, wall=180, cpu=360,
                    rss_cap=4 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
                stdout = out.read_text(errors='replace')
                stderr = err.read_text(errors='replace')
                passed = (supervision.closed(entry) and entry.get('exit_code') == 0 and
                          'fixture_status=PASS' in stdout and not re.search(
                              r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr))
                record['checks'].append(dict(mode=mode, target=name, process=process, command=command,
                    binary_sha256=fixtures[name]['sha256'], passed=passed, exit_code=entry.get('exit_code'),
                    ownership_closed=supervision.closed(entry),
                    cases=[x for x in stdout.splitlines() if x.startswith(('case ', 'fixture_status='))],
                    samples=[dict(token.split('=', 1) for token in x.split()[1:])
                             for x in stdout.splitlines() if x.startswith('sample ')]))
                guards.require(passed, name + ': ' + (stderr or stdout)[-8192:])
                if ordered:
                    lines = stdout.splitlines()
                    if name == bench:
                        samples = record['checks'][-1]['samples']
                        guards.require(len(samples) == (36 if mode == 'perf' else 12) and all(
                            int(x['process']) == process for x in samples), 'incomplete benchmark output')
                    else:
                        count = 2 if mode == 'smoke' else 11
                        guards.require(sum(x.startswith('case cells=') for x in lines) == count,
                                       'incomplete numerical cases')
                        if name.endswith('ordered_exp_test'):
                            guards.require(sum(x.startswith('case exact_FP32_online_state=PASS') for x in lines) == count and
                                sum(x.startswith('case exact_prefix33_bits=PASS') for x in lines) == count and
                                'case state_admission_rejections=9 status=PASS' in lines, 'incomplete state/admission gates')
            if ordered and mode == 'perf':
                measured = [dict(sample, process=check['process']) for check in record['checks']
                            if check['mode'] == 'perf' for sample in check['samples']]
                guards.require(len(measured) == 108, 'incomplete3-process samples')
                summary = []
                for cap, pattern in ((33, 'gather'), (2051, 'gather'), (2051, 'clustered')):
                    arms = []
                    for arm in (0, 1):
                        group = [s for s in measured if int(s['cap']) == cap and s['pattern'] == pattern
                                 and int(s['variant']) == arm]
                        values = [float(s['milliseconds']) for s in group]
                        guards.require(len(values) == 18 and all(math.isfinite(v) and v > 0 for v in values), 'missing samples')
                        arms.append(dict(arm=arm, count=len(values), median_ms=statistics.median(values),
                            sample_sd_ms=statistics.stdev(values), process_medians_ms=[statistics.median(
                                float(s['milliseconds']) for s in group if s['process'] == process) for process in range(3)]))
                    summary.append(dict(cap=cap, pattern=pattern, arms=arms,
                                        ratio=arms[1]['median_ms'] / arms[0]['median_ms']))
                ratio=statistics.geometric_mean(s['ratio'] for s in summary)
                record['component_summary'] = dict(shapes=summary, geometric_mean_latency_ratio=ratio,
                    advance_to_model=ratio <= .95 and all(s['ratio'] <= 1.05 for s in summary) and all(
                        a < b for s in summary if s['cap'] == 2051 for a,b in zip(
                            s['arms'][1]['process_medians_ms'],s['arms'][0]['process_medians_ms'])))
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
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
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and
                                    record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            result_path.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'passed', 'complete', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
