#!/usr/bin/env python3
"""Main-owned GU/SwiGLU component qualification; ownership mode is CPU-only."""
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
ROOT = B / 'moe-gu-swiglu-component-v2'
ENGINE = B / 'main-native-gdn-engine'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/moe-gu-swiglu-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
TARGET = 'sycl_moe_gu_swiglu_bench'
ROWS = (1, 3, 17, 33, 129, 2334, 5963, 8191, 8192)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('ownership', 'smoke', 'validate', 'perf'),
                   'use measure_gu_swiglu.py ownership|smoke|validate|perf')
    mode = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        build = json.loads((ENGINE / 'record.json').read_text())
        guards.require(build['complete'] and build['passed'] and build['ownership_closed'] and
                       not build['active'], 'closed build required')
        for group in [build['project'], *build['dependencies']]:
            for name, expected in group['source_sha256'].items():
                guards.require(digest(Path(group['root']) / name) == expected, 'source changed: ' + name)
        fixture = build['fixtures'][TARGET]
        guards.require(digest(fixture['path']) == fixture['sha256'], 'fixture changed')
        if mode == 'ownership':
            guards.require(not ROOT.exists(), 'inspect/retire the previous ownership job before reuse')
        else:
            guards.require(RESULT.read_bytes() == subprocess.check_output(
                ['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P), 'commit the preceding phase first')
            prior = json.loads(RESULT.read_text())
            guards.require(prior['passed'] and prior['ownership_closed'] and
                prior['source_commit'] == build['source_commit'] and prior['fixture'] == fixture and
                prior['mode'] == {'smoke':'ownership', 'validate':'smoke', 'perf':'validate'}[mode],
                'matching passed preceding phase required')
            guards.require(all(dict(tile=t, layout='matrix') in prior['packed_lane_xor1_eligible']
                               for t in ('compact', 'wide')), 'both matrix packed ownership proofs required')
        settings = json.loads(SETTINGS.read_text())
        for path, item in settings['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime changed: ' + path)
        ROOT.mkdir(mode=0o700, exist_ok=mode != 'ownership')
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir(exist_ok=mode != 'ownership')
        env = dict(settings['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        if mode == 'perf':
            for key in list(env):
                if key.startswith(('UR_LOG_', 'IGC_')) or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE'):
                    del env[key]
        record = dict(mode=mode, source_commit=build['source_commit'], fixture=fixture,
            complete=False, passed=False, active=False, GPU_executed=False,
            model_performance_measured=False, model_quality_qualified=False,
            physical262144_RESTORE_qualified=False, cases=[],
            consumer='Actual GU fragment ownership before selecting a fused epilogue; retire raw output after interpretation.',
            checks=[],
            prospective_gate='All36 ownership cases complete. Packed matrix ownership and actual SG16/local-index topology required. '
                'Exact BF16 capture, independent small finite integer dot oracle, finite saturation/NaN classification, '
                'maximum1 FP16 ULP, guards and immutable inputs for capture/no-capture kernels;13 invalid admissions. '
                'Only after traced smoke and full validation:3 independent rotated clean timing processes. '
                'Consider model integration only if wall-time ratio geomean<0.98 and each large shape/process ratio<=1.02.',
            limits='Component only; queue intervals include bubbles, not hardware busy time. '
                'Resident reused operands exclude dequant/routing/transfers. No whole-model or physical-context qualification.',
            removed_logical_GU_write_read_bytes_per_row=5120)
        if mode != 'ownership':
            record = prior
        record.update(mode=mode, complete=False, passed=False, GPU_executed=mode != 'ownership', environment=env)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + mode), persist)
        try:
            for process in range(3 if mode == 'perf' else 1):
                command = [fixture['path'], '--bench' if mode == 'perf' else '--' + mode]
                if mode == 'perf': command.append(str(process))
                entry, out, err = owner.run(mode + str(process), command, env, P, wall=240 if mode != 'ownership' else 90,
                    cpu=480 if mode != 'ownership' else 120, rss_cap=4 << 30,
                    file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
                stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
                events = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
                checked = dict(mode=mode, process=process, command=command, exit_code=entry.get('exit_code'),
                    ownership_closed=supervision.closed(entry), stdout_sha256=digest(out), stderr_sha256=digest(err),
                    events=[e for e in events if e.get('round', 0) >= 0 or entry.get('exit_code') != 0])
                record['checks'].append(checked)
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0 and
                    'fixture_status=PASS mode=' + ('bench' if mode == 'perf' else mode) in stdout and
                    not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr),
                    'GU fixture failed: ' + stdout[-2048:] + stderr[-2048:])
                if mode == 'ownership':
                    for line in stdout.splitlines():
                        if line.startswith('case gu_ownership '):
                            record['cases'].append({k: int(v) if v.isdigit() else v
                                for k, v in (word.split('=', 1) for word in line.split()[2:])})
                    expected = {(r, t, m) for r in ROWS for t in ('compact', 'wide') for m in ('matrix', 'output_copy')}
                    actual = {(c['rows'], c['tile'], c['layout']) for c in record['cases']}
                    guards.require(len(record['cases']) == 36 and actual == expected and
                        all(c['status'] == 'PASS' for c in record['cases']), 'ownership cases incomplete')
                    record['lane_xor1_eligible'] = [dict(tile=t, layout=m) for t in ('compact', 'wide')
                        for m in ('matrix', 'output_copy') if all(c['partner_mismatch'] == 0
                            for c in record['cases'] if c['tile'] == t and c['layout'] == m)]
                    record['packed_lane_xor1_eligible'] = [dict(tile=t, layout=m) for t in ('compact', 'wide')
                        for m in ('matrix', 'output_copy') if all(not any(c[k] for k in
                            ('partner_mismatch','packed_missing','packed_multiple','packed_orientation_mismatch'))
                            for c in record['cases'] if c['tile'] == t and c['layout'] == m)]
                else:
                    topology = [e for e in events if e['kind'] == 'topology']
                    correctness = [e for e in events if e['kind'] == 'correctness']
                    guards.require(len(topology) == 1 and topology[0]['status'] == 'PASS', 'topology missing')
                    guards.require(len(correctness) == {'smoke':4,'validate':18,'perf':5}[mode] and all(
                        e['status'] == 'PASS' and e['max_half_ulp'] <= 1 and e['BF16_capture_exact'] and
                        e['guards'] and e['inputs_immutable'] for e in correctness), 'correctness coverage missing')
                    if mode == 'smoke':
                        guards.require([e['rejected'] for e in events if e['kind'] == 'admissions'] == [13], 'admission coverage')
                    if mode == 'perf':
                        samples = [e for e in events if e['kind'] == 'sample']
                        expected = {(r,n,a) for r in (64,160,2334,5963,8192) for n in range(-1,5) for a in (0,1)}
                        guards.require(len(samples) == 60 and {(e['rows'],e['round'],e['arm']) for e in samples} == expected
                            and all(e['process'] == process and e['status'] == 'PASS' and e['max_half_ulp'] <= 1
                                and all(math.isfinite(e[k]) and e[k] > 0 for k in ('wall_ms','queue_interval_ms'))
                                for e in samples), 'timing sample coverage')
                persist()
            if mode == 'perf':
                record['summary'] = []
                for rows in (64,160,2334,5963,8192):
                    ratios=[]
                    for c in record['checks']:
                        if c['mode'] != 'perf': continue
                        arms={a:[e for e in c['events'] if e['kind']=='sample' and e['rows']==rows and e['arm']==a]
                              for a in (0,1)}
                        ratios.append({k:statistics.median(e[k] for e in arms[1]) /
                            statistics.median(e[k] for e in arms[0]) for k in ('wall_ms','queue_interval_ms')})
                    record['summary'].append(dict(rows=rows, per_process_ratios=ratios))
                ratios=[r['wall_ms'] for s in record['summary'] for r in s['per_process_ratios']]
                record['wall_ratio_geomean']=math.exp(statistics.mean(math.log(r) for r in ratios))
                record['component_performance_gate_passed']=record['wall_ratio_geomean'] < .98 and all(
                    r['wall_ms']<=1.02 for s in record['summary'] if s['rows']>=2334 for r in s['per_process_ratios'])
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['passed'] = bool(record['passed'] and record['ownership_closed'])
            after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
            record['fresh_faults']=[x for x in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if mode != 'ownership':
                try: record['driver_after']=guards.driver_guard()
                except Exception as error:
                    record['driver_guard_error']=str(error); record['passed']=False
            record['passed']=bool(record['passed'] and record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('complete', 'passed', 'ownership_closed', 'lane_xor1_eligible', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
