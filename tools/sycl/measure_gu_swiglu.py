#!/usr/bin/env python3
"""Main-owned GU/SwiGLU component qualification; ownership mode is CPU-only."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'moe-gu-swiglu-component-v1'
ENGINE = B / 'main-native-gdn-engine'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/moe-gu-swiglu-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
TARGET = 'sycl_moe_gu_swiglu_bench'
ROWS = (1, 3, 17, 33, 129, 2334, 5963, 8191, 8192)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(sys.argv[1:] == ['ownership'], 'use measure_gu_swiglu.py ownership')
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
        guards.require(not ROOT.exists(), 'inspect/retire the previous ownership job before reuse')
        settings = json.loads(SETTINGS.read_text())
        for path, item in settings['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime changed: ' + path)
        ROOT.mkdir(mode=0o700)
        for name in ('tmp', 'cache'):
            (ROOT / name).mkdir()
        env = dict(settings['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        record = dict(mode='ownership', source_commit=build['source_commit'], fixture=fixture,
            complete=False, passed=False, active=False, GPU_executed=False,
            model_performance_measured=False, model_quality_qualified=False,
            physical262144_RESTORE_qualified=False, cases=[],
            consumer='Actual GU fragment ownership before selecting a fused epilogue; retire raw output after interpretation.',
            prospective_gate='All36 tile/layout/row cases have unique complete GU and hidden ownership. '
                'Only a layout with zero adjacent-pair mismatch for every row may use same-index lane-XOR1 exchange.',
            limits='Host layout proof only; no GPU execution, numerical or performance qualification.')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / 'owned-ownership', persist)
        try:
            command = [fixture['path'], '--ownership']
            entry, out, err = owner.run('ownership', command, env, P, wall=90, cpu=120,
                rss_cap=2 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
            stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
            for line in stdout.splitlines():
                if line.startswith('case gu_ownership '):
                    record['cases'].append({k: int(v) if v.isdigit() else v
                        for k, v in (word.split('=', 1) for word in line.split()[2:])})
            expected = {(r, t, m) for r in ROWS for t in ('compact', 'wide') for m in ('matrix', 'output_copy')}
            actual = {(c['rows'], c['tile'], c['layout']) for c in record['cases']}
            record.update(command=command, exit_code=entry.get('exit_code'),
                stdout_sha256=digest(out), stderr_sha256=digest(err))
            guards.require(supervision.closed(entry) and entry.get('exit_code') == 0 and len(record['cases']) == 36
                and actual == expected and all(c['status'] == 'PASS' for c in record['cases'])
                and 'fixture_status=PASS mode=ownership GPU_submissions=0' in stdout,
                'ownership fixture failed: ' + stderr[-4096:])
            record['lane_xor1_eligible'] = [dict(tile=t, layout=m) for t in ('compact', 'wide')
                for m in ('matrix', 'output_copy') if all(c['partner_mismatch'] == 0
                    for c in record['cases'] if c['tile'] == t and c['layout'] == m)]
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['passed'] = bool(record['passed'] and record['ownership_closed'])
            persist()
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('complete', 'passed', 'ownership_closed', 'lane_xor1_eligible', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
