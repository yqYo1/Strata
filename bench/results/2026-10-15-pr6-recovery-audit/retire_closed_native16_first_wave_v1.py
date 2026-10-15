#!/usr/bin/env python3
"""Retire only this closed wave; keep v2 archives and the old two-queue control."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(json.loads((A / 'own-native16-model-smoke-v2.json').read_text())['passed'], 'fixed paired model gate')
        require(json.loads((A / 'own-native16-first-long-comparison-v1.json').read_text())['complete'], 'long pair recorded')
        target = A / 'closed-native16-first-wave-retention-v1.json'
        require(not target.exists(), 'fresh deletion manifest')
        mapping = {
            'main-own-native16-prefill-build-root-v1': 'own-native16-prefill-build-v1.json',
            'main-own-native16-ops-build-root-v1': 'own-native16-ops-build-v1.json',
            'main-own-native16-ops-correctness-root-v1': 'own-native16-ops-correctness-v1.json',
            'main-own-native16-gemm-correctness-root-v1': 'own-native16-gemm-correctness-v1.json',
            'main-own-native16-gemm-correctness-root-v2': 'own-native16-gemm-correctness-v2.json',
            'main-own-native16-regression-correctness-root-v1': 'own-native16-regression-correctness-v1.json',
            'main-own-native16-regression-correctness-root-v2': 'own-native16-regression-correctness-v2.json',
            'main-own-native16-regression-followup-correctness-root-v2': 'own-native16-regression-followup-correctness-v2.json',
            'main-own-native16-regression-followup2-correctness-root-v2': 'own-native16-regression-followup2-correctness-v2.json',
            'main-own-native16-model-smoke-root-v1': 'own-native16-model-smoke-v1.json',
            'main-own-native16-model-smoke-root-v2': 'own-native16-model-smoke-v2.json',
            'main-own-registry-teardown-repro-root-v1': 'own-registry-teardown-repro-v1.json',
            'main-own-native16-clean-fp32-r0-root-v2': 'own-native16-clean-fp32-r0-v2.json',
            'main-own-native16-clean-bf16-r0-root-v2': 'own-native16-clean-bf16-r0-v2.json',
        }
        keep = B / 'main-own-native16-prefill-build-root-v1/build/sycl_gemm_test'
        record = dict(complete=False, recipe_sha256=sha(Path(__file__)), entries=[], receipts={},
            replacements=['own-native16-model-smoke-v2.json', 'own-registry-teardown-repro-v1.json',
                'own-native16-separate-queue-timeout-v2.stack.txt', 'own-native16-first-long-comparison-v1.json'],
            retained=[dict(path=str(keep), sha256=sha(keep),
                consumer='One old full-regression executable for the unresolved separate-queue v1-pass/v2-timeout question; no automatic rerun',
                byte_budget=128 << 20, next_review='At closure or supersession of the separate-queue investigation'),
                dict(path=str(B / 'main-own-native16-prefill-build-root-v2/build'),
                consumer='Current source-matched engine and archives for R802/R803 and native16 diagnostics'),
                dict(path='Both original long FP32 NLL control roots, untouched',
                consumer='Unresolved two isolated FP32 NLL differences; this cleanup does not retire those comparator fixtures')],
            limit='All original PASS/FAIL outcomes and individual samples remain unchanged in Git. Synthetic129 tensors are superseded by the new natural32K screen; no existing comparator requires them. This is not quality acceptance or a claim of recovered physical free space.')
        roots = []
        for dirname, receipt in mapping.items():
            r = A / receipt
            blob = subprocess.check_output(['git', 'show', 'HEAD:' + str(r.relative_to(P))], cwd=P)
            require(hashlib.sha256(blob).hexdigest() == sha(r), 'committed receipt unchanged')
            v = json.loads(blob)
            require(not v['active'], 'run active')
            for c in v['commands']:
                require(c.get('direct_child_reaped') and c.get('session_empty') and not c.get('survivors'), 'owned job not retired')
                for identity in c['owners']:
                    now = module.identity(identity['pid'])
                    require(now is None or now['start_ticks'] != identity['start_ticks'], 'old owner present')
            record['receipts'][receipt] = dict(sha256=sha(r), passed=v['passed'], complete=v['complete'])
            root = B / dirname
            require(root.is_dir() and not root.is_symlink(), 'exact private root')
            roots.append(root)
            for f in sorted(root.rglob('*')):
                require(not f.is_symlink(), 'unexpected symlink')
                if not f.is_file() or f == keep:
                    continue
                st = f.stat()
                entry = dict(path=str(f), bytes=st.st_size, sha256=sha(f),
                    identity=[st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns],
                    evidence=receipt, reason='Closed, superseded build/capture or redundant successful log; source, recipe, samples, oracle/fault/retirement status retained in committed receipt.')
                if f.suffix in ('.stderr', '.stdout', '.txt') and st.st_size < 8 << 20:
                    lines = f.read_text(errors='replace').splitlines()
                    interesting = [l for l in lines if re.search(r'(?i)warning:|error:|double free|corrupt|SIGABRT|GEMM16 (device|tuple)|fixture=PASS|native16_prefill PASS', l)]
                    entry['diagnostic_lines'] = interesting[:100]
                    entry['matching_line_count'] = len(interesting)
                record['entries'].append(entry)
        require(keep.stat().st_size <= 128 << 20, 'bounded retained control')
        record['files'] = len(record['entries'])
        record['removed_logical_bytes'] = sum(e['bytes'] for e in record['entries'])
        target.write_text(json.dumps(record, indent=2) + '\n')
        for e in record['entries']:
            f = Path(e['path']); st = f.stat()
            require([st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] == e['identity'], 'file changed before unlink')
            f.unlink()
        for root in roots:
            for current, dirs, files in os.walk(root, topdown=False):
                try:
                    Path(current).rmdir()
                except OSError:
                    pass
        record['complete'] = True
        target.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record[k] for k in ('complete', 'files', 'removed_logical_bytes')}))


if __name__ == '__main__':
    main()
