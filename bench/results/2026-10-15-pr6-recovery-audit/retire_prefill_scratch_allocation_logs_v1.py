#!/usr/bin/env python3
"""Compact closed allocation diagnostics before retiring full UR histories."""
import collections
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compact_path = A/'prefill-scratch-allocation-log-evidence-v1.json'
        manifest_path = A/'prefill-scratch-allocation-log-retention-v1.json'
        require(not compact_path.exists() and not manifest_path.exists(), 'fresh retention result')
        owner = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        pending, evidence = [], []
        for v in (1, 2):
            receipt = A/f'prefill-scratch-allocation-v{v}.json'
            r = json.loads(receipt.read_text())
            require(not r['active'] and r['passed'] == (v == 2) and r['component_exit_code'] == (0 if v == 2 else 88), 'original closed outcome')
            require(not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'], 'original fault outcome')
            for c in r['commands']:
                require(owner.closed(c), 'command not closed')
                for old in c['owners']:
                    now = owner.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owner present')
            command = next(c for c in r['commands'] if c['label'] == 'actual-scratch-allocation')
            path = B/f'main-prefill-scratch-allocation-root-v{v}/owned/actual-scratch-allocation.stderr'
            before = command['logs'][path.name]
            require(path.stat().st_size == before['bytes'] and sha(path) == before['sha256'], 'original log identity')
            lines = path.read_text().splitlines()
            returns = [m.groups() for line in lines if (m := re.search(r'<--- (ur\w+)\(.*\) -> (UR_RESULT_\w+);$', line))]
            enters = re.findall(r'^\s*---> (ur\w+)\s*$', '\n'.join(lines), flags=re.M)
            counts = collections.Counter(fn for fn, status in returns)
            require(collections.Counter(enters) == counts and all(s == 'UR_RESULT_SUCCESS' for _,s in returns), 'complete successful API entries/results')
            relevant = [line for line in lines if
                re.search(r'<--- ur(?:USM\w*|EnqueueUSM\w*|QueueFinish|QueueRelease|ContextRelease|AdapterRelease)\(', line) or
                line.startswith(('strata prefill:', 'scratch allocation fail-stop:'))]
            if v == 1:
                require(not any('USM' in fn for fn in counts) and r['runtime_errors'] == [lines[-1]], 'failure occurred before allocation/copy')
            evidence.append(dict(version=v, receipt=receipt.name, receipt_sha256=sha(receipt),
                original_log=dict(path=str(path), **before),
                original_passed=r['passed'], original_exit=r['component_exit_code'],
                API_counts=dict(sorted(counts.items())), API_returns_total=len(returns),
                all_API_returns_success=True, retained_lifetime_and_failure_lines=relevant,
                stdout=r['stdout'], carve_lines=r.get('carve_lines', []),
                fresh_faults=r['fresh_faults'], boot_unchanged=r['boot_unchanged'],
                driver_dump_exists=r['driver_dump_exists']))
            pending.append((path, receipt, sha(receipt), before))
        compact = dict(passed=True, complete=True, GPU_executed=False, recipe_sha256=sha(Path(__file__)),
            scope='Closed fixture failure/progress/allocation lifetime API summary only; no inference, hidden correctness or pool/leak attribution.', runs=evidence)
        with compact_path.open('x') as out:
            json.dump(compact, out, indent=2); out.write('\n')
        require(json.loads(compact_path.read_text()) == compact, 'compact evidence roundtrip')
        manifest = dict(passed=False, complete=False, GPU_executed=False,
            recipe_sha256=sha(Path(__file__)), replacement=compact_path.name,
            replacement_sha256=sha(compact_path), deleted=[],
            reason='First allocation diagnostic is resolved as fixture8MiB slack expectation; corrected exact-carve/short-loan GPU check passed. Retain distinct failure error and all API counts plus allocation/copy/free/drain/release lines; full successful API history has no current consumer.')
        with manifest_path.open('x') as out:
            json.dump(manifest, out, indent=2); out.write('\n')
        for path, receipt, identity, before in pending:
            require(sha(receipt) == identity and sha(path) == before['sha256'], 'identity before unlink')
            path.unlink()
            manifest['deleted'].append(dict(path=str(path), **before, original_receipt_sha256=identity))
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        manifest.update(passed=True, complete=True, removed_bytes=sum(e['bytes'] for e in manifest['deleted']),
            retained_bytes=compact_path.stat().st_size, finished_utc=owner.utc())
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ('passed','removed_bytes','retained_bytes')}))


if __name__ == '__main__':
    main()
