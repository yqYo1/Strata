#!/usr/bin/env python3
"""Retire closed successful traces and the resolved component-loader trace.

Never touches current NativeDense model captures or any source/binary/model.
"""
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
        compact = A/'qsa-clean-dense-component-log-compact-v1.json'
        target = A/'qsa-clean-dense-component-log-retention-v1.json'
        require(not compact.exists() and not target.exists(), 'fresh closed-run retention')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        comparison_path = A/'qsa-device-batch-clean-comparison-v1.json'
        comparison = json.loads(comparison_path.read_text())
        require(comparison['passed'] and comparison['complete'], 'six clean samples compared')
        model_path = A/'native-dense-bank-model-1-v1.json'
        model = json.loads(model_path.read_text())
        require(model['passed'] and model['complete'] and not model['active'], 'current GPU model closed')
        for c in model['commands']:
            require(module.closed(c), 'current model commands closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'current model owner retired')
        roots = []
        for s in comparison['samples']:
            roots.append((A/s['receipt'], B/f"main-qsa-device-batch-clean-{s['arm']}-r{s['repetition']}-root-v1", False))
        for mode in ('normal','oom-null','unknown-null','unknown-nonnull','oom-nonnull','success-null'):
            roots.append((A/f'native-dense-bank-component-{mode}-v2.json', B/f'main-native-dense-bank-component-{mode}-root-v2', False))
        roots.append((A/'native-dense-bank-component-normal-v1.json', B/'main-native-dense-bank-component-normal-root-v1', True))
        evidence = dict(passed=True, complete=True, GPU_executed=False, recipe_sha256=sha(Path(__file__)),
            references={comparison_path.name:sha(comparison_path),model_path.name:sha(model_path)}, runs=[])
        deleted = []
        def plan(path, digest, reason):
            require(path.is_file() and not path.is_symlink() and sha(path) == digest, 'exact closed artifact '+str(path))
            deleted.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest, reason=reason))
        for receipt,root,resolved_failure in roots:
            r = json.loads(receipt.read_text())
            require(not r['active'] and not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'], 'closed healthy run')
            require((not r['passed'] and r['exit_code'] == -6) if resolved_failure else (r['passed'] and r['complete']), 'original status preserved')
            evidence['references'][receipt.name] = sha(receipt)
            item = dict(receipt=receipt.name, original_passed=r['passed'], source_binary_sha256=r['binary_sha256'],
                fresh_faults=r['fresh_faults'], boot_unchanged=r['boot_unchanged'], driver_dump_exists=r['driver_dump_exists'],
                head_lifecycle=r.get('head_lifecycle'), placement=r.get('placement_lines'),
                failure_excerpt=r.get('runtime_errors') if resolved_failure else r.get('mechanism_excerpt'), API={})
            for c in r['commands']:
                require(module.closed(c), 'owned command closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owner retired')
                for name,v in c['logs'].items():
                    path = root/'owned'/name
                    require(path.stat().st_size == v['bytes'], 'log extent')
                    if name.startswith('native-dense-bank-normal') and name.endswith('.stderr'):
                        text = path.read_text()
                        entries = re.findall(r'^\s*---> (ur\w+)\s*$', text, re.M)
                        returns = re.findall(r'<--- (ur\w+)\(.*\) -> (UR_RESULT_\w+);$', text, re.M)
                        counts = collections.Counter(fn for fn,result in returns)
                        require(counts == collections.Counter(entries) and returns and
                            all(result == 'UR_RESULT_SUCCESS' for _,result in returns), 'complete successful UR API entry/results')
                        item['API'][name] = dict(counts=dict(sorted(counts.items())), total_returns=len(returns), all_success=True)
                    plan(path, v['sha256'], 'Closed original status and individual outcomes retained; complete successful API evidence or resolved loader failure extracted. Repeated process/journal/protocol text has no remaining consumer.')
            if 'qsa_batch_arm' in r:
                path = root/'owned/engine.stderr'
                plan(path, r['native_log_sha256'], 'Six clean samples compared; placement/head lifecycle, request metrics and owner/fault evidence committed.')
                for v in r['request_journals']:
                    plan(root/v['path'], v['sha256'], 'Full successful API request journal validated; body hash, actual read/reuse counts and all individual results retained in original receipt.')
            duplicate = root/'record.json'
            if duplicate.exists():
                require(json.loads(duplicate.read_text()) == r, 'external record duplicates committed receipt')
                plan(duplicate, sha(duplicate), 'Closed duplicate record; resume from committed original receipt and report.')
            evidence['runs'].append(item)
        with compact.open('x') as output:
            json.dump(evidence, output, indent=2); output.write('\n')
        require(json.loads(compact.read_text()) == evidence, 'verified compact evidence')
        manifest = dict(passed=False, complete=False, GPU_executed=False, recipe_sha256=sha(Path(__file__)),
            replacement=compact.name, replacement_sha256=sha(compact), deleted=deleted,
            current_model_raw_retained_for='Main same-native FP32 residualGPU0 vs32768 selected-row comparison; new capture budget64MiB paired.')
        with target.open('x') as output:
            json.dump(manifest, output, indent=2); output.write('\n')
        for item in deleted:
            path = Path(item['path'])
            require(sha(path) == item['sha256'], 'identity before unlink')
            path.unlink()
        manifest.update(passed=True, complete=True, removed_bytes=sum(x['bytes'] for x in deleted), finished_utc=module.utc())
        target.write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(dict(passed=True, retired_files=len(deleted), removed_bytes=manifest['removed_bytes'])), flush=True)


if __name__ == '__main__':
    main()
