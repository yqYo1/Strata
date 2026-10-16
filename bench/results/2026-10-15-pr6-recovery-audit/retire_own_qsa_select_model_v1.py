#!/usr/bin/env python3
"""Retire one closed successful exact model comparison; preserve reference."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main(arm):
    require(arm in (0, 1, 2), 'declared arm')
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rp = A / f'own-qsa-select-model-a{arm}-v1.json'
        cp = A / f'own-qsa-select-model-a{arm}-comparison-v1.json'
        r, c = [json.loads(p.read_text()) for p in (rp, cp)]
        require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults']
                and r['boot_unchanged'] and not r['driver_dump_exists'], 'healthy closed model')
        require(c['passed'] and c['complete'] and c['sampled_FP32_values_equal']
                and c['sampled_FP32_values_finite'], 'actual selected rows exact finite')
        for command in r['commands']:
            require(module.closed(command), 'closed command')
            for old in command['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'owner absent')
        reference_path = A / r['selected_capture_reference']['receipt']
        reference = json.loads(reference_path.read_text())
        keep = Path(reference['boundary']['raw_path'])
        require(keep.stat().st_size == reference['boundary']['raw_bytes'] and
                sha(keep) == reference['boundary']['raw_sha256'] == r['boundary']['raw_sha256'],
                'retain one exact original reference, candidate is redundant')
        root = B / f'main-own-qsa-select-model-a{arm}-root-v1'
        files = [f for f in sorted(root.rglob('*')) if f.is_file()]
        require(all(not f.is_symlink() and f != keep for f in files), 'only private closed files')
        evidence = dict(model_receipt=rp.name, model_sha256=sha(rp), comparison=cp.name,
            comparison_sha256=sha(cp), recipe_sha256=sha(Path(__file__)),
            actual_input=78453, selected_rows=768, same_run_output_rows=768,
            same_run_stored_rows=752, same_run_input_rows=752,
            driver_faults=0, native_server_exit0=True, all_owned_retired=True,
            queue_QSA_ms=r['qsa_profile']['rows'][-1]['queue_ms'],
            individual_diagnostic_PP=r['benchmark_result']['results']['prefill_only']['prompt_tps'],
            retained_reference=dict(path=str(keep), bytes=keep.stat().st_size, sha256=sha(keep),
                consumer='Main remaining selector arms/current defaultOFF same78453 comparison',
                budget_bytes=32 << 20, next_review='after selector decision or reference replacement'))
        compact = A / f'closed-own-qsa-select-model-a{arm}-compact-v1.json'
        target = A / f'closed-own-qsa-select-model-a{arm}-retention-v1.json'
        require(not compact.exists() and not target.exists(), 'fresh retention')
        compact.write_text(json.dumps(evidence, indent=2) + '\n')
        manifest = dict(passed=True, complete=False, active=False,
            original_receipts={rp.name: sha(rp), cp.name: sha(cp)},
            compact=dict(path=compact.name, sha256=sha(compact)),
            files=[dict(path=str(f), bytes=f.stat().st_size, sha256=sha(f),
                reason='Successful closed model trace/journal/generated wrapper/duplicate record/capture: '
                       'all compact measured samples and exact numerical/fault/ownership results retained; '
                       'candidate tensor byte-identical to retained reference, no remaining raw consumer.') for f in files],
            original_PASS_samples_unchanged=True, finished_utc=module.utc())
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        for v in manifest['files']:
            f = Path(v['path'])
            require(f.stat().st_size == v['bytes'] and sha(f) == v['sha256'], 'exact retirement identity')
            f.unlink()
        require(all(sha(A / n) == h for n, h in manifest['original_receipts'].items()), 'original receipts unchanged')
        manifest.update(complete=True, files_retired=len(files), logical_bytes_retired=sum(v['bytes'] for v in manifest['files']))
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({k: manifest[k] for k in ('files_retired', 'logical_bytes_retired')}), flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'arm0|1|2')
    main(int(sys.argv[1]))
