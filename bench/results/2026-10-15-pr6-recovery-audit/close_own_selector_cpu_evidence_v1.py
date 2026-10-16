#!/usr/bin/env python3
"""Compact only closed successful selector/CPU diagnostics under the GPU lock."""
import fcntl
import json
from pathlib import Path
import re
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
        target = A / 'closed-own-selector-cpu-retention-v1.json'
        compact = A / 'closed-own-selector-cpu-compact-v1.json'
        require(not target.exists() and not compact.exists(), 'fresh closure')
        comparison = json.loads((A / 'own-selector-clean-comparison-v1.json').read_text())
        require(comparison['passed'] and comparison['complete'], 'all6 clean comparison')
        pairs = [(f'own-selector-clean-{a}-r{r}-v1.json', f'main-own-selector-clean-{a}-r{r}-root-v1')
            for a, r in comparison['actual_order']]
        pairs += [('own-prefill-cpu-wait-build-v1.json', 'main-own-prefill-cpu-wait-build-root-v1'),
                  ('own-prefill-cpu-wait-model-v1.json', 'main-own-prefill-cpu-wait-model-root-v1')]
        proof = dict(original_results_unchanged=True, records={}, comparison_sha256=sha(A / 'own-selector-clean-comparison-v1.json'))
        paths = []
        for name, directory in pairs:
            r = json.loads((A / name).read_text()); root = B / directory
            require(r['passed'] and r['complete'] and not r['active'] and not r.get('fresh_faults')
                and r.get('boot_unchanged') and not r.get('driver_dump_exists'), 'healthy closed record')
            for command in r['commands']:
                require(module.closed(command) and command['exit_code'] == 0, 'normal owned closure')
                for old in command['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owned identity alive')
            require(sha(root / 'record.json') == sha(A / name), 'record is duplicate of durable receipt')
            entry = dict(receipt_sha256=sha(A / name), root=str(root), raw_files=[])
            native = root / 'owned/engine.stderr'
            if native.exists():
                text = native.read_text()
                require(sha(native) == r['native_log_sha256'], 'native log identity')
                entry['selected_native_progress'] = [s for s in text.splitlines() if s.startswith((
                    'strata prefill: layer-major candidate,', 'strata prefill: FP16 MoE compact',
                    'strata prefill cpu wait:', 'strata trace: read ', 'strata serve: prompt ',
                    'strata trace: layer-major layer '))]
                if name == 'own-prefill-cpu-wait-model-v1.json':
                    require(list(map(int, re.findall(r'layer-major layer (\d+) of 48: completed 10 chunks', text))) == list(range(48)), 'all48 layers10chunks')
                    require(text.count(r['CPU_wait_diagnostic']['line']) == 1, 'actual CPU line retained')
                for v in r['request_journals']:
                    require(sha(root / v['path']) == v['sha256'], 'closed verified request journal')
                suite = list((root / 'stock-results').glob('*/results.json'))
                require(len(suite) == 1 and sha(suite[0]) == r['benchmark_result_sha256'], 'full suite retained in receipt')
                require(json.loads(suite[0].read_text()) == r['benchmark_result'], 'exact structured suite equality')
                require(json.loads((root / 'engine-exit.json').read_text()) == r['result']['real_engine_exit'], 'actual native exit retained')
                entry['engine_start'] = json.loads((root / 'engine-start.json').read_text())
                entry['wrapper_source'] = (root / 'engine-wrapper.py').read_text()
            else:
                entry['build_stdout'] = (root / 'owned/incremental-sycl-build.stdout').read_text()
                entry['small_GPU_evidence'] = r['gpu_cases']
                entry['host_cases'] = r['host_cases']
                entry['successful_API_returns'] = {n: (root / ('owned/' + n + '.stderr')).read_text().count('UR_RESULT_SUCCESS') for n in r['gpu_cases']}
            for path in root.rglob('*'):
                if not path.is_file(): continue
                require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'private artifact path')
                item = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                    reason='Closed success; source/recipes/individual timings/CPU accounting/actual lifecycle/faults retained in Git receipts and compact evidence; no raw comparison consumer remains.', status='planned')
                entry['raw_files'].append(dict(path=str(path.relative_to(root)), bytes=item['bytes'], sha256=item['sha256']))
                paths.append(item)
            proof['records'][name] = entry
        compact.write_text(json.dumps(proof, indent=2) + '\n')
        manifest = dict(status='planned', owner='Main', recipe_sha256=sha(Path(__file__)), compact_sha256=sha(compact),
            original_results_unchanged=True, physical_free_space_claim=False, deletions=paths)
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        for item in paths:
            path = Path(item['path'])
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'identity before deletion')
            path.unlink(); item['status'] = 'removed'
        manifest.update(status='complete', removed_logical_bytes=sum(v['bytes'] for v in paths), files=len(paths))
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({k: manifest[k] for k in ('status', 'files', 'removed_logical_bytes')}), flush=True)


if __name__ == '__main__': main()
