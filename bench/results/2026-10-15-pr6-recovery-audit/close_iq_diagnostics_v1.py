#!/usr/bin/env python3
"""Extract verified route/progress evidence, then retire closed successful raw data."""
import collections
import fcntl
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BASE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(HERE / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def routes(text):
    kernels, count, candidate = {}, collections.Counter(), collections.Counter()
    created = re.compile(r'urKernelCreate\(.*\.pKernelName = 0x[0-9a-f]+ \(([^)]+)\), \.phKernel = 0x[0-9a-f]+ \((0x[0-9a-f]+)\)\) -> UR_RESULT_SUCCESS')
    launch = re.compile(r'<--- urEnqueueKernelLaunch(?:WithArgsExp)?\(.*?\.hKernel = (0x[0-9a-f]+),')
    for line in text.splitlines():
        if m := created.search(line):
            kernels[m[2]] = m[1]
        if m := launch.search(line):
            require(m[1] in kernels, 'launch kernel lacks observed create identity')
            name = kernels[m[1]]
            count[name] += 1
            if p := re.search(r'parallel_iq_launchILi(\d+)ELb([01])', name):
                candidate[f'type{p[1]}_GU{p[2]}'] += 1
    return dict(total_launches=sum(count.values()), candidate_launches=sum(candidate.values()),
                candidate_by_type_and_role=dict(candidate), kernels=dict(count),
                errors=[s for s in text.splitlines() if re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|FAIL ', s)])


def main():
    module = load_owner(PROJECT / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (HERE / 'iq-diagnostics-retention-v1.json').exists(), 'closure collector already used')
        records = {}
        for name in ('owned-prefill-stall-model-v1.json', 'parallel-iq-component-quick-v1.json', 'parallel-iq-component-full-v1.json'):
            p = HERE / name
            r = json.loads(p.read_text())
            require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults'] and
                    r['boot_unchanged'] and not r['driver_dump_exists'], 'closed success gate: ' + name)
            for command in r['commands']:
                require(module.closed(command), 'owner not normally closed')
                for pin in command['owners']:
                    now = module.identity(pin['pid'])
                    require(now is None or now['start_ticks'] != pin['start_ticks'] or now['state'] == 'Z', 'same owned process remains live')
            records[name] = dict(receipt_sha256=sha(p), record=r)
        quick = BASE / 'main-parallel-iq-component-quick-root-v1'
        proof = dict(schema='main-iq-diagnostic-evidence-v1', quick_routes={}, numerical_scope='Complete half bits including NaNs, guards, inputs and six F32 arrays. Not model quality/performance.',
                     model_scope='Three exact32K inputs only, preceding historical chat/agent omitted; no stall observed does not resolve original full-suite failure.')
        for arm in ('off', 'on'):
            path = quick / ('owned/device-' + arm + '.stderr')
            result = routes(path.read_text())
            require(result['total_launches'] == 132 and not result['errors'], 'unexpected traced launches/errors')
            require(result['candidate_launches'] == (18 if arm == 'on' else 0), 'wrong actual candidate route')
            if arm == 'on':
                require(result['candidate_by_type_and_role'] == {f'type{t}_GU{g}': 1 if g else 2 for t in (18,22,21,23,20,42) for g in (0,1)}, 'candidate type/role launch counts')
            proof['quick_routes'][arm] = dict(**result, trace_sha256=sha(path), trace_bytes=path.stat().st_size)
        model = BASE / 'main-owned-prefill-stall-model-root-v1'
        model_text = (model / 'native-owned/model.stderr').read_text()
        errors = [s for s in model_text.splitlines() if re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|retirement failed|stopping with owners live', s)]
        require(not errors, 'model runtime diagnostic errors')
        loading = re.findall(r'layer-major layer (\d+) of 48: loading', model_text)
        complete = re.findall(r'layer-major layer (\d+) of 48: completed 8 chunks', model_text)
        require(list(map(int, loading)) == list(range(48)) * 3 and list(map(int, complete)) == list(range(48)) * 3, 'model layer progress gaps')
        proof['model'] = dict(three_all48layer_walks=True, runtime_errors=errors,
                             stderr_sha256=sha(model / 'native-owned/model.stderr'),
                             selected_progress=[s for s in model_text.splitlines() if s.startswith(('strata trace: read ', 'strata serve: prompt ', 'strata serve: decode expert cache hit rate:', 'strata serve: KV streaming:'))])
        (HERE / 'iq-diagnostic-evidence-v1.json').write_text(json.dumps(proof, indent=2) + '\n')
        paths = [model / 'native-owned/model.stdout', model / 'native-owned/model.stderr']
        for mode in ('quick', 'full'):
            root = BASE / ('main-parallel-iq-component-' + mode + '-root-v1')
            comparison = json.loads((root / 'comparison.json').read_text())
            committed = HERE / ('parallel-iq-component-' + mode + '-arrays-v1.json')
            require(sha(committed) == sha(root / 'comparison.json'), 'compact comparator copy changed')
            for arm in ('off', 'on'):
                for name, pin in comparison['files'][arm].items():
                    p = root / arm / name
                    require(p.is_file() and not p.is_symlink() and p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'capture pin changed')
                    paths.append(p)
                paths.extend([root / ('owned/device-' + arm + '.stdout'), root / ('owned/device-' + arm + '.stderr')])
        manifest = dict(schema='iq-diagnostics-retention-v1', recipe_sha256=sha(Path(__file__)),
                        owner='Main', status='planned', source_receipts={n:v['receipt_sha256'] for n,v in records.items()},
                        preserved=['iq-diagnostic-evidence-v1.json', *records, 'parallel-iq-component-quick-arrays-v1.json', 'parallel-iq-component-full-arrays-v1.json'],
                        reason='Successful complete comparisons and route/progress evidence verified. Raw synthetic arrays and repeated API/layer traces have no remaining comparison/RESTORE or unresolved-numerical consumer.',
                        physical_free_space_claim=False, deletions=[])
        for path in paths:
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(BASE), 'retention path escape')
            manifest['deletions'].append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path), status='planned'))
        target = HERE / 'iq-diagnostics-retention-v1.json'
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        for item in manifest['deletions']:
            p = Path(item['path'])
            require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'retention identity changed before unlink')
            p.unlink()
            item['status'] = 'removed'
        manifest.update(status='complete', removed_logical_bytes=sum(x['bytes'] for x in manifest['deletions']))
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(dict(routes='PASS actual18candidate vs0', removed_bytes=manifest['removed_logical_bytes'], files=len(paths))))


if __name__ == '__main__':
    main()
