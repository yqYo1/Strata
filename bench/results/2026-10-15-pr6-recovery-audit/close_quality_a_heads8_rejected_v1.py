#!/usr/bin/env python3
"""Retire closed diagnostics; keep all five live-consumer NLL comparison files."""
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        compact = A/'closed-quality-a-heads8-compact-v1.json'
        target = A/'closed-quality-a-heads8-retention-v1.json'
        require(not compact.exists() and not target.exists(),'fresh closure')
        bench = json.loads((A/'own-heads8-isolated-bench-v1.json').read_text())
        require(bench['passed'] and bench['complete'] and all(v['ratio_baseline_over_heads8'] < .5 for v in bench['summary'].values()),'Main rejection: sampled component is over2x slower, no native adoption')
        definitions = [
            ('own-residual-quality-fp32-a-v1.json','main-own-residual-quality-root-v1/fp32-a',{'recipe.json','closed.json','faults.json','nll.bin','nll.bin.meta'}),
            ('own-heads8-isolated-build-v1.json','main-own-heads8-isolated-build-root-v1',set()),
            ('own-heads8-isolated-correctness-v1.json','main-own-heads8-isolated-correctness-root-v1',set()),
            ('own-heads8-isolated-correctness-v2.json','main-own-heads8-isolated-correctness-root-v2',set()),
            ('own-heads8-isolated-bench-v1.json','main-own-heads8-isolated-bench-root-v1',set())]
        evidence = dict(original_result_statuses_unchanged=True,cases={},preserved_NLL_comparison_files=[],
                        rejected_component='Heads8 slower than baseline for nq128/256 in3 independent processes perarm/shape; canonical native/source remained unchanged.',
                        retained_source_fixture='own-prompt-heads8-v2/files/tests/sycl/prompt_heads8.cpp: source oracle consumer for next independent attention proposal; no compiled fixture consumer remains.',
                        representative_dispatch_sha256=sha(A/'own-heads8-actual-dispatch-v1.json'),
                        manual_owned_stop_sha256=sha(A/'own-heads8-isolated-stop-v1.json'))
        deletions = []
        for receipt_name,root_name,keep in definitions:
            source = A/receipt_name
            r = json.loads(source.read_text())
            root = B/root_name
            require(not r['active'] and sha(root/'record.json') == sha(source),'closed exact durable receipt')
            failed = receipt_name == 'own-heads8-isolated-correctness-v1.json'
            require((not r['passed']) if failed else (r['passed'] and r['complete']),'original pass/fail retained')
            for c in r['commands']:
                require(module.closed(c) and (c['exit_code'] in (0,-15) if failed else c['exit_code'] == 0),'owned command terminal')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'],'owner alive')
            if r['GPU_executed']:
                require(not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'],'closed GPU health')
            e = dict(receipt_sha256=sha(source),passed=r['passed'],complete=r['complete'])
            if receipt_name.startswith('own-residual-quality'):
                require(sha(root/'nll.bin') == r['raw_sha256'] and sha(root/'nll.bin.meta') == r['metadata_sha256'] and sha(root/'closed.json') == r['closure_sha256'],'current NLL comparator inputs intact')
                closed = json.loads((root/'closed.json').read_text())
                require(sha(root/'faults.json') == closed['faults']['receipt_sha256'] and sha(root/'recipe.json') == closed['recipe_sha256'],'external NLL recipe/health proof')
                stderr = root/'owned/native-fp32-a.stderr'
                stdout = root/'owned/native-fp32-a.stdout'
                require(sha(stderr) == r['native_stderr_sha256'] and sha(stdout) == r['native_stdout_sha256'],'native trace pins')
                text = stderr.read_text()
                e['progress_and_lifecycle'] = [l for l in text.splitlines() if re.search(r'layer-major candidate|completed 10 chunks|strata dense prefill NLL:|strata prefill dense:',l)]
                require(sum('completed 10 chunks' in l for l in e['progress_and_lifecycle']) == 48,'all48 completion windows retained')
                e['diagnostic_stdout_summary'] = [l for l in stdout.read_text().splitlines() if not l.startswith('prompt  :')]
            elif failed:
                e.update(failure_stderr=(root/'owned/smoke-0.stderr').read_text(),
                         failure_scope='UR adapter initialization error then CPU spinning, owned SIGTERM exit-15, no fresh GPU fault/dump. Layer delimiter corrected only in new v2 recipe; original failure unchanged.')
            elif 'correctness-v2' in receipt_name:
                for label,v in r['cases'].items():
                    require((root/('owned/'+label+'.stdout')).read_text() == v['stdout'] and sha(root/('owned/'+label+'.stderr')) == v['stderr_sha256'],'case stdout/trace proof')
                    require(sha(root/(label+'.bin')) == v['dump_sha256'],'case dump pin')
                for phase in ('smoke','full'):
                    require((root/(phase+'-0.bin')).read_bytes() == (root/(phase+'-1.bin')).read_bytes(),'reverify full dumps before retirement')
                e['trace_success_returns'] = {str(arm):(root/f'owned/smoke-{arm}.stderr').read_text().count('UR_RESULT_SUCCESS') for arm in (0,1)}
            elif 'isolated-bench' in receipt_name:
                reference = {}
                require(len(r['cases']) == 12 and r['all_process_outputs_byteexact'],'all independent samples retained')
                for v in r['cases']:
                    label = f"bench-{v['sequence']}-nq{v['nq']}-arm{v['arm']}"
                    dump = root/(label+'.bin')
                    require(sha(dump) == v['dump_sha256'] and (root/('owned/'+label+'.stdout')).read_text() == v['stdout'] and sha(root/('owned/'+label+'.stderr')) == v['stderr_sha256'],'timing/output pins')
                    data = dump.read_bytes()
                    if v['nq'] not in reference:
                        reference[v['nq']] = data
                    require(data == reference[v['nq']],'all full benchmark outputs still exact')
            evidence['cases'][receipt_name] = e
            for path in root.rglob('*'):
                if not path.is_file():
                    continue
                require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),'private raw artifact')
                item = dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
                if path.relative_to(root).as_posix() in keep:
                    evidence['preserved_NLL_comparison_files'].append(item)
                else:
                    item.update(status='planned',reason='Closed job; result/owners/health/all samples/oracles/exact comparisons/source/flags preserved. Rejected component has no remaining compiled/raw consumer; NLL raw comparator inputs excluded from deletion.')
                    deletions.append(item)
        compact.write_text(json.dumps(evidence,indent=2)+'\n')
        manifest = dict(status='planned',recipe_sha256=sha(Path(__file__)),compact_sha256=sha(compact),
                        owner='Main',original_result_statuses_unchanged=True,physical_free_space_claim=False,deletions=deletions)
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        for item in deletions:
            path = Path(item['path'])
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],'identity before unlink')
            path.unlink()
            item['status'] = 'removed'
        for _,root_name,keep in definitions:
            root = B/root_name
            for path in sorted((p for p in root.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                if not any(path.iterdir()):
                    path.rmdir()
            if not keep:
                root.rmdir()
        manifest.update(status='complete',files=len(deletions),removed_logical_bytes=sum(v['bytes'] for v in deletions))
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({k:manifest[k] for k in ('status','files','removed_logical_bytes')}))


if __name__ == '__main__':
    main()
