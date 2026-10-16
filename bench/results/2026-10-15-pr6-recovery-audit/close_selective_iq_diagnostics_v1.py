#!/usr/bin/env python3
"""Keep compact complete evidence and retire closed successful synthetic captures."""
import fcntl
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
BASE=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(HERE/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha
from close_iq_diagnostics_v1 import routes


def main():
    module=load_owner(PROJECT/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (BASE/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target=HERE/'selective-iq-diagnostics-retention-v1.json'
        require(not target.exists(),'closure already used')
        records={}
        for name in ['selective-iq-component-quick-v1.json','selective-iq-component-full-v1.json',
                     'iq-timing-progress-v4.json','selective-iq-model-v1.json']:
            r=json.loads((HERE/name).read_text())
            require(not r['active'] and not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'],'closed fault gate')
            for c in r['commands']:
                require(module.closed(c) and c['exit_code']==0,'normal owned closure required')
                for pin in c['owners']:
                    now=module.identity(pin['pid'])
                    require(now is None or now['start_ticks']!=pin['start_ticks'] or now['state']=='Z','owned process still live')
            records[name]=sha(HERE/name)
        assessment=json.loads((HERE/'selective-iq-model-assessment-v2.json').read_text())
        require(assessment['passed'] and assessment['complete'] and assessment['census_validation']['passed'],'model census/equality gate')
        require(assessment['original_receipt_sha256']==records['selective-iq-model-v1.json'],'original failure identity')
        proof=dict(schema='main-selective-iq-diagnostic-evidence-v1',quick_routes={},source_receipts=records,
                   actual_model_assessment_sha256=sha(HERE/'selective-iq-model-assessment-v2.json'),
                   scope='Actual selective product route, full array comparisons and closed model/census. No clean speed/task-quality claim.')
        paths=[]
        for mode in ['quick','full']:
            root=BASE/('main-selective-iq-component-'+mode+'-root-v1')
            compare=root/'comparison.json'
            compact=HERE/('selective-iq-component-'+mode+'-arrays-v1.json')
            require(sha(compare)==sha(compact),'compact comparison differs')
            result=json.loads(compare.read_text())
            require(result['array_comparison']=='PASS','array comparison failure')
            for arm in ['off','on']:
                if mode=='quick':
                    trace=root/('owned/device-'+arm+'.stderr')
                    actual=routes(trace.read_text())
                    require(not actual['errors'] and actual['total_launches']==132,'actual traced route/errors')
                    expected={} if arm=='off' else {'type21_GU1':1,'type22_GU1':1,'type23_GU1':1,'type20_GU0':2,'type42_GU0':2}
                    require(actual['candidate_by_type_and_role']==expected,'selective role dispatch mismatch')
                    proof['quick_routes'][arm]=dict(**actual,trace_sha256=sha(trace),trace_bytes=trace.stat().st_size)
                for name,pin in result['files'][arm].items():
                    path=root/arm/name
                    require(path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],'raw array identity differs')
                    paths.append(path)
                paths.extend([root/('owned/device-'+arm+'.stdout'),root/('owned/device-'+arm+'.stderr')])
        model=BASE/'main-selective-iq-model-root-v1/native-owned'
        text=(model/'model.stderr').read_text()
        require(list(map(int,re.findall(r'layer-major layer (\d+) of 48: loading',text)))==list(range(48)),'model layer loading gaps')
        require(list(map(int,re.findall(r'layer-major layer (\d+) of 48: completed 8 chunks',text)))==list(range(48)),'model layer completion gaps')
        require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|retirement failed|stopping with owners live',text),'model runtime errors')
        proof['model']=dict(all48layers_completed=True,stderr_sha256=sha(model/'model.stderr'),
                            selected_progress=[s for s in text.splitlines() if s.startswith(('strata trace: read ', 'strata serve: prompt ',
                              'strata serve: decode expert cache hit rate:', 'strata serve: KV streaming:'))])
        paths.extend([model/'model.stdout',model/'model.stderr'])
        # Diagnostic v4 already keeps its full70 phase transcript and numeric result.
        diagnostic=BASE/'main-iq-timing-progress-root-v4/native-owned'
        trace=diagnostic/'diagnostic.stderr'
        t=trace.read_text()
        require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_',t),'diagnostic API error')
        proof['progress_v4']=dict(trace_sha256=sha(trace),trace_bytes=trace.stat().st_size,
                                  successful_API_returns=t.count('-> UR_RESULT_SUCCESS;'),
                                  phase_count=sum(s.startswith('iq-progress-v3 ') for s in t.splitlines()),
                                  retained_result='iq-timing-progress-v4.json')
        paths.extend([trace,diagnostic/'diagnostic.stdout'])
        proof_path=HERE/'selective-iq-diagnostic-evidence-v1.json'
        proof_path.write_text(json.dumps(proof,indent=2)+'\n')
        manifest=dict(schema='selective-iq-retention-v1',status='planned',recipe_sha256=sha(Path(__file__)),
                      owner='Main',preserved=list(records)+['selective-iq-model-assessment-v2.json',
                      'selective-iq-component-quick-arrays-v1.json','selective-iq-component-full-arrays-v1.json',proof_path.name],
                      original_statuses_unchanged=True,physical_free_space_claim=False,deletions=[])
        for path in paths:
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(BASE),'retention path escape')
            manifest['deletions'].append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),status='planned',
              reason='Complete arrays, actual route/progress and census evidence retained; no remaining raw comparison/RESTORE/numerical consumer.'))
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        for item in manifest['deletions']:
            path=Path(item['path'])
            require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'identity changed before unlink')
            path.unlink();item['status']='removed'
        manifest.update(status='complete',removed_logical_bytes=sum(i['bytes'] for i in manifest['deletions']))
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(files=len(paths),removed_logical_bytes=manifest['removed_logical_bytes'],actual_ON_candidate_launches=7)),flush=True)


if __name__=='__main__':
    main()
