#!/usr/bin/env python3
"""Main-only retirement of closed duplicate tensors/traces; no source/model deletion."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha


def main():
    spec=importlib.util.spec_from_file_location('owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=D/'retention-v6-manifest.json';assert not output.exists()
        manifest=dict(complete=False,owner='Main',recipe_sha256=sha(Path(__file__)),GPU_executed=False,
            original_status_unchanged=True,deleted=[],evidence={},preserved=[],
            scope='Only explicitly enumerated closed diagnostic logs and redundant success tensors. Models/source/binaries and original unresolved divergence captures stay.',
            allocated_bytes_are_not_available_space=True)
        pending={}
        def persist():output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        def load(name,success=True):
            path=D/name;d=json.loads(path.read_text())
            assert not d['active'] and (not success or (d['passed'] and d['complete']))
            for c in d.get('commands',[]):
                assert c['direct_child_reaped'] and c['session_empty']
                for i in c['owners']:
                    now=module.identity(i['pid'])
                    assert now is None or now['start_ticks']!=i['start_ticks'] or now['state']=='Z'
            manifest['evidence'][name]=sha(path)
            return d
        def add(path,h,reason,replacement):
            path=Path(path);assert path.is_relative_to(B) and path.is_file() and not path.is_symlink()
            assert sha(path)==h,(path,'hash changed')
            assert path not in pending
            s=path.stat();pending[path]=dict(path=str(path),bytes=s.st_size,sha256=h,
                device=s.st_dev,inode=s.st_ino,allocated_bytes=s.st_blocks*512,reason=reason,replacement=replacement)
        # The old fixed and divergent raw rows remain the numerical failure fixtures.
        original=load('boundary-audit-long-fixed-v1.json')
        arm=original['arms'][0];canonical=Path(arm['boundary']['raw_path'])
        assert canonical.is_file() and sha(canonical)==arm['boundary']['raw_sha256']
        manifest['preserved'].append(dict(path=str(canonical),sha256=sha(canonical),consumer='Canonical old fixed rows paired with unresolved old auto divergence'))
        for name in ['phase-audit-small-v1.json','phase-audit-long-fixed-v1.json','phase-audit-long-auto-v1.json','boundary-audit-phase-off-long-auto-v2.json']:
            d=load(name)
            for arm in d['arms']:
                for f in arm['fixtures']:
                    add(f['path'],f['sha256'],'Completed successful selected/first-logit comparison; hashes/results committed',name)
                if 'boundary' in arm:
                    v=arm['boundary']
                    if 'long' in name:assert v['raw_sha256']==sha(canonical)
                    for field,hfield in [('text_path','text_sha256'),('raw_path','raw_sha256')]:
                        add(v[field],v[hfield],'Completed source/store/input comparison; long successful raw rows duplicate preserved canonical fixed rows',name+' + phase-boundary-controls-v1.json')
                if 'phase' in arm:
                    v=arm['phase']
                    if name=='phase-audit-long-fixed-v1.json' or (name=='phase-audit-small-v1.json' and arm['label']=='fp32-phase'):
                        manifest['preserved'].append(dict(path=v['raw_path'],sha256=v['raw_sha256'],consumer='One canonical phase capture per small/long shape; future exact phase comparison'))
                    else:
                        for field,hfield in [('text_path','text_sha256'),('raw_path','raw_sha256')]:
                            add(v[field],v[hfield],'Exact duplicate phase data; one canonical small/long phase capture retained',name+' + layer26-phase-pair-v1.json')
        # Successful raw traces and tables have exact compact consumers in Git.
        for name in ['kv-copy-groups-validation-v2.json','kv-copy-groups-graph-benchmark-128-v3.json',
                     'kv-copy-groups-graph-benchmark-512-v3.json','kv-resolver-profile-v1.json','model-device-profile-v1.json']:
            d=load(name)
            for c in d['commands']:
                if not any(s in c['label'] for s in ['kv-','model-profile']):continue
                # Only the actual component/model label, not journal/help utilities.
                if c['label'] not in ['kv-validation','kv-graph-benchmark','kv-profile','model-profile']:continue
                for fname,v in c['logs'].items():
                    path=B/('main-primary-kv-copy-groups-validation-root-v2' if name=='kv-copy-groups-validation-v2.json' else
                        'main-primary-kv-copy-groups-graph-benchmark-128-root-v3' if 'benchmark-128' in name else
                        'main-primary-kv-copy-groups-graph-benchmark-512-root-v3' if 'benchmark-512' in name else
                        'main-primary-kv-resolver-profile-root-v1' if name=='kv-resolver-profile-v1.json' else
                        'main-primary-model-device-profile-root-v1')/'owned'/fname
                    if path.suffix=='.stdout' and 'samples' in d:
                        assert [json.loads(x) for x in path.read_text().splitlines() if x.strip()]==d['samples']
                    if path.suffix=='.stderr' and path.stat().st_size<4096:
                        manifest.setdefault('short_diagnostics',{})[str(path)]=path.read_text()
                    add(path,v['sha256'],'All compared samples/correctness and useful profiler/ledger evidence preserved; raw protocol is redundant',name)
            for f in d.get('provider_files',[]):
                path=Path(f['path'])
                if 'text' in f:assert hashlib.sha256(f['text'].encode()).hexdigest()==f['sha256']
                elif path.name=='run_config.json':manifest.setdefault('provider_configuration',{})[str(path)]=json.loads(path.read_text())
                else:raise RuntimeError('unextracted provider file '+str(path))
                add(path,f['sha256'],'Exact provider table/configuration extracted and verified',name+' / profile summaries')
        # Failed full-API attempt remains failed; only its unnecessary bulk trace retires.
        failed=load('prompt-direct-small-v1.json',success=False)
        health=load('prompt-direct-small-faultcheck-v1.json');success=load('prompt-direct-small-v2.json')
        assert health['original_receipt_sha256']==sha(D/'prompt-direct-small-v1.json')
        assert not health['fresh_faults'] and not success['fresh_faults']
        c=next(c for c in failed['commands'] if c['label']=='direct-small')
        assert c['exit_code']==-25 and 'log cap' in c['error']
        for fname,v in c['logs'].items():
            path=B/'main-primary-prompt-direct-small-root-v1/owned'/fname
            if path.suffix=='.stderr':
                with path.open('rb') as f:f.seek(max(0,path.stat().st_size-4096));excerpt=f.read().decode(errors='replace')
                manifest['failed_log_budget_excerpt']=dict(path=str(path),tail=excerpt,error=c['error'],native_exit_code=-25,
                    conclusion='Known raw-log budget/SIGXFSZ incomplete diagnostic; original failure status preserved, later independent faultcheck and same numerical retry PASS')
            add(path,v['sha256'],'Known log-budget failure has owner/error/exit, representative API tail, faultcheck and replacement numerical result; full success-API history answers no open question','prompt-direct-small-v1.json + faultcheck-v1.json + v2.json')
        manifest['planned']=list(pending.values());persist()
        allocated=set();logical=0;physical=0
        for path,item in pending.items():
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
            key=item['device'],item['inode'];logical+=item['bytes']
            if key not in allocated:physical+=item['allocated_bytes'];allocated.add(key)
            path.unlink();manifest['deleted'].append(item);persist()
        manifest.update(complete=True,deleted_files=len(pending),removed_logical_bytes=logical,
            removed_unique_allocated_bytes=physical)
        persist();print(json.dumps(dict(complete=True,files=len(pending),logical_bytes=logical,unique_allocated_bytes=physical)))


if __name__=='__main__':main()
