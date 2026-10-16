#!/usr/bin/env python3
"""Main-only retirement of verified closed Direct/observer duplicates."""
import fcntl,hashlib,importlib.util,json,re
from pathlib import Path
from check_gemm_orientation import B,D,OWNER,sha

def main():
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=D/'retention-v7-manifest.json';assert not output.exists()
        manifest=dict(complete=False,GPU_executed=False,owner='Main',recipe_sha256=sha(Path(__file__)),evidence={},deleted=[],preserved=[],diagnostic_excerpts={},original_status_unchanged=True,
            scope='Closed recent Direct and CPUobserver logs and successful duplicate tensors only. No model/source/binary/original divergent fixture deletion. Logical/allocation bytes are not available-space claims.')
        pending={}
        def persist():output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        def load(name,success=True):
            p=D/name;r=json.loads(p.read_text());assert not r['active']
            if success:assert r['passed'] and r['complete']
            for c in r.get('commands',[]):
                assert c['direct_child_reaped'] and c['session_empty']
                for i in c['owners']:
                    now=m.identity(i['pid']);assert now is None or now['start_ticks']!=i['start_ticks'] or now['state']=='Z'
            manifest['evidence'][name]=sha(p);return r
        def add(p,h,reason,replacement):
            p=Path(p);assert p.is_relative_to(B) and p.resolve().is_relative_to(B.resolve()) and p.is_file() and not p.is_symlink() and p not in pending
            assert sha(p)==h;s=p.stat();pending[p]=dict(path=str(p),bytes=s.st_size,sha256=h,device=s.st_dev,inode=s.st_ino,allocated_bytes=s.st_blocks*512,reason=reason,replacement=replacement)
        canonical=load('selected-prefill-small-v1.json');control=next(a for a in canonical['arms'] if a['dtype']=='fp32')
        hashes=set()
        for f in control['fixtures']:
            p=Path(f['path']);assert p.is_file() and sha(p)==f['sha256'];hashes.add(f['sha256'])
            manifest['preserved'].append(dict(path=str(p),sha256=f['sha256'],consumer='One canonical small selected-head/first-logit numerical comparison fixture'))
        small=load('prompt-direct-small-v2.json')
        for f in small['fixtures']:
            assert f['sha256'] in hashes;add(f['path'],f['sha256'],'Successful Direct tensor exactly duplicates preserved canonical source', 'prompt-direct-small-v2.json + selected-prefill-small-v1.json')
        roots={
            'prompt-direct-small-v2.json':'main-primary-prompt-direct-small-root-v2',
            'model-device-profile-v2.json':'main-primary-model-device-profile-root-v2',
            'cpu-gu-reuse-build-v2.json':'main-primary-cpu-gu-reuse-build-root-v2',
            'cpu-gu-reuse-build-v3.json':'main-primary-cpu-gu-reuse-build-root-v3',
            'cpu-gu-reuse-small-v1.json':'main-primary-cpu-gu-reuse-small-root-v1',
            'cpu-gu-reuse-long-v1.json':'main-primary-cpu-gu-reuse-long-root-v1'}
        for name,root in roots.items():
            r=load(name,success=name!='cpu-gu-reuse-build-v2.json')
            if 'fresh_faults' in r:assert not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists']
            if name=='cpu-gu-reuse-build-v2.json':assert not r['passed'] and 'format-security' in r['error'] and not r['GPU_executed']
            for c in r['commands']:
                for filename,v in c['logs'].items():
                    p=B/root/'owned'/filename;text=p.read_text()
                    if p.suffix=='.stderr':
                        if len(text)<=4096:excerpt=text
                        else:excerpt='\n'.join(s for s in text.splitlines() if s.startswith(('strata mtp:','strata prefill:','strata trace: layer-major')) or re.search(r'(?i)warning|error|FAIL|exception|abort',s))
                        assert len(excerpt.encode())<=65536;manifest['diagnostic_excerpts'][str(p)]=excerpt
                    if name.startswith('cpu-gu-reuse-') and 'build' not in name and c['label'].startswith('observer-') and p.suffix=='.stderr':
                        blocks=re.findall(r'^strata decode joined begin .*?^strata decode joined close [^\n]*\n?',text,re.M|re.S)
                        assert len(blocks)==1
                        request=next(x for x in r['requests'] if int(x['observer_on'])==int(c['label'].split('-')[-1]))
                        assert blocks[0].splitlines()==request['ledger']['lines']
                        if request['observer_on']:
                            encoded=re.search(r'^expert_jobs_uleb128_base64 ([^\n]+)$',text,re.M)[1]
                            assert encoded==request['recurrence']['expert_counts_base64']
                    add(p,v['sha256'],'Closed command identity/status/counts/hash retained; useful warning/lifetime excerpts extracted; no current full-history consumer',name)
            for f in r.get('provider_files',[]):
                p=Path(f['path'])
                if 'text' in f:assert hashlib.sha256(f['text'].encode()).hexdigest()==f['sha256']
                elif p.name=='run_config.json':manifest.setdefault('provider_configuration',{})[str(p)]=json.loads(p.read_text())
                else:raise RuntimeError('Unextracted provider file '+str(p))
                add(p,f['sha256'],'Official provider table/config retained exactly in compact receipt/manifest',name+' + prompt-direct-profile-pair-v1.json')
        manifest['planned']=list(pending.values());persist();seen=set();logical=allocated=0
        for p,item in pending.items():
            s=p.stat();assert (s.st_dev,s.st_ino,s.st_size)==(item['device'],item['inode'],item['bytes']) and sha(p)==item['sha256']
            logical+=item['bytes'];key=s.st_dev,s.st_ino
            if key not in seen:allocated+=item['allocated_bytes'];seen.add(key)
            p.unlink();manifest['deleted'].append(item);persist()
        manifest.update(complete=True,deleted_files=len(pending),removed_logical_bytes=logical,removed_unique_allocated_bytes=allocated);persist()
        print(json.dumps(dict(complete=True,files=len(pending),logical_bytes=logical,unique_allocated_bytes=allocated)))
if __name__=='__main__':main()
