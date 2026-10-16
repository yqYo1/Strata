#!/usr/bin/env python3
"""Retire closed successful duplicates and resolved CPU build output only."""
import fcntl
import importlib.util
import json
import subprocess
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,sha

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target=D/'retention-v5-manifest.json';assert not target.exists()
        roots={
            'main-primary-boundary-service-build-root-v1':'boundary-service-build-v1.json',
            'main-primary-boundary-audit-small-root-v1':'boundary-audit-small-v1.json',
            'main-primary-boundary-audit-long-fixed-root-v1':'boundary-audit-long-fixed-v1.json',
            'main-primary-decode-format-service-normal-root-v1':'decode-format-service-normal-v1.json',
            'main-primary-existing-iq-single-exact-root-v1':'existing-iq-single-exact-v1.json',
            'main-primary-iq3s-rowpair-probe-root-v1':'iq3s-rowpair-probe-v1.json',
            'main-primary-iq3s-rowpair-probe-root-v2':'iq3s-rowpair-probe-v2.json'}
        extras=['boundary-audit-long-pair-v1.json','boundary-audit-long-pair.ja.md',
            'decode-format-service-summary-v1.json','decode-format-service-extra-v1.json','decode-format-service.ja.md',
            'iq3s-rowpair-probe-builds-v1.json','iq3s-rowpair-probe.ja.md']
        evidence={};records={}
        for name in list(roots.values())+extras:
            path=D/name;relative=str(path.relative_to(P))
            assert subprocess.check_output(['git','show','HEAD:'+relative],cwd=P)==path.read_bytes()
            evidence[relative]=sha(path)
        assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
        spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for root,name in roots.items():
            r=json.loads((D/name).read_text());records[root]=r
            assert not r['active']
            if root.endswith('rowpair-probe-root-v1'):
                assert not r['passed'] and not r['complete'] # resolved missing-header build failure
            else:assert r['passed'] and r['complete']
            if 'fresh_faults' in r:assert not r['fresh_faults'] and not r['driver_dump_exists'] and r['boot_unchanged']
            for c in r['commands']:
                assert m.closed(c)
                for i in c['owners']:
                    current=m.identity(i['pid'])
                    assert current is None or current['start_ticks']!=i['start_ticks'] or current['state']=='Z'
        entries={}
        def add(path,expected,reason,replacement):
            assert path.is_file() and not path.is_symlink() and sha(path)==expected
            st=path.stat();entries[str(path)]=dict(path=str(path),bytes=st.st_size,sha256=expected,
                device=st.st_dev,inode=st.st_ino,links=st.st_nlink,allocated_bytes=st.st_blocks*512,
                reason=reason,replacement=replacement)
        for root,r in records.items():
            for c in r['commands']:
                for n,info in c['logs'].items():
                    add(B/root/'owned'/n,info['sha256'],'Closed full process log; useful evidence committed.',roots[root])
            add(B/root/'record.json',sha(D/roots[root]),'Working receipt duplicates committed authoritative bytes.',roots[root])
            # These roots contain only retired CPU build/test output, not the product build.
            if root in ['main-primary-boundary-service-build-root-v1','main-primary-existing-iq-single-exact-root-v1',
                        'main-primary-iq3s-rowpair-probe-root-v1','main-primary-iq3s-rowpair-probe-root-v2']:
                for f in (B/root).rglob('*'):
                    if f.is_file() and not f.is_symlink() and str(f) not in entries:
                        add(f,sha(f),'CPU fixtures/objects/probe binary no longer consumed; source, exact build commands/results retained.',
                            roots[root]+'; iq3s-rowpair-probe-builds-v1.json where applicable')
        for job,reference_name in [('small','selected-prefill-small-v1.json'),('long-fixed','selected-prefill-long-v1.json')]:
            root='main-primary-boundary-audit-'+job+'-root-v1';reference=json.loads((D/reference_name).read_text())
            for arm in records[root]['arms']:
                old=next(a for a in reference['arms'] if a['dtype']==arm['dtype'])
                assert arm['numerical_passed'] and arm['same_dtype_generated_ids_equal']
                expected={'first-logits.bin':old['first_logits_sha256'],'selected-logits.bin':old['selected_logits_sha256'],
                    'selected-logits.bin.meta':old['metadata_sha256'],'selected-logits.bin.meta.partial':old['metadata_sha256']}
                for fixture in arm['fixtures']:
                    f=Path(fixture['path']);assert fixture['sha256']==expected[f.name]
                    add(f,fixture['sha256'],'Byte-identical successful candidate; canonical old control retained.',reference_name)
                if job=='small' and arm['audit_on']:
                    for key in ['text','raw']:
                        info=arm['boundary'];f=Path(info[key+'_path'])
                        add(f,info[key+'_sha256'],'Small boundary coverage/copy equality qualified; no current raw consumer.',roots[root])
        result=dict(active=True,complete=False,recipe_sha256=sha(Path(__file__)),evidence=evidence,
            entries=list(entries.values()),deleted=[],original_statuses_unchanged=True,
            preserved='Both long per-layer raw captures and auto diagnostic text/selected/first, original non-audited auto failure and old canonical controls.',
            logical_bytes=sum(x['bytes'] for x in entries.values()),
            allocation_scope='Unique removed inode st_blocks only; not actual freed filesystem bytes (shared extents/snapshots).')
        def persist():target.write_text(json.dumps(result,indent=2)+'\n')
        persist()
        try:
            for row in result['entries']:
                f=Path(row['path']);st=f.stat()
                assert (st.st_dev,st.st_ino,st.st_size)==(row['device'],row['inode'],row['bytes']) and sha(f)==row['sha256']
                f.unlink();result['deleted'].append(row['path']);persist()
            unique={(x['device'],x['inode']):x for x in result['entries']}
            result['unique_inode_allocated_bytes']=sum(x['allocated_bytes'] for x in unique.values());result['complete']=True
        finally:result['active']=False;persist()
        print(json.dumps(dict(complete=result['complete'],files=len(result['deleted']),logical_bytes=result['logical_bytes'],
            unique_inode_allocated_bytes=result.get('unique_inode_allocated_bytes'))),flush=True)

if __name__=='__main__':main()
