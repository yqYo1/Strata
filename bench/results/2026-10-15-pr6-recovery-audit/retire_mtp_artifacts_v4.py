#!/usr/bin/env python3
"""Retire closed successful duplicates; preserve the unresolved long divergence."""
import fcntl
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=D/'retention-v4-manifest.json'
        assert not manifest.exists()
        roots={
            'main-primary-mtp-expert-release-build-root-v1':'mtp-expert-release-build-v1.json',
            'main-primary-mtp-expert-release-small-root-v1':'mtp-expert-release-small-v1.json',
            'main-primary-mtp-expert-release-long-fixed-root-v1':'mtp-expert-release-long-fixed-v1.json',
            'main-primary-decode-ledger-fp32-fixed-normal-root-v1':'decode-ledger-fp32-fixed-normal-v1.json'}
        evidence={}
        records={}
        for root,name in roots.items():
            path=D/name;relative=str(path.relative_to(P));raw=path.read_bytes()
            assert subprocess.check_output(['git','show','HEAD:'+relative],cwd=P)==raw
            record=json.loads(raw)
            assert record['passed'] and record['complete'] and not record['active']
            assert all(c['observation_complete'] and c['direct_child_reaped'] and c['session_empty'] and
                       c['exit_code']==0 and not c['cleanup'] and not c.get('errors') for c in record['commands'])
            if 'fresh_faults' in record:
                assert not record['fresh_faults'] and record['boot_unchanged'] and not record['driver_dump_exists']
            evidence[relative]=sha(path);records[root]=record
        for name in ['decode-ledger-final-summary-v1.json','decode-ledger-final.ja.md','decode-ledger-extra-diagnostics-v1.json']:
            path=D/name;relative=str(path.relative_to(P))
            assert subprocess.check_output(['git','show','HEAD:'+relative],cwd=P)==path.read_bytes()
            evidence[relative]=sha(path)
        assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
        spec=importlib.util.spec_from_file_location('owner',OWNER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for record in records.values():
            for command in record['commands']:
                for identity in command['owners']:
                    current=module.identity(identity['pid'])
                    assert current is None or current['start_ticks']!=identity['start_ticks'] or current['state']=='Z'
        entries={}
        def add(path,expected,reason,replacement):
            path=Path(path);assert path.is_file() and not path.is_symlink()
            assert sha(path)==expected
            st=path.stat()
            entries[str(path)]=dict(path=str(path),bytes=st.st_size,sha256=expected,
                device=st.st_dev,inode=st.st_ino,links=st.st_nlink,allocated_bytes=st.st_blocks*512,
                reason=reason,replacement=replacement)
        for root,record in records.items():
            for command in record['commands']:
                for filename,info in command['logs'].items():
                    add(B/root/'owned'/filename,info['sha256'],
                        'Successful full log: progress/counters/ownership/faults and hashes extracted and committed.',roots[root])
            add(B/root/'record.json',sha(D/roots[root]),'Duplicate working receipt; authoritative identical bytes are committed.',roots[root])
            # Only caches private to these closed processes; never global caches.
            for directory in [B/root/'cache',B/root/'fp32/cache',B/root/'fp16/cache']:
                if not directory.exists():continue
                for path in directory.rglob('*'):
                    if path.is_file() and not path.is_symlink():
                        assert path.resolve().is_relative_to((B/root).resolve())
                        add(path,sha(path),'Private successful-run compilation cache, no current consumer.','Recorded binary/source/toolchain identity and reproducible commands')
        for root,reference_name in [
            ('main-primary-mtp-expert-release-small-root-v1','selected-prefill-small-v1.json'),
            ('main-primary-mtp-expert-release-long-fixed-root-v1','selected-prefill-long-v1.json')]:
            reference=json.loads((D/reference_name).read_text())
            for arm in records[root]['arms']:
                old=next(a for a in reference['arms'] if a['dtype']==arm['dtype'])
                assert arm['same_dtype_first_logits_equal'] and arm['same_dtype_selected_logits_equal'] and arm['same_dtype_generated_ids_equal']
                expected={'first-logits.bin':old['first_logits_sha256'],'selected-logits.bin':old['selected_logits_sha256'],
                          'selected-logits.bin.meta':old['metadata_sha256'],'selected-logits.bin.meta.partial':old['metadata_sha256']}
                for fixture in arm['fixtures']:
                    path=Path(fixture['path']);assert fixture['sha256']==expected[path.name]
                    add(path,fixture['sha256'],'Byte-identical successful candidate capture; canonical old control remains.',reference_name)
        summary=json.loads((D/'decode-ledger-final-summary-v1.json').read_text())
        assert summary['first_extraction_matches_final'] and summary['all_owned_processes_closed']
        add(D/'decode-ledger-first-request-v1.json',summary['first_extraction_sha256'],
            'Preliminary duplicate transcript; exact first block verified against final complete receipt.','decode-ledger-final-summary-v1.json and decode-ledger-fp32-fixed-normal-v1.json')
        path=D/'decode-ledger-first-request.ja.md'
        add(path,sha(path),'Preliminary prose superseded by complete three-request report.','decode-ledger-final.ja.md')
        r=dict(active=True,complete=False,recipe_sha256=sha(Path(__file__)),evidence=evidence,
               entries=list(entries.values()),deleted=[],original_measurement_statuses_unchanged=True,
               retained_unresolved='main-primary-mtp-expert-release-long-root-v1: selected/first-logit divergence fixtures retained for boundary audit',
               logical_bytes=sum(x['bytes'] for x in entries.values()),
               allocation_scope='Unique removed inode st_blocks; not causal available-space recovery, compressed/shared extents and snapshots may retain storage.')
        def persist():manifest.write_text(json.dumps(r,indent=2)+'\n')
        persist()
        try:
            for entry in r['entries']:
                path=Path(entry['path']);assert sha(path)==entry['sha256']
                st=path.stat();assert (st.st_dev,st.st_ino,st.st_size)==(entry['device'],entry['inode'],entry['bytes'])
                path.unlink();r['deleted'].append(entry['path']);persist()
            unique={(x['device'],x['inode']):x for x in r['entries']}
            r['unique_inode_allocated_bytes']=sum(x['allocated_bytes'] for x in unique.values())
            r['complete']=True
        finally:
            r['active']=False;persist()
        print(json.dumps(dict(complete=r['complete'],files=len(r['deleted']),logical_bytes=r['logical_bytes'],
                              unique_inode_allocated_bytes=r.get('unique_inode_allocated_bytes'))),flush=True)


if __name__=='__main__':
    main()
