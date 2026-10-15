#!/usr/bin/env python3
"""Retire two successful timelines after their exact compact consumers close."""
import fcntl
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha


def main():
    target=A/'prefill-window-raw-retention-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not target.exists(),'fresh deletion manifest')
        host=A/'prefill-window-interval-summary-host-v2.json'
        gap=A/'prefill-window-gap-api-summary-v1.json'
        copy=A/'prefill-copy-accounting-main-v1.json'
        for p in [host,gap,copy]:require(json.loads(p.read_text())['passed'],'closed analysis consumer')
        owner=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        delete=[];retain=[]
        for mode,receipt_name in [('component','prefill-window-profile-component-v3-compact.json'),
                                  ('long','prefill-window-profile-long-v4.json')]:
            receipt=A/receipt_name;d=json.loads(receipt.read_text())
            require(d['passed'] and d['complete'] and not d['active'] and not d['fresh_faults']
                    and not d['driver_dump_exists'] and d['boot_unchanged'],'closed successful collection')
            for c in d['commands']+d['control_commands']:
                require(owner.closed(c),'owned command not closed')
                for old in c['owners']:
                    now=owner.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'owned process still present')
            summary=A/f'prefill-window-{mode}-interval-summary-v2.json'
            s=json.loads(summary.read_text())
            require(s['passed'] and s['receipt_sha256']==sha(receipt),'exact summary receipt')
            p=Path(s['trace']['path'])
            require(p.is_relative_to(B) and not p.is_symlink() and sha(p)==s['trace']['sha256'],
                    'exact owned raw timeline')
            st=p.stat()
            delete.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,
                sha256=s['trace']['sha256'],device=st.st_dev,inode=st.st_ino,deleted=False,
                reason='Complete event/SHA accounting, independent timing-table crosscheck and interval consumers finished; compact evidence replaces successful raw timeline.',
                replacement=str(summary),replacement_sha256=sha(summary)))
            for f in d['provider_files']:
                if Path(f['path']).name in ('device_timing.txt','host_timing.txt') and Path(f['path']).exists():
                    retain.append(dict(path=f['path'],bytes=f['bytes'],sha256=f['sha256'],
                        consumer='Independent timing table fixture for interval-summary qualification',
                        next_review='When the profiler harness/reference is replaced; not a complete event timeline.'))
        original=json.loads((A/'prefill-window-profile-component-v3-compact.json').read_text())['original_receipt']
        retain.append(dict(path=original['retained_path'],bytes=original['bytes'],sha256=original['sha256'],
            consumer='test_prefill_profile_ready_v4.py actual operation-category conservation fixture',
            byte_budget=48<<20,next_review='Retire with replacement of the v4 host-parser/category fixture.'))
        for mode in ('component','long'):
            p=A/f'prefill-window-{mode}-interval-summary-v1.json'
            retain.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
                consumer='qualify_prefill_interval_summary_v2.py original-v1 aggregate conservation fixture',
                byte_budget=6<<20,next_review='Retire when this cross-version analyzer fixture is replaced; v1 long remains output-budget failure.'))
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,
            recipe_sha256=sha(Path(__file__)),evidence={str(p):sha(p) for p in [host,gap,copy]},
            deleted=delete,retained=retain,original_run_status_unchanged=True,
            logical_bytes_to_retire=sum(x['bytes'] for x in delete))
        def persist():target.write_text(json.dumps(record,indent=2)+'\n')
        persist()
        try:
            for item in delete:
                p=Path(item['path']);st=p.stat()
                require((st.st_dev,st.st_ino,st.st_size)==(item['device'],item['inode'],item['bytes']),
                        'timeline identity changed')
                p.unlink();item['deleted']=True;persist()
            record.update(passed=True,complete=True,
                logical_bytes_retired=sum(x['bytes'] for x in delete if x['deleted']),
                allocated_bytes_retired=sum(x['allocated_bytes'] for x in delete if x['deleted']))
        finally:
            record['active']=False;persist()
        print(json.dumps(dict(passed=True,raw_timelines_retired=len(delete),
            logical_bytes_retired=record['logical_bytes_retired'])))


if __name__=='__main__':main()
