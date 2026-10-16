#!/usr/bin/env python3
"""Wait read-only for the observed foreign benchmark, then attempt one owned pair."""
import datetime
import hashlib
import importlib.util
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
P=A.parents[2]
PID=3431768
START_TICKS=52544839
WAIT_SECONDS=7200


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=A/'qsa-tilepack-profile-wait-v1.json'
    assert not out.exists()
    child=A/'run_qsa_bf16_prefix_tilepack_profile_pair_v2.py'
    pair=A/'qsa-bf16-prefix-tilepack-profile-pair-v2.json'
    assert not pair.exists()
    owner_path=A/'direct_owner_transient_artifacts_v1.py'
    assert sha(owner_path)=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec=importlib.util.spec_from_file_location('wait_identity_owner',owner_path)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    files=[Path(__file__),child,A/'run_qsa_bf16_prefix_tilepack_profile_census_v2.py',
           A/'profile_sysfs_observation_v1.py']
    pins={str(f):sha(f) for f in files}
    record=dict(schema='strata-read-only-foreign-owner-wait-v1',active=True,passed=False,complete=False,
        status='waiting',expected_foreign_identity=dict(pid=PID,start_ticks=START_TICKS),
        wait_budget_seconds=WAIT_SECONDS,source_pins=pins,observations=[],active_controller=None,
        foreign_process_signals_sent=False,GPU_executed=False,
        policy='Use a pidfd only for exit observation. Never reap, signal, stop or change the foreign benchmark. After exit, run one fresh diagnostic pair; its independent model-owner, lock, runtime, source and fault guards remain mandatory. No retry.')
    def save():
        record['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        out.write_text(json.dumps(record,indent=2)+'\n')
    save();fd=None
    try:
        identity=owner.identity(PID)
        if identity and identity['start_ticks']==START_TICKS and identity['state']!='Z':
            try:
                fd=os.pidfd_open(PID)
            except ProcessLookupError:
                record['foreign_exit_evidence']='pidfd_open ESRCH after initial identity observation'
            if fd is not None:
                after=owner.identity(PID)
                if after and after['start_ticks']==START_TICKS and after['state']!='Z':
                    watch=select.poll();watch.register(fd,select.POLLIN)
                    deadline=time.monotonic()+WAIT_SECONDS
                    while not watch.poll(0):
                        now=time.monotonic();current=owner.identity(PID)
                        record['observations'].append(dict(monotonic=now,
                            same_identity=current is not None and current['start_ticks']==START_TICKS,
                            state=current['state'] if current else None))
                        save()
                        if now>=deadline:
                            record['status']='wait_budget_expired';save();return 3
                        print(json.dumps(dict(status='waiting_for_foreign_benchmark',pid=PID,
                            remaining_seconds=round(deadline-now))),flush=True)
                        watch.poll(min(30000,max(1,int((deadline-now)*1000))))
                    record['foreign_exit_evidence']='pidfd POLLIN for exact observed identity'
                else:
                    record['foreign_exit_evidence']='original PID/start identity no longer live after pidfd pin'
        else:
            record['foreign_exit_evidence']='original PID/start identity absent, replaced or zombie'
        if fd is not None:
            os.close(fd);fd=None
        record['status']='foreign_exit_observed';save()
        assert all(sha(f)==v for f,v in pins.items())
        argv=[sys.executable,'-B',str(child)]
        proc=subprocess.Popen(argv,cwd=P)
        record['active_controller']=dict(pid=proc.pid,argv=argv)
        record['status']='paired_profile_running';save()
        code=proc.wait();record['active_controller']=None
        record['controller_exit_code']=code
        result=json.loads(pair.read_text()) if pair.exists() else None
        if result is not None:
            record['pair_receipt']=dict(path=str(pair),sha256=sha(pair))
            for item in result['completed']:
                f=Path(item['path'])
                if f.is_file():
                    r=json.loads(f.read_text())
                    record['GPU_executed']|=bool(r.get('GPU_executed'))
        assert code==0 and result and result['passed'] and result['complete'] and result['active_controller'] is None
        record.update(passed=True,complete=True,status='passed')
        return 0
    except BaseException as error:
        record.update(status='failed',error=repr(error));return 1
    finally:
        if fd is not None:os.close(fd)
        record['active']=record['active_controller'] is not None
        save()
        print(json.dumps(dict(passed=record['passed'],complete=record['complete'],
            status=record['status'],receipt=str(out))),flush=True)


if __name__=='__main__':
    sys.exit(main())
