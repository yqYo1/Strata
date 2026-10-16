#!/usr/bin/env python3
"""Run exactly two fresh diagnostic processes, with independent owned closure."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

A=Path(__file__).resolve().parent
P=A.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=A/'qsa-bf16-prefix-tilepack-profile-pair-v1.json'
    assert not out.exists()
    child=A/'run_qsa_bf16_prefix_tilepack_profile_census_v1.py'
    sensor=A/'profile_sysfs_observation_v1.py'
    prior=A/'own-qsa-bf16-prefix-tilepack-clean-bf16-prefix-tilepack-r2-v2.json'
    pins={str(f):sha(f) for f in (Path(__file__),child,sensor)}
    order=['bf16-prefix-optimized','bf16-prefix-tilepack']
    record=dict(schema='strata-tilepack-profile-pair-v1',passed=False,complete=False,
        source_pins=pins,order=order,completed=[],active_controller=None,
        performance_adopted=False,scope='Independent paired diagnostic runs; not clean timing. No parallel build/test/cleanup.')
    def save():
        out.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for arm in order:
            assert all(sha(f)==v for f,v in pins.items())
            target=A/f'own-qsa-bf16-prefix-tilepack-profile-census-{arm}-r0-v1.json'
            assert not target.exists()
            argv=[sys.executable,str(child),'profile',arm,'0','--prior',str(prior),'--prior-sha256',sha(prior)]
            print(json.dumps(dict(start=arm)),flush=True)
            proc=subprocess.Popen(argv,cwd=P)
            record['active_controller']=dict(pid=proc.pid,argv=argv);save()
            code=proc.wait();record['active_controller']=None
            item=dict(arm=arm,controller_exit=code,path=str(target));record['completed'].append(item);save()
            assert code==0 and target.is_file()
            result=json.loads(target.read_text())
            assert result['passed'] and result['complete'] and result['ownership_closed'] and not result['fresh_faults']
            item['sha256']=sha(target);save();prior=target
        record.update(passed=True,complete=True)
    except BaseException as error:
        record['error']=repr(error)
    finally:
        record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print(json.dumps(dict(passed=record['passed'],complete=record['complete'],receipt=str(out))),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
