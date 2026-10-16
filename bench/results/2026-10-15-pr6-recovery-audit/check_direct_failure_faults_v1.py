#!/usr/bin/env python3
"""Use a fresh bounded owner after the original diagnostic exhausted its log cap."""
import fcntl
import importlib.util
import json
from check_gemm_orientation import P, B, D, OWNER, DEVICE, FAULT, sha
from pathlib import Path


def main():
    old_path=D/'prompt-direct-small-v1.json';old=json.loads(old_path.read_text())
    assert not old['active'] and not old['passed'] and not old['complete']
    assert old['commands'][1]['exit_code']==-25 and old['commands'][1]['session_empty']
    assert all(c['direct_child_reaped'] and c['session_empty'] for c in old['commands'])
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=B/'main-primary-prompt-direct-faultcheck-root-v1';assert not out.exists();out.mkdir(mode=0o700)
        r=dict(active=True,passed=False,complete=False,commands=[],GPU_executed=False,original_status_unchanged=True,
            original_receipt_sha256=sha(old_path),recipe_sha256=sha(Path(__file__)))
        def persist():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=m.Owner(out/'owned',persist);r['commands']=owner.commands;persist()
        env=json.loads((D/'kv-copy-groups-build-v2.json').read_text())['environment']
        try:
            e,o,s=owner.run('journal-after-cap',['journalctl','-k','-b','--after-cursor',old['journal_before_cursor'],'--no-pager','-o','json'],env,P,wall=30,text_cap=1<<20,total_cap=8<<20,file_cap=1<<20)
            assert m.closed(e) and e['exit_code']==0,(e.get('error'),s.read_text())
            r['fresh_faults']=[v for v in [json.loads(x) for x in o.read_text().splitlines() if x.strip()] if FAULT.search(str(v.get('MESSAGE','')))]
            r['driver_dump_exists']=(DEVICE/'devcoredump').exists()
            r['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
            r['driver_unchanged']=(DEVICE/'driver').resolve().name=='xe'
            assert not r['fresh_faults'] and not r['driver_dump_exists'] and r['boot_unchanged'] and r['driver_unchanged']
            r.update(passed=True,complete=True)
        except BaseException as error:r['error']=repr(error);raise
        finally:
            r['active']=False;persist();(D/'prompt-direct-small-faultcheck-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],fresh_faults=len(r['fresh_faults']))))


if __name__=='__main__':main()
