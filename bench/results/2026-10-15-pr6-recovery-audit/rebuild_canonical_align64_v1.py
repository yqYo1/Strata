#!/usr/bin/env python3
"""Serial incremental rebuild for the recorded pack-alignment rejection."""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha,boot_id


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=dict(build=A/'canonical-bank-profile-build-v1.json',
                   failed=A/'canonical-bank-component-normal-v1.json',
                   normal=A/'canonical-bank-component-native-normal-v1.json',
                   host=A/'coarse-profile-host-v1.json',fix=A/'canonical-align64-fix-v1/pins.json')
        refs={k:json.loads(p.read_text()) for k,p in paths.items()}
        old,failed,normal,host,fix=(refs[k] for k in ('build','failed','normal','host','fix'))
        require(old['passed'] and old['complete'] and not old['active'] and normal['passed'] and
                host['passed'] and host['host_arms']==13,'preceding build/native/host gates')
        require(not failed['passed'] and not failed['active'] and failed['exit_code']==-6 and
                failed['runtime_errors']==['FAIL canonical capture: exact owner size or bounded RAM limit mismatch'],
                'exact original capture rejection')
        for key in ('build','failed','normal','host'):
            r=refs[key]
            for c in r['commands']:
                require(module.closed(c),'prior command owner closed')
                for identity in c['owners']:
                    now=module.identity(identity['pid'])
                    require(now is None or now['start_ticks']!=identity['start_ticks'],'prior owner present')
            if key in ('failed','normal'):
                require(r['boot_id']==boot_id() and r['boot_unchanged'] and not r['fresh_faults'] and
                        not r['driver_dump_exists'],'healthy sameboot component closure')
        require(sha(A/'canonical-align64-fix-v1/source.diff')==fix['source_diff_sha256'],'fix diff pin')
        for n,h in old['source_pins'].items():
            expected=fix['files'][n]['proposed_sha256'] if n in fix['files'] else h
            require(sha(P/n)==expected,'exact bounded source correction '+n)
        require(all(sha(Path(n))==h for n,h in host['source_pins'].items()),'unchanged qualified coarse observer/shim')
        sources=dict(old['source_pins']);sources.update({n:v['proposed_sha256'] for n,v in fix['files'].items()})
        root=B/'main-canonical-bank-profile-align64-build-root-v1';output=A/'canonical-bank-profile-align64-build-v1.json'
        require(not root.exists() and not output.exists(),'fresh correction receipt')
        build=Path(old['build_path']);products=[]
        for n,h in old['binaries'].items():
            f=build/n;require(sha(f)==h,'prior fixture/product identity')
            products.append(dict(path=str(f),bytes=f.stat().st_size,before_sha256=h,
                                 reason='Rebuild after captured pack64 alignment correction; original build and failed component receipts retained. No model/RESTORE capture consumes these initial candidate binaries.'))
        root.mkdir(mode=0o700)
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],
                    environment=old['environment'],source_pins=sources,build_path=str(build),
                    references={p.name:sha(p) for p in paths.values()},recipe_sha256=sha(Path(__file__)),
                    superseded_build_products=products,compiler=old['compiler'],compiler_sha256=old['compiler_sha256'],
                    scope='Correct original captured pack alignment64 separately from internal bank stride256. '
                          'All7 candidate/manual binaries rebuilt; new policy executed. Coarse13arm host gate '
                          'reused only with unchanged actual observer/shim source. No GPU/model execution.',
                    performance_qualified=False,quality_qualified=False,restore_qualified=False,physical_262144_qualified=False)
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        try:
            c,out,err=owner.run('rebuild-align64',['cmake','--build',str(build),'--target',*old['binaries'],
                                                '--parallel','2'],record['environment'],P,
                                wall=1200,cpu=14400,rss_cap=32<<30,text_cap=16<<20,total_cap=1<<30,file_cap=512<<20)
            require(module.closed(c) and c['exit_code']==0,'correction rebuild: '+err.read_text()[-3000:]+'\n'+out.read_text()[-3000:])
            hostenv=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
            c,out,err=owner.run('canonical-align64-policy',[str(build/'canonical_bank_policy_test')],hostenv,P,
                                wall=30,cpu=60,rss_cap=1<<30,text_cap=64<<10,total_cap=1<<30,file_cap=64<<10)
            require(module.closed(c) and c['exit_code']==0 and out.read_text().strip()==
                    'HOST_ONLY canonical policy/HC/PLE classification; no allocation or model pass' and not err.read_text(),
                    'actual64/256 alignment policy')
            record['canonical_policy_passed']=True
            require(all(sha(P/n)==h for n,h in sources.items()),'corrected source stable')
            for item in products:item['after_sha256']=sha(Path(item['path']))
            record.update(binary=str(build/'strata'),binary_sha256=sha(build/'strata'),
                          binaries={n:sha(build/n) for n in old['binaries']},passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error);raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):record.update(passed=False,complete=False)
            persist()
            with output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],GPU_executed=False,binary_sha256=record.get('binary_sha256'))))


if __name__=='__main__':main()
