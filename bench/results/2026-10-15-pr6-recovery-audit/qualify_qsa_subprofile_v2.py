#!/usr/bin/env python3
"""Main-owned serial patch integration, host fixture and incremental SYCL build."""
import fcntl
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S=A/'qsa-subprofile-v2'
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha,boot_id


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths={k:A/n for k,n in dict(build='canonical-bank-profile-align64-build-v1.json',
            model='canonical-bank-profile-r40960-model-v1.json',comparison='canonical-bank-profile-r40960-model-comparison-v1.json').items()}
        refs={k:json.loads(p.read_text()) for k,p in paths.items()};old=refs['build']
        for r in refs.values():
            require(r['passed'] and r['complete'] and not r['active'],'preceding model/comparison/build closure')
            for c in r.get('commands',[]):
                require(module.closed(c),'prior command closed')
                for identity in c['owners']:
                    now=module.identity(identity['pid'])
                    require(now is None or now['start_ticks']!=identity['start_ticks'],'prior owner present')
        model=refs['model']
        require(not model['fresh_faults'] and model['boot_unchanged'] and not model['driver_dump_exists'] and model['boot_id']==boot_id(),'healthy sameboot model closure')
        require(refs['comparison']['sampled_FP32_values_equal'],'exact placement comparator')
        manifest_path=S/'manifest.json';manifest=json.loads(manifest_path.read_text())
        require(sha(manifest_path)=='4377c1cc3ed18809bfc3769f01c91aef6723590b3ba89bdaa9c6d311338a2414','reviewed proposal manifest')
        require(sha(S/'source.diff')=='dba9d301385f8d166df50ecd3022d32f0cadf95020a7d82d45816ef488bf37ff','reviewed full patch')
        require(all(sha(S/n)==v['sha256'] for n,v in manifest['files'].items()),'immutable proposal files')
        require(all(sha(P/n)==h for n,h in old['source_pins'].items()),'actual old compiled source stable before patch')
        base=json.loads((S/'base-manifest.json').read_text())['source_sha256']
        require(all(sha(P/n)==h for n,h in base.items()),'same actual reviewed byte base')
        changed={str(f.relative_to(S/'proposed')):sha(f) for f in (S/'proposed').rglob('*') if f.is_file()}
        require(set(changed)=={'include/strata/prefill/layermajor_qsa_profile.hpp','src/prefill/prefill.cpp','src/prefill/layer_major.inc'},'only reviewed product changes')
        root=B/'main-qsa-subprofile-build-root-v2';target=A/'qsa-subprofile-build-host-v2.json'
        require(not root.exists() and not target.exists(),'fresh build/host qualification')
        root.mkdir(mode=0o700);build=Path(old['build_path']);exe=Path(old['binary'])
        require(sha(exe)==old['binary_sha256'],'qualified previous native')
        retained=root/'reference-strata-1694';shutil.copy2(exe,retained)
        require(sha(retained)==old['binary_sha256'],'retained reference copy verified')
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],
            references={p.name:sha(p) for p in paths.values()},recipe_sha256=sha(Path(__file__)),
            proposal_manifest_sha256=sha(manifest_path),source_diff_sha256=sha(S/'source.diff'),
            environment=old['environment'],source_pins=dict(old['source_pins']),build_path=str(build),
            compiler=old['compiler'],compiler_sha256=old['compiler_sha256'],canonical_policy_passed=old['canonical_policy_passed'],
            transformations={n:dict(base_sha256=base.get(n),proposed_sha256=h) for n,h in changed.items()},
            retained_reference_binary=dict(path=str(retained),bytes=retained.stat().st_size,sha256=sha(retained),
                consumer='Main same-workload observer overhead/clean baseline if needed;32MiB budget, review after fine-model comparison'),
            superseded_build_products=[dict(path=str(exe),bytes=exe.stat().st_size,before_sha256=sha(exe),replacement=str(retained),reason='Incremental independent bounded QSA observation; retain prior native for current matched reference consumer')],
            performance_qualified=False,quality_qualified=False,restore_qualified=False,physical_262144_qualified=False,
            scope='Main reviewed complete independent three-file source diff; integrate exact patch after closed model, host shim lifetime/count/error fixture, rebuild actual SYCL engine. No model or GPU execution here.')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def host(label,argv,wall=60):
            c,out,err=owner.run(label,argv,env,P,wall=wall,cpu=120,rss_cap=2<<30,text_cap=4<<20,total_cap=1<<30,file_cap=8<<20)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-2000:]);return out,err
        try:
            host('patch-check',['git','apply','--check',str(S/'source.diff')])
            host('patch-apply',['git','apply',str(S/'source.diff')])
            record['source_applied']=True
            require(all(sha(P/n)==h for n,h in changed.items()),'exact reviewed proposed source applied')
            record['source_pins'].update(changed)
            require(sha(P/'include/strata/prefill/layermajor_coarse_profile.hpp')==base['include/strata/prefill/layermajor_coarse_profile.hpp'],'unchanged coarse-v2 formatter')
            cc=Path('/usr/bin/c++').resolve();record['host_compiler']=dict(path=str(cc),sha256=sha(cc))
            fixture=root/'host-fixture'
            host('host-compile',[str(cc),'-std=c++17','-O2','-Wall','-Wextra','-I'+str(S/'fixtures/shim'),'-I'+str(P/'include'),str(S/'fixtures/qsa_subprofile_fixture.cpp'),'-o',str(fixture)])
            out,err=host('host-fixture',[str(fixture)])
            lines=[line for line in err.read_text().splitlines() if line.startswith('strata lm qsa profile: v=1 ')]
            require(not out.read_text() and len(lines)==23 and sum('complete=1 errors=0' in line for line in lines)==2,'expected23 host outcomes with2 full geometries')
            record.update(host_fixture_passed=True,host_fixture_sha256=sha(fixture),host_cases=len(lines),host_complete_geometries=2,host_output_sha256=sha(err),host_output_bytes=err.stat().st_size)
            c,out,err=owner.run('incremental-sycl-build',['cmake','--build',str(build),'--target','strata','--parallel','2'],record['environment'],P,
                wall=1200,cpu=14400,rss_cap=32<<30,text_cap=16<<20,total_cap=1<<30,file_cap=512<<20)
            require(module.closed(c) and c['exit_code']==0,'SYCL rebuild: '+err.read_text()[-3000:]+'\n'+out.read_text()[-3000:])
            require(all(sha(P/n)==h for n,h in record['source_pins'].items()),'compiled source stable')
            record['superseded_build_products'][0]['after_sha256']=sha(exe)
            record.update(binary=str(exe),binary_sha256=sha(exe),binaries={n:sha(build/n) for n in old['binaries']},passed=True,complete=True)
        except BaseException as error:record['error']=repr(error);raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):record.update(passed=False,complete=False)
            persist()
            with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],host_cases=record['host_cases'],binary_sha256=record['binary_sha256'])),flush=True)


if __name__=='__main__':main()
