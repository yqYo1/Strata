#!/usr/bin/env python3
"""Main integrates the previously reviewed own NLL proposal with exact CMake rebase."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S=A/'own-dense-prefill-nll-v3'
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha,boot_id


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=json.loads((A/'own-host-transfer-component-v1.json').read_text())
        require(prior['passed'] and prior['complete'] and not prior['active'] and not prior['fresh_faults'],'closed transfer comparison')
        for c in prior['commands']:
            require(module.closed(c) and c['exit_code']==0,'prior owners retired')
            for old in c['owners']:
                now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'prior owned process alive')
        require(all(sha(P/n)==h for n,h in prior['source_pins'].items()),'exact preceding sources')
        require(sha(S/'source.diff')=='fca81a0f6aac0ae805b438ab5368cf5f91120179e33c2df5fd100d8679f20689','reviewed own patch')
        require(sha(S/'manifest.json')=='e917011fdb210265df32125ed35fec8d6beb46bc6e1f1f472dd7b24d94fc3464','reviewed own manifest')
        require(sha(S/'provenance.json')=='4acefc64ac2d2bab14b7a8935dd3e325f1d7f96cc3d3aff5760f619845ac7f3b','own source provenance')
        manifest=json.loads((S/'manifest.json').read_text())
        extras=[
            'add_executable(sycl_host_transfer_probe EXCLUDE_FROM_ALL tools/sycl/host_transfer_probe.cpp)\ntarget_link_libraries(sycl_host_transfer_probe PRIVATE strata_sycl_runtime)\n',
            '  add_executable(sycl_gemm_device_probe EXCLUDE_FROM_ALL tools/sycl/gemm_device_probe.cpp)\n  target_link_libraries(sycl_gemm_device_probe PRIVATE strata_prefill)\n  target_compile_options(sycl_gemm_device_probe PRIVATE -fno-fast-math -ffp-contract=off)\n']
        def without_extras(text):
            for extra in extras:
                require(text.count(extra)==1,'one exact current target block preserved');text=text.replace(extra,'')
            return text
        import hashlib
        for v in manifest['changes']:
            n=v['path'];require(sha(S/'files'/n)==v['proposed_sha256'],'proposal file unchanged')
            if n=='cmake/sycl_backend.cmake':
                require(hashlib.sha256(without_extras((P/n).read_text()).encode()).hexdigest()==v['base_sha256'],'CMake differs only by two independent Main diagnostic targets')
            else:require(not (P/n).exists() if v['base_sha256'] is None else sha(P/n)==v['base_sha256'],'exact touched base')
        root=B/'main-own-dense-prefill-nll-build-root-v3';target=A/'own-dense-prefill-nll-build-v3.json'
        require(not root.exists() and not target.exists(),'fresh integration build')
        root.mkdir(mode=0o700)
        old=json.loads((A/'own-prefill-host-profile-build-v3.json').read_text());build=Path(old['build_path'])
        require(sha(build/'strata')==prior['native_binary_sha256_before'] and sha(old['compiler'])==old['compiler_sha256'],'native/toolchain pin')
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],boot_id=boot_id(),
            recipe_sha256=sha(Path(__file__)),source_diff_sha256=sha(S/'source.diff'),proposal_manifest_sha256=sha(S/'manifest.json'),provenance_sha256=sha(S/'provenance.json'),
            source_base_commit='7d2d344f',source_pins=dict(prior['source_pins']),build_path=str(build),environment=old['environment'],
            compiler=old['compiler'],compiler_sha256=old['compiler_sha256'],native_binary_sha256_before=sha(build/'strata'),
            rebase='Exact two Main diagnostic target blocks preserved; rest CMake and generator match immutable own proposal base.',
            model_quality_qualified=False,FP16_adopted=False,physical_262144_qualified=False,scope='Own default-OFF dense full-vocabulary NLL diagnostic integration/build/host fixture only; no model/GPU qualification')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        hostenv=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def run(label,argv,env=hostenv,wall=60,cpu=120,rss=2<<30,file_cap=8<<20):
            c,out,err=owner.run(label,argv,env,P,wall=wall,cpu=cpu,rss_cap=rss,text_cap=64<<20,total_cap=1<<30,file_cap=file_cap)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-6000:]);return out.read_text(),err.read_text()
        try:
            for name in ('src/program/generate.cpp','cmake/sycl_backend.cmake'):
                out,_=run('base-membership-'+str(len(record.get('base_commit_membership',{}))),['git','show','7d2d344f:'+name])
                require(hashlib.sha256(out.encode()).hexdigest()==sha(P/name),'actual current commit touched membership')
                record.setdefault('base_commit_membership',{})[name]=sha(P/name)
            run('patch-check',['git','apply','--check',str(S/'source.diff')])
            run('patch-apply',['git','apply',str(S/'source.diff')])
            for v in manifest['changes']:
                n=v['path']
                if n=='cmake/sycl_backend.cmake':require(hashlib.sha256(without_extras((P/n).read_text()).encode()).hexdigest()==v['proposed_sha256'],'rebased CMake exact proposal plus independent targets')
                else:require(sha(P/n)==v['proposed_sha256'],'exact applied source')
                record['source_pins'][n]=sha(P/n)
            require(all(sha(P/n)==h for n,h in record['source_pins'].items()),'only reviewed source changes')
            record['source_applied']=True;persist()
            out,err=run('build-nll',['cmake','--build',str(build),'--target','strata','dense_prefill_nll_host_test','--parallel','2'],old['environment'],wall=1500,cpu=18000,rss=32<<30,file_cap=512<<20)
            record.update(binary=str(build/'strata'),binary_sha256=sha(build/'strata'),host_binary_sha256=sha(build/'dense_prefill_nll_host_test'))
            require(record['binary_sha256']!=record['native_binary_sha256_before'],'new diagnostic linked')
            out,err=run('host-nll',[str(build/'dense_prefill_nll_host_test')],wall=60,cpu=120,rss=2<<30)
            record['host_fixture_stdout']=out;record['host_fixture_stderr']=err
            require('PASS' in out and not err,'actual host oracle passed')
            require(all(sha(P/n)==h for n,h in record['source_pins'].items()),'build source frozen')
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error);raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):record.update(passed=False,complete=False)
            persist()
            with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],binary_sha256=record['binary_sha256'],host=record['host_fixture_stdout'],GPU_executed=False)),flush=True)


if __name__=='__main__':main()
