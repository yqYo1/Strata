#!/usr/bin/env python3
"""Main-owned integration/build only; all GPU tests are separate serial jobs."""
import fcntl
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S=A/'own-prompt-k-cellmajor-v2'
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha,boot_id


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths={k:A/n for k,n in dict(build='qsa-subprofile-build-host-v2.json',model='qsa-subprofile-model-v2.json',comparison='qsa-subprofile-model-comparison-v2.json').items()}
        refs={k:json.loads(p.read_text()) for k,p in paths.items()};old=refs['build']
        for r in refs.values():
            require(r['passed'] and r['complete'] and not r['active'],'preceding qualification closed')
            for c in r.get('commands',[]):
                require(module.closed(c),'prior command closed')
                for identity in c['owners']:
                    now=module.identity(identity['pid']);require(now is None or now['start_ticks']!=identity['start_ticks'],'prior owner absent')
        require(not refs['model']['fresh_faults'] and refs['model']['boot_unchanged'] and not refs['model']['driver_dump_exists'] and refs['model']['boot_id']==boot_id(),'healthy sameboot model')
        require(refs['comparison']['sampled_FP32_values_equal'],'actual selected parity')
        manifest=json.loads((S/'manifest.json').read_text())
        require(manifest['base_commit']=='3d873f9ca050db105122b8b18c8a8e1275aecdde' and manifest['diff_sha256']==sha(S/'source.diff')=='578353f48ca7d39591b70a495b0fbe3d8d06661f6e7647a3fc9ce932322637f2','reviewed exact diff')
        require(all(sha(P/n)==h for n,h in old['source_pins'].items()),'actual compiled base stable')
        changed={n:v['proposal_sha256'] for n,v in manifest['files'].items()}
        require(set(changed)=={'src/kernels/sycl/prompt_attention.cpp','tests/sycl/prompt_k_cellmajor.cpp','cmake/sycl_backend.cmake'},'exact reviewed three-file scope')
        for n,v in manifest['files'].items():
            require(sha(S/'files'/n)==v['proposal_sha256'],'reviewed proposed bytes')
            require((not (P/n).exists()) if v['base_sha256'] is None else sha(P/n)==v['base_sha256'],'immutable patch base')
        root=B/'main-own-prompt-k-cellmajor-build-root-v2';target=A/'own-prompt-k-cellmajor-build-v2.json'
        require(not root.exists() and not target.exists(),'fresh build')
        root.mkdir(mode=0o700);build=Path(old['build_path']);exe=Path(old['binary'])
        require(sha(exe)==old['binary_sha256'],'previous native identity')
        retained=root/'reference-strata-f01e';shutil.copy2(exe,retained);require(sha(retained)==old['binary_sha256'],'reference copy verified')
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],references={p.name:sha(p) for p in paths.values()},
            recipe_sha256=sha(Path(__file__)),source_diff_sha256=sha(S/'source.diff'),proposal_manifest_sha256=sha(S/'manifest.json'),
            transformations=manifest['files'],environment=old['environment'],source_pins=dict(old['source_pins']),build_path=str(build),
            compiler=old['compiler'],compiler_sha256=old['compiler_sha256'],canonical_policy_passed=old['canonical_policy_passed'],
            retained_reference_binary=dict(path=str(retained),bytes=retained.stat().st_size,sha256=sha(retained),consumer='Main old/native defaultOFF regression/whole-prefill comparator;32MiB budget, review after candidate rejection/adoption'),
            component_qualified=False,performance_qualified=False,quality_qualified=False,restore_qualified=False,physical_262144_qualified=False,
            scope='Independent own-source Mode1 Direct1 K-gather lane remap, defaultOFF; full Main patch review and actual SYCL build only. No Xe source or GPU execution.')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        try:
            for label,argv in [('patch-check',['git','apply','--check',str(S/'source.diff')]),('patch-apply',['git','apply',str(S/'source.diff')])]:
                c,out,err=owner.run(label,argv,env,P,wall=60,cpu=60,rss_cap=2<<30,text_cap=1<<20,total_cap=1<<30,file_cap=1<<20)
                require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-2000:])
            require(all(sha(P/n)==h for n,h in changed.items()),'exact reviewed bytes applied')
            record['source_pins'].update(changed);record['source_applied']=True
            c,out,err=owner.run('incremental-sycl-build',['cmake','--build',str(build),'--target','strata','sycl_prompt_k_cellmajor_test','--parallel','2'],record['environment'],P,
                wall=1200,cpu=14400,rss_cap=32<<30,text_cap=16<<20,total_cap=1<<30,file_cap=512<<20)
            require(module.closed(c) and c['exit_code']==0,'SYCL build: '+err.read_text()[-3000:]+'\n'+out.read_text()[-3000:])
            require(all(sha(P/n)==h for n,h in record['source_pins'].items()),'compiled source stable')
            fixture=build/'sycl_prompt_k_cellmajor_test'
            record.update(passed=True,complete=True,binary=str(exe),binary_sha256=sha(exe),component_binary=str(fixture),component_binary_sha256=sha(fixture))
        except BaseException as error:record['error']=repr(error);raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):record.update(passed=False,complete=False)
            persist()
            with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],binary_sha256=record['binary_sha256'],component_binary_sha256=record['component_binary_sha256'])),flush=True)


if __name__=='__main__':main()
