#!/usr/bin/env python3
"""Main-owned FP16 storage build and host arithmetic qualification."""
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha

SEED = D/'layer-major-build-v2.json'
O = B/'main-primary-layer-major-storage-build-root-v1'
CHANGED = {'src/prefill/layer_major.inc', 'src/prefill/prefill.cpp',
           'src/prefill/sycl/postops.cpp', 'src/prefill/layer_major_plan_test.cpp',
           'include/strata/prefill/layer_major_plan.hpp', 'include/strata/prefill/kernels.hpp',
           'tests/sycl/prefill_postops.cpp'}


def main():
    seed=json.loads(SEED.read_text())
    failed_path=D/'layer-major-dispatch-build-v1.json'
    failed=json.loads(failed_path.read_text())
    assert seed['passed'] and seed['complete'] and not seed['active']
    assert not failed['passed'] and not failed['complete'] and not failed['active']
    assert len(failed['commands'])==1 and failed['commands'][0]['exit_code']==1
    assert all(c['direct_child_reaped'] and c['session_empty'] for c in failed['commands'])
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert {n for n,pin in seed['source_pins'].items() if sha(P/n)!=pin}==(CHANGED & set(seed['source_pins']))
        build=Path(seed['build_path'])
        assert all(sha(build/n)==pin for n,pin in seed['binaries'].items())
        assert sha(Path(seed['control_binary']['path']))==seed['control_binary']['sha256']
        assert not O.exists();O.mkdir(mode=0o700)
        proposal=D/'layer-major-fp16-storage-v1'
        env=dict(seed['environment'])
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        r=dict(active=True,passed=False,complete=False,GPU_executed=False,commands=[],
            environment=env,build_path=str(build),seed_receipt_sha256=sha(SEED),
            preserved_compile_failure_sha256=sha(failed_path),recipe_sha256=sha(Path(__file__)),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
            source_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--',*sorted(CHANGED)],cwd=P)).hexdigest(),
            source_pins={n:sha(P/n) for n in sorted(set(seed['source_pins'])|CHANGED)},
            control_binary=seed['control_binary'],previous_binaries=seed['binaries'],
            proposal_sha256={n:sha(proposal/n) for n in ['source.diff','implementation.txt','source-base-sha256.json','dependency-base-sha256.json']},
            main_adjustments=['Correct int64 NE trace format diagnosed by the prior compiler failure.',
                'Test memcpy failures terminate before local host DMA vectors unwind; Buffer get/set share the same checked helper.'],
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,
            restore_qualified=False,adopted=False,build_byte_budget=6 << 30,
            scope='FP32-default layer-major residual storage with opt-in FP16 save/restore, '
                'short-prompt dispatcher and bounded per-layer trace. CPU gate only; '
                'GPU conversion and model comparison must run separately after this build.')

        def persist():
            (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')

        owner=module.Owner(O/'owned',persist);r['commands']=owner.commands;persist()

        def run(label,argv,wall):
            entry,out,err=owner.run(label,argv,env,P,wall=wall,
                cpu=wall*len(os.sched_getaffinity(0)),text_cap=64 << 20,
                rss_cap=48 << 30,total_cap=1 << 30,file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code']==0, (
                label,entry.get('error'),err.read_text()[-6000:],out.read_text()[-4000:])
            return out,err

        try:
            run('build-layer-major-storage',['cmake','--build',str(build),'--target',
                'strata','prefill_layer_major_plan_test','sycl_prefill_postops_test','--parallel','2'],1500)
            out,err=run('host-layer-major-storage-plan',[str(build/'prefill_layer_major_plan_test')],30)
            assert out.read_text()=='layer-major host arithmetic/order/lease fixtures PASS (no kernels)\n'
            r['host_contracts_passed']=True;r['host_test_output']=out.read_text()
            r['binaries']={n:sha(build/n) for n in ['strata','prefill_layer_major_plan_test','sycl_prefill_postops_test']}
            commands=json.loads((build/'compile_commands.json').read_text())
            r['compile_flag_proof']={}
            for n in sorted(CHANGED-{'src/prefill/layer_major.inc','include/strata/prefill/layer_major_plan.hpp','include/strata/prefill/kernels.hpp'}):
                vals=[c for c in commands if c['file']==str(P/n)];assert len(vals)==1
                if n.endswith('layer_major_plan_test.cpp'):
                    assert '-std=c++20' in vals[0]['command'] and '-fsycl' not in vals[0]['command']
                else:
                    assert '-fsycl-id-queries-range=size_t' in vals[0]['command']
                r['compile_flag_proof'][n]=vals[0]
            r['compile_commands_sha256']=sha(build/'compile_commands.json')
            r['build_logical_bytes']=sum(p.stat().st_size for p in build.rglob('*') if p.is_file())
            assert r['build_logical_bytes']<=r['build_byte_budget']
            assert all(sha(P/n)==pin for n,pin in r['source_pins'].items())
            assert sha(failed_path)==r['preserved_compile_failure_sha256']
            r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error);raise
        finally:
            r['active']=False;persist()
            (D/'layer-major-storage-build-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=True,GPU_executed=False,binaries=r['binaries'])),flush=True)


if __name__=='__main__':
    main()
