#!/usr/bin/env python3
"""Main-owned build/host gates after review; preserve the qualified control."""
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha

O = B/'main-primary-layer-major-build-root-v1'
SEED = D/'gemm-orientation-build-v2.json'
CHANGED = {'src/prefill/prefill.cpp', 'include/strata/prefill/prefill.hpp',
           'include/strata/core/expert_cache.hpp', 'src/core/expert_cache.cpp', 'CMakeLists.txt'}
NEW = {'src/prefill/layer_major.inc', 'include/strata/prefill/layer_major_plan.hpp',
       'src/prefill/layer_major_plan_test.cpp'}


def main():
    seed = json.loads(SEED.read_text())
    assert seed['passed'] and seed['complete'] and not seed['active']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ['natural-layout-long-pair-v2.json', 'unitrace-clock-source-v1.json']:
            gate = json.loads((D/name).read_text())
            assert gate['passed'] and gate['complete'] and not gate['active']
            assert all(c['direct_child_reaped'] and c['session_empty'] for c in gate['commands'])
        assert [n for n, pin in seed['source_pins'].items() if sha(P/n) != pin and n not in CHANGED] == []
        build = Path(seed['build_path'])
        assert sha(build/'strata') == seed['production_binary_sha256']
        assert not O.exists()
        O.mkdir(mode=0o700)
        control = O/'control-strata'
        shutil.copy2(build/'strata', control)
        assert sha(control) == seed['production_binary_sha256']
        env = dict(seed['environment'])
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        proposal = D/'layer-major-candidate-v1'
        r = dict(active=True, passed=False, complete=False, commands=[], GPU_executed=False,
            environment=env, build_path=str(build), seed_receipt_sha256=sha(SEED),
            source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
            source_diff_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff','HEAD','--',*sorted(CHANGED)],cwd=P)).hexdigest(),
            source_pins={n:sha(P/n) for n in sorted(set(seed['source_pins'])|NEW|CHANGED)},
            proposal_sha256={n:sha(proposal/n) for n in ['implementation.txt','product.patch','source-base-sha256.json']},
            main_adjustment='Allow residual dump only at final model layer, with checked drain/readback. Original proposal rejected this diagnostic; permits matched-input numeric validation without intermediate-layer captures.',
            recipe_sha256=sha(Path(__file__)), control_binary=dict(path=str(control),sha256=sha(control),bytes=control.stat().st_size,
                consumer='Main same-input default/layer-major numerical and formal bench.sh performance comparison',
                byte_budget=128 << 20,next_review='after candidate is rejected or qualified and performance comparison closes'),
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
            adopted=False, build_byte_budget=6 << 30,
            scope='Current-root opt-in per-layer expert reuse with residual GPU prefix/host tail. No decode cache/graph/MTP release. Default OFF; compile/CPU gates only.')

        def persist():
            (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')

        owner = module.Owner(O/'owned',persist)
        r['commands'] = owner.commands
        persist()

        def run(label,argv,wall=1500):
            entry,out,err=owner.run(label,argv,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                text_cap=64 << 20,rss_cap=48 << 30,total_cap=1 << 30,file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code']==0,(label,entry.get('error'),err.read_text()[-6000:],out.read_text()[-2000:])
            return out,err

        try:
            run('configure-layer-major-host-tests',['cmake','-S',str(P),'-B',str(build),'-DSTRATA_BUILD_PREFILL_HOST_TESTS=ON'],120)
            run('build-layer-major',['cmake','--build',str(build),'--target','strata','prefill_layer_major_plan_test','--parallel','2'])
            out,err=run('host-layer-major-contracts',[str(build/'prefill_layer_major_plan_test')],30)
            assert 'layer-major host arithmetic/order/lease fixtures PASS (no kernels)' in out.read_text()
            r['host_contracts_passed']=True
            r['host_test_output']=out.read_text()
            r['binaries']={n:sha(build/n) for n in ['strata','prefill_layer_major_plan_test']}
            commands=json.loads((build/'compile_commands.json').read_text())
            r['compile_flag_proof']={}
            for n in ['src/prefill/prefill.cpp','src/core/expert_cache.cpp','src/prefill/layer_major_plan_test.cpp']:
                vals=[c for c in commands if c['file']==str(P/n)]
                assert len(vals)==1 and '-fsycl-id-queries-range=size_t' in vals[0]['command']
                r['compile_flag_proof'][n]=vals[0]
            r['compile_commands_sha256']=sha(build/'compile_commands.json')
            r['build_logical_bytes']=sum(p.stat().st_size for p in build.rglob('*') if p.is_file())
            assert r['build_logical_bytes'] <= r['build_byte_budget']
            assert all(sha(P/n)==pin for n,pin in r['source_pins'].items())
            assert sha(control)==r['control_binary']['sha256']
            r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error)
            raise
        finally:
            r['active']=False
            persist()
            (D/'layer-major-build-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=True,host_contracts_passed=True,GPU_executed=False,binaries=r['binaries'])),flush=True)


if __name__=='__main__':
    main()
