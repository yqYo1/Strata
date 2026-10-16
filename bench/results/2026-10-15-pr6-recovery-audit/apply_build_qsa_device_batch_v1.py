#!/usr/bin/env python3
"""Main serial QSA device-batch integration/build after the alias model pair.

Baseline libraries remain immutable. No GPU execution in this recipe.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha


def main():
    root = B/'main-qsa-device-batch-build-root-v1'
    target = A/'qsa-device-batch-build-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(),'fresh integration/build')
        pair_path = A/'prefill-scratch-alias-model-comparison-v1.json'
        pair = json.loads(pair_path.read_text())
        require(pair['passed'] and pair['complete'] and not pair['active'] and
            pair['sampled_layer_residuals_bitwise_equal'],'closed same-native alias gate')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for arm in (0,1):
            path = A/('prefill-scratch-alias-model-0-v1.json' if arm == 0 else 'prefill-scratch-alias-model-1-v2.json')
            old = json.loads(path.read_text())
            require(pair['references'][path.name]==sha(path) and old['passed'] and old['complete'] and
                not old['active'] and not old['fresh_faults'] and old['boot_unchanged'] and
                not old['driver_dump_exists'],'closed model/fault gate')
            for c in old['commands']:
                require(module.closed(c),'closed owned commands')
                for identity in c['owners']:
                    now = module.identity(identity['pid'])
                    require(now is None or now['start_ticks']!=identity['start_ticks'],'retired prior owners')
            require(all(sha(P/n)==h for n,h in old['source_pins'].items()),'unchanged model baseline source')
        proposal = A/'qsa-indexer-device-batch-v2'
        manifest_path = proposal/'source-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        patch = proposal/'source.diff'
        require(sha(patch)==manifest['source_diff_sha256']=='6732a090a727a1ede3aa2ca5165bb4c060070f625ec90696a49c533aff5c7c87','reviewed proposal')
        for x in manifest['files']:
            path=P/x['path'];base=x['base_sha256']
            require(not path.exists() if base is None else sha(path)==base,'exact patch base '+x['path'])
        review=A/'research-qsa-batch-v2-seed-boundary-round695-v1.txt'
        require(sha(review)=='822d477426369c7134cd57ed834fc64cc495e4208144db6caad350ff4f9d7310','independent source review')
        seed_path=B/'main-primary-native-build-root-v1/record.json'
        seed=json.loads(seed_path.read_text())
        require(seed['passed'] and seed['complete'] and not seed['active'],'original actual configure recipe')
        env=dict(seed['environment']);compiler=seed['compiler']['path']
        require(sha(Path(compiler))==seed['compiler']['sha256'],'pinned compiler')
        require(not any('TOKEN' in k or 'API_KEY' in k or k=='LD_PRELOAD' or
            k.startswith(('UNITRACE_','PTI_')) for k in env),'bounded build environment')
        configure=list(next(c for c in seed['commands'] if c['label']=='configure-primary-native')['argv'])
        build=root/'build';configure[configure.index('-B')+1]=str(build)
        configure+=['-DSTRATA_PREFILL_MMQ=OFF','-DSTRATA_BUILD_PREFILL_HOST_TESTS=ON']
        root.mkdir(mode=0o700)
        r=dict(passed=False,complete=False,active=True,applied=False,GPU_executed=False,
            commands=[],recipe_sha256=sha(Path(__file__)),references={p.name:sha(p) for p in (pair_path,manifest_path,review)},
            patch_sha256=sha(patch),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
            environment=env,compiler=compiler,compiler_sha256=sha(Path(compiler)),build_path=str(build),
            default_enabled=False,performance_qualified=False,quality_qualified=False,restore_qualified=False,
            physical_model_262144_qualified=False,configure_argv=configure,
            build_byte_budget=2<<30,
            build_consumer='Main actual QSA component direct/dispatch/state/guard/scoring comparison, then long-model and clean PP candidate',
            review_point='After component and model comparisons; retire superseded binaries/verbose successful logs.',
            scope='Actual production QSA batch compile/link plus production host planner checks only. No GPU/performance/state/RESTORE/model capacity qualification.')
        def persist(): (root/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);r['commands']=owner.commands;persist()
        def run(label,argv,wall=90):
            c,out,err=owner.run(label,argv,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                rss_cap=32<<30,text_cap=8<<20,total_cap=2<<30,file_cap=256<<20)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-5000:]+'\n'+out.read_text()[-2500:])
            print('PASS '+label,flush=True)
            return out.read_text()
        try:
            run('check-QSA-patch',['git','apply','--check',str(patch)])
            run('apply-QSA-patch',['git','apply',str(patch)]);r['applied']=True
            for x in manifest['files']:
                require(sha(P/x['path'])==x['proposed_sha256'],'exact proposed source '+x['path'])
            names=subprocess.check_output(['git','ls-files','--','src','include','cmake','CMakeLists.txt','third_party'],cwd=P,text=True).splitlines()
            names=sorted(set(names)|{x['path'] for x in manifest['files']})
            r['source_pins']={n:sha(P/n) for n in names if (P/n).is_file()};persist()
            run('configure-QSA',configure,300)
            run('build-QSA',['cmake','--build',str(build),'--target','strata',
                'sycl_qsa_indexer_device_batch_test','qsa_indexer_device_batch_plan_test','--parallel','2'],1800)
            output=run('host-production-QSA-plan',[str(build/'qsa_indexer_device_batch_plan_test')])
            require('PASS native QSA device batch host plan/flag contracts; no GPU qualification' in output,'complete host assertions')
            r['host_stdout']=output
            cc=json.loads((build/'compile_commands.json').read_text());r['compile_commands_sha256']=sha(build/'compile_commands.json')
            proof={}
            for name in ('src/kernels/sycl/native_qsa_indexer.cpp','src/prefill/prefill.cpp','tests/sycl/qsa_indexer_device_batch.cpp'):
                matches=[c for c in cc if c['file']==str(P/name)]
                require(len(matches)==1 and '-fsycl-id-queries-range=size_t' in matches[0]['command'],'actual safe resolver '+name)
                proof[name]=matches[0]
            for name in ('src/kernels/sycl/native_qsa_indexer.cpp','src/prefill/prefill.cpp'):
                require('-fno-fast-math' in proof[name]['command'] and '-ffp-contract=off' in proof[name]['command'],'reference arithmetic compilation '+name)
            require(all(sha(P/n)==h for n,h in r['source_pins'].items()),'source stability')
            r['compile_flag_proof']=proof
            r['binaries']={n:sha(build/n) for n in ('strata','sycl_qsa_indexer_device_batch_test','qsa_indexer_device_batch_plan_test')}
            r.update(binary=str(build/'strata'),binary_sha256=r['binaries']['strata'],
                fixture=str(build/'sycl_qsa_indexer_device_batch_test'),
                fixture_sha256=r['binaries']['sycl_qsa_indexer_device_batch_test'],passed=True,complete=True)
        except BaseException as e:
            r['error']=repr(e);raise
        finally:
            r.update(active=owner.active is not None,finished_utc=module.utc());persist()
            target.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],GPU_executed=False,binary_sha256=r.get('binary_sha256'))))


if __name__=='__main__': main()
