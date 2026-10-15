#!/usr/bin/env python3
"""Main serial CPU contracts and product build; refuses while the Direct profile is live."""
import fcntl
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seed_path=D/'kv-copy-groups-build-v2.json'
        closed_path=D/'model-device-profile-v2.json'
        seed,closed=[json.loads(p.read_text()) for p in [seed_path,closed_path]]
        assert all(d['passed'] and d['complete'] and not d['active'] for d in [seed,closed])
        assert not closed['fresh_faults'] and closed['boot_unchanged'] and not closed['driver_dump_exists']
        assert closed['requests'][0]['reference_ids_equal']
        assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
        spec=importlib.util.spec_from_file_location('owner',OWNER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for c in closed['commands']:
            assert module.closed(c)
            for i in c['owners']:
                now=module.identity(i['pid']);assert now is None or now['start_ticks']!=i['start_ticks'] or now['state']=='Z'
        assert all(sha(P/n)==h for n,h in seed['source_pins'].items())
        build=Path(seed['build_path']);assert all(sha(build/n)==h for n,h in seed['binaries'].items())
        proposal=D/'cpu-gu-reuse-observer-v2'
        pins=json.loads((proposal/'pins.json').read_text())
        patch=proposal/'source.diff'
        for n in ['include/strata/core/expert_source.hpp','include/strata/core/decode_service_budget.hpp','src/core/expert_source.cpp']:
            assert sha(P/n)==pins['inputs_sha256'][n]
        subprocess.run(['git','apply','--check',str(patch)],cwd=P,check=True)
        output=B/'main-primary-cpu-gu-reuse-build-root-v2';assert not output.exists();output.mkdir(mode=0o700)
        env=dict(seed['environment'],ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        result=dict(active=True,passed=False,complete=False,GPU_executed=False,commands=[],environment=env,
            recipe_sha256=sha(Path(__file__)),seed_receipt_sha256=sha(seed_path),closed_profile_sha256=sha(closed_path),
            applied_patch_sha256=sha(patch),proposal_manifest_sha256=sha(proposal/'manifest.json'),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
            build_path=str(build),previous_binaries=seed['binaries'],CPU_outputs={},adopted=False,
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
            scope='Default-OFF issuer-only observer, independent host reference contracts and actual product build. No model recurrence result, cache allocation, or speed claim.')
        def persist():(output/'record.json').write_text(json.dumps(result,indent=2)+'\n')
        owner=module.Owner(output/'owned',persist);result['commands']=owner.commands;persist()
        def run(label,argv,wall=90):
            e,o,s=owner.run(label,argv,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),rss_cap=48<<30,
                text_cap=64<<20,total_cap=256<<20,file_cap=64<<20)
            assert module.closed(e) and e['exit_code']==0,(label,e.get('error'),s.read_text()[-5000:])
            return dict(stdout=o.read_text(),stderr=s.read_text())
        try:
            result['host_compiler_version']=run('host-compiler-version',['/usr/bin/c++','--version'])
            result['host_compiler_sha256']=sha(Path('/usr/bin/c++'))
            subprocess.run(['git','apply',str(patch)],cwd=P,check=True)
            header='include/strata/core/cpu_gu_reuse_observer.hpp'
            assert sha(P/header)==sha(proposal/'cpu_gu_reuse_observer.hpp')
            fixture=P/'tests/cpu/cpu_gu_reuse_observer.cpp';assert not fixture.exists()
            text=(proposal/'cpu_mock.cpp').read_text().replace('#include "cpu_gu_reuse_observer.hpp"','#include "strata/core/cpu_gu_reuse_observer.hpp"')
            text=text.replace('// Main-only CPU test proposal. No product/runtime/model dependencies.','// Independent fixed-domain LRU, provenance, conservation and bounded receipt checks.')
            # Clarify that the completion call belongs after the grouping loop.
            text=text.replace('for(int j=0;j<nj;++j)grouped->touch(0,origin[j],nt[j]);grouped->completed(nj,3);',
                'for(int j=0;j<nj;++j) { grouped->touch(0,origin[j],nt[j]); }\n    grouped->completed(nj,3);')
            fixture.write_text(text)
            sources=set(seed['source_pins'])|set(pins['proposed_product_paths'])|{str(fixture.relative_to(P))}
            result['source_pins']={n:sha(P/n) for n in sorted(sources)};persist()
            for sanitized in [False,True]:
                label='reuse-asan-ubsan' if sanitized else 'reuse';exe=output/label
                flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitized else ['-O2']
                run('compile-'+label,['/usr/bin/c++','-std=c++17','-UNDEBUG','-Wall','-Wextra','-Werror',*flags,
                    '-I'+str(P/'include'),str(fixture),'-o',str(exe)])
                result['CPU_outputs'][label]=run('run-'+label,[str(exe)],300)
                assert 'PASS' in result['CPU_outputs'][label]['stdout']
                result['CPU_outputs'][label]['binary_sha256']=sha(exe);persist();print('CPU PASS '+label,flush=True)
            exe=output/'joined'
            run('compile-joined',['/usr/bin/c++','-std=c++17','-UNDEBUG','-O2','-Wall','-Wextra','-Werror',
                '-I'+str(P/'include'),str(P/'tests/cpu/decode_joined_mtp4.cpp'),'-o',str(exe)])
            result['CPU_outputs']['joined']=run('run-joined',[str(exe)]);result['host_contracts_passed']=True;persist()
            run('build-gu-reuse',['cmake','--build',str(build),'--target','strata','--parallel','2'],1500)
            result['binaries']={n:sha(build/n) for n in seed['binaries']}
            commands=json.loads((build/'compile_commands.json').read_text())
            result['compile_flag_proof']={}
            for n in ['src/core/expert_source.cpp','src/program/generate.cpp']:
                matches=[c for c in commands if c['file']==str(P/n)];assert len(matches)==1
                assert '-fsycl-id-queries-range=size_t' in matches[0]['command']
                result['compile_flag_proof'][n]=matches[0]
            result['compile_commands_sha256']=sha(build/'compile_commands.json')
            assert all(sha(P/n)==h for n,h in result['source_pins'].items())
            result.update(passed=True,complete=True)
        except BaseException as error:result['error']=repr(error);raise
        finally:
            result['active']=False;persist();(D/'cpu-gu-reuse-build-v2.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=result['passed'],GPU_executed=False,binaries=result.get('binaries'))),flush=True)


if __name__=='__main__':main()
