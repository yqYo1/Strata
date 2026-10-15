#!/usr/bin/env python3
"""Main serial opt-in source integration and a fresh complete SYCL build.

The baseline archives/binaries stay immutable. No GPU test is run here.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha


def main():
    root=B/'main-prefill-head-release-build-root-v1'
    target=A/'prefill-head-release-build-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(),'fresh build')
        host=A/'prefill-owner-helpers-host-v3.json'
        h=json.loads(host.read_text())
        require(h['passed'] and h['complete'] and not h['active'] and len(h['results'])==4,'CPU helper gate')
        closed=json.loads((A/'prefill-window-profile-long-v4.json').read_text())
        require(closed['passed'] and closed['complete'] and not closed['active'] and not closed['fresh_faults']
                and closed['boot_unchanged'] and not closed['driver_dump_exists'],'latest GPU closure')
        module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for c in closed['commands']+closed['control_commands']+h['commands']:
            require(module.closed(c),'previous owned command not closed')
            for old in c['owners']:
                now=module.identity(old['pid'])
                require(now is None or now['start_ticks']!=old['start_ticks'],'previous owned process active')
        proposal=A/'prefill-head-release-v1'
        patch=proposal/'source.diff';pins=json.loads((proposal/'pins.json').read_text())
        require(sha(patch)=='737b9a346e719aec22fc069915e86da97e5caebe4ec4ca27ffa5308ed49262bc','reviewed source diff')
        for name,x in pins.items():
            p=P/name
            require(not p.exists() if x['base_sha256'] is None else sha(p)==x['base_sha256'],
                    'proposal base changed: '+name)
        seed_path=B/'main-primary-native-build-root-v1/record.json'
        seed=json.loads(seed_path.read_text())
        require(seed['passed'] and seed['complete'] and not seed['active'],'original complete configure recipe')
        compiler=seed['compiler']['path']
        env=dict(seed['environment'])
        require(not any('TOKEN' in k or 'API_KEY' in k or k=='LD_PRELOAD' or
                        k.startswith(('UNITRACE_','PTI_')) for k in env),'bounded build environment')
        configure=list(next(c for c in seed['commands'] if c['label']=='configure-primary-native')['argv'])
        build=root/'build';configure[configure.index('-B')+1]=str(build)
        configure+=['-DSTRATA_PREFILL_MMQ=OFF','-DSTRATA_BUILD_PREFILL_HOST_TESTS=ON']
        current_build=json.loads((A/'selective-iq-native-build-v1.json').read_text())
        configured=configure[next(i for i,x in enumerate(configure) if x.startswith('-DCMAKE_CXX_COMPILER='))].split('=',1)[1]
        require(compiler==current_build['compiler']==configured and
                sha(Path(compiler))==current_build['compiler_sha256']==seed['compiler']['sha256'],
                'same previously pinned compiler')
        root.mkdir(mode=0o700)
        r=dict(passed=False,complete=False,active=True,GPU_executed=False,applied=False,
            recipe_sha256=sha(Path(__file__)),proposal_sha256=sha(patch),proposal_pins_sha256=sha(proposal/'pins.json'),
            CPU_receipt_sha256=sha(host),seed_receipt_sha256=sha(seed_path),source_revision=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=P,text=True).strip(),environment=env,
            compiler=compiler,compiler_sha256=sha(Path(compiler)),
            build_path=str(build),configure_argv=configure,commands=[],default_enabled=False,
            scope='Fresh complete SYCL compile/link of exact reviewed head-release proposal. CPU helper contracts qualified separately; no GPU, memory recovery, model, quality, performance, physical262144 or RESTORE qualification.',
            build_consumer='Main controlled baseline/default-off and head-release GPU lifecycle comparison',
            build_byte_budget=2<<30,review_point='After lifecycle/capacity/clean-prefill comparison; retire superseded builds and successful logs.')
        def persist():(root/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);r['commands']=owner.commands;persist()
        def run(label,args,wall=90):
            entry,out,err=owner.run(label,args,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                rss_cap=32<<30,text_cap=8<<20,total_cap=2<<30,file_cap=256<<20)
            require(module.closed(entry) and entry['exit_code']==0,
                    label+' failed: '+err.read_text()[-5000:]+'\n'+out.read_text()[-2500:])
            print('PASS '+label,flush=True)
        try:
            run('check-reviewed-head-patch',['git','apply','--check',str(patch)])
            run('apply-reviewed-head-patch',['git','apply',str(patch)])
            r['applied']=True
            for name,x in pins.items():require(sha(P/name)==x['proposed_sha256'],'exact proposed output: '+name)
            names=subprocess.check_output(['git','ls-files','--','src','include','cmake','CMakeLists.txt','third_party'],cwd=P,text=True).splitlines()
            sources=sorted(set(names)|set(pins))
            r['source_pins']={name:sha(P/name) for name in sources if (P/name).is_file()};persist()
            run('configure-head-release',configure,300)
            run('build-head-release',['cmake','--build',str(build),'--target','strata','--parallel','2'],1800)
            require(all(sha(P/name)==digest for name,digest in r['source_pins'].items()),'sources changed during build')
            commands=json.loads((build/'compile_commands.json').read_text())
            r['compile_commands_sha256']=sha(build/'compile_commands.json')
            proof={}
            for name in ['src/core/native_head.cpp','src/core/mtp.cpp','src/core/verify.cpp',
                         'src/program/generate.cpp','src/prefill/prefill.cpp','src/kernels/sycl/iq.cpp']:
                matches=[c for c in commands if c['file']==str(P/name)]
                require(len(matches)==1,'actual resolver: '+name)
                require('-fsycl-id-queries-range=size_t' in matches[0]['command'],'range-size safety flag: '+name)
                proof[name]=matches[0]
            require('-fno-fast-math' in proof['src/prefill/prefill.cpp']['command'] and
                    '-ffp-contract=off' in proof['src/prefill/prefill.cpp']['command'],'FP32 reference compilation policy')
            r['compile_flag_proof']=proof
            r['binary']=str(build/'strata');r['binary_sha256']=sha(build/'strata')
            r['binaries']={p.name:sha(p) for p in build.glob('libstrata*.a')}
            r['binaries']['strata']=r['binary_sha256']
            r.update(passed=True,complete=True)
        except BaseException as e:
            r['error']=repr(e);raise
        finally:
            r['active']=owner.active is not None;r['finished_utc']=module.utc();persist()
            target.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],GPU_executed=False,binary_sha256=r.get('binary_sha256'))))


if __name__=='__main__':main()
