#!/usr/bin/env python3
"""Main serial host observer gates, with no provider initialization or GPU work."""
import fcntl
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha

MODES=('normal32','create-1','create-7','create-12','record-1','record-2',
       'query-error','elapsed-error','nan','negative','double-begin','early-return','destroy-error')


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gate_path=A/'canonical-bank-profile-build-v1.json';gate=json.loads(gate_path.read_text())
        require(gate['passed'] and gate['complete'] and not gate['active'],'fresh actual ABI build before host gates')
        require(all(sha(P/n)==h for n,h in gate['source_pins'].items()),'compiled source identity')
        for c in gate['commands']:require(module.closed(c),'build owner closed')
        root=B/'main-coarse-profile-host-root-v1';output=A/'coarse-profile-host-v1.json'
        require(not root.exists() and not output.exists(),'fresh host qualification root')
        root.mkdir(mode=0o700)
        fixture=A/'coarse-profile-host-fixture-v1'
        compiler=shutil.which('c++');require(compiler is not None,'host C++ compiler')
        files=(fixture/'fixture.cpp',fixture/'cuda_runtime.h',P/'include/strata/prefill/layermajor_coarse_profile.hpp')
        record=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],samples=[],
                    recipe_sha256=sha(Path(__file__)),references={gate_path.name:sha(gate_path)},
                    source_pins={str(p):sha(p) for p in files},compiler=compiler,compiler_sha256=sha(Path(compiler)),
                    scope='Actual observer header with a host-only API shim. Fixed12event lifetimes, '
                          '12256markers at32chunks/48layers, API/error/incomplete and output bounds. '
                          'No actual SYCL/runtime, real queue scheduling, model, quality, RESTORE or physical-context proof.')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def run(label,argv,wall,text):
            c,out,err=owner.run(label,argv,env,P,wall=wall,cpu=wall*2,rss_cap=2<<30,
                                text_cap=text,total_cap=16<<20,file_cap=max(text,16<<20) if label=='compile' else text)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-3000:])
            return out,err
        try:
            policy=Path(gate['build_path'])/'canonical_bank_policy_test'
            require(sha(policy)==gate['binaries']['canonical_bank_policy_test'],'actual canonical policy executable')
            out,err=run('canonical-policy',[str(policy)],30,64<<10)
            require(out.read_text().strip()=='HOST_ONLY canonical policy/HC/PLE classification; no allocation or model pass' and
                    not err.read_text(),'canonical policy geometry/status gates')
            record['canonical_policy_passed']=True
            exe=root/'profile-host'
            run('compile',[compiler,'-std=c++20','-Wall','-Wextra','-Werror',
                           '-I'+str(fixture),'-I'+str(P/'include'),str(fixture/'fixture.cpp'),'-o',str(exe)],120,1<<20)
            record['binary_sha256']=sha(exe)
            for mode in MODES:
                out,err=run(mode,[str(exe),mode],30,64<<10)
                stdout=out.read_text();stderr=err.read_text();lines=stderr.splitlines()
                require(stdout.startswith('HOST_ONLY mode='+mode+' ') and 'HOST_FIXTURE_FAIL' not in stderr,'host result')
                require(len(lines)==50 and all(line.startswith('strata lm profile: ') for line in lines) and
                        len(stderr.encode())<64<<10,'bounded complete observer output')
                expected=1 if mode=='normal32' else 0
                require(f'v=2 complete={expected} ' in lines[0],'correct profile incomplete/success status')
                require(('errors=0 ' in lines[0])==(mode in ('normal32','early-return')),'error accounting')
                record['samples'].append(dict(mode=mode,exit_code=0,stdout=stdout.strip(),
                                              header=lines[0],aggregate=lines[-1],stderr_bytes=len(stderr.encode()),
                                              stderr_lines=len(lines),stderr_sha256=sha(err)))
            require(all(sha(Path(n))==h for n,h in record['source_pins'].items()),'observer/fixture source stable')
            record.update(passed=True,complete=True,host_arms=len(MODES))
        except BaseException as error:
            record['error']=repr(error);raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):record.update(passed=False,complete=False)
            persist()
            with output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],host_arms=record['host_arms'],GPU_executed=False)))


if __name__=='__main__':main()
