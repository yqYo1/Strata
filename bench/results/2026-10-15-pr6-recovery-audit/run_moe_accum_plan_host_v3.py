#!/usr/bin/env python3
"""Serialize host-only capacity checks after the active clean GPU cohort closes."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-moe-window')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if not __debug__:
        raise RuntimeError('Run without Python -O')
    dst = A/'moe-accum-plan-host-v3.json'
    root = B/'main-moe-accum-plan-host-v3'
    assert not dst.exists() and not root.exists()
    helper = A/'direct_owner_transient_artifacts_v1.py'
    assert sha(helper) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('accum_plan_owner', helper)
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    source = P/'tests/moe_accum_plan.cpp'
    header = P/'include/strata/prefill/moe_accum_plan.hpp'
    compiler = Path('/usr/bin/g++').resolve(strict=True)
    pins = {str(f): sha(f) for f in [source,header,P/'tests/moe_accum_routes.cpp',P/'include/strata/prefill/moe_window_routes.hpp',compiler,Path(__file__),helper]}
    rec = dict(schema='strata-moe-accum-plan-host-v3', passed=False, complete=False, active=False,
               phase='waiting_for_measurement_lock', parent_identity=owned.identity(os.getpid()),
               GPU_executed=False, source_pins=pins, commands=[],
               scope='Capacity/offset/guard-reservation and rejection tests only. Host routes/uniqueness and capacity only. No GPU kernel, alias lifetime, precision, inference adoption or performance qualification.')
    owner = None
    def save():
        rec['active'] = bool(owner and owner.active is not None)
        rec['ownership_closed'] = bool(owner is None or (owner.active is None and all(owned.closed(c) for c in owner.commands)))
        dst.write_text(json.dumps(rec,indent=2)+'\n')
    save()
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            path = A/'moe-accum-model-clean-cohort-v1.json'
            cohort = json.loads(path.read_text())
            assert cohort['passed'] and cohort['complete'] and cohort['active_sample'] is None and len(cohort['completed'])==6
            for item in cohort['completed']:
                assert sha(item['path']) == item['sha256']
                sample = json.loads(Path(item['path']).read_text())
                assert sample['passed'] and sample['complete'] and sample['ownership_closed'] and not sample['active'] and not sample['fresh_faults'] and not sample['driver_dump_exists']
                for command in sample['commands']:
                    assert owned.closed(command) and command['exit_code']==0
                    for old in command['owners']+[command['direct_identity']]:
                        now = owned.identity(old['pid'])
                        assert now is None or now['start_ticks']!=old['start_ticks']
            assert all(sha(f)==h for f,h in pins.items())
            rec['closed_GPU_cohort'] = dict(path=str(path),sha256=sha(path))
            root.mkdir(mode=0o700)
            (root/'tmp').mkdir()
            rec['phase']='host_analysis_and_capacity_tests'
            owner=owned.Owner(root/'owned',save)
            rec['commands']=owner.commands
            env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
                     PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root/'tmp'))
            rec['environment']=env
            def run(label,argv):
                command,out,err=owner.run(label,argv,env,P,wall=120,cpu=120,rss_cap=1<<30,
                    text_cap=16<<20,file_cap=32<<20,total_cap=64<<20)
                assert owned.closed(command) and command['exit_code']==0,(label,command['exit_code'],str(err))
                return out.read_text()
            rec['compiler_version']=run('compiler-version',[str(compiler),'--version'])
            results={}
            for mode in ('release','ubsan'):
                exe=root/f'moe_accum_plan_{mode}'
                flags=['-std=c++20','-O2','-DNDEBUG','-Wall','-Wextra','-Werror','-pedantic']
                if mode=='ubsan': flags+=['-fsanitize=undefined','-fno-sanitize-recover=undefined']
                run(f'compile-{mode}',[str(compiler),*flags,'-I'+str(P/'include'),str(source),'-o',str(exe)])
                output=run(f'test-{mode}',[str(exe)])
                assert sum(line.startswith('case rows=') and line.endswith(' PASS') for line in output.splitlines())==9
                assert 'negative_checks=34 PASS' in output and 'moe_accum_plan PASS GPU=UNRUN numeric=UNQUALIFIED lifetime=EXTERNAL_REQUIRED' in output
                results[mode]=dict(binary=str(exe),binary_sha256=sha(exe),stdout=output)
                route_exe=root/f'moe_accum_routes_{mode}'
                run(f'compile-routes-{mode}',[str(compiler),*flags,'-I'+str(P/'include'),str(P/'tests/moe_accum_routes.cpp'),'-o',str(route_exe)])
                route_output=run(f'test-routes-{mode}',[str(route_exe)])
                assert sum(line.startswith('routes rows=') for line in route_output.splitlines())==18
                assert 'moe_accum_routes PASS GPU=UNRUN' in route_output
                results['routes-'+mode]=dict(binary=str(route_exe),binary_sha256=sha(route_exe),stdout=route_output)
            assert all(sha(f)==h for f,h in pins.items())
            rec.update(results=results,passed=True,complete=True,phase='closed')
        except BaseException as error:
            import traceback
            rec.update(error=repr(error),traceback=traceback.format_exc(),passed=False,complete=False,phase='failed')
        finally:
            save()
    print(json.dumps({k:rec.get(k) for k in ['passed','complete','ownership_closed','phase','error']}),flush=True)
    return 0 if rec['passed'] else 1

if __name__=='__main__':
    sys.exit(main())
