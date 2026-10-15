#!/usr/bin/env python3
"""Main serial CPU contracts for exact proposed helpers, without root edits."""
import fcntl
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha


def main():
    target=A/'prefill-owner-helpers-host-v3.json'
    root=B/'main-prefill-owner-helpers-host-root-v3'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(),'fresh host experiment')
        closed=json.loads((A/'prefill-window-profile-long-v4.json').read_text())
        require(closed['passed'] and closed['complete'] and not closed['active'],'GPU job closed')
        module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for c in closed['commands']+closed['control_commands']:
            require(module.closed(c),'previous command closure')
            for old in c['owners']:
                now=module.identity(old['pid'])
                require(now is None or now['start_ticks']!=old['start_ticks'],'GPU job still active')
        root.mkdir(mode=0o700)
        env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
                 PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        cases=[('head-scope',A/'prefill-head-release-v1/host_scope_fixture_main_v2.cpp',
                A/'prefill-head-release-v1/prefill_head_scope.hpp'),
               ('KV-prefix',A/'layer-major-kv-prefix-reuse-v1/layer_major_kv_prefix_reuse_test.cpp',
                A/'layer-major-kv-prefix-reuse-v1/layer_major_kv_prefix_reuse.hpp')]
        pins=json.loads((A/'prefill-head-release-v1/pins.json').read_text())
        require(sha(cases[0][2])==pins['include/strata/core/prefill_head_scope.hpp']['proposed_sha256'],
                'exact proposed production head helper')
        require(sha(cases[1][2])=='236b59d28b07e2c39069f458d62d4d470b69263e079eec33987fdbceda585aab',
                'exact proposed production KV helper')
        original=A/'prefill-head-release-v1/host_scope_fixture.cpp'
        old='if(!target)std::terminate();draft=true;move(Phase::detached,Phase::restored);'
        new='if(!target)std::terminate();\n        draft=true;move(Phase::detached,Phase::restored);'
        require(original.read_text().count(old)==1 and
                cases[0][1].read_text()==original.read_text().replace(old,new),
                'only a newline changed in the misleading-indentation fixture')
        overlay=root/'include/strata/prefill/layer_major_kv_prefix_reuse.hpp'
        overlay.parent.mkdir(parents=True)
        overlay.write_bytes(cases[1][2].read_bytes())
        require(sha(overlay)==sha(cases[1][2]),'exact product include path overlay')
        r=dict(passed=False,complete=False,active=True,GPU_executed=False,root_source_changed=False,
            recipe_sha256=sha(Path(__file__)),environment=env,commands=[],results=[],
            previous_failed_receipts={n:sha(A/n) for n in ('prefill-owner-helpers-host-v1.json','prefill-owner-helpers-host-v2.json')},
            fixture_fix=dict(original_sha256=sha(original),new_sha256=sha(cases[0][1]),
                reason='GCC -Werror=misleading-indentation rejected a one-line if followed by unconditional state writes; split one newline, no logic or warning-policy change.'),
            source_pins={str(p.relative_to(P)):sha(p) for _,s,h in cases for p in (s,h)},
            scope='Normal and ASan/UBSan CPU execution of the byte-identical proposed helper headers. Does not instantiate production GPU owners or qualify DMA, graph retirement, model parity, capacity, quality, performance, physical262144 or RESTORE.',
            retention=dict(consumer='Main proposal integration and actual SYCL build review',byte_budget=128<<20,
                next_review='After product-source integration/build qualification; retain compact receipts and retire redundant build logs/binaries.'))
        def persist():(root/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);r['commands']=owner.commands;persist()
        try:
            for sanitized in (False,True):
                for name,source,header in cases:
                    label=name+('-asan-ubsan' if sanitized else '-normal');exe=root/label
                    flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitized else ['-O2']
                    command=['/usr/bin/c++','-std=c++17','-UNDEBUG','-Wall','-Wextra','-Werror',*flags,
                             '-I'+str(root/'include'),'-I'+str(header.parent),str(source),'-o',str(exe)]
                    for action,args in [('compile',command),('run',[str(exe)])]:
                        entry,out,err=owner.run(action+'-'+label,args,env,P,wall=90,cpu=180,
                            rss_cap=4<<30,text_cap=4<<20,total_cap=128<<20,file_cap=32<<20)
                        require(module.closed(entry) and entry['exit_code']==0,
                            action+' '+label+' failed: '+err.read_text()[-4000:])
                        if action=='run':
                            r['results'].append(dict(case=label,stdout=out.read_text(),stderr=err.read_text(),
                                binary_sha256=sha(exe),sanitizers=sanitized))
                    print('CPU contract PASS '+label,flush=True);persist()
            require(all(sha(P/n)==h for n,h in r['source_pins'].items()),'sources unchanged')
            r.update(passed=True,complete=True)
        except BaseException as e:
            r['error']=repr(e);raise
        finally:
            r['active']=owner.active is not None;r['finished_utc']=module.utc();persist()
            require(not r['active'],'host command still active')
            target.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],cases=len(r['results']),GPU_executed=False)))


if __name__=='__main__':main()
