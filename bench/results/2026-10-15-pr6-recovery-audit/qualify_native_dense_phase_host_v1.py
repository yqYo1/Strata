#!/usr/bin/env python3
"""Main-only host gates over proposed policy/resolver/actual graph-retire body."""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
V=A/'prefill-native-dense-phase-v2'
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=V/'proposal-manifest.json'
        proposal=json.loads(manifest.read_text())
        require(proposal['files']['source.diff']==sha(V/'source.diff'),'reviewed patch identity')
        require(all(sha(V/n)==h for n,h in proposal['files'].items()),'immutable full proposal')
        for n,h in proposal['base_files'].items():
            require(not (P/n).exists() if h is None else sha(P/n)==h,'exact production base: '+n)
        root=B/'main-native-dense-phase-host-root-v1'
        target=A/'native-dense-phase-host-v1.json'
        require(not root.exists() and not target.exists(),'fresh owned host gate')
        root.mkdir(mode=0o700)
        source=V/'proposed/src/core/session.cpp'
        text=source.read_text();start=text.index('void session_graphs_retire_dense(')
        body=text[start:text.index('\n}\n',start)+3]
        include=root/'actual_session_graphs_retire_dense.inc'
        include.write_text(body)
        compiler=Path('/usr/bin/g++')
        fixture=A/'native_dense_graph_retire_host_fixture_v1.cpp'
        module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        record=dict(schema='native-dense-phase-proposed-host-gates-v1',passed=False,complete=False,
            active=True,GPU_executed=False,commands=[],samples=[],recipe_sha256=sha(Path(__file__)),
            proposal_manifest_sha256=sha(manifest),source_diff_sha256=sha(V/'source.diff'),
            compiler=str(compiler),compiler_sha256=sha(compiler),fixture_sha256=sha(fixture),
            extracted_function_sha256=sha(include),proposed_session_source_sha256=sha(source),
            physical_262144_qualified=False,quality_qualified=False,performance_qualified=False,
            actual_generator_transaction_qualified=False,
            scope='Actual proposed inline fault policy and resolver, exact extracted proposed session graph-retirement function with fake CUDA result/handle providers. Host control-flow/all-array coverage only; not driver completion, full generator lease, component payload, model or memory-recovery proof.')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def run(label,args,expected=0):
            c,out,err=owner.run(label,args,env,P,wall=30,cpu=30,rss_cap=2<<30,
                text_cap=1<<20,total_cap=16<<20,file_cap=4<<20)
            require(module.closed(c) and c['exit_code']==expected,label+': '+err.read_text())
            return out.read_text(),err.read_text()
        try:
            record['compiler_version']=run('compiler-version',[str(compiler),'--version'])[0]
            common=[str(compiler),'-std=c++20','-O2','-Wall','-Wextra','-Werror',
                '-I',str(V/'proposed/include'),'-I',str(P/'include')]
            for name in ('host_fault_policy_fixture','host_composite_resolver_fixture'):
                exe=root/name
                run('compile-'+name,[*common,str(V/(name+'.cpp')),'-o',str(exe)])
                stdout,stderr=run(name,[str(exe)])
                record['samples'].append(dict(case=name,binary_sha256=sha(exe),passed=True,stdout=stdout,stderr=stderr))
            exe=root/'graph-retire'
            run('compile-actual-graph-retire',[*common,'-DSTRATA_ENABLE_SYCL','-I',str(root),str(fixture),'-o',str(exe)])
            record['graph_binary_sha256']=sha(exe)
            cases=['empty','full','null-stream','sync-failure']+[f'destroy-{i}' for i in range(1,38)]
            for case in cases:
                negative=case not in ('empty','full')
                stdout,stderr=run('graph-'+case,[str(exe),case],-6 if negative else 0)
                if negative:
                    require('session: uncertain dense graph retirement; owners retained' in stderr and
                        'FAIL retirement returned' not in stderr and 'PASS ' not in stdout,'precise fail-stop provenance')
                else:
                    require(f'PASS graph_count={0 if case=="empty" else 37} all_arrays_reset=1 idempotent=1' in stdout,'complete graph-array coverage')
                record['samples'].append(dict(case=case,expected_exit=-6 if negative else 0,
                    passed=True,stdout=stdout,stderr=stderr))
            require(sha(source)==record['proposed_session_source_sha256'],'source immutable during host gate')
            record.update(passed=True,complete=True,total_cases=len(record['samples']))
        except BaseException as error:
            record['error']=repr(error)
            raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False,complete=False)
            persist()
            with target.open('x') as out:json.dump(record,out,indent=2);out.write('\n')
        print(json.dumps(dict(passed=record['passed'],total_cases=record['total_cases'])))


if __name__=='__main__':main()
