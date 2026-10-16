#!/usr/bin/env python3
"""Main-owned build and CPU gates for bounded numerical/service diagnostics."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha

SEED=D/'layer-major-storage-build-v1.json'
APPLICATION=D/'diagnostics-application-v1.json'
O=B/'main-primary-current-diagnostics-build-root-v1'
CHANGED={'src/core/mtp.cpp','src/program/generate.cpp','src/core/expert_source.cpp',
         'include/strata/core/expert_source.hpp','src/kernels/cpu/pool.cpp',
         'include/strata/kernels/cpu/pool.hpp'}


def main():
    seed=json.loads(SEED.read_text());application=json.loads(APPLICATION.read_text())
    assert seed['passed'] and seed['complete'] and not seed['active'] and application['complete']
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        small=json.loads((D/'layer-major-storage-small-v1.json').read_text())
        assert small['passed'] and small['complete'] and not small['active'] and not small['fresh_faults']
        assert all(c['direct_child_reaped'] and c['session_empty'] for c in small['commands'])
        assert {n for n,pin in seed['source_pins'].items() if sha(P/n)!=pin}==CHANGED
        assert all(sha(P/n)==pin for n,pin in application['changed_source_pins'].items())
        build=Path(seed['build_path'])
        assert all(sha(build/n)==pin for n,pin in seed['binaries'].items())
        assert not O.exists();O.mkdir(mode=0o700)
        env=dict(seed['environment']);assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        sources=sorted(set(seed['source_pins'])|set(application['changed_source_pins']))
        r=dict(active=True,passed=False,complete=False,GPU_executed=False,commands=[],
            build_path=str(build),environment=env,recipe_sha256=sha(Path(__file__)),
            seed_receipt_sha256=sha(SEED),application_receipt_sha256=sha(APPLICATION),
            source_pins={n:sha(P/n) for n in sources},control_binary=seed['control_binary'],
            previous_binaries=seed['binaries'],host_contracts_passed=False,
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,
            restore_qualified=False,adopted=False,
            scope='Build actual root with checked MTP record lease, default-off MTP4 joined decode '
                  'observer and standalone selected-position head logits. CPU-only contract/size '
                  'fixtures; no runtime pool scheduling, GPU or model quality qualification.')

        def persist():
            (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')

        owner=module.Owner(O/'owned',persist);r['commands']=owner.commands;persist()

        def run(label,argv,wall=60):
            entry,out,err=owner.run(label,argv,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                text_cap=64 << 20,rss_cap=48 << 30,total_cap=1 << 30,file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code']==0, (
                label,entry.get('error'),err.read_text()[-6000:],out.read_text()[-4000:])
            return out,err

        try:
            run('build-current-diagnostics',['cmake','--build',str(build),'--target','strata',
                'draft_uploads_host_test','--parallel','2'],1500)
            compile_base=['/usr/bin/c++','-std=c++17','-O2','-UNDEBUG','-Wall','-Wextra','-Werror',
                '-pthread','-I'+str(P/'include')]
            fixtures=[('decode-joined-mtp4',P/'tests/cpu/decode_joined_mtp4.cpp'),
                ('prefill-logits',P/'tests/cpu/prefill_logits.cpp'),
                ('decode-ledger-size',D/'mtp4-decode-observer-v1/files/sizeof.cpp')]
            r['fixture_sources']={str(s):sha(s) for _,s in fixtures}
            for label,source in fixtures:
                run('compile-'+label,[*compile_base,str(source),'-o',str(O/label)])
            fixture_dir=O/'prefill-logits-fixtures';fixture_dir.mkdir()
            r['CPU_outputs']={}
            for label,args in [('decode-joined-mtp4',[]),('prefill-logits',[str(fixture_dir)]),
                               ('decode-ledger-size',[])]:
                out,err=run('run-'+label,[str(O/label),*args],30)
                r['CPU_outputs'][label]=dict(stdout=out.read_text(),stderr=err.read_text(),
                    binary_sha256=sha(O/label))
            out,err=run('run-draft-upload-owner-contracts',[str(build/'draft_uploads_host_test')],30)
            r['CPU_outputs']['draft_uploads']=dict(stdout=out.read_text(),stderr=err.read_text(),
                binary_sha256=sha(build/'draft_uploads_host_test'))
            r['binaries']={n:sha(build/n) for n in ['strata','prefill_layer_major_plan_test','sycl_prefill_postops_test']}
            r['compile_commands_sha256']=sha(build/'compile_commands.json')
            commands=json.loads((build/'compile_commands.json').read_text())
            r['compile_flag_proof']={}
            for n in sorted(CHANGED):
                if not n.endswith('.cpp'):continue
                vals=[c for c in commands if c['file']==str(P/n)];assert len(vals)==1
                if n!='src/kernels/cpu/pool.cpp':
                    assert '-fsycl-id-queries-range=size_t' in vals[0]['command']
                r['compile_flag_proof'][n]=vals[0]
            assert all(sha(P/n)==pin for n,pin in r['source_pins'].items())
            r.update(host_contracts_passed=True,passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error);raise
        finally:
            r['active']=False;persist()
            (D/'current-diagnostics-build-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=True,GPU_executed=False,binaries=r['binaries'],
            sizeof_output=r['CPU_outputs']['decode-ledger-size']['stdout'])),flush=True)


if __name__=='__main__':
    main()
