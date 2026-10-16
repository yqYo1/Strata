#!/usr/bin/env python3
"""Main serial application/build of default-off boundary and format diagnostics."""
import fcntl
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,sha

SEED=D/'mtp-expert-release-build-v1.json'
O=B/'main-primary-boundary-service-build-root-v1'

def main():
    seed=json.loads(SEED.read_text())
    assert seed['passed'] and seed['complete'] and not seed['active']
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert all(sha(P/n)==h for n,h in seed['source_pins'].items())
        closed=json.loads((D/'decode-ledger-fp32-fixed-normal-v1.json').read_text())
        assert closed['passed'] and closed['complete'] and not closed['active'] and not closed['fresh_faults']
        for c in closed['commands']:
            assert module.closed(c)
            for i in c['owners']:
                live=module.identity(i['pid'])
                assert live is None or live['start_ticks']!=i['start_ticks'] or live['state']=='Z'
        build=Path(seed['build_path'])
        assert all(sha(build/n)==h for n,h in seed['binaries'].items())
        patches={D/'layer-major-boundary-audit-v2/source.diff':'24fad8321ee9c7d655df42a9be9cd694ade1ae535b0a5cc65231ea0d64e8ee53',
            D/'decode-format-service-v2/source.diff':'1825ae2c93b8db73df10cb96f5da1089a69ee4c04008cd7ad6787d826b8027d6'}
        assert all(sha(p)==h for p,h in patches.items())
        bp=json.loads((D/'layer-major-boundary-audit-v2/pins.json').read_text())
        fp=json.loads((D/'decode-format-service-v2/pins.json').read_text())
        for n,v in bp['files'].items():
            assert not (P/n).exists() if v['new_file'] else sha(P/n)==v['base_sha256']
        assert all(sha(P/n)==h for n,h in fp['source_sha256'].items())
        for patch in patches: subprocess.run(['git','apply','--check',str(patch)],cwd=P,check=True)
        assert not O.exists();O.mkdir(mode=0o700)
        env=dict(seed['environment']);assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        sources=sorted(set(seed['source_pins'])|set(bp['files'])|set(fp['source_sha256'])|{'tests/cpu/prefill_boundary_audit.cpp'})
        r=dict(active=True,passed=False,complete=False,GPU_executed=False,commands=[],environment=env,
            recipe_sha256=sha(Path(__file__)),seed_receipt_sha256=sha(SEED),build_path=str(build),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
            applied_patches={str(p.relative_to(P)):h for p,h in patches.items()},previous_binaries=seed['binaries'],
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
            adopted=False,scope='Diagnostic source integration, CPU contracts/sanitizers and actual SYCL build only.')
        def persist(): (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(O/'owned',persist);r['commands']=owner.commands;persist()
        def run(label,args,wall=60):
            entry,out,err=owner.run(label,args,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                text_cap=64<<20,rss_cap=48<<30,total_cap=512<<20,file_cap=64<<20)
            assert module.closed(entry) and entry['exit_code']==0,(label,entry.get('error'),err.read_text()[-6000:],out.read_text()[-4000:])
            return out,err
        try:
            for patch in patches: subprocess.run(['git','apply',str(patch)],cwd=P,check=True)
            assert all(sha(P/n)==v['proposed_sha256'] for n,v in bp['files'].items())
            r['source_pins']={n:sha(P/n) for n in sources};persist()
            fixture=D/'decode-format-service-v2/fixture.cpp'
            r['fixture_proposal_sha256']=sha(fixture)
            compile_base=['/usr/bin/c++','-std=c++17','-UNDEBUG','-Wall','-Wextra','-Werror','-pthread','-I'+str(P/'include')]
            fixtures=[('decode-joined',P/'tests/cpu/decode_joined_mtp4.cpp'),('decode-format',fixture),
                ('boundary',P/'tests/cpu/prefill_boundary_audit.cpp')]
            r['CPU_outputs']={}
            for sanitized in [False,True]:
                flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitized else ['-O2']
                for name,source in fixtures:
                    if sanitized and name=='decode-joined': continue # fork/failstop covered by normal existing fixture
                    label=name+('-asan-ubsan' if sanitized else '')
                    exe=O/label
                    run('compile-'+label,[*compile_base,*flags,str(source),'-o',str(exe)],90)
                    args=[]
                    if name=='boundary': args=[str(O/(label+'-rows'))]
                    out,err=run('run-'+label,[str(exe),*args],30)
                    r['CPU_outputs'][label]=dict(stdout=out.read_text(),stderr=err.read_text(),binary_sha256=sha(exe))
                    print('CPU PASS '+label,flush=True);persist()
            r['host_contracts_passed']=True;persist()
            run('build-boundary-service',['cmake','--build',str(build),'--target','strata','--parallel','2'],1500)
            r['binaries']={n:sha(build/n) for n in seed['binaries']}
            commands=json.loads((build/'compile_commands.json').read_text())
            r['compile_commands_sha256']=sha(build/'compile_commands.json');r['compile_flag_proof']={}
            for name in ['src/prefill/prefill.cpp','src/core/expert_source.cpp','src/program/generate.cpp']:
                entries=[c for c in commands if c['file']==str(P/name)];assert len(entries)==1
                assert '-fsycl-id-queries-range=size_t' in entries[0]['command']
                r['compile_flag_proof'][name]=entries[0]
            assert all(sha(P/n)==h for n,h in r['source_pins'].items())
            r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error);raise
        finally:
            r['active']=False;persist();(D/'boundary-service-build-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],binaries=r.get('binaries'),GPU_executed=False)),flush=True)

if __name__=='__main__': main()
