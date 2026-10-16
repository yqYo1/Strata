#!/usr/bin/env python3
"""Main serial build for checked temporary MTP expert retirement."""
import fcntl
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,sha

SEED=D/'current-diagnostics-build-v1.json'
CANDIDATE=D/'mtp-expert-prefill-release-v2'
O=B/'main-primary-mtp-expert-release-build-root-v1'
CHANGED={'include/strata/core/mtp.hpp','src/core/mtp.cpp','src/program/generate.cpp'}


def main():
    seed=json.loads(SEED.read_text());proposal=json.loads((CANDIDATE/'rebase-v3.json').read_text())
    assert seed['passed'] and seed['complete'] and not seed['active'] and proposal['complete']
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        closed=json.loads((D/'selected-prefill-long-v1.json').read_text())
        assert closed['passed'] and closed['complete'] and not closed['active'] and not closed['fresh_faults']
        assert all(c['direct_child_reaped'] and c['session_empty'] for c in closed['commands'])
        assert {n for n,v in seed['source_pins'].items() if sha(P/n)!=v}==CHANGED
        assert all(sha(P/n)==v['proposed_sha256'] for n,v in proposal['files'].items())
        assert sha(CANDIDATE/'source.diff')==proposal['source_diff_sha256']
        build=Path(seed['build_path']);assert all(sha(build/n)==pin for n,pin in seed['binaries'].items())
        assert not O.exists();O.mkdir(mode=0o700)
        env=dict(seed['environment']);assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        r=dict(active=True,passed=False,complete=False,GPU_executed=False,commands=[],environment=env,
            build_path=str(build),seed_receipt_sha256=sha(SEED),recipe_sha256=sha(Path(__file__)),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
            source_pins={n:sha(P/n) for n in sorted(set(seed['source_pins'])|set(proposal['files']))},
            source_patch_sha256=sha(CANDIDATE/'source.diff'),proposal_receipt_sha256=sha(CANDIDATE/'rebase-v3.json'),
            previous_binaries=seed['binaries'],control_binary=seed['control_binary'],
            host_contracts_passed=False,performance_qualified=False,quality_qualified=False,
            physical_262144_qualified=False,restore_qualified=False,adopted=False,
            scope='Opt-in675MiB MTP expert detach/restore with immutable675MiB RAM source, '
                  'checked normal/coupled graph retirement and actual VRAM telemetry. '
                  'Build and source-identity SHA CPU vectors only; actual retirement/restore/model '
                  'numerics, repeated serve, decode cost and performance require Main GPU gates.')
        def persist():(O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(O/'owned',persist);r['commands']=owner.commands;persist()
        def run(label,argv,wall):
            entry,out,err=owner.run(label,argv,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                text_cap=64 << 20,rss_cap=48 << 30,total_cap=1 << 30,file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code']==0,(label,entry.get('error'),err.read_text()[-6000:],out.read_text()[-4000:])
            return out,err
        try:
            run('build-mtp-expert-release',['cmake','--build',str(build),'--target','strata','--parallel','2'],1500)
            test=O/'mtp-expert-source-test'
            run('compile-mtp-source-vectors',['/usr/bin/c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                '-I'+str(P/'include'),str(P/'tools/test_mtp_expert_source.cpp'),'-o',str(test)],60)
            out,err=run('run-mtp-source-vectors',[str(test)],30)
            assert out.read_text()=='PASS MTP expert source SHA256 known-answer streaming vectors\n'
            r['host_contracts_passed']=True;r['host_test_output']=out.read_text();r['host_test_binary_sha256']=sha(test)
            r['binaries']={n:sha(build/n) for n in seed['binaries']}
            commands=json.loads((build/'compile_commands.json').read_text());r['compile_flag_proof']={}
            for n in ['src/core/mtp.cpp','src/program/generate.cpp']:
                vals=[c for c in commands if c['file']==str(P/n)];assert len(vals)==1
                assert '-fsycl-id-queries-range=size_t' in vals[0]['command']
                r['compile_flag_proof'][n]=vals[0]
            r['compile_commands_sha256']=sha(build/'compile_commands.json')
            assert all(sha(P/n)==v for n,v in r['source_pins'].items())
            r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error);raise
        finally:
            r['active']=False;persist();(D/'mtp-expert-release-build-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=True,GPU_executed=False,binaries=r['binaries'])),flush=True)


if __name__=='__main__':main()
