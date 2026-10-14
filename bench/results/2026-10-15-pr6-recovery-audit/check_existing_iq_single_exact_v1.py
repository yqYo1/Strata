#!/usr/bin/env python3
"""Main CPU gate for existing IQ3_XXS/IQ3_S/IQ2_S exact helpers."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,sha

O=B/'main-primary-existing-iq-single-exact-root-v1'

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        closed=json.loads((D/'decode-format-service-normal-v1.json').read_text())
        assert closed['passed'] and closed['complete'] and not closed['active'] and not closed['fresh_faults']
        assert all(c['direct_child_reaped'] and c['session_empty'] and c['observation_complete'] for c in closed['commands'])
        seed=json.loads((D/'boundary-service-build-v1.json').read_text());build=Path(seed['build_path'])
        assert all(sha(P/n)==h for n,h in seed['source_pins'].items())
        exe=build/'strata';before=sha(exe);assert before==seed['binaries']['strata']
        assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
        spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for c in closed['commands']:
            for i in c['owners']:
                now=m.identity(i['pid']);assert now is None or now['start_ticks']!=i['start_ticks'] or now['state']=='Z'
        assert not O.exists();O.mkdir(mode=0o700)
        env=dict(seed['environment']);assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        r=dict(active=True,passed=False,complete=False,GPU_executed=False,product_changed=False,commands=[],environment=env,
            recipe_sha256=sha(Path(__file__)),source_pins=seed['source_pins'],strata_sha256=before,
            scope='Existing exact CPU helper384GGML parity cases for types18,21,22/length256..2560/row canaries. Model setting and performance unqualified.')
        def persist():(O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=m.Owner(O/'owned',persist);r['commands']=owner.commands;persist()
        def run(label,args,wall):
            e,out,err=owner.run(label,args,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                text_cap=64<<20,rss_cap=8<<30,total_cap=256<<20,file_cap=64<<20)
            assert m.closed(e) and e['exit_code']==0,(label,err.read_text()[-4000:],out.read_text()[-2000:])
            return out,err
        try:
            run('build-existing-exact',['cmake','--build',str(build),'--target','iq_single_exact_test','--parallel','2'],300)
            out,err=run('run-existing-exact',[str(build/'iq_single_exact_test')],30)
            assert out.read_text()=='IQ single exact PASS: 384 cases, bitwise GGML parity and row canaries\n'
            r['stdout']=out.read_text();r['stderr']=err.read_text();r['test_binary_sha256']=sha(build/'iq_single_exact_test')
            commands=json.loads((build/'compile_commands.json').read_text())
            r['compile_flag_proof']=[c for c in commands if c['file']==str(P/'src/kernels/cpu/iq_single_avx2.cpp')]
            assert len(r['compile_flag_proof'])==1 and '-march=znver3' in r['compile_flag_proof'][0]['command']
            assert sha(exe)==before and all(sha(P/n)==h for n,h in seed['source_pins'].items())
            r.update(passed=True,complete=True)
        except BaseException as error:r['error']=repr(error);raise
        finally:
            r['active']=False;persist();(D/'existing-iq-single-exact-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=True,cases=384,strata_unchanged=True,GPU_executed=False)),flush=True)

if __name__=='__main__':main()
