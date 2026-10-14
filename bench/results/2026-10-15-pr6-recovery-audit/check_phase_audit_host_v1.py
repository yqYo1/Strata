#!/usr/bin/env python3
"""Main CPU checks of the unapplied phase helper and exact comparator."""
import fcntl
import importlib.util
import json
import os
import re
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,sha

O=B/'main-primary-phase-audit-host-root-v1'
C=D/'layer26-phase-audit-v1'

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        seed=json.loads((D/'boundary-service-build-v1.json').read_text())
        closed=json.loads((D/'decode-format-existing-exact-normal-v1.json').read_text())
        assert closed['passed'] and closed['complete'] and not closed['active'] and not closed['fresh_faults']
        assert all(sha(P/n)==v for n,v in seed['source_pins'].items())
        assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
        spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for c in closed['commands']:
            assert m.closed(c)
            for i in c['owners']:
                now=m.identity(i['pid']);assert now is None or now['start_ticks']!=i['start_ticks'] or now['state']=='Z'
        assert not O.exists();O.mkdir(mode=0o700)
        env=dict(seed['environment']);assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        r=dict(active=True,passed=False,complete=False,GPU_executed=False,product_patch_applied=False,
            commands=[],environment=env,recipe_sha256=sha(Path(__file__)),source_pins=seed['source_pins'],
            fixture_sha256=sha(D/'phase-audit-main-fixture-v1.cpp'),header_sha256=sha(C/'layer26_phase_audit.hpp.in'),
            comparator_sha256=sha(C/'compare.py'),comparisons=[],scope='CPU helper/negative comparator only; no product SYCL build, actual phase tensors or GPU DMA lifetime qualification.')
        def persist():(O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=m.Owner(O/'owned',persist);r['commands']=owner.commands;persist()
        def run(label,args,expected=0,wall=60):
            e,out,err=owner.run(label,args,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                rss_cap=8<<30,text_cap=64<<20,total_cap=256<<20,file_cap=64<<20)
            assert m.closed(e) and e['exit_code']==expected,(label,e.get('error'),err.read_text()[-4000:],out.read_text()[-2000:])
            return out,err
        try:
            r['fixtures']={}
            for sanitize in [False,True]:
                label='asan-ubsan' if sanitize else 'normal';exe=O/label;folder=O/(label+'-data');folder.mkdir(mode=0o700)
                flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitize else ['-O2']
                run('compile-'+label,['/usr/bin/c++','-std=c++20','-UNDEBUG','-Wall','-Wextra','-Werror',*flags,
                    '-I'+str(D),str(D/'phase-audit-main-fixture-v1.cpp'),'-o',str(exe)])
                out,err=run('run-'+label,[str(exe),str(folder)])
                match=re.search(r'artifacts=(.+)',out.read_text());assert match
                r['fixtures'][label]=dict(path=match[1],stdout=out.read_text(),stderr=err.read_text(),binary_sha256=sha(exe));persist()
            normal=Path(r['fixtures']['normal']['path']);sanitized=Path(r['fixtures']['asan-ubsan']['path'])
            def compare(label,left,right,expected):
                out,err=run('compare-'+label,['/usr/bin/python3',str(C/'compare.py'),str(left),str(right)],expected)
                result=json.loads(out.read_text() if expected!=2 else err.read_text())
                assert result['coverage_complete']==(expected!=2)
                assert result['raw_memcmp']==('equal' if expected==0 else 'different' if expected==1 else 'not_compared')
                r['comparisons'].append(dict(label=label,expected_exit=expected,result=result));persist()
            compare('same-normal',normal/'case-0',normal/'case-0',0)
            compare('same-low',normal/'case-1',normal/'case-1',0)
            compare('cross-sanitized',normal/'case-0',sanitized/'case-0',0)
            text=(normal/'case-0').read_text();raw=Path(str(normal/'case-0')+'.raw').read_bytes()
            mutations={
                'changed-byte':(text,bytes([raw[0]^1])+raw[1:],1),
                'truncated':(text,raw[:-1],2),'extra-byte':(text,raw+b'x',2),
                'footer-missing':('\n'.join(text.splitlines()[:-1])+'\n',raw,2),
                'wrong-dtype':(text.replace('dtype=fp16','dtype=bf16',1),raw,2),
                'wrong-extent':(text.replace('bytes=5120','bytes=5119',1),raw,2),
                'duplicate-record':(text.replace(text.splitlines()[3]+'\n',text.splitlines()[3]+'\n'+text.splitlines()[3]+'\n',1),raw,2),
                'wrong-selection':(text.replace('selection=0 row=1','selection=0 row=2',1),raw,2),
                'wrong-offset':(text.replace('offset=0 ','offset=1 ',1),raw,2)}
            for label,(changed,body,expected) in mutations.items():
                f=O/('mutated-'+label);f.write_text(changed);Path(str(f)+'.raw').write_bytes(body)
                compare(label,normal/'case-0',f,expected)
            compare('low-schema-mismatch',normal/'case-0',normal/'case-1',2)
            assert all(sha(P/n)==v for n,v in seed['source_pins'].items())
            r.update(passed=True,complete=True)
        except BaseException as error:r['error']=repr(error);raise
        finally:
            r['active']=False;persist();(D/'phase-audit-host-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=True,GPU_executed=False,product_patch_applied=False,comparisons=len(r['comparisons']))),flush=True)

if __name__=='__main__':main()
