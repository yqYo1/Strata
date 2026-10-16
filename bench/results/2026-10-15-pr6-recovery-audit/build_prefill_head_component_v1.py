#!/usr/bin/env python3
"""Link the isolated head fixture to the newly built production libraries."""
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
    root=B/'main-prefill-head-component-build-root-v1'
    receipt=A/'prefill-head-component-build-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not root.exists() and not receipt.exists(),'fresh component build')
        source=A/'prefill_head_roundtrip_fixture_v1.cpp'
        seed_path=A/'prefill-head-release-build-v1.json';d=json.loads(seed_path.read_text())
        require(d['passed'] and d['complete'] and not d['active'],'complete product build')
        require(all(sha(P/n)==h for n,h in d['source_pins'].items()),'exact product source')
        build=Path(d['build_path']);compiler=d['compiler']
        require(sha(Path(compiler))==d['compiler_sha256'],'compiler identity')
        require(all(sha(build/n)==h for n,h in d['binaries'].items()),'production library identities')
        root.mkdir(mode=0o700);exe=root/'head-roundtrip'
        libraries=[build/n for n in ['libstrata_prefill.a','libstrata_spec.a','libstrata_engine.a',
            'libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a','libstrata_kernels_cpu.a',
            'ggml/src/libggml-cpu.a','ggml/src/libggml-base.a']]
        mkl=Path('/opt/intel/oneapi/mkl/2026.1/lib')
        libraries += [mkl/n for n in ['libmkl_sycl_blas.so','libmkl_intel_ilp64.so','libmkl_sequential.so','libmkl_core.so']]
        command=[compiler,'-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-id-queries-range=size_t',
            '-fno-fast-math','-ffp-contract=off','-fsycl-allow-device-image-dependencies','-DSTRATA_ENABLE_SYCL=1',
            '-Wall','-Wextra','-Werror','-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),
            str(source),*map(str,libraries),'-Wl,-rpath='+str(mkl),'-lm','-ldl','-lpthread','-o',str(exe)]
        r=dict(passed=False,complete=False,active=True,GPU_executed=False,
            recipe_sha256=sha(Path(__file__)),source_sha256=sha(source),product_build_sha256=sha(seed_path),
            compiler=compiler,compiler_sha256=sha(Path(compiler)),environment=d['environment'],commands=[],
            linked_inputs={str(p):sha(p) for p in libraries},scope='Build-only isolated NativeHead production API fixture. No runtime or model qualification.')
        def persist():(root/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        owner=module.Owner(root/'owned',persist);r['commands']=owner.commands;persist()
        try:
            c,out,err=owner.run('link-head-component',command,d['environment'],P,wall=300,cpu=1200,
                rss_cap=16<<30,text_cap=4<<20,total_cap=512<<20,file_cap=128<<20)
            require(module.closed(c) and c['exit_code']==0,'component link: '+err.read_text()[-5000:])
            require(sha(source)==r['source_sha256'],'fixture source changed during build')
            r.update(passed=True,complete=True,binary=str(exe),binary_sha256=sha(exe))
        except BaseException as e:r['error']=repr(e);raise
        finally:
            r['active']=owner.active is not None;r['finished_utc']=module.utc();persist()
            receipt.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],GPU_executed=False,binary_sha256=r.get('binary_sha256'))))


if __name__=='__main__':main()
