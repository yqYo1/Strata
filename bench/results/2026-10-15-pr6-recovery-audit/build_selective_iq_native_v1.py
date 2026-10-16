#!/usr/bin/env python3
"""Build actual selective-IQ product objects, public fixture and native separately."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
BASE=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(HERE/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
    module=load_owner(PROJECT/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    root=BASE/'main-selective-iq-native-build-root-v1'
    baseline=BASE/'main-primary-native-build-root-v1/build'
    with (BASE/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        health=json.loads((HERE/'iq-timing-progress-v4.json').read_text())
        require(health['passed'] and health['complete'] and not health['active'] and not health['fresh_faults'],'progress diagnostic gate')
        old=json.loads((HERE/'parallel-iq-component-build-v1.json').read_text())
        census=json.loads((HERE/'next-diagnostics-build-v1.json').read_text())
        require(census['passed'] and not census['active'],'census compile gate')
        prefill_obj=Path(census['census_object'])
        require(sha(prefill_obj)==census['census_object_sha256'],'compiled census changed')
        for n,h in census['source_pins'].items():
            require(sha(PROJECT/n)==h,'reviewed input changed: '+n)
        libraries=[baseline/n for n in ['libstrata_prefill.a','libstrata_spec.a','libstrata_engine.a',
                   'libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a','libstrata_kernels_cpu.a',
                   'ggml/src/libggml-cpu.a','ggml/src/libggml-base.a']]
        gen_obj=baseline/'CMakeFiles/strata.dir/src/program/generate.cpp.o'
        mkl=Path('/opt/intel/oneapi/mkl/2026.1')
        mkl_libs=[mkl/('lib/'+n) for n in ['libmkl_sycl_blas.so','libmkl_intel_ilp64.so','libmkl_sequential.so','libmkl_core.so']]
        source=PROJECT/'src/kernels/sycl/iq.cpp'
        header=PROJECT/'include/strata/sycl/iq_parallel_decode.hpp'
        fixture=HERE/'parallel-iq-dequant-v1/gpu_fixture.cpp'
        require(sha(header)==sha(HERE/'parallel-iq-dequant-v1/proposed/include/strata/sycl/iq_parallel_decode.hpp'),'host-tested decoder header differs')
        require(not root.exists(),'fresh build root')
        root.mkdir(mode=0o700)
        record=dict(schema='main-selective-iq-native-build-v1',passed=False,complete=False,active=True,
                    GPU_executed=False,commands=[],source_pins={**census['source_pins'],str(source.relative_to(PROJECT)):sha(source),
                    str(header.relative_to(PROJECT)):sha(header),str(fixture.relative_to(PROJECT)):sha(fixture)},
                    recipe_sha256=sha(Path(__file__)),environment=old['environment'],compiler=old['compiler'],
                    compiler_sha256=sha(Path(old['compiler'])),linked_inputs={str(p):sha(p) for p in [prefill_obj,gen_obj,*libraries,*mkl_libs]},
                    baseline_binary_sha256=sha(baseline/'strata'),default_enabled=False,
                    selector='STRATA_SYCL_PARALLEL_IQ_DEQUANT=1; GU21/22/23 and generic20/42 only',performance_adopted=False)

        def persist():
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')

        owner=module.Owner(root/'owned',persist)
        record['commands']=owner.commands
        cc=json.loads((baseline/'compile_commands.json').read_text())
        matches=[c for c in cc if c['file']==str(source)]
        require(len(matches)==1,'actual IQ source resolver')
        argv=[c for c in shlex.split(matches[0]['command']) if not c.startswith('-DSTRATA_VERSION=')]
        iq_obj=root/'iq-role-selective.o'
        argv[argv.index('-o')+1]=str(iq_obj)
        common=[old['compiler'],'-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-id-queries-range=size_t',
                '-fno-fast-math','-ffp-contract=off','-fsycl-allow-device-image-dependencies']
        tail=['-Wl,-rpath='+str(mkl/'lib'),*map(str,mkl_libs),'-lm','-ldl','-lpthread']
        fixture_exe=root/'selective_iq_fixture'
        native=root/'strata'
        fixture_cmd=[*common,'-DSTRATA_ENABLE_SYCL=1','-I'+str(PROJECT/'include'),
                     '-I'+str(PROJECT/'include/strata/sycl/compat'),str(fixture),str(iq_obj),
                     *map(str,[baseline/'libstrata_kernels.a',baseline/'libstrata_core.a',baseline/'libstrata_sycl_runtime.a']),
                     '-Wl,-Map='+str(root/'fixture.map'),'-o',str(fixture_exe)]
        native_cmd=[*common,str(gen_obj),str(prefill_obj),str(iq_obj),*map(str,libraries),
                    '-Wl,-Map='+str(root/'native.map'),*tail,'-o',str(native)]
        try:
            for label,cmd in [('compile-selective-product-IQ',argv),('link-public-fixture',fixture_cmd),('link-native',native_cmd)]:
                c,out,err=owner.run(label,cmd,old['environment'],baseline,wall=600,cpu=600*len(os.sched_getaffinity(0)),
                                   rss_cap=32<<30,text_cap=8<<20,total_cap=2<<30,file_cap=512<<20)
                require(module.closed(c) and c['exit_code']==0,label+' failed: '+err.read_text()[-4000:])
            for name in ['fixture.map','native.map']:
                text=(root/name).read_text()
                require('iq.cpp.o' not in text,'old IQ archive selected')
                if name=='native.map':require('prefill.cpp.o' not in text,'old prefill archive selected')
            require(all(sha(PROJECT/n)==h for n,h in record['source_pins'].items()),'source changed during build')
            record.update(passed=True,complete=True,binary=str(native),binary_sha256=sha(native),
                          fixture=str(fixture_exe),fixture_sha256=sha(fixture_exe),iq_object_sha256=sha(iq_obj),
                          link_map_hashes={n:sha(root/n) for n in ['fixture.map','native.map']})
        except BaseException as error:
            record['error']=repr(error)
            raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            persist()
            (HERE/'selective-iq-native-build-v1.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(passed=record['passed'],GPU_executed=False)),flush=True)


if __name__=='__main__':
    main()
