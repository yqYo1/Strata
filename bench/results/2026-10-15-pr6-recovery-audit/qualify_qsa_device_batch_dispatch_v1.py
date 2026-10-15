#!/usr/bin/env python3
"""Link actual production dispatch and test immutable flags in fresh processes."""
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha


def main():
    root=B/'main-qsa-device-batch-dispatch-root-v1';target=A/'qsa-device-batch-dispatch-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(),'fresh production dispatch qualification')
        build_path=A/'qsa-device-batch-build-v1.json';build=json.loads(build_path.read_text())
        matrix_path=A/'qsa-device-batch-matrix-v1.json';matrix=json.loads(matrix_path.read_text())
        require(all(r['passed'] and r['complete'] and not r['active'] for r in (build,matrix)),'actual build/matrix gates')
        module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for r in (build,matrix):
            for c in r['commands']:
                require(module.closed(c),'previous command closure')
                for old in c['owners']:
                    now=module.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'previous owner still present')
        require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'qualified actual source')
        b=Path(build['build_path']);compiler=build['compiler']
        libraries=[b/n for n in ['libstrata_prefill.a','libstrata_spec.a','libstrata_engine.a',
            'libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a','libstrata_kernels_cpu.a',
            'ggml/src/libggml-cpu.a','ggml/src/libggml-base.a']]
        mkl=Path('/opt/intel/oneapi/mkl/2026.1/lib')
        libraries += [mkl/n for n in ['libmkl_sycl_blas.so','libmkl_intel_ilp64.so','libmkl_sequential.so','libmkl_core.so']]
        source=A/'qsa_device_batch_dispatch_fixture_v1.cpp';root.mkdir(mode=0o700);exe=root/'dispatch'
        argv=[compiler,'-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-id-queries-range=size_t',
            '-fsycl-allow-device-image-dependencies','-fno-fast-math','-ffp-contract=off',
            '-DSTRATA_ENABLE_SYCL=1','-Wall','-Wextra','-Werror','-I'+str(P/'include'),
            '-I'+str(P/'include/strata/sycl/compat'),str(source),*map(str,libraries),
            '-Wl,-rpath='+str(mkl),'-Wl,-Map='+str(exe)+'.map','-lm','-ldl','-lpthread','-o',str(exe)]
        r=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],samples=[],
            recipe_sha256=sha(Path(__file__)),references={p.name:sha(p) for p in (build_path,matrix_path)},
            fixture_sha256=sha(source),source_pins=build['source_pins'],compiler=compiler,
            compiler_sha256=sha(Path(compiler)),linked_inputs={str(p):sha(p) for p in libraries},
            scope='Production append_batch zero-row dispatch under10 immutable environment processes; no queue, device allocation or kernel launch allowed. Not positive-batch mechanics, model, RESTORE, quality or speed qualification.')
        def persist():(root/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);r['commands']=owner.commands;persist()
        try:
            c,out,err=owner.run('link-production-dispatch',argv,build['environment'],P,
                wall=300,cpu=1800,rss_cap=16<<30,text_cap=8<<20,total_cap=1<<30,file_cap=256<<20)
            require(module.closed(c) and c['exit_code']==0,'link: '+err.read_text()[-5000:])
            linkmap=Path(str(exe)+'.map').read_text()
            require('native_qsa_indexer.cpp.o' in linkmap,'actual production indexer object selected')
            r.update(binary=str(exe),binary_sha256=sha(exe),link_map_sha256=sha(Path(str(exe)+'.map')))
            for i,flag in enumerate([None,'0','1','','01','true','-1','2',' 1','1 ']):
                reject=flag not in (None,'0','1')
                env=dict(matrix['environment']);env.pop('STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH',None)
                if flag is not None:env['STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH']=flag
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                    UR_LOG_TRACING='level:info;flush:info;output:stderr',XDG_CACHE_HOME=str(root/'cache'))
                c,out,err=owner.run('production-dispatch-'+str(i),[str(exe),'reject' if reject else 'noop'],
                    env,P,wall=60,cpu=120,rss_cap=2<<30,text_cap=4<<20,total_cap=1<<30,file_cap=256<<20)
                text=err.read_text();sample=dict(flag=flag,expected_rejection=reject,exit_code=c.get('exit_code'),
                    stdout=out.read_text(),stderr_bytes=err.stat().st_size,stderr_sha256=sha(err))
                r['samples'].append(sample);persist()
                require(module.closed(c) and c['exit_code']==0 and sample['stdout'].startswith('PASS production dispatch '),'actual dispatch: '+text[-5000:])
                require(not re.search(r'---> ur(?:ContextCreate|QueueCreate|USM\w+|Enqueue\w+)|UR_RESULT_ERROR_',text),
                    'zero/invalid dispatch initialized or submitted device work')
                sample['passed']=True;persist()
            require(all(sha(P/n)==h for n,h in r['source_pins'].items()),'source stability')
            r.update(passed=True,complete=True)
        except BaseException as e:
            r['error']=repr(e);raise
        finally:
            r.update(active=owner.active is not None,finished_utc=module.utc());persist()
            target.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],processes=len(r['samples']),GPU_executed=False)))


if __name__=='__main__':main()
