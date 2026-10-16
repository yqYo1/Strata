#!/usr/bin/env python3
"""Main-owned isolated subgroup fixture/IR build; never changes product source."""
import fcntl
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
BASE=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(HERE/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
    module=load_owner(PROJECT/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    proposal=HERE/'iq3xxs-subgroup-materializer-v2'
    root=BASE/'main-iq3xxs-subgroup-build-root-v2'
    # A live measurement owns this same lock; contention stops before any build.
    with (BASE/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for relative,pin in json.loads((proposal/'source-pins.json').read_text()).items():
            require(sha(PROJECT/relative)==pin,'subgroup reviewed base changed: '+relative)
        for relative,pin in json.loads((proposal/'artifact-sha256.json').read_text()).items():
            require(sha(proposal/relative)==pin,'subgroup proposal changed: '+relative)
        baseline=json.loads((HERE/'selective-iq-native-build-v1.json').read_text())
        public=json.loads((HERE/'selective-iq-component-full-v1.json').read_text())
        model=json.loads((HERE/'selective-iq-model-assessment-v2.json').read_text())
        for gate in [baseline,public,model]:
            require(gate['passed'] and gate['complete'] and not gate['active'],'closed selective baseline gate')
        old=json.loads((HERE/'parallel-iq-component-build-v1.json').read_text())
        require(all(sha(Path(p))==h for p,h in old['linked_archives'].items()),'baseline archive changed')
        compiler=Path(baseline['compiler'])
        require(sha(compiler)==baseline['compiler_sha256'],'compiler changed')
        require(not root.exists(),'fresh build root required')
        root.mkdir(mode=0o700)
        sources=[proposal/'gpu_fixture.cpp',proposal/'proposed/src/kernels/sycl/iq.cpp',
                 proposal/'proposed/include/strata/sycl/iq3xxs_subgroup.hpp',
                 HERE/'parallel-iq-dequant-v1/fixture_common.hpp',PROJECT/'include/strata/sycl/iq_parallel_decode.hpp']
        source_pins={str(p.relative_to(PROJECT)):sha(p) for p in sources}
        record=dict(schema='main-iq3xxs-subgroup-build-v2',passed=False,complete=False,active=True,
                    GPU_executed=False,commands=[],source_pins=source_pins,
                    reviewed_base_pins=json.loads((proposal/'source-pins.json').read_text()),
                    linked_archives=old['linked_archives'],compiler=str(compiler),compiler_sha256=sha(compiler),
                    environment=baseline['environment'],recipe_sha256=sha(Path(__file__)),
                    product_applied=False,performance_adopted=False,IR_review_complete=False,
                    scope='Separate candidate public-entrypoint fixture and device IR only. No GPU/model/speed qualification.')

        def persist():
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')

        owner=module.Owner(root/'owned',persist)
        record['commands']=owner.commands
        flags=[str(compiler),'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1',
               '-fsycl','-fsycl-id-queries-range=size_t','-fno-fast-math','-ffp-contract=off',
               '-Wall','-Wextra','-Werror','-I'+str(proposal/'proposed/include'),
               '-I'+str(PROJECT/'include'),'-I'+str(PROJECT/'include/strata/sycl/compat')]
        exe=root/'iq3xxs_subgroup_fixture'
        ir=root/'iq3xxs-subgroup.ll'
        commands=[('build-subgroup-fixture',[*flags,str(sources[0]),str(sources[1]),
                    *old['linked_archives'],'-fsycl-allow-device-image-dependencies',
                    '-Wl,-Map='+str(root/'link.map'),'-ldl','-lpthread','-o',str(exe)]),
                  ('emit-subgroup-device-IR',[*flags,'-fsycl-device-only','-S','-emit-llvm',str(sources[1]),'-o',str(ir)])]
        try:
            for label,argv in commands:
                c,out,err=owner.run(label,argv,baseline['environment'],PROJECT,wall=600,
                                   cpu=600*len(os.sched_getaffinity(0)),rss_cap=32<<30,
                                   text_cap=8<<20,total_cap=1<<30,file_cap=512<<20)
                require(module.closed(c) and c['exit_code']==0,label+' failed: '+err.read_text()[-4000:])
            require('iq.cpp.o' not in (root/'link.map').read_text(),'old IQ archive member selected')
            require(all(sha(PROJECT/p)==h for p,h in source_pins.items()),'candidate changed during build')
            require(all(sha(PROJECT/p)==h for p,h in record['reviewed_base_pins'].items()),'base changed during build')
            record.update(passed=True,complete=True,binary=str(exe),binary_sha256=sha(exe),
                          device_IR=str(ir),device_IR_bytes=ir.stat().st_size,device_IR_sha256=sha(ir),
                          link_map_sha256=sha(root/'link.map'))
        except BaseException as error:
            record['error']=repr(error)
            raise
        finally:
            record.update(active=owner.active is not None,finished_utc=module.utc())
            persist()
            (HERE/'iq3xxs-subgroup-build-v2.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(passed=record['passed'],GPU_executed=False,IR_review_complete=False)),flush=True)


if __name__=='__main__':
    main()
