#!/usr/bin/env python3
"""Main-owned build of the independently authored private GDN chunk64 candidate."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

if not __debug__:
    raise RuntimeError('Qualification requires Python assertions')
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
L=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
HOST=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
def pin(path):
    path=Path(path)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),realpath=str(path.resolve()),bytes=path.stat().st_size,sha256=digest)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def main():
    assert len(sys.argv)==1
    root=B/'main-gdn-chunk64-build-v2';destination=A/'own-gdn-chunk64-build-v2.json'
    assert not root.exists() and not destination.exists()
    package=A/'own-gdn-chunk64-v2';manifest_path=package/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    owner_path=A/'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256']=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owner_module=module('chunk64_build_owner',owner_path)
    guards_package=A/'own-vllm-xe2-qualification-v1';guard_meta=guards_package/'manifest.json'
    assert pin(guard_meta)['sha256']=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    meta=json.loads(guard_meta.read_text());guard_source=guards_package/'main_controller.py'
    assert pin(guard_source)['sha256']==meta['package_files']['main_controller.py']['sha256']
    guards=module('chunk64_build_guards',guard_source)
    compiler=pin(L/'bin/clang++')
    assert compiler['sha256']=='1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
    engine_path=A/'qsa-score-gemm-entry-engine-build-v2.json'
    engine=json.loads(engine_path.read_text())
    assert engine['passed'] and engine['complete'] and engine['ownership_closed'] and not engine['active']
    library=Path(engine['binary']['path']).parent/'libstrata_sycl_runtime.a'
    # This fixture needs no engine/kernel archive: only current project runtime.
    paths=[Path(__file__),manifest_path,owner_path,guard_meta,guard_source,engine_path,library,L/'bin/clang++']
    paths += [package/name for name in manifest['files']]
    paths += sorted((P/'include/strata/sycl').rglob('*.hpp'))
    paths += [P/'include/strata/kernels/bf16_bits.hpp',P/'include/strata/kernels/f16_bits.hpp',P/'include/strata/prefill/kernels.hpp']
    frozen={str(path):pin(path) for path in paths}
    for name,expected in manifest['files'].items():assert pin(package/name)['sha256']==expected['sha256']
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resource_guard=guards.no_model_owners(meta['known_model_files'])
        root.mkdir(mode=0o700);(root/'source').mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
        for name in manifest['files']:
            shutil.copyfile(package/name,root/'source'/name)
            assert pin(root/'source'/name)['sha256']==manifest['files'][name]['sha256']
            frozen[str(root/'source'/name)]=pin(root/'source'/name)
        record=dict(schema='strata-gdn-chunk64-build-v2',passed=False,complete=False,active=False,
                    GPU_executed=False,commands=[],frozen_files=frozen,compiler=compiler,
                    source_manifest=pin(manifest_path),base_commit=manifest['base_commit'],
                    source_snapshot={name:pin(root/'source'/name) for name in manifest['files']},
                    library_inputs={str(library):pin(library)},resource_guard=resource_guard,
                    provenance='Main-authored new Strata source and fixture; mathematical chunk design references pinned separately. No external kernel source copied. Intel compiler installation path is dependency assets only; no XeStrata source.',
                    scope='Compile and link private BF16 GDN chunk64 path. No product/source/default or model/runtime qualification.',
                    production_selected=False,model_execution=False,model_PP_measured=False,
                    physical262144_qualified=False,RESTORE_qualified=False)
        owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=owner_module.Owner(root/'owned',persist);record['commands']=owner.commands
        def verify():
            for path,value in frozen.items():assert pin(path)==value,path
        try:
            verify()
            binary=root/'sycl_gdn_chunk64'
            env={**HOST,'PATH':str(L/'bin')+':/usr/bin:/bin','LD_LIBRARY_PATH':str(L/'lib'),
                 'TMPDIR':str(root/'tmp'),'XDG_CACHE_HOME':str(root/'cache')}
            argv=[compiler['path'],'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1',
                  '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),
                  '-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t',
                  '-Wall','-Wextra','-Werror',str(root/'source/candidate.cpp'),
                  str(root/'source/fixture.cpp'),str(library),'-o',str(binary)]
            command,out,err=owner.run('compile-link',argv,env,P,wall=300,cpu=900,
                                      rss_cap=8<<30,text_cap=64<<20,file_cap=128<<20,total_cap=1<<30)
            assert owner_module.closed(command)
            record.update(compile_stdout=out.read_text(),compile_stderr=err.read_text())
            assert command['exit_code']==0,'compiler exit'
            verify();record.update(binary=pin(binary),passed=True,complete=True)
        except BaseException as error:
            import traceback
            record.update(error=repr(error),traceback=traceback.format_exc())
        finally:
            record['ownership_closed']=owner.active is None and all(owner_module.closed(c) for c in owner.commands)
            if not record['ownership_closed']:record.update(passed=False,complete=False)
            persist();destination.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(destination),passed=record['passed'],complete=record['complete'],error=record.get('error'))))
        return 0 if record['passed'] else 1
if __name__=='__main__':sys.exit(main())
