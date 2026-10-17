#!/usr/bin/env python3
"""Main-owned build of the unchanged QSA scorer/selector stage timing fixture."""
import datetime, fcntl, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
spec = __import__('importlib.util', fromlist=['util'])
s = spec.spec_from_file_location('qsa_stage_build_base', A / 'build_qsa_prefix16_v1.py')
base = spec.module_from_spec(s); s.loader.exec_module(base)
P, B, L, pin, module = base.P, base.B, base.L, base.pin, base.module

def main():
    root = B / 'main-qsa-select-stage-build-v1'
    dst = A / 'qsa-select-stage-build-v1.json'
    assert not root.exists() and not dst.exists()
    prior_path = A / 'qsa-prefix16-build-v1.json'
    prior = json.loads(prior_path.read_text())
    assert prior['passed'] and prior['complete'] and prior['ownership_closed'] and not prior['active']
    engine_path = A / 'moe-accum-engine-build-v2.json'
    engine = json.loads(engine_path.read_text())
    assert engine['passed'] and engine['complete'] and engine['ownership_closed']
    kernel_source = P / 'src/kernels/sycl/qsa_select.cpp'
    assert pin(kernel_source)['sha256'] == '377ad7713ebf9af859cd02614871de705a998af3f5f5cf5f551b881a43d29bfa'
    compiler = pin(L / 'bin/clang++')
    assert compiler['sha256'] == '1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb'
    libs = [Path(engine['binary']['path']).parent / n for n in ('libstrata_kernels.a', 'libstrata_core.a', 'libstrata_sycl_runtime.a')]
    for p in libs: assert pin(p) == prior['frozen_files'][str(p)]
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert pin(owner_path)['sha256'] == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = module('qsa_stage_build_owner', owner_path)
    package = A / 'own-vllm-xe2-qualification-v1'
    meta = json.loads((package / 'manifest.json').read_text())
    guard_path = package / 'main_controller.py'
    assert pin(guard_path)['sha256'] == meta['package_files']['main_controller.py']['sha256']
    guards = module('qsa_stage_build_guards', guard_path)
    fixture = P / 'tests/sycl/qsa_select_stage_bench.cpp'
    inputs = [Path(__file__), A/'build_qsa_prefix16_v1.py', engine_path, prior_path, owner_path, guard_path,
              kernel_source, fixture, L/'bin/clang++', *libs, *list((P/'include').rglob('*.hpp'))]
    frozen = {str(p):pin(p) for p in inputs}
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        guards.no_model_owners(meta['known_model_files'])
        root.mkdir(mode=0o700); (root/'tmp').mkdir()
        env = dict(PATH=str(L/'bin')+':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                   PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(L/'lib'),TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
        rec = dict(schema='strata-qsa-select-stage-build-v1',passed=False,complete=False,active=False,GPU_executed=False,
                   frozen_files=frozen,compiler=compiler,environment=env,commands=[],products={},
                   provenance='Main-authored fixture using repository Runtime/Allocation and public QSA APIs; independent FP64 sampled scorer diagnostic and cell-ranking top-k oracle. No XeStrata implementation used.',
                   scope='Link the exact existing FP32 scorer and radix4 top-k from the measured engine archives; no product kernel changes.')
        owner=None
        def save():
            rec['active']=bool(owner and owner.active is not None)
            dst.write_text(json.dumps(rec,indent=2)+'\n')
        owner=owned.Owner(root/'owned',save);rec['commands']=owner.commands
        def verify():
            for p,v in frozen.items(): assert pin(p)==v,p
        def run(name,argv):
            c,so,se=owner.run(name,argv,env,P,wall=300,cpu=900,rss_cap=12<<30,text_cap=16<<20,file_cap=256<<20,total_cap=2<<30)
            assert owned.closed(c) and c['exit_code']==0,(name,se.read_text()[-6000:])
            return so.read_text()
        try:
            verify()
            exe=root/'sycl_qsa_select_stage_bench'
            flags=[compiler['path'],'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1',
                   '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-fno-fast-math','-ffp-contract=off',
                   '-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror']
            run('link-fixture',flags+[str(fixture),*map(str,libs),'-o',str(exe)])
            rec['binary']=pin(exe)
            rec['dynamic']=run('readelf',['/usr/bin/readelf','-d',str(exe)])
            verify();rec.update(passed=True,complete=True)
        except BaseException as e:
            import traceback
            rec.update(error=repr(e),traceback=traceback.format_exc())
        finally:
            rec['ownership_closed']=owner.active is None and all(owned.closed(c) for c in owner.commands)
            rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps(dict(receipt=str(dst),passed=rec['passed'],error=rec.get('error'))),flush=True)
        return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
